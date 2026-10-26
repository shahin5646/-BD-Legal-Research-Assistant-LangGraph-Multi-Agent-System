"""Graph node functions for BD Legal Research Assistant.

Contains all node handlers for:
  - Analyst creation & human review
  - Interview conduction (questions, answers, web search with counter-arguments)
  - Legal memo section writing with confidence scoring & cross-verification
  - Report assembly (body, introduction, conclusion, final assembly)
  - Analyst debate & final critique
"""

import os
import logging
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from langgraph.types import interrupt
from langchain_tavily import TavilySearch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import openai
import httpx
from pydantic import BaseModel, Field

from src.utils.models import get_llm
from src.utils.objects import Analyst, Perspectives
from src.utils.states import (
    GenerateAnalystsState,
    InterviewState,
    ResearchGraphState,
    SearchQuery,
    NoveltyAssessment,
)
from src.utils.prompts import (
    analyst_instructions,
    question_instructions,
    answer_instructions,
    search_instructions,
    counter_search_instructions,
    section_writer_instructions,
    report_writer_instructions,
    intro_conclusion_instructions,
)
from src.utils.security import security
from src.utils.scoring import score_section, format_confidence_badge, format_reliability_footer
from src.utils.verification import extract_claims, verify_claim, format_verification_markers
from src.utils.debate import run_debate, format_debate_section
from src.utils.cost_tracker import tracker

logger = logging.getLogger(__name__)

# Feature flags (configurable via environment variables)
_ENABLE_DEBATE = os.getenv("ENABLE_DEBATE", "true").lower() in ("true", "1", "yes")
_ENABLE_SCORING = os.getenv("ENABLE_SCORING", "true").lower() in ("true", "1", "yes")
_ENABLE_VERIFICATION = os.getenv("ENABLE_VERIFICATION", "true").lower() in ("true", "1", "yes")
_ENABLE_CRITIQUE = os.getenv("ENABLE_CRITIQUE", "true").lower() in ("true", "1", "yes")
_ENABLE_NOVELTY_CHECK = os.getenv("ENABLE_NOVELTY_CHECK", "true").lower() in ("true", "1", "yes")

# Shared helpers
_TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
tavily_search = TavilySearch(max_results=_TAVILY_MAX_RESULTS)
_search_system_message = SystemMessage(content=search_instructions)


def _is_transient_error(exc: BaseException) -> bool:
    """Predicate to determine if an exception is transient and safe to retry.

    Explicitly excludes 400, 401, 403, 404 client errors.
    """
    if isinstance(exc, (
        openai.AuthenticationError,
        openai.BadRequestError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
    )):
        return False

    if isinstance(exc, (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
        httpx.TimeoutException,
        httpx.NetworkError,
    )):
        return True

    msg = str(exc)
    if any(code in msg for code in ("429", "500", "502", "503", "504")):
        return True

    return False


@retry(
    stop=stop_after_attempt(int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))),
    wait=wait_exponential(
        multiplier=float(os.getenv("RETRY_INITIAL_INTERVAL", "1.0")),
        max=30,
    ),
    retry=retry_if_exception(_is_transient_error),
    before_sleep=lambda rs: logger.warning(
        "LLM call failed (attempt %d), retrying…", rs.attempt_number
    ),
)
def _invoke_llm(messages, *, structured=None):
    """Invoke the LLM with automatic retry on transient failures."""
    target = structured if structured else get_llm()
    response = target.invoke(messages)
    if structured and response is None:
        raise ValueError("LLM failed to produce structured output — retrying")
    return response


@retry(
    stop=stop_after_attempt(int(os.getenv("SEARCH_RETRY_ATTEMPTS", "4"))),
    wait=wait_exponential(
        multiplier=float(os.getenv("RETRY_INITIAL_INTERVAL", "1.0")),
        max=30,
    ),
    retry=retry_if_exception(_is_transient_error),
    before_sleep=lambda rs: logger.warning(
        "Search failed (attempt %d), retrying…", rs.attempt_number
    ),
)
def _search_tavily(query: str) -> list[dict]:
    """Execute a Tavily search with retry on transient network/rate-limit errors."""
    data = tavily_search.invoke({"query": query})
    results = data.get("results", data)
    tracker.record_search()
    if not results:
        logger.warning("Search returned no results for query: %s", query)
    return results


def _format_search_docs(search_docs: list[dict]) -> str:
    """Format raw Tavily results into XML-style context string, dropping injected docs."""
    valid_docs = []
    for doc in search_docs:
        content = doc.get("content", "")
        if not content:
            continue
        # Drop document if prompt injection is detected
        is_safe, _ = security.sanitizer.check(content)
        if not is_safe:
            logger.warning("Dropping untrusted search document due to injection: %s", doc.get("url", ""))
            continue
        url = doc.get("url", "")
        valid_docs.append(f'<Document href="{url}">\n{content}\n</Document>')
    return "\n\n---\n\n".join(valid_docs)


def _coerce_analyst(analyst) -> Analyst:
    """Ensure analyst is an Analyst instance (handles dict from serialization)."""
    if isinstance(analyst, dict):
        return Analyst.model_validate(analyst)
    return analyst


_NOVELTY_PROMPT = SystemMessage(content="""Score the novelty of the LAST expert answer in this conversation.

Compare it to all previous answers in the conversation:
- 0.0 = completely repetitive, no new information at all
- 0.3 = mostly repetitive with minor additions
- 0.5 = some new information mixed with repetition
- 0.7 = mostly new information and insights
- 1.0 = entirely new insights not mentioned before

Focus on: new legal provisions cited, new case law, new practical implications.
Be strict — if the expert is restating the same Acts/Sections, score low.""")


def _assess_novelty(messages: list) -> float:
    """Score how much new information the last expert answer provided (0.0-1.0)."""
    try:
        model = get_llm()
        structured = model.with_structured_output(NoveltyAssessment)
        recent = messages[-4:] if len(messages) > 4 else messages
        result = _invoke_llm([_NOVELTY_PROMPT] + recent, structured=structured)
        score = result.novelty_score if result else 0.5
        logger.info("Novelty assessment: %.2f (%s)", score, result.reasoning[:60] if result else "n/a")
        return score
    except Exception:
        logger.warning("Novelty assessment failed, defaulting to 0.5")
        return 0.5


# Analyst creation & human review nodes

def create_analysts(state: GenerateAnalystsState):
    """Create analyst personas via structured LLM output, strictly capped."""
    tracker.start_timer()

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    # Security: sanitize topic input
    is_allowed, topic, sec_notes = security.check_input(topic)
    if not is_allowed:
        logger.warning("Topic blocked by security: %s", sec_notes)
        raise ValueError(f"Input rejected: {sec_notes}")
    if sec_notes:
        logger.info("Security notes for topic: %s", sec_notes)

    logger.info("Creating up to %d analysts for topic=%r", max_analysts, topic)

    structured_llm = get_llm().with_structured_output(Perspectives)
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )

    analysts = _invoke_llm(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate the set of analysts.")],
        structured=structured_llm,
    )

    # Strictly enforce analyst cap
    selected_analysts = analysts.analysts[:max_analysts]
    logger.info("Created %d analysts (capped to %d)", len(selected_analysts), max_analysts)
    return {"analysts": selected_analysts}


def human_feedback(state: GenerateAnalystsState):
    """Pause for human review of generated analysts."""
    feedback = interrupt({
        "question": "Are these analysts okay?",
        "analysts": [
            analyst.model_dump() if hasattr(analyst, "model_dump") else analyst
            for analyst in state.get("analysts", [])
        ],
        "instructions": "Return feedback to regenerate analysts, or return empty/perfect/continue to approve.",
    })

    if feedback is None:
        return {"human_analyst_feedback": None}

    if isinstance(feedback, str):
        feedback = feedback.strip()
        if feedback == "" or feedback.lower() in {"perfect", "continue", "approved", "yes"}:
            logger.info("Analysts approved by human")
            return {"human_analyst_feedback": None}

        # Security: sanitize human feedback before passing to LLM
        is_allowed, cleaned_feedback, sec_notes = security.check_input(feedback)
        if not is_allowed:
            logger.warning("Feedback blocked by security: %s", sec_notes)
            return {"human_analyst_feedback": None}
        if sec_notes:
            logger.info("Security notes for feedback: %s", sec_notes)

        logger.info("Human requested analyst revisions: %s", cleaned_feedback)
        return {"human_analyst_feedback": cleaned_feedback}

    return {"human_analyst_feedback": None}


# Interview nodes

def generate_question(state: InterviewState):
    """Generate the next interview question from the analyst persona."""
    analyst = _coerce_analyst(state["analyst"])
    messages = state["messages"]

    system_message = question_instructions.format(goals=analyst.persona)
    question = _invoke_llm([SystemMessage(content=system_message)] + messages)

    logger.debug("Analyst %s asked: %s", analyst.name, question.content[:80])
    return {"messages": [question]}


def generate_answer(state: InterviewState):
    """Generate an expert answer grounded in web-search context, then assess novelty."""
    analyst = _coerce_analyst(state["analyst"])
    messages = state["messages"]
    context = state["context"]

    # Format context cleanly as string to avoid Python repr formatting
    context_str = "\n\n".join(str(c) for c in context) if isinstance(context, list) else str(context)

    system_message = answer_instructions.format(goals=analyst.persona, context=context_str)
    answer = _invoke_llm([SystemMessage(content=system_message)] + messages)
    answer.name = "expert"

    if _ENABLE_NOVELTY_CHECK:
        novelty_score = _assess_novelty(messages + [answer])
    else:
        novelty_score = 0.5

    logger.debug("Expert answered (%d chars, novelty=%.2f)", len(answer.content), novelty_score)
    return {"messages": [answer], "novelty_scores": [novelty_score]}


def save_interview(state: InterviewState):
    """Serialize the interview message history to a transcript string."""
    messages = state["messages"]
    interview = get_buffer_string(messages)

    logger.info("Saved interview transcript (%d chars)", len(interview))
    return {"interview": interview}


def search_web(state: InterviewState):
    """Generate a search query from the conversation and retrieve web results."""
    structured_llm = get_llm().with_structured_output(SearchQuery)
    search_query = _invoke_llm(
        [_search_system_message] + state["messages"],
        structured=structured_llm,
    )

    logger.info("Web search query: %s", search_query.search_query)

    try:
        search_docs = _search_tavily(search_query.search_query)
        formatted = _format_search_docs(search_docs)
    except Exception:
        logger.exception("Search failed after all retries")
        formatted = "No search results available."

    return {"context": [formatted]}


def search_web2(state: InterviewState):
    """Counter-argument search — finds opposing legal perspectives and exceptions."""
    _counter_msg = SystemMessage(content=counter_search_instructions)
    structured_llm = get_llm().with_structured_output(SearchQuery)
    search_query = _invoke_llm(
        [_counter_msg] + state["messages"],
        structured=structured_llm,
    )

    logger.info("Counter-search query: %s", search_query.search_query)

    try:
        search_docs = _search_tavily(search_query.search_query)
        formatted = _format_search_docs(search_docs)
    except Exception:
        logger.exception("Counter-search failed after all retries")
        formatted = "No counter-argument results available."

    return {"context": [formatted]}


def write_section(state: InterviewState):
    """Write a report section, score grounding, and cross-check key claims."""
    interview = state.get("interview", "")
    context = state.get("context", [])
    analyst = _coerce_analyst(state["analyst"])

    context_str = "\n\n".join(str(c) for c in context) if isinstance(context, list) else str(context)

    # 1. Synthesize both interview dialogue and source context
    system_message = section_writer_instructions.format(focus=analyst.description)
    prompt_payload = (
        f"Expert Interview Transcript:\n\n{interview}\n\n"
        f"---\n\n"
        f"Source Documents:\n\n{context_str}"
    )

    section = _invoke_llm([
        SystemMessage(content=system_message),
        HumanMessage(content=prompt_payload),
    ])
    section_text = section.content

    # 2. Source grounding assessment
    if _ENABLE_SCORING:
        score = score_section(section_text)
        section_text += format_confidence_badge(score)

    # 3. Source cross-check
    if _ENABLE_VERIFICATION:
        claims = extract_claims(interview)
        if claims:
            results = [verify_claim(claim, context_str) for claim in claims[:3]]
            section_text = format_verification_markers(section_text, results)
            logger.info("Cross-checked %d claims for analyst %s", len(results), analyst.name)

    logger.info("Wrote section for analyst %s (%d chars)", analyst.name, len(section_text))
    return {"sections": [section_text]}


# Report assembly nodes

def write_report(state: ResearchGraphState):
    """Consolidate analyst memos into the main report body."""
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join(sections)

    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = _invoke_llm([
        SystemMessage(content=system_message),
        HumanMessage(content="Write a report based upon these memos."),
    ])

    logger.info("Report body written (%d chars)", len(report.content))
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):
    """Write the report introduction."""
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join(sections)

    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections,
    )
    intro = _invoke_llm([
        SystemMessage(content=instructions),
        HumanMessage(content="Write the report introduction"),
    ])

    logger.info("Introduction written (%d chars)", len(intro.content))
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    """Write the report conclusion."""
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join(sections)

    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections,
    )
    conclusion = _invoke_llm([
        SystemMessage(content=instructions),
        HumanMessage(content="Write the report conclusion"),
    ])

    logger.info("Conclusion written (%d chars)", len(conclusion.content))
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """Assemble introduction, body, conclusion, debate notes, metadata, and disclaimer."""
    content = state["content"]

    sources = None
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except ValueError:
            logger.warning("Could not cleanly split sources from report body")
            sources = None

    final_report = (
        state["introduction"]
        + "\n\n---\n\n"
        + content
        + "\n\n---\n\n"
        + state["conclusion"]
    )

    # Include analyst debate notes if present
    debate_notes = state.get("debate_notes", "")
    if debate_notes:
        final_report += "\n\n---\n\n" + debate_notes

    if sources is not None:
        final_report += "\n\n## Sources\n" + sources

    # Security: validate final report output for PII leakage / harmful exploits
    final_report, warnings = security.check_output(final_report)
    is_blocked = any("blocked" in w.lower() for w in warnings)
    if warnings:
        logger.warning("Security warnings in final report: %s", warnings)

    # Cost dashboard
    tracker.analysts_used = len(state.get("analysts", []))
    final_report += tracker.to_markdown()

    # Reliability disclaimer
    final_report += format_reliability_footer()

    logger.info("Final report assembled (%d chars)", len(final_report))
    return {
        "final_report": final_report,
        "metadata": tracker.to_dict(),
        "is_blocked": is_blocked,
    }


# Debate & critique nodes

def debate_sections(state: ResearchGraphState):
    """Each analyst critiques the other analysts' sections for contradictions and gaps."""
    if not _ENABLE_DEBATE:
        logger.info("Debate disabled via ENABLE_DEBATE flag")
        return {"debate_notes": ""}

    analysts = state.get("analysts", [])
    sections = state.get("sections", [])

    if len(analysts) < 2 or len(sections) < 2:
        logger.info("Skipping debate: need at least 2 analysts and sections")
        return {"debate_notes": ""}

    results, consensus = run_debate(analysts, sections)
    debate_md = format_debate_section(results, consensus)

    total_points = sum(len(r.points) for r in results)
    logger.info("Debate complete: %d points from %d reviews", total_points, len(results))
    return {"debate_notes": debate_md}


class ReportCritique(BaseModel):
    """Quality and coherence scorecard for the final report."""

    clarity_score: float = Field(description="Clarity and readability (1-10).")
    completeness_score: float = Field(description="Completeness of legal analysis (1-10).")
    coherence_score: float = Field(description="Logical coherence across sections (1-10).")
    citation_score: float = Field(description="Proper citation quality and statutory references (1-10).")
    overall_score: float = Field(description="Overall research quality (1-10).")
    strengths: list[str] = Field(default_factory=list, description="Top positive legal findings.")
    weaknesses: list[str] = Field(default_factory=list, description="Top areas for future investigation.")


_CRITIQUE_PROMPT = SystemMessage(content="""You are a senior legal editor auditing a research report
on Bangladesh law for quality, coherence, and statutory citation depth.

Score the report on these dimensions (1 = poor, 10 = excellent):
1. **Clarity** — Is the report clear and accessible for practicing legal counsel?
2. **Completeness** — Does it cover the key statutory and regulatory domains?
3. **Coherence** — Do the sections integrate without conflicting findings?
4. **Citation quality** — Are statutory sections and court precedents properly referenced?
5. **Overall** — Overall utility of the legal analysis.

Provide a constructive quality assessment scorecard. Do NOT recommend rewriting or modifying the text.""")


def critique_report(state: ResearchGraphState):
    """Score the final report on quality and append an evaluative scorecard.

    Never performs auto-rewriting to avoid wiping citations, debate notes, or hallucinating.
    """
    if not _ENABLE_CRITIQUE:
        logger.info("Critique disabled via ENABLE_CRITIQUE flag")
        return {"final_report": state["final_report"]}

    final_report = state["final_report"]

    try:
        model = get_llm()
        structured = model.with_structured_output(ReportCritique)
        critique = _invoke_llm(
            [_CRITIQUE_PROMPT, HumanMessage(content=f"Review this legal research report:\n\n{final_report[:8000]}")],
            structured=structured,
        )

        if critique:
            logger.info(
                "Report critique: clarity=%.1f, completeness=%.1f, coherence=%.1f, "
                "citation=%.1f, overall=%.1f",
                critique.clarity_score, critique.completeness_score,
                critique.coherence_score, critique.citation_score,
                critique.overall_score,
            )

            scorecard = (
                "\n\n## Quality Assessment\n"
                f"| Dimension | Score |\n"
                f"|---|---|\n"
                f"| Clarity | {critique.clarity_score:.0f}/10 |\n"
                f"| Completeness | {critique.completeness_score:.0f}/10 |\n"
                f"| Coherence | {critique.coherence_score:.0f}/10 |\n"
                f"| Citation Quality | {critique.citation_score:.0f}/10 |\n"
                f"| **Overall** | **{critique.overall_score:.0f}/10** |\n"
            )

            if critique.weaknesses:
                scorecard += "\n**Areas for further review:**\n"
                for w in critique.weaknesses:
                    scorecard += f"- {w}\n"

            # Pure evaluative assessment: append scorecard, NEVER rewrite
            final_report += scorecard
            return {"final_report": final_report}

    except Exception:
        logger.warning("Report critique failed, returning report as-is")

    return {"final_report": final_report}