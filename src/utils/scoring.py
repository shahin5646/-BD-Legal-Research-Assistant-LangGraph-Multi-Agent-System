"""Source grounding and citation density scoring for report sections.

Analyzes each section's fact density — how many claims are backed
by cited sources vs. unsourced assertions. Formats a source grounding
indicator so readers can judge citation backing at a glance.
"""

import logging
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, HumanMessage

from src.utils.models import get_llm

logger = logging.getLogger(__name__)


class SectionScore(BaseModel):
    """LLM-assessed citation density metrics for a report section."""

    confidence: float = Field(
        description="Overall grounding score from 0.0 (unsourced) to 1.0 (well-sourced)."
    )
    sourced_claims: int = Field(
        description="Number of factual claims backed by a cited source."
    )
    unsourced_claims: int = Field(
        description="Number of factual claims with no citation."
    )
    source_quality: str = Field(
        description="Dominant source type: 'primary_law' (Acts/Ordinances), "
                    "'case_law' (court judgments), 'secondary' (commentary/analysis), "
                    "or 'web' (general web sources)."
    )
    assessment: str = Field(
        description="One-sentence summary of the section's citation backing."
    )


_SCORING_PROMPT = SystemMessage(content="""You are a legal research quality auditor.

Analyze the following legal report section and score its citation grounding.

Count:
1. **Sourced claims** — factual statements that have a citation [1], [2], etc.
2. **Unsourced claims** — factual/legal statements with NO citation.

Assess **source quality**:
- "primary_law" if most citations are to Acts, Ordinances, Sections
- "case_law" if most citations are to court judgments
- "secondary" if most citations are to commentary, analysis, articles
- "web" if most citations are generic web URLs

Calculate **grounding confidence**:
- 1.0 = every claim is sourced, sources are primary law or case law
- 0.7 = most claims sourced, mix of source types
- 0.5 = roughly half sourced
- 0.3 = mostly unsourced
- 0.0 = no sources at all

Be strict and precise in your counting.""")


def score_section(section_text: str) -> SectionScore:
    """Score a report section for citation density and source quality."""
    try:
        model = get_llm()
        structured = model.with_structured_output(SectionScore)
        result = structured.invoke([
            _SCORING_PROMPT,
            HumanMessage(content=f"Score this section:\n\n{section_text}"),
        ])
        if result:
            logger.info(
                "Section scored: confidence=%.2f, sourced=%d, unsourced=%d, quality=%s",
                result.confidence, result.sourced_claims, result.unsourced_claims, result.source_quality,
            )
            return result
    except Exception:
        logger.warning("Section scoring failed, using default score")

    return SectionScore(
        confidence=0.5,
        sourced_claims=0,
        unsourced_claims=0,
        source_quality="unknown",
        assessment="Scoring unavailable.",
    )


def format_confidence_badge(score: SectionScore) -> str:
    """Format a reliability badge that a non-technical reader can understand."""
    if score.confidence >= 0.7:
        emoji = "🟢"
        action = "Safe to rely on — sources support it."
    elif score.confidence >= 0.4:
        emoji = "🟡"
        action = "Check the key points before relying on it."
    else:
        emoji = "🔴"
        action = "Verify independently before relying on it."

    return (
        f"\n\n> {emoji} **Reliability: {score.confidence:.0%}** — {action}\n"
        f"> *{score.assessment}*"
    )


def format_reliability_footer() -> str:
    """One-line disclaimer appended once at the end of the final report."""
    return "\n\n---\n*Reliability scores are automated estimates, not guarantees. Always verify before citing.*"
