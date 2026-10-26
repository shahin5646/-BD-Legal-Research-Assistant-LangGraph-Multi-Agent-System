"""Integration tests: End-to-end graph execution using MemorySaver checkpointer

Verifies that the full deep_agent graph completes successfully through human-in-the-loop
pause and resume, parallel interview execution, debate, report synthesis, and quality critique.
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.deep_agent import builder as deep_agent_builder
from src.utils.objects import Analyst, Perspectives
from src.utils.states import SearchQuery, NoveltyAssessment
from src.utils.scoring import SectionScore
from src.utils.verification import ClaimExtraction, VerificationResult
from src.utils.nodes import ReportCritique


class FakeChatModel:
    """Deterministic fake chat model for end-to-end graph testing."""

    def __init__(self):
        self.model_name = "gpt-4o-mini"
        self.temperature = 0

    def with_structured_output(self, schema):
        mock = MagicMock()

        def _fake_invoke(messages):
            if schema == Perspectives:
                return Perspectives(analysts=[
                    Analyst(
                        name="Farhana Ahmed",
                        role="Corporate Law Specialist",
                        affiliation="Supreme Court Bar",
                        description="Focus on Companies Act 1994 and BIDA rules",
                        legal_domain="Corporate Law",
                    )
                ])
            elif schema == SearchQuery:
                return SearchQuery(search_query="Bangladesh Companies Act foreign ownership")
            elif schema == NoveltyAssessment:
                return NoveltyAssessment(novelty_score=0.8, reasoning="Cites new statutory provisions")
            elif schema == SectionScore:
                return SectionScore(
                    confidence=0.9,
                    sourced_claims=2,
                    unsourced_claims=0,
                    source_quality="primary_law",
                    assessment="Well sourced with primary legislation",
                )
            elif schema == ClaimExtraction:
                return ClaimExtraction(claims=["Foreign ownership permitted up to 100% under Companies Act 1994"])
            elif schema == VerificationResult:
                return VerificationResult(
                    claim="Foreign ownership permitted up to 100% under Companies Act 1994",
                    status="verified",
                    evidence="Confirmed by Section 2 and BIDA investment guidelines",
                )
            elif schema == ReportCritique:
                return ReportCritique(
                    clarity_score=9.0,
                    completeness_score=8.5,
                    coherence_score=9.0,
                    citation_score=9.0,
                    overall_score=8.8,
                    strengths=["Comprehensive statutory references"],
                    weaknesses=["Could explore sector-specific exceptions"],
                )
            return None

        mock.invoke = _fake_invoke
        return mock

    def invoke(self, messages):
        content_str = str(messages)
        if "question_instructions" in content_str or "interviewing a legal expert" in content_str:
            return AIMessage(content="What are the foreign equity restrictions under Bangladesh law? Thank you so much for your help!")
        elif "answer_instructions" in content_str or "legal expert being interviewed" in content_str:
            return AIMessage(content="Under The Companies Act, 1994, 100% foreign equity is permitted for most sectors [1].\n\n[1] http://bdlaws.minlaw.gov.bd/act-details/1994")
        elif "section_writer_instructions" in content_str or "legal memo" in content_str:
            return AIMessage(content="## Corporate Legal Memo\n\n### Issue\nForeign equity restrictions.\n\n### Relevant Law\nCompanies Act 1994, Section 2.\n\n### Analysis\nForeign investors may hold up to 100% equity.\n\n### Conclusion\nSubsidiary formation is permitted.\n\n### Sources\n[1] http://bdlaws.minlaw.gov.bd/act-details/1994")
        elif "report_writer_instructions" in content_str or "consolidated legal research report" in content_str:
            return AIMessage(content="## Legal Analysis\n\nForeign companies may establish wholly owned subsidiaries under Bangladesh law.\n\n## Sources\n[1] http://bdlaws.minlaw.gov.bd/act-details/1994")
        elif "intro_conclusion_instructions" in content_str:
            # Differentiate intro vs conclusion by prompt instructions
            for m in messages:
                if hasattr(m, "content") and "introduction" in m.content.lower():
                    return AIMessage(content="# Foreign Investment Law in Bangladesh\n\n## Introduction\nThis report examines the legal framework for foreign direct investment.")
            return AIMessage(content="## Conclusion\nIn summary, Bangladesh law offers a liberal investment regime with minimal restrictions.")
        return AIMessage(content="Standard legal analysis response.")


class TestFullGraphExecution:
    """Test full execution of the deep_agent graph with interrupt and resume."""

    @patch("src.utils.nodes._search_tavily")
    @patch("src.utils.models.get_llm")
    def test_deep_agent_full_run(self, mock_get_llm, mock_search):
        fake_llm = FakeChatModel()
        mock_get_llm.return_value = fake_llm

        mock_search.return_value = [
            {
                "url": "http://bdlaws.minlaw.gov.bd/act-details/1994",
                "content": "The Companies Act 1994: Section 2 permits foreign companies to establish subsidiaries.",
            }
        ]

        checkpointer = MemorySaver()
        app = deep_agent_builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-session-1"}}

        initial_state = {
            "topic": "Can foreign companies own 100% shares of a Bangladesh subsidiary?",
            "max_analysts": 1,
            "interview_turns": 1,
            "sections": [],
        }

        # Step 1: Run until human_feedback interrupt
        events = list(app.stream(initial_state, config=config))
        assert len(events) > 0

        # Verify paused at human_feedback
        snapshot = app.get_state(config)
        assert snapshot.next == ("human_feedback",)

        # Step 2: Resume with approval
        resume_events = list(app.stream(Command(resume="approved"), config=config))
        assert len(resume_events) > 0

        # Step 3: Verify graph finished at END
        final_snapshot = app.get_state(config)
        assert final_snapshot.next == ()

        final_state = final_snapshot.values
        final_report = final_state.get("final_report", "")

        # Verify key report sections exist
        assert "# Foreign Investment Law in Bangladesh" in final_report
        assert "## Introduction" in final_report
        assert "## Legal Analysis" in final_report
        assert "## Conclusion" in final_report
        assert "## Quality Assessment" in final_report
        assert "## Research Metadata" in final_report
        assert "Reliability" in final_report or "Source grounding" in final_report or "disclaimer" in final_report.lower()

        # Verify metadata dictionary populated
        assert "metadata" in final_state
        assert final_state["metadata"]["analysts_used"] == 1
        assert final_state["is_blocked"] is False
