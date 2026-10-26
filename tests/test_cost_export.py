"""Unit tests for cost tracking and multi-format export."""

import json
import pytest
from src.utils.cost_tracker import ResearchMetrics, reset_tracker, tracker
from src.utils.export import ReportExport, export_report, export_to_json
from src.utils.objects import Analyst


class TestResearchMetrics:
    def test_initial_state(self):
        m = ResearchMetrics()
        assert m.total_input_tokens == 0
        assert m.total_output_tokens == 0
        assert m.total_searches == 0
        assert m.llm_calls == 0
        assert m.analysts_used == 0

    def test_record_llm_call(self):
        m = ResearchMetrics()
        m.record_llm_call(input_tokens=100, output_tokens=50)
        assert m.llm_calls == 1
        assert m.total_input_tokens == 100
        assert m.total_output_tokens == 50

    def test_record_multiple_calls(self):
        m = ResearchMetrics()
        m.record_llm_call(input_tokens=100, output_tokens=50)
        m.record_llm_call(input_tokens=200, output_tokens=100)
        assert m.llm_calls == 2
        assert m.total_input_tokens == 300
        assert m.total_output_tokens == 150

    def test_record_search(self):
        m = ResearchMetrics()
        m.record_search()
        m.record_search()
        assert m.total_searches == 2

    def test_estimated_cost(self):
        m = ResearchMetrics()
        m.total_input_tokens = 1_000_000
        m.total_output_tokens = 1_000_000
        # Default: 0.15/M input, 0.60/M output
        assert m.estimated_cost == pytest.approx(0.75, rel=0.01)

    def test_to_markdown(self):
        m = ResearchMetrics()
        m.analysts_used = 3
        m.record_llm_call(input_tokens=500, output_tokens=200)
        m.record_search()
        md = m.to_markdown()
        assert "## Research Metadata" in md
        assert "Legal Analysts" in md
        assert "3" in md
        assert "LLM Calls" in md
        assert "Web Searches" in md
        assert "$" in md  # cost

    def test_reset_tracker(self):
        tracker.record_llm_call(input_tokens=100, output_tokens=50)
        tracker.record_search()
        assert tracker.llm_calls > 0
        reset_tracker()
        assert tracker.llm_calls == 0
        assert tracker.total_searches == 0

    def test_to_dict(self):
        m = ResearchMetrics()
        m.analysts_used = 2
        m.record_llm_call(input_tokens=200, output_tokens=100)
        m.record_search()
        d = m.to_dict()
        assert d["analysts_used"] == 2
        assert d["llm_calls"] == 1
        assert d["total_searches"] == 1
        assert d["total_input_tokens"] == 200
        assert d["total_output_tokens"] == 100
        assert "elapsed_seconds" in d



class TestReportExport:
    def test_create_export(self):
        export = ReportExport(
            title="Test Report",
            topic="BD corporate law",
            analysts=[{"name": "Test", "role": "Tester"}],
            sections=["## Section 1\nContent."],
            introduction="# Title\n## Introduction\nIntro text.",
            conclusion="## Conclusion\nConclusion text.",
            final_report="Full report text.",
            generated_at="2026-01-01T00:00:00Z",
        )
        assert export.title == "Test Report"
        assert len(export.sections) == 1

    def test_export_report_from_state(self):
        analyst = Analyst(
            affiliation="Test Firm",
            name="Adv. Test",
            role="Tester",
            description="A tester.",
            legal_domain="Corporate",
        )
        state = {
            "topic": "Foreign investment in BD",
            "analysts": [analyst],
            "sections": ["## S1\nContent"],
            "introduction": "# FDI in Bangladesh\n## Introduction\nIntro.",
            "conclusion": "## Conclusion\nDone.",
            "debate_notes": "No debate.",
            "final_report": "Full report.",
        }
        export = export_report(state)
        assert export.title == "FDI in Bangladesh"
        assert export.topic == "Foreign investment in BD"
        assert len(export.analysts) == 1
        assert export.analysts[0]["name"] == "Adv. Test"

    def test_export_to_json(self):
        state = {
            "topic": "Test",
            "analysts": [],
            "sections": [],
            "introduction": "# Test\n## Introduction\nIntro.",
            "conclusion": "## Conclusion\nDone.",
            "final_report": "Report.",
        }
        json_str = export_to_json(state)
        data = json.loads(json_str)
        assert data["topic"] == "Test"
        assert "generated_at" in data
        assert isinstance(data["sections"], list)

    def test_export_extracts_title_from_h1(self):
        state = {
            "topic": "Test",
            "analysts": [],
            "sections": [],
            "introduction": "# Legal Analysis of Tax Implications\n## Introduction\nText.",
            "conclusion": "",
            "final_report": "",
        }
        export = export_report(state)
        assert export.title == "Legal Analysis of Tax Implications"

    def test_export_fallback_title(self):
        state = {
            "topic": "Test",
            "analysts": [],
            "sections": [],
            "introduction": "No heading here.",
            "conclusion": "",
            "final_report": "",
        }
        export = export_report(state)
        assert export.title == "Legal Research Report"
