"""Unit tests for the analyst debate module."""

import pytest
from src.utils.debate import (
    DebatePoint,
    DebateResult,
    DebateRound,
    format_debate_section,
    _build_consensus,
)


class TestDebateModels:
    def test_debate_point(self):
        point = DebatePoint(
            target_section_title="Corporate Law Analysis",
            critique_type="contradiction",
            description="Section cites Companies Act 1994 Section 2, but this was amended in 2020.",
            suggested_fix="Update to reflect the 2020 amendment.",
        )
        assert point.critique_type == "contradiction"
        assert "Companies Act" in point.description

    def test_debate_result(self):
        result = DebateResult(
            reviewer_name="Adv. Fatima Rahman",
            points=[
                DebatePoint(
                    target_section_title="Tax Section",
                    critique_type="gap",
                    description="Missing VAT Act 2012 reference.",
                    suggested_fix="Add Section 3 of VAT Act.",
                ),
            ],
        )
        assert result.reviewer_name == "Adv. Fatima Rahman"
        assert len(result.points) == 1

    def test_debate_round(self):
        round_ = DebateRound(
            debates=[
                DebateResult(reviewer_name="A", points=[]),
                DebateResult(reviewer_name="B", points=[]),
            ],
            consensus_summary="All sections consistent.",
        )
        assert len(round_.debates) == 2

    def test_empty_debate_result(self):
        result = DebateResult(reviewer_name="Test", points=[])
        assert len(result.points) == 0


class TestBuildConsensus:
    def test_no_results(self):
        consensus = _build_consensus([])
        assert "No debate points" in consensus

    def test_with_results(self):
        results = [
            DebateResult(
                reviewer_name="A",
                points=[
                    DebatePoint(
                        target_section_title="S1",
                        critique_type="contradiction",
                        description="D1",
                        suggested_fix="F1",
                    ),
                    DebatePoint(
                        target_section_title="S2",
                        critique_type="gap",
                        description="D2",
                        suggested_fix="F2",
                    ),
                ],
            ),
        ]
        consensus = _build_consensus(results)
        assert "2 points" in consensus
        assert "1 review" in consensus
        assert "contradiction" in consensus
        assert "gap" in consensus


class TestFormatDebateSection:
    def test_empty_results(self):
        result = format_debate_section([], "No issues.")
        assert result == ""

    def test_formats_debate_with_emojis(self):
        results = [
            DebateResult(
                reviewer_name="Adv. Kamal",
                points=[
                    DebatePoint(
                        target_section_title="Corporate Section",
                        critique_type="contradiction",
                        description="Companies Act citation is wrong.",
                        suggested_fix="Use Section 3 instead.",
                    ),
                    DebatePoint(
                        target_section_title="Tax Section",
                        critique_type="gap",
                        description="Missing VAT provisions.",
                        suggested_fix="Add VAT Act 2012.",
                    ),
                ],
            ),
        ]
        md = format_debate_section(results, "Some disagreements found.")
        assert "## Analyst Review Notes" in md
        assert "⚡" in md  # contradiction emoji
        assert "📋" in md  # gap emoji
        assert "Adv. Kamal" in md
        assert "Companies Act citation is wrong" in md
        assert "Add VAT Act 2012" in md

    def test_weak_evidence_emoji(self):
        results = [
            DebateResult(
                reviewer_name="Test",
                points=[
                    DebatePoint(
                        target_section_title="S1",
                        critique_type="weak_evidence",
                        description="No citation.",
                        suggested_fix="Add one.",
                    ),
                ],
            ),
        ]
        md = format_debate_section(results, "Weak evidence found.")
        assert "⚠️" in md

    def test_outdated_law_emoji(self):
        results = [
            DebateResult(
                reviewer_name="Test",
                points=[
                    DebatePoint(
                        target_section_title="S1",
                        critique_type="outdated_law",
                        description="Repealed.",
                        suggested_fix="Update.",
                    ),
                ],
            ),
        ]
        md = format_debate_section(results, "Outdated law found.")
        assert "🕐" in md

    def test_no_points_no_output(self):
        """Results with empty points list should produce minimal output."""
        results = [DebateResult(reviewer_name="Test", points=[])]
        md = format_debate_section(results, "Nothing to report.")
        assert "## Analyst Review Notes" in md
        assert "Nothing to report" in md
