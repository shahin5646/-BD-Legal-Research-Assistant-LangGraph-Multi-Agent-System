"""Unit tests for confidence scoring."""

import pytest
from src.utils.scoring import SectionScore, format_confidence_badge, format_reliability_footer


class TestSectionScore:
    def test_create_valid_score(self):
        score = SectionScore(
            confidence=0.85,
            sourced_claims=8,
            unsourced_claims=2,
            source_quality="primary_law",
            assessment="Well-sourced with primary legal references.",
        )
        assert score.confidence == 0.85
        assert score.sourced_claims == 8
        assert score.source_quality == "primary_law"

    def test_default_score(self):
        """Verify the fallback score structure."""
        score = SectionScore(
            confidence=0.5,
            sourced_claims=0,
            unsourced_claims=0,
            source_quality="unknown",
            assessment="Scoring unavailable.",
        )
        assert score.confidence == 0.5


class TestFormatConfidenceBadge:
    def test_high_confidence_green_with_action(self):
        score = SectionScore(
            confidence=0.85,
            sourced_claims=8,
            unsourced_claims=1,
            source_quality="primary_law",
            assessment="Excellent sourcing.",
        )
        badge = format_confidence_badge(score)
        assert "🟢" in badge
        assert "Reliability: 85%" in badge
        assert "Safe to rely on" in badge
        assert "Excellent sourcing." in badge
        # Should NOT contain technical details
        assert "Sourced claims" not in badge
        assert "Unsourced" not in badge
        assert "Source type" not in badge

    def test_medium_confidence_yellow_with_action(self):
        score = SectionScore(
            confidence=0.5,
            sourced_claims=3,
            unsourced_claims=4,
            source_quality="secondary",
            assessment="Mixed sourcing.",
        )
        badge = format_confidence_badge(score)
        assert "🟡" in badge
        assert "Reliability: 50%" in badge
        assert "Check the key points" in badge

    def test_low_confidence_red_with_action(self):
        score = SectionScore(
            confidence=0.2,
            sourced_claims=1,
            unsourced_claims=8,
            source_quality="web",
            assessment="Poorly sourced.",
        )
        badge = format_confidence_badge(score)
        assert "🔴" in badge
        assert "Reliability: 20%" in badge
        assert "Verify independently" in badge

    def test_badge_is_markdown_blockquote(self):
        score = SectionScore(
            confidence=0.7,
            sourced_claims=5,
            unsourced_claims=2,
            source_quality="case_law",
            assessment="Good case law coverage.",
        )
        badge = format_confidence_badge(score)
        assert badge.strip().startswith(">")  # blockquote format

    def test_assessment_in_italics(self):
        score = SectionScore(
            confidence=0.7,
            sourced_claims=5,
            unsourced_claims=2,
            source_quality="case_law",
            assessment="Good case law coverage.",
        )
        badge = format_confidence_badge(score)
        assert "*Good case law coverage.*" in badge


class TestFormatReliabilityFooter:
    def test_footer_contains_disclaimer(self):
        footer = format_reliability_footer()
        assert "automated estimate" in footer.lower()
        assert "not" in footer.lower()
        assert "guarantee" in footer.lower()

    def test_footer_starts_with_separator(self):
        footer = format_reliability_footer()
        assert "---" in footer

    def test_footer_in_italics(self):
        footer = format_reliability_footer()
        # Footer text should be wrapped in *...*
        assert footer.strip().endswith("*")
