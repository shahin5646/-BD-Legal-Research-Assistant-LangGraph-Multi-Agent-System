"""Unit tests for Pydantic models and data objects."""

import pytest
from pydantic import ValidationError

from src.utils.objects import Analyst, Perspectives
from src.utils.states import SearchQuery


class TestAnalyst:
    def test_create_valid_analyst(self, sample_analyst):
        assert sample_analyst.name == "Adv. Fatima Rahman"
        assert sample_analyst.role == "Corporate Law Specialist"
        assert sample_analyst.legal_domain == "Corporate"

    def test_persona_property(self, sample_analyst):
        persona = sample_analyst.persona
        assert "Adv. Fatima Rahman" in persona
        assert "Corporate Law Specialist" in persona
        assert "Bangladesh Supreme Court Bar Association" in persona
        assert "Corporate" in persona  # legal_domain
        assert "Companies Act 1994" in persona

    def test_legal_domain_default(self):
        """legal_domain defaults to 'General' when not provided."""
        analyst = Analyst(
            affiliation="Test Firm",
            name="Test",
            role="Tester",
            description="A tester.",
        )
        assert analyst.legal_domain == "General"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            Analyst(affiliation="MIT", name="Test")  # missing role, description

    def test_model_dump_roundtrip(self, sample_analyst):
        data = sample_analyst.model_dump()
        restored = Analyst.model_validate(data)
        assert restored == sample_analyst
        assert restored.legal_domain == "Corporate"

    def test_model_dump_includes_legal_domain(self, sample_analyst):
        data = sample_analyst.model_dump()
        assert "legal_domain" in data
        assert data["legal_domain"] == "Corporate"


class TestPerspectives:
    def test_create_perspectives(self, sample_analysts):
        perspectives = Perspectives(analysts=sample_analysts)
        assert len(perspectives.analysts) == 2

    def test_empty_analysts_list(self):
        perspectives = Perspectives(analysts=[])
        assert len(perspectives.analysts) == 0


class TestSearchQuery:
    def test_create_search_query(self):
        sq = SearchQuery(search_query="Bangladesh Companies Act 1994 Section 2")
        assert sq.search_query == "Bangladesh Companies Act 1994 Section 2"

    def test_default_none(self):
        sq = SearchQuery()
        assert sq.search_query is None
