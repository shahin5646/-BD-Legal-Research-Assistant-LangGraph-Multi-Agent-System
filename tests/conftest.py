"""Shared test fixtures — mock LLM, sample legal analysts, etc."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from src.utils.objects import Analyst, Perspectives


@pytest.fixture
def sample_analyst():
    """A reusable BD legal analyst fixture."""
    return Analyst(
        affiliation="Bangladesh Supreme Court Bar Association",
        name="Adv. Fatima Rahman",
        role="Corporate Law Specialist",
        description="Focuses on company formation, foreign investment, and BIDA regulations under the Companies Act 1994.",
        legal_domain="Corporate",
    )


@pytest.fixture
def sample_analysts(sample_analyst):
    """List of legal analysts for graph-level tests."""
    return [
        sample_analyst,
        Analyst(
            affiliation="Dhaka District Bar Association",
            name="Adv. Kamal Hossain",
            role="Criminal Law Analyst",
            description="Focuses on the Penal Code 1860, CrPC 1898, and anti-corruption laws.",
            legal_domain="Criminal",
        ),
    ]


@pytest.fixture
def mock_llm_response():
    """A basic AIMessage mock."""
    return AIMessage(content="This is a mock LLM response.", name="expert")


@pytest.fixture
def mock_llm(mock_llm_response):
    """Patch the LLM to return a deterministic response."""
    mock = MagicMock()
    mock.invoke.return_value = mock_llm_response
    mock.with_structured_output.return_value = mock
    return mock
