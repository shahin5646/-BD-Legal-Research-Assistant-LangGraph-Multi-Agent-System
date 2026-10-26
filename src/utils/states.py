"""State definitions for the BD Legal Research Assistant graphs.

Defines the TypedDicts and Pydantic models used as graph state
across the analyst creation, interview, and deep research graphs.
"""

from typing import List, Optional, Any
from typing_extensions import Annotated, NotRequired, TypedDict

import operator
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from src.utils.objects import Analyst


class GenerateAnalystsState(TypedDict):
    topic: str  # Legal research topic
    max_analysts: int  # Number of analysts to generate
    human_analyst_feedback: NotRequired[Optional[str]]  # Human feedback for revision
    analysts: NotRequired[List[Analyst]]  # Generated analyst personas


class InterviewState(MessagesState):
    max_num_turns: int  # Max interview turns (hard ceiling for this interview)
    context: Annotated[list, operator.add]  # Accumulated source documents
    analyst: Analyst  # Analyst conducting this interview
    interview: str  # Serialized interview transcript
    sections: list  # Written section output
    novelty_scores: Annotated[list, operator.add]  # Per-turn novelty scores (0.0-1.0)


class SearchQuery(BaseModel):
    """Structured output for web search query generation."""
    search_query: Optional[str] = Field(default=None, description="Search query for retrieval.")


class NoveltyAssessment(BaseModel):
    """LLM-assessed novelty score for an expert answer turn."""
    novelty_score: float = Field(
        description="How much new information this answer provided (0.0=repetitive, 1.0=entirely new insights)."
    )
    reasoning: str = Field(
        description="Brief explanation of why this novelty score was given."
    )


class ResearchGraphState(TypedDict):
    topic: str  # Legal research topic
    max_analysts: int  # Number of analysts
    interview_turns: NotRequired[int]  # Configured turns per interview (parent state key avoiding collisions)
    human_analyst_feedback: NotRequired[Optional[str]]  # Human feedback
    analysts: List[Analyst]  # Legal analyst personas
    sections: Annotated[list, operator.add]  # Sections from all interviews
    debate_notes: str  # Analyst debate output (markdown)
    introduction: str  # Report introduction
    content: str  # Report body
    conclusion: str  # Report conclusion
    final_report: str  # Assembled final report
    metadata: NotRequired[dict[str, Any]]  # Cost, token, and timing metrics
    is_blocked: NotRequired[bool]  # Security violation flag