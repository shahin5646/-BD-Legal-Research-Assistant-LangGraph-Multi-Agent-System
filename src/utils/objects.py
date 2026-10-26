"""Pydantic models for the BD Legal Research Assistant.

Defines the structured output schemas used by the LLM for
analyst persona generation and search query extraction.
"""

from pydantic import BaseModel, Field
from typing import List


class Analyst(BaseModel):
    """A legal research analyst persona with a specific BD law specialization."""

    affiliation: str = Field(
        description="Law firm, court, bar council, or institution the analyst is affiliated with.",
    )
    name: str = Field(
        description="Name of the legal analyst.",
    )
    role: str = Field(
        description="Legal specialization (e.g., Corporate Law Specialist, Criminal Law Analyst).",
    )
    description: str = Field(
        description="Specific focus area within Bangladesh law and key concerns.",
    )
    legal_domain: str = Field(
        default="General",
        description="Primary area of BD law (Constitutional, Criminal, Corporate, Labour, Land, Family, Tax, Banking, Environmental, Cyber, IP).",
    )

    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Affiliation: {self.affiliation}\n"
            f"Legal Domain: {self.legal_domain}\n"
            f"Description: {self.description}\n"
        )


class Perspectives(BaseModel):
    """Collection of legal analyst personas generated for a research topic."""

    analysts: List[Analyst] = Field(
        description="List of legal analysts, each specializing in a relevant area of Bangladesh law.",
    )
