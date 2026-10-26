"""Analyst debate — multi-agent adversarial collaboration.

After all interviews are complete, each analyst reviews the OTHER analysts'
sections and flags contradictions, gaps, and weak reasoning. The debate
points are injected into the report body so the final report reflects
cross-analyst scrutiny.
"""

import logging

from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, HumanMessage

from src.utils.models import get_llm
from src.utils.objects import Analyst

logger = logging.getLogger(__name__)


class DebatePoint(BaseModel):
    """A single critique from one analyst about another's section."""

    target_section_title: str = Field(description="Title of the section being critiqued.")
    critique_type: str = Field(
        description="Type of issue: 'contradiction', 'gap', 'weak_evidence', or 'outdated_law'."
    )
    description: str = Field(description="What the issue is.")
    suggested_fix: str = Field(description="How to address the issue.")


class DebateResult(BaseModel):
    """All critique points from one analyst reviewing another's section."""

    reviewer_name: str = Field(description="Name of the analyst providing critique.")
    points: list[DebatePoint] = Field(description="List of critique points.")


class DebateRound(BaseModel):
    """Aggregated debate results from all analyst pairings."""

    debates: list[DebateResult] = Field(description="All debate results.")
    consensus_summary: str = Field(
        description="One-paragraph summary of key disagreements and consensus areas."
    )


_DEBATE_PROMPT = """You are {reviewer_name}, a {reviewer_role} specializing in {reviewer_domain}.

Another analyst wrote the following section for a legal research report.
Review it critically from your area of expertise and identify:

1. **Contradictions** — Does anything conflict with your legal knowledge?
2. **Gaps** — Are there missing Acts, Sections, or legal provisions that should be cited?
3. **Weak evidence** — Are claims made without proper legal citations?
4. **Outdated law** — Has any cited law been amended, repealed, or superseded?

Be specific. Cite the exact provision or precedent that supports your critique.

Section to review:
{section}"""


def run_debate(analysts: list, sections: list[str]) -> tuple[list[DebateResult], str]:
    """Each analyst reviews sections they did NOT write.

    Returns (list of DebateResults, consensus summary string).
    """
    all_results = []

    for i, reviewer in enumerate(analysts):
        # Coerce dict to Analyst if needed
        if isinstance(reviewer, dict):
            reviewer = Analyst.model_validate(reviewer)

        # Review all sections except your own (section index matches analyst index)
        other_sections = [(j, s) for j, s in enumerate(sections) if j != i]

        for section_idx, section_text in other_sections:
            prompt = _DEBATE_PROMPT.format(
                reviewer_name=reviewer.name,
                reviewer_role=reviewer.role,
                reviewer_domain=reviewer.legal_domain,
                section=section_text[:3000],  # Cap to avoid token overflow
            )

            try:
                model = get_llm()
                structured = model.with_structured_output(DebateResult)
                result = structured.invoke([
                    SystemMessage(content=prompt),
                    HumanMessage(content="Provide your critique of this section."),
                ])
                if result:
                    result.reviewer_name = reviewer.name
                    all_results.append(result)
                    logger.info(
                        "%s raised %d points on section %d",
                        reviewer.name, len(result.points), section_idx,
                    )
            except Exception:
                logger.warning("Debate failed for %s on section %d", reviewer.name, section_idx)

    # Generate consensus summary
    consensus = _build_consensus(all_results)

    return all_results, consensus


def _build_consensus(results: list[DebateResult]) -> str:
    """Summarize the debate into a consensus paragraph."""
    if not results:
        return "No debate points were raised. All sections appear consistent."

    total_points = sum(len(r.points) for r in results)
    point_types = {}
    for r in results:
        for p in r.points:
            point_types[p.critique_type] = point_types.get(p.critique_type, 0) + 1

    type_summary = ", ".join(f"{count} {t}" for t, count in sorted(point_types.items(), key=lambda x: -x[1]))

    return (
        f"The analyst debate raised {total_points} points across {len(results)} reviews "
        f"({type_summary}). Key areas of disagreement and gaps have been flagged in the report."
    )


def format_debate_section(results: list[DebateResult], consensus: str) -> str:
    """Format debate results as a markdown section for the report."""
    if not results:
        return ""

    lines = ["## Analyst Review Notes\n"]
    lines.append(f"*{consensus}*\n")

    for result in results:
        if not result.points:
            continue
        for point in result.points:
            icon = {
                "contradiction": "⚡",
                "gap": "📋",
                "weak_evidence": "⚠️",
                "outdated_law": "🕐",
            }.get(point.critique_type, "📌")

            lines.append(
                f"- {icon} **{point.critique_type.replace('_', ' ').title()}** "
                f"({result.reviewer_name}): {point.description}\n"
                f"  - *Fix*: {point.suggested_fix}"
            )

    return "\n".join(lines)
