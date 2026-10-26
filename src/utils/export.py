"""Multi-format export — structured JSON output alongside markdown.

Generates a machine-readable JSON export of the research report
with all metadata (analysts, sections, scores, debate points, cost).
Useful for API consumers, dashboards, or downstream processing.
"""

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReportExport(BaseModel):
    """Structured export format for the full research report."""

    title: str = Field(description="Report title extracted from the introduction.")
    topic: str = Field(description="Original research topic.")
    analysts: list[dict] = Field(description="List of analyst personas used.")
    sections: list[str] = Field(description="Individual section texts.")
    introduction: str = Field(description="Report introduction.")
    conclusion: str = Field(description="Report conclusion.")
    debate_notes: str = Field(default="", description="Analyst debate output.")
    final_report: str = Field(description="Assembled final report (markdown).")
    generated_at: str = Field(description="ISO timestamp of generation.")
    metadata: dict = Field(default_factory=dict, description="Cost, token, and timing metrics.")


def export_report(state: dict) -> ReportExport:
    """Build a structured ReportExport from the graph state."""

    # Extract title from introduction (first # heading)
    intro = state.get("introduction", "")
    title = "Legal Research Report"
    for line in intro.split("\n"):
        if line.startswith("# "):
            title = line.lstrip("# ").strip()
            break

    analysts_data = []
    for a in state.get("analysts", []):
        if hasattr(a, "model_dump"):
            analysts_data.append(a.model_dump())
        elif isinstance(a, dict):
            analysts_data.append(a)

    export = ReportExport(
        title=title,
        topic=state.get("topic", ""),
        analysts=analysts_data,
        sections=state.get("sections", []),
        introduction=state.get("introduction", ""),
        conclusion=state.get("conclusion", ""),
        debate_notes=state.get("debate_notes", ""),
        final_report=state.get("final_report", ""),
        generated_at=datetime.now(timezone.utc).isoformat(),
        metadata=state.get("metadata", {}),
    )

    logger.info("Report exported: %s (%d sections, %d analysts)",
                title, len(export.sections), len(export.analysts))
    return export


def export_to_json(state: dict) -> str:
    """Export the report as a formatted JSON string."""
    report = export_report(state)
    return report.model_dump_json(indent=2)
