"""Deep research agent — full pipeline from analyst creation through report generation.

Orchestrates:
1. Analyst creation with human review
2. Parallel interviews via Send() API
3. Analyst debate — cross-critique of each other's sections
4. Report body, introduction, and conclusion (in parallel)
5. Final report assembly with debate notes
6. Self-critique with quality scorecard (optional rewrite)

Exposed as the ``deep_agent`` graph in langgraph.json.
"""

from langgraph.graph import StateGraph, START, END

from src.answering_questions import interview_builder
from src.utils.edges import initiate_all_interviews
from src.utils.nodes import (
    create_analysts,
    human_feedback,
    debate_sections,
    write_report,
    write_introduction,
    write_conclusion,
    finalize_report,
    critique_report,
)
from src.utils.states import ResearchGraphState

# Build the outer research graph
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("debate_sections", debate_sections)
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)
builder.add_node("critique_report", critique_report)

# Flow
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback",
    initiate_all_interviews,
    ["create_analysts", "conduct_interview"],
)

# After interviews → debate, then write in parallel
builder.add_edge("conduct_interview", "debate_sections")
builder.add_edge("debate_sections", "write_report")
builder.add_edge("debate_sections", "write_introduction")
builder.add_edge("debate_sections", "write_conclusion")

# Assemble and critique
builder.add_edge(
    ["write_conclusion", "write_report", "write_introduction"],
    "finalize_report",
)
builder.add_edge("finalize_report", "critique_report")
builder.add_edge("critique_report", END)

graph = builder.compile()