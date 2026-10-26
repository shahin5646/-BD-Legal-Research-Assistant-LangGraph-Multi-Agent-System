"""Analyst creation graph — generates analyst personas with human-in-the-loop review.

Exposed as the ``create_analysts`` graph in langgraph.json.
"""

from langgraph.graph import START, END, StateGraph

from src.utils.edges import should_continue
from src.utils.nodes import create_analysts, human_feedback
from src.utils.states import GenerateAnalystsState

builder = StateGraph(GenerateAnalystsState)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

graph = builder.compile()
