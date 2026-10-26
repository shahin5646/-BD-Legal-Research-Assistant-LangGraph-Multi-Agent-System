"""Tests that the three graphs compile without errors."""

import pytest


class TestGraphCompilation:
    """Verify that each graph compiles and has the expected structure."""

    def test_analyst_graph_compiles(self):
        from src.agent import graph
        assert graph is not None
        # The graph should have the core nodes
        node_names = set(graph.get_graph().nodes.keys())
        assert "create_analysts" in node_names
        assert "human_feedback" in node_names

    def test_interview_graph_compiles(self):
        from src.answering_questions import question_answer_graph
        assert question_answer_graph is not None
        node_names = set(question_answer_graph.get_graph().nodes.keys())
        assert "ask_question" in node_names
        assert "answer_question" in node_names
        assert "search_web" in node_names
        assert "save_interview" in node_names
        assert "write_section" in node_names

    def test_deep_agent_graph_compiles(self):
        from src.deep_agent import graph
        assert graph is not None
        node_names = set(graph.get_graph().nodes.keys())
        assert "create_analysts" in node_names
        assert "conduct_interview" in node_names
        assert "write_report" in node_names
        assert "finalize_report" in node_names
