"""Unit tests for edge (routing) functions."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.utils.edges import should_continue, route_messages, initiate_all_interviews
from src.utils.objects import Analyst


# ---------------------------------------------------------------------------
# should_continue
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def test_returns_end_when_no_feedback(self):
        state = {"human_analyst_feedback": None}
        assert should_continue(state) == "__end__"

    def test_returns_end_when_feedback_key_missing(self):
        state = {}
        assert should_continue(state) == "__end__"

    def test_returns_create_analysts_when_feedback_given(self):
        state = {"human_analyst_feedback": "Add a finance analyst"}
        assert should_continue(state) == "create_analysts"

    def test_returns_end_for_empty_string_feedback(self):
        """Empty string is falsy, should route to end."""
        state = {"human_analyst_feedback": ""}
        assert should_continue(state) == "__end__"


# ---------------------------------------------------------------------------
# route_messages
# ---------------------------------------------------------------------------

class TestRouteMessages:
    def _make_messages(self, num_expert_answers, include_thank_you=False):
        """Helper to build a realistic message list."""
        messages = []
        for i in range(num_expert_answers):
            messages.append(HumanMessage(content=f"Question {i + 1}"))
            messages.append(AIMessage(content=f"Answer {i + 1}", name="expert"))
        if include_thank_you:
            messages.append(HumanMessage(content="Thank you so much for your help!"))
            messages.append(AIMessage(content="You're welcome!", name="expert"))
        return messages

    def test_continues_when_under_max_turns(self):
        messages = self._make_messages(1)
        state = {"messages": messages, "max_num_turns": 3}
        assert route_messages(state) == "ask_question"

    def test_saves_when_max_turns_reached(self):
        messages = self._make_messages(2)
        state = {"messages": messages, "max_num_turns": 2}
        assert route_messages(state) == "save_interview"

    def test_saves_when_over_max_turns(self):
        messages = self._make_messages(5)
        state = {"messages": messages, "max_num_turns": 3}
        assert route_messages(state) == "save_interview"

    def test_saves_on_thank_you_phrase(self):
        messages = self._make_messages(0, include_thank_you=True)
        state = {"messages": messages, "max_num_turns": 10}
        assert route_messages(state) == "save_interview"

    def test_uses_default_max_turns(self):
        """When max_num_turns is missing, default to 2."""
        messages = self._make_messages(2)
        state = {"messages": messages}
        assert route_messages(state) == "save_interview"

    def test_single_turn_continues(self):
        messages = self._make_messages(1)
        state = {"messages": messages}  # default max_num_turns=2
        assert route_messages(state) == "ask_question"

    # -- Adaptive depth (novelty-based early stopping) --

    def test_stops_early_on_low_novelty(self):
        """Stop when 2 consecutive turns have novelty below threshold."""
        messages = self._make_messages(1)
        state = {
            "messages": messages,
            "max_num_turns": 10,
            "novelty_scores": [0.8, 0.2, 0.1],  # last 2 are low
        }
        assert route_messages(state) == "save_interview"

    def test_continues_with_high_novelty(self):
        """Don't stop when novelty is above threshold."""
        messages = self._make_messages(1)
        state = {
            "messages": messages,
            "max_num_turns": 10,
            "novelty_scores": [0.8, 0.7, 0.6],
        }
        assert route_messages(state) == "ask_question"

    def test_continues_with_one_low_score(self):
        """Don't stop on just 1 low novelty score — need 2 consecutive."""
        messages = self._make_messages(1)
        state = {
            "messages": messages,
            "max_num_turns": 10,
            "novelty_scores": [0.1],  # only 1 low score
        }
        assert route_messages(state) == "ask_question"

    def test_continues_without_novelty_scores(self):
        """Backward compatible: works when novelty_scores is absent."""
        messages = self._make_messages(1)
        state = {"messages": messages, "max_num_turns": 10}
        assert route_messages(state) == "ask_question"

    def test_mixed_novelty_continues(self):
        """Don't stop when last 2 scores are mixed (one high, one low)."""
        messages = self._make_messages(1)
        state = {
            "messages": messages,
            "max_num_turns": 10,
            "novelty_scores": [0.1, 0.6],  # last is above threshold
        }
        assert route_messages(state) == "ask_question"


# ---------------------------------------------------------------------------
# initiate_all_interviews
# ---------------------------------------------------------------------------

class TestInitiateAllInterviews:
    def test_returns_create_analysts_when_feedback(self, sample_analysts):
        state = {
            "human_analyst_feedback": "Need more diversity",
            "analysts": sample_analysts,
            "topic": "AI Safety",
        }
        assert initiate_all_interviews(state) == "create_analysts"

    def test_returns_send_objects_when_no_feedback(self, sample_analysts):
        state = {
            "human_analyst_feedback": None,
            "analysts": sample_analysts,
            "topic": "AI Safety",
        }
        result = initiate_all_interviews(state)
        assert isinstance(result, list)
        assert len(result) == len(sample_analysts)

    def test_returns_send_objects_when_feedback_missing(self, sample_analysts):
        state = {
            "analysts": sample_analysts,
            "topic": "AI Safety",
        }
        result = initiate_all_interviews(state)
        assert isinstance(result, list)
        assert len(result) == len(sample_analysts)
