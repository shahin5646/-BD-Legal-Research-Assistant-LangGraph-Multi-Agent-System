import os
import logging
from typing import Literal
from langchain.messages import AIMessage, HumanMessage
from langgraph.types import Send

from src.utils.states import GenerateAnalystsState, InterviewState, ResearchGraphState

logger = logging.getLogger(__name__)


def should_continue(state: GenerateAnalystsState) -> Literal["create_analysts", "__end__"]:
    """Return the next node to execute based on human feedback."""
    feedback = state.get('human_analyst_feedback')
    if feedback:
        return "create_analysts"
    return "__end__"


def route_messages(state: InterviewState, name: str = "expert"):
    """Route between continuing the interview or saving it.

    Stops the interview when:
    1. Max turns reached (hard ceiling)
    2. Analyst says "Thank you so much for your help" (natural end)
    3. Adaptive: 2 consecutive turns with low novelty (repetitive answers)
    """
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    # Check the number of expert answers
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    # Hard ceiling: end if expert has answered max turns
    if num_responses >= max_num_turns:
        logger.info("Interview ended: max turns (%d) reached", max_num_turns)
        return 'save_interview'

    # Adaptive depth: stop early if info is repetitive
    novelty_scores = state.get("novelty_scores", [])
    novelty_threshold = float(os.getenv("NOVELTY_THRESHOLD", "0.3"))
    if len(novelty_scores) >= 2 and all(s < novelty_threshold for s in novelty_scores[-2:]):
        logger.info(
            "Interview ended early: low novelty for 2 consecutive turns (scores: %s)",
            novelty_scores[-2:],
        )
        return 'save_interview'

    # Natural end: analyst signals completion
    if len(messages) >= 2:
        last_question = messages[-2]
        if "Thank you so much for your help" in last_question.content:
            return 'save_interview'

    return "ask_question"


def initiate_all_interviews(state: ResearchGraphState):
    """The map step where we run each interview sub-graph in parallel using Send API.

    Maps parent state 'interview_turns' into child state 'max_num_turns' avoiding
    parent state update collisions.
    """
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"

    topic = state["topic"]
    turns = state.get("interview_turns") or int(os.getenv("DEFAULT_MAX_TURNS", "2"))

    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "max_num_turns": turns,
            "messages": [HumanMessage(
                content=f"So you said you were writing an article on {topic}?"
            )],
        })
        for analyst in state["analysts"]
    ]
