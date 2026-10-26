"""Research cost and performance tracking.

Captures token usage, search counts, and estimated cost across all LLM invocations
(both plain and structured) using a LangChain BaseCallbackHandler, avoiding parallel
state race conditions.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# Standard list pricing (per 1M tokens) for known models
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


@dataclass
class ResearchMetrics:
    """Accumulated metrics for a research run."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_searches: int = 0
    llm_calls: int = 0
    analysts_used: int = 0
    model_name: str = ""
    start_time: Optional[float] = None

    def start_timer(self):
        """Set the start time once at the start of a run."""
        if self.start_time is None:
            self.start_time = time.time()

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def estimated_cost(self) -> Optional[float]:
        """Estimated USD cost based on token counts.

        Returns None if model pricing is unknown and no env override is provided.
        """
        env_input = os.getenv("PRICE_INPUT_PER_1M")
        env_output = os.getenv("PRICE_OUTPUT_PER_1M")

        if env_input is not None and env_output is not None:
            try:
                p_in = float(env_input)
                p_out = float(env_output)
                return (
                    self.total_input_tokens * p_in / 1_000_000
                    + self.total_output_tokens * p_out / 1_000_000
                )
            except ValueError:
                pass

        # Check model name in known table
        model_key = (self.model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")).lower()
        pricing = None
        for k, v in MODEL_PRICING.items():
            if k in model_key:
                pricing = v
                break

        if pricing is not None:
            return (
                self.total_input_tokens * pricing["input"] / 1_000_000
                + self.total_output_tokens * pricing["output"] / 1_000_000
            )

        return None

    def record_llm_call(self, input_tokens: int = 0, output_tokens: int = 0):
        """Record an LLM call directly if not using callback."""
        self.llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_search(self):
        """Record a single web search."""
        self.total_searches += 1

    def reset(self):
        """In-place reset of all metrics for a fresh run."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_searches = 0
        self.llm_calls = 0
        self.analysts_used = 0
        self.model_name = ""
        self.start_time = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a dictionary for JSON exports."""
        return {
            "analysts_used": self.analysts_used,
            "llm_calls": self.llm_calls,
            "total_searches": self.total_searches,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.estimated_cost,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "model_name": self.model_name or os.getenv("LLM_MODEL", "gpt-4o-mini"),
        }

    def to_markdown(self) -> str:
        """Format metrics as a markdown table for the report."""
        elapsed = self.elapsed_seconds
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds:.1f}s"
        cost_val = self.estimated_cost
        cost_str = f"${cost_val:.4f}" if cost_val is not None else "n/a"

        return (
            "\n\n## Research Metadata\n"
            "| Metric | Value |\n"
            "|---|---|\n"
            f"| Legal Analysts | {self.analysts_used} |\n"
            f"| LLM Calls | {self.llm_calls} |\n"
            f"| Web Searches | {self.total_searches} |\n"
            f"| Input Tokens | {self.total_input_tokens:,} |\n"
            f"| Output Tokens | {self.total_output_tokens:,} |\n"
            f"| Estimated Cost | {cost_str} |\n"
            f"| Time Elapsed | {time_str} |\n"
        )


class ResearchMetricsCallback(BaseCallbackHandler):
    """LangChain callback to automatically capture usage metadata from all LLM calls."""

    def __init__(self, metrics: ResearchMetrics):
        super().__init__()
        self.metrics = metrics

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.metrics.start_timer()
        if "name" in serialized and not self.metrics.model_name:
            self.metrics.model_name = serialized["name"]

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.metrics.llm_calls += 1

        in_tok = 0
        out_tok = 0

        # 1. Standard llm_output token_usage
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            in_tok = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            out_tok = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        # 2. Check generation metadata (ChatOpenAI & structured output fallback)
        if in_tok == 0 and out_tok == 0 and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    usage = getattr(msg, "usage_metadata", None) if msg else None
                    if usage:
                        in_tok += usage.get("input_tokens", 0)
                        out_tok += usage.get("output_tokens", 0)

        self.metrics.total_input_tokens += in_tok
        self.metrics.total_output_tokens += out_tok


# Module-level instance
tracker = ResearchMetrics()
metrics_callback = ResearchMetricsCallback(tracker)


def reset_tracker():
    """In-place reset of the shared tracker."""
    tracker.reset()
