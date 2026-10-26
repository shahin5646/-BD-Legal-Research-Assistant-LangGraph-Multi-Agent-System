"""LLM configuration — single source of truth for model instantiation and access.

Reads from environment variables defined in .env / .env.example:
  INMETECH_API_KEY / OPENAI_API_KEY
  OPENAI_BASE_URL    (optional, defaults according to active key)
  LLM_MODEL          (default: gpt-4o-mini)
  LLM_TEMPERATURE    (default: 0)
  LLM_TIMEOUT        (default: 60)
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.utils.cost_tracker import metrics_callback

load_dotenv()

logger = logging.getLogger(__name__)


def get_llm(callbacks: Optional[list] = None) -> ChatOpenAI:
    """Create and return a configured ChatOpenAI instance with single-point accessor."""
    openai_key = os.getenv("OPENAI_API_KEY")
    inmetech_key = os.getenv("INMETECH_API_KEY")

    api_key = openai_key or inmetech_key
    if not api_key:
        raise ValueError("Missing API key: please set OPENAI_API_KEY or INMETECH_API_KEY in your .env file.")

    explicit_base_url = os.getenv("OPENAI_BASE_URL")
    if explicit_base_url:
        base_url = explicit_base_url
    elif inmetech_key and not openai_key:
        base_url = "https://api.inmetech.com/api/v1"
    else:
        # Standard OpenAI API endpoint
        base_url = None

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))

    cb = callbacks if callbacks is not None else [metrics_callback]

    client = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=0,  # Retries are handled centrally by tenacity predicate
        callbacks=cb,
    )
    return client


# Default singleton instance (uses metrics_callback)
llm = get_llm()
