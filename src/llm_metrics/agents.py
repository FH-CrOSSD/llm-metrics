"""
Agent / LLM helper layer.

Wraps the local Ollama endpoint behind a thin interface so the graph nodes
stay clean.  All LLM calls go through ``call_llm`` which accepts messages
and returns the assistant reply text.

Debate agent personas are defined in ``config.py`` via ``DebateAgent``.
"""

from __future__ import annotations

import json
from typing import Any

import requests

DEFAULT_LLM_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:27b"


def call_llm(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    llm_host: str = DEFAULT_LLM_HOST,
    llm_auth_token: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Send *messages* to Ollama and return the assistant content string."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(
        llm_host.rstrip("/") + "/api/chat",
        json=payload,
        timeout=300,
        headers={"Authorization": f"Bearer {llm_auth_token}"} if llm_auth_token else None,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


# ── Prompt templates ────────────────────────────────────────────────────

SINGLE_AGENT_SYSTEM = (
    "You are a software-repository health analyst. "
    "Your role is to evaluate repository data and compute the specific metric "
    "requested by the user. "
    "Base evaluations strictly on the data provided. If required data is missing, "
    "state this clearly inside the explanation field.\n"
    "You MUST respond with ONLY a JSON object — no markdown, no prose before "
    "or after it. The object must have exactly two keys:\n"
    '  "metric": a single number within the stated scale\n'
    '  "explanation": a concise string summarising your reasoning\n'
    "Example of the required output format:\n"
    '{"metric": 7, "explanation": "The project demonstrates ..."}'
)
