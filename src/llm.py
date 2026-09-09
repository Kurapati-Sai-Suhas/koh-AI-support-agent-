"""Thin facade over the provider chain, plus the critical/optional distinction.

The rest of the codebase imports this module and never touches `providers.py`
directly, so swapping or reordering providers changes nothing else.

CRITICAL vs OPTIONAL
--------------------
Not every LLM call deserves to break a request:

  critical   the drafted customer reply, the judge's score. If every provider
             fails, the caller must know — it falls back to the extractive
             backend or escalates, explicitly.
  optional   advisory critique, enrichment. If it fails, we log it internally
             and carry on: `try_chat_json` returns None instead of raising.

This is why an outage degrades the system to "extractive replies, still fully
auditable" rather than to a stack trace.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: F401  (imported for its .env loading side effect)
from providers import LLMError, get_manager, reset_manager  # noqa: F401

__all__ = ["available", "backend_name", "chat", "chat_json", "try_chat_json",
           "provider_status", "LLMError", "reset_manager"]


def available() -> bool:
    return get_manager().available()


def backend_name() -> str:
    return get_manager().label()


def provider_status() -> list[dict]:
    """Which providers are configured and reachable. Never includes a key."""
    return get_manager().status()


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 800,
         json_object: bool = False) -> str:
    content, _ = get_manager().chat(messages, temperature=temperature,
                                    max_tokens=max_tokens, json_mode=json_object)
    return content


def chat_json(messages: list[dict], temperature: float = 0.2,
              max_tokens: int = 800) -> dict:
    """CRITICAL path: raises LLMError if every provider fails."""
    obj, _ = get_manager().chat_json(messages, temperature=temperature,
                                     max_tokens=max_tokens)
    return obj


def chat_json_with_provider(messages: list[dict], temperature: float = 0.2,
                            max_tokens: int = 800) -> tuple[dict, str]:
    """As above, but also returns which provider actually served the request."""
    return get_manager().chat_json(messages, temperature=temperature,
                                   max_tokens=max_tokens)


def try_chat_json(messages: list[dict], temperature: float = 0.2,
                  max_tokens: int = 800) -> tuple[dict | None, str | None]:
    """OPTIONAL path: never raises. Returns (None, None) when unavailable."""
    try:
        return get_manager().chat_json(messages, temperature=temperature,
                                       max_tokens=max_tokens)
    except LLMError:
        return None, None
