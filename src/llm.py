"""Thin OpenAI-compatible LLM client + an explicit offline fallback.

Works unchanged against NVIDIA NIM (https://integrate.api.nvidia.com/v1),
OpenAI, Together, Groq, or a local vLLM server — set LLM_BASE_URL / LLM_API_KEY
/ LLM_MODEL in .env.

Written on urllib rather than the openai SDK on purpose: one less dependency to
install inside the 15-minute reproduction budget, and the request shape is
visible in the code, which matters when the interviewer asks what is actually
being sent to the model.

`available()` is checked everywhere before use. When no key is configured the
system does NOT silently degrade into pretending: every artefact it produces is
stamped with backend="template"/"heuristic" so no number in the report can be
mistaken for an LLM result.
"""
from __future__ import annotations
import json
import random
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# --- rate limiting -----------------------------------------------------------
# Free inference tiers (NVIDIA NIM in particular) rate-limit aggressively. A
# 200-row golden-set build hammers the endpoint and gets HTTP 429 on nearly every
# call, which silently degrades the whole run to the fallback backend. A minimum
# spacing between requests plus retry-with-backoff turns that from a data-quality
# failure into a slower-but-correct run.
MIN_INTERVAL_S = float(__import__("os").getenv("LLM_MIN_INTERVAL", "1.5"))
MAX_RETRIES = int(__import__("os").getenv("LLM_MAX_RETRIES", "5"))
_lock = threading.Lock()
_last_call = [0.0]


def _throttle():
    with _lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def available() -> bool:
    return bool(C.LLM_API_KEY)


def backend_name() -> str:
    return f"llm:{C.LLM_MODEL}" if available() else "offline"


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 700,
         json_object: bool = False) -> str:
    if not available():
        raise LLMError("no LLM_API_KEY configured")
    payload = {
        "model": C.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_object:
        # Supported by OpenAI + NIM; harmless extra key on servers that ignore it.
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        C.LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {C.LLM_API_KEY}"},
        method="POST",
    )
    body = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=C.LLM_TIMEOUT) as r:
                body = json.load(r)
            break
        except urllib.error.HTTPError as e:
            # 429 = rate limited, 5xx = transient. Back off and retry; anything
            # else (401, 404, 410) is a configuration error that retrying cannot fix.
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                delay = (float(retry_after) if retry_after and str(retry_after).isdigit()
                         else (2 ** attempt) + random.random())
                time.sleep(min(delay, 30))
                continue
            raise LLMError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((2 ** attempt) + random.random())
                continue
            raise LLMError(str(e)) from e
    if body is None:
        raise LLMError("no response after retries")
    choice = body["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content")
    # Reasoning models split their output: the chain of thought goes to
    # `reasoning_content` and can consume the entire max_tokens budget, leaving
    # `content` empty with finish_reason="length". Treat that as a real error
    # rather than returning None into a .strip() downstream.
    if not content:
        if choice.get("finish_reason") == "length":
            raise LLMError("model hit max_tokens before emitting an answer "
                           "(reasoning model — raise max_tokens)")
        content = msg.get("reasoning_content") or ""
    if not content:
        raise LLMError(f"empty response (finish_reason={choice.get('finish_reason')})")
    return content


def chat_json(messages: list[dict], **kw) -> dict:
    """Call the model and parse a JSON object out of the reply.

    Models wrap JSON in prose or fences often enough that a bare json.loads is a
    reliability bug, so we strip fences and fall back to the outermost {...}.
    """
    raw = chat(messages, json_object=True, **kw)
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        txt = txt[4:] if txt.lower().startswith("json") else txt
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            return json.loads(txt[i:j + 1])
        raise LLMError(f"could not parse JSON from: {raw[:200]!r}")
