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
import sys
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


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
    try:
        with urllib.request.urlopen(req, timeout=C.LLM_TIMEOUT) as r:
            body = json.load(r)
        return body["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
    except Exception as e:
        raise LLMError(str(e)) from e


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
