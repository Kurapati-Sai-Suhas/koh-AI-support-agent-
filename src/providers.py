"""LLM provider abstraction with automatic failover.

    ProviderManager
        |-- OllamaProvider        local, no quota, fast once warm
        |-- GeminiProvider        cloud, fast, free tier has RPM/daily quota
        `-- OpenAICompatProvider  any OpenAI-shaped endpoint (NVIDIA NIM, vLLM, ...)

Callers ask the manager for a completion and never learn which provider served it
beyond a short label. The manager tries providers in order, and moves to the next
one only on a *retryable* failure.

WHY FAILOVER IS ERROR-CLASSIFIED, NOT BLIND
-------------------------------------------
Retrying a 401 or a 404 is pointless — the configuration is wrong and will still
be wrong in two seconds. Retrying a 429 or a 503 is exactly right. So errors are
classified into:

    RETRY_SAME   429 / 5xx / timeout  -> brief backoff, retry this provider,
                                          then fail over
    FAIL_OVER    auth / model-missing / bad-request -> skip straight to the next
                                          provider, no retry
    FATAL        nothing left to try

Each provider is attempted at most `max_attempts` times, and the chain is walked
at most once, so there is no infinite fallback loop.

SECRETS: keys are read from the environment only. `label()` returns
"provider:model" and never the key. Error strings are scrubbed before they are
allowed anywhere near a log or an API response.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))


class LLMError(RuntimeError):
    """Raised when every configured provider failed."""


@dataclass
class ProviderError(Exception):
    provider: str
    kind: str          # "retry_same" | "fail_over"
    detail: str

    def __str__(self) -> str:
        return f"{self.provider}: {self.kind}: {self.detail}"


def _scrub(text: str) -> str:
    """Remove anything key-shaped before an error string is surfaced."""
    text = re.sub(r"(AIza[0-9A-Za-z_\-]{10,})", "<redacted>", text)
    text = re.sub(r"(nvapi-[0-9A-Za-z_\-]{10,})", "<redacted>", text)
    text = re.sub(r"(sk-[0-9A-Za-z_\-]{10,})", "<redacted>", text)
    text = re.sub(r"([?&]key=)[^&\s]+", r"\1<redacted>", text)
    return text


def _post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# --------------------------------------------------------------------------- #
class LLMProvider:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def label(self) -> str:
        raise NotImplementedError

    def chat(self, messages: list[dict], temperature: float, max_tokens: int,
             json_mode: bool) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Local Ollama. No quota, no key, but only as good as the installed model."""

    name = "ollama"

    def __init__(self):
        self.base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama3:latest")
        self._checked: bool | None = None

    def available(self) -> bool:
        if self._checked is not None:
            return self._checked
        try:
            with urllib.request.urlopen(self.base + "/api/tags", timeout=5) as r:
                tags = json.load(r)
            names = {m.get("name", "") for m in tags.get("models", [])}
            # Accept an exact match or the same model under a different tag.
            self._checked = bool(names) and (
                self.model in names
                or any(n.split(":")[0] == self.model.split(":")[0] for n in names))
        except Exception:
            self._checked = False
        return self._checked

    def label(self) -> str:
        return f"ollama:{self.model}"

    def chat(self, messages, temperature, max_tokens, json_mode) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            body = _post(self.base + "/api/chat", payload,
                         {"Content-Type": "application/json"}, DEFAULT_TIMEOUT)
        except urllib.error.HTTPError as e:
            detail = _scrub(str(e.read()[:200]))
            kind = "retry_same" if e.code in (429, 500, 502, 503, 504) else "fail_over"
            raise ProviderError(self.name, kind, f"HTTP {e.code}: {detail}")
        except Exception as e:
            # Connection refused / timeout: the daemon is down, so retrying the
            # same provider is unlikely to help within a request's lifetime.
            raise ProviderError(self.name, "fail_over", _scrub(str(e)))
        content = (body.get("message") or {}).get("content") or ""
        if not content.strip():
            raise ProviderError(self.name, "retry_same", "empty content")
        return content


class GeminiProvider(LLMProvider):
    """Google Gemini via the public generativelanguage REST endpoint."""

    name = "gemini"

    def __init__(self):
        self.key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.base = os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

    def available(self) -> bool:
        return bool(self.key)

    def label(self) -> str:
        return f"gemini:{self.model}"

    def chat(self, messages, temperature, max_tokens, json_mode) -> str:
        # Gemini has no "system" role: fold system turns into the first user turn.
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        parts = [m["content"] for m in messages if m.get("role") != "system"]
        text = ("\n\n".join([system] + parts) if system else "\n\n".join(parts))
        payload = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens},
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        url = f"{self.base}/models/{self.model}:generateContent"
        try:
            body = _post(url, payload,
                         {"Content-Type": "application/json",
                          "x-goog-api-key": self.key},   # header, never the URL
                         DEFAULT_TIMEOUT)
        except urllib.error.HTTPError as e:
            detail = _scrub(str(e.read()[:250]))
            if e.code in (429, 500, 502, 503, 504):
                kind = "retry_same"          # rate limit / transient
            else:
                kind = "fail_over"           # 400 bad request, 403 key, 404 model
            raise ProviderError(self.name, kind, f"HTTP {e.code}: {detail}")
        except Exception as e:
            raise ProviderError(self.name, "retry_same", _scrub(str(e)))

        cands = body.get("candidates") or []
        if not cands:
            fb = (body.get("promptFeedback") or {}).get("blockReason")
            raise ProviderError(self.name, "fail_over", f"no candidates (blockReason={fb})")
        cand = cands[0]
        content = "".join(p.get("text", "")
                          for p in (cand.get("content") or {}).get("parts", []))
        if not content.strip():
            raise ProviderError(self.name, "retry_same",
                                f"empty content (finishReason={cand.get('finishReason')})")
        return content


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-shaped /chat/completions endpoint (NVIDIA NIM, vLLM, Together)."""

    name = "openai_compat"

    def __init__(self):
        self.key = os.getenv("LLM_API_KEY", "")
        self.base = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "")

    def available(self) -> bool:
        return bool(self.key and self.base and self.model)

    def label(self) -> str:
        return f"openai_compat:{self.model}"

    def chat(self, messages, temperature, max_tokens, json_mode) -> str:
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            body = _post(self.base + "/chat/completions", payload,
                         {"Content-Type": "application/json",
                          "Authorization": f"Bearer {self.key}"}, DEFAULT_TIMEOUT)
        except urllib.error.HTTPError as e:
            detail = _scrub(str(e.read()[:200]))
            kind = "retry_same" if e.code in (429, 500, 502, 503, 504) else "fail_over"
            raise ProviderError(self.name, kind, f"HTTP {e.code}: {detail}")
        except Exception as e:
            raise ProviderError(self.name, "retry_same", _scrub(str(e)))
        choice = body["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content")
        if not content:
            # Reasoning models put the chain of thought in reasoning_content and
            # can exhaust max_tokens before emitting an answer.
            if choice.get("finish_reason") == "length":
                raise ProviderError(self.name, "retry_same",
                                    "hit max_tokens before answering")
            content = msg.get("reasoning_content") or ""
        if not content.strip():
            raise ProviderError(self.name, "fail_over", "empty content")
        return content


# --------------------------------------------------------------------------- #
REGISTRY = {"ollama": OllamaProvider, "gemini": GeminiProvider,
            "openai_compat": OpenAICompatProvider}
DEFAULT_ORDER = "ollama,gemini"


@dataclass
class ProviderManager:
    """Tries providers in order; fails over on retryable errors only."""

    order: list[str] = field(default_factory=list)
    max_attempts: int = 2         # per provider, before failing over
    providers: list[LLMProvider] = field(default_factory=list)
    last_errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.order:
            self.order = [p.strip() for p in
                          os.getenv("LLM_PROVIDER_ORDER", DEFAULT_ORDER).split(",")
                          if p.strip()]
        self.providers = [REGISTRY[n]() for n in self.order if n in REGISTRY]

    def active(self) -> list[LLMProvider]:
        return [p for p in self.providers if p.available()]

    def available(self) -> bool:
        return bool(self.active())

    def label(self) -> str:
        a = self.active()
        return a[0].label() if a else "offline"

    def status(self) -> list[dict]:
        return [{"provider": p.name, "model": getattr(p, "model", ""),
                 "available": p.available()} for p in self.providers]

    def chat(self, messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 800, json_mode: bool = False) -> tuple[str, str]:
        """-> (content, provider_label). Raises LLMError only if all providers fail."""
        self.last_errors = []
        active = self.active()
        if not active:
            raise LLMError("no LLM provider is available")
        for prov in active:
            for attempt in range(self.max_attempts):
                try:
                    return prov.chat(messages, temperature, max_tokens, json_mode), prov.label()
                except ProviderError as e:
                    self.last_errors.append(str(e))
                    if e.kind == "retry_same" and attempt < self.max_attempts - 1:
                        time.sleep(min((2 ** attempt) + random.random(), 15))
                        continue
                    break            # fail over to the next provider
                except Exception as e:  # never let an unexpected bug kill the chain
                    self.last_errors.append(f"{prov.name}: unexpected: {_scrub(str(e))}")
                    break
        raise LLMError("all providers failed: " + " | ".join(self.last_errors[-4:]))

    def chat_json(self, messages: list[dict], **kw) -> tuple[dict, str]:
        """Same, but parses a JSON object out of the reply.

        A model that returns prose-wrapped or fenced JSON is a reliability bug in
        practice, so parsing is defensive rather than a bare json.loads.
        """
        raw, label = self.chat(messages, json_mode=True, **kw)
        txt = raw.strip()
        if txt.startswith("```"):
            txt = txt.split("```")[1]
            txt = txt[4:] if txt.lower().startswith("json") else txt
        try:
            return json.loads(txt), label
        except json.JSONDecodeError:
            i, j = txt.find("{"), txt.rfind("}")
            if i >= 0 and j > i:
                try:
                    return json.loads(txt[i:j + 1]), label
                except json.JSONDecodeError:
                    pass
            raise LLMError(f"{label} returned unparseable JSON: {txt[:160]!r}")


_MANAGER: ProviderManager | None = None


def get_manager() -> ProviderManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ProviderManager()
    return _MANAGER


def reset_manager() -> None:
    """Force re-detection (used by tests and after env changes)."""
    global _MANAGER
    _MANAGER = None
