"""Provider failover tests. No network: providers are stubbed.

Covers the behaviour the README claims:
  - failover happens on retryable errors (429/5xx/timeout)
  - failover does NOT waste attempts on auth/config errors
  - both providers down degrades in a controlled way, no infinite loop
  - keys never appear in a label, status or error string
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from providers import (ProviderManager, LLMProvider, ProviderError,  # noqa: E402
                       LLMError, _scrub)


class StubProvider(LLMProvider):
    """Scripted provider: each entry is either a reply string or a ProviderError."""

    def __init__(self, name, script, model="stub"):
        self.name = name
        self.model = model
        self.script = list(script)
        self.calls = 0

    def available(self):
        return True

    def label(self):
        return f"{self.name}:{self.model}"

    def chat(self, messages, temperature, max_tokens, json_mode):
        self.calls += 1
        item = self.script.pop(0) if self.script else "{}"
        if isinstance(item, Exception):
            raise item
        return item


def mgr(*provs, max_attempts=2):
    m = ProviderManager(order=[], max_attempts=max_attempts)
    m.providers = list(provs)
    return m


def rate_limited(name):
    return ProviderError(name, "retry_same", "HTTP 429: Too Many Requests")


def auth_error(name):
    return ProviderError(name, "fail_over", "HTTP 401: invalid key")


# ---------- happy path -------------------------------------------------------
def test_primary_serves_and_secondary_is_never_called():
    """No wasted calls: the fallback must not be invoked when primary works."""
    a = StubProvider("ollama", ['{"ok":1}'])
    b = StubProvider("gemini", ['{"ok":2}'])
    out, label = mgr(a, b).chat_json([{"role": "user", "content": "hi"}])
    assert out == {"ok": 1}
    assert label == "ollama:stub"
    assert b.calls == 0


# ---------- TEST G: provider failure -> fallback -----------------------------
def test_rate_limited_primary_falls_over_to_secondary():
    a = StubProvider("gemini", [rate_limited("gemini"), rate_limited("gemini")])
    b = StubProvider("ollama", ['{"ok":"served by fallback"}'])
    out, label = mgr(a, b).chat_json([{"role": "user", "content": "hi"}])
    assert out["ok"] == "served by fallback"
    assert label == "ollama:stub"
    assert a.calls == 2      # retried the rate limit before failing over
    assert b.calls == 1


def test_auth_error_does_not_retry_same_provider():
    """Retrying a bad key is pointless — fail over immediately."""
    a = StubProvider("gemini", [auth_error("gemini"), '{"never":"reached"}'])
    b = StubProvider("ollama", ['{"ok":1}'])
    out, _ = mgr(a, b).chat_json([{"role": "user", "content": "hi"}])
    assert out == {"ok": 1}
    assert a.calls == 1      # exactly one attempt, no retry


def test_failover_works_in_either_direction():
    a = StubProvider("ollama", [ProviderError("ollama", "fail_over", "connection refused")])
    b = StubProvider("gemini", ['{"ok":1}'])
    out, label = mgr(a, b).chat_json([{"role": "user", "content": "hi"}])
    assert out == {"ok": 1} and label == "gemini:stub"


# ---------- TEST H: both providers fail -> controlled failure ----------------
def test_all_providers_failing_raises_llmerror_not_a_crash():
    a = StubProvider("gemini", [rate_limited("gemini")] * 5)
    b = StubProvider("ollama", [rate_limited("ollama")] * 5)
    with pytest.raises(LLMError) as e:
        mgr(a, b).chat_json([{"role": "user", "content": "hi"}])
    assert "all providers failed" in str(e.value)


def test_chain_is_walked_at_most_once_no_infinite_loop():
    a = StubProvider("gemini", [rate_limited("gemini")] * 10)
    b = StubProvider("ollama", [rate_limited("ollama")] * 10)
    m = mgr(a, b, max_attempts=2)
    with pytest.raises(LLMError):
        m.chat_json([{"role": "user", "content": "hi"}])
    assert a.calls == 2 and b.calls == 2   # bounded, not looping


def test_no_provider_available_raises_cleanly():
    m = ProviderManager(order=[])
    m.providers = []
    assert not m.available()
    assert m.label() == "offline"
    with pytest.raises(LLMError):
        m.chat([{"role": "user", "content": "hi"}])


# ---------- malformed output -------------------------------------------------
def test_fenced_json_is_parsed():
    a = StubProvider("ollama", ['```json\n{"reply":"hi"}\n```'])
    out, _ = mgr(a).chat_json([{"role": "user", "content": "x"}])
    assert out == {"reply": "hi"}


def test_json_embedded_in_prose_is_recovered():
    a = StubProvider("ollama", ['Sure! Here it is: {"reply":"hi"} hope that helps'])
    out, _ = mgr(a).chat_json([{"role": "user", "content": "x"}])
    assert out == {"reply": "hi"}


def test_unparseable_output_raises_llmerror():
    a = StubProvider("ollama", ["not json at all"])
    with pytest.raises(LLMError):
        mgr(a).chat_json([{"role": "user", "content": "x"}])


# ---------- secrets ----------------------------------------------------------
@pytest.mark.parametrize("secret", [
    "AIzaSyA1234567890abcdefghijklmnop",
    "nvapi-QHnra37E6ATEOoWCtF9Iq0000000",
    "sk-abcdefghijklmnopqrstuvwxyz1234",
])
def test_scrub_removes_key_shapes(secret):
    assert secret not in _scrub(f"HTTP 401: bad key {secret} rejected")
    assert "<redacted>" in _scrub(f"HTTP 401: bad key {secret} rejected")


def test_scrub_removes_key_query_param():
    assert "<redacted>" in _scrub("https://api.example.com/v1?key=AIzaSecretValue123")


def test_status_never_contains_a_key():
    m = ProviderManager(order=["ollama", "gemini", "openai_compat"])
    blob = repr(m.status()) + m.label()
    for marker in ("AIza", "nvapi-", "sk-"):
        assert marker not in blob
