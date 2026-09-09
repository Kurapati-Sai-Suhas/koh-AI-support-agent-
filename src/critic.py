"""Tool: critique_reply() — the draft must pass checks the drafter cannot fake.

Research anchor: Huang et al., "Large Language Models Cannot Self-Correct
Reasoning Yet" (ICLR 2024) — intrinsic self-correction, where a model revises
itself with no external signal, does not reliably improve and can degrade output.
So this critic is NOT "ask the LLM if its answer was good". Every verdict-changing
check is deterministic and computed against external evidence:

  1. forbidden commitments  -> regex over the draft (risk_engine.check_forbidden)
  2. lexical grounding      -> content-word overlap with the retrieved replies
  3. invented links         -> any URL not present in the precedents
  4. channel fit            -> Twitter length limit

An optional LLM critique can be layered on top when a key is configured, but it
can only ever *lower* the verdict (PASS -> REVISE), never raise it. Advisory, not
authoritative.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_engine import check_forbidden
from text_utils import URL_RE

STOP = set("""a an the and or but if then than that this these those is are was were be been
being am do does did doing have has had having i you he she it we they me him her us them my
your his its our their to of in on at for with without from by as so not no nor can could
will would shall should may might must about into over under again further once here there
all any both each few more most other some such only own same very just now please thanks
thank hi hey hello""".split())
GROUNDING_MIN = 0.18  # share of the draft's content words seen in the precedents


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'-]{2,}", (text or "").lower())
            if w not in STOP}


def lexical_grounding(reply: str, cases: list[dict]) -> float:
    """Share of the draft's content words that appear in the precedent replies.

    A blunt proxy for faithfulness — it measures word reuse, not semantic
    entailment, and will over-credit a paraphrase-free copy. Stated as a
    limitation in the report; an NLI entailment check is the one-week upgrade.
    """
    words = _content_words(reply)
    if not words:
        return 0.0
    evidence = _content_words(" ".join(c["historical_reply"] for c in cases))
    return round(len(words & evidence) / len(words), 4)


def critique_reply(reply: str, cases: list[dict], risk_level: str) -> dict:
    """-> {verdict: PASS|REVISE|ESCALATE, issues[], grounding_score}"""
    issues: list[str] = []
    verdict = "PASS"

    if not (reply or "").strip():
        return {"verdict": "ESCALATE", "issues": ["empty_draft"], "grounding_score": 0.0}

    forbidden = check_forbidden(reply)
    if forbidden:
        issues += [f"policy_violation:{f}" for f in forbidden]
        verdict = "ESCALATE"          # a promise we cannot keep never goes out

    g = lexical_grounding(reply, cases)
    if g < GROUNDING_MIN:
        issues.append(f"low_grounding:{g:.2f}")
        verdict = "ESCALATE" if verdict == "ESCALATE" else "REVISE"

    reply_urls = set(URL_RE.findall(reply))
    evidence_urls = set(URL_RE.findall(" ".join(c["historical_reply"] for c in cases)))
    invented = reply_urls - evidence_urls
    if invented:
        issues.append("invented_link")
        verdict = "ESCALATE"

    if len(reply) > 280:
        issues.append("over_twitter_length")
        verdict = "ESCALATE" if verdict == "ESCALATE" else "REVISE"

    # High-risk cases are held to a stricter grounding bar.
    if risk_level == "HIGH" and g < GROUNDING_MIN * 1.5:
        issues.append("high_risk_insufficient_grounding")
        verdict = "ESCALATE"

    return {"verdict": verdict, "issues": issues, "grounding_score": g}
