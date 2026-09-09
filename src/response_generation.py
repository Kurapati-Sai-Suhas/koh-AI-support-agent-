"""Tool: generate_reply() — grounded drafting, with a deterministic fallback.

Grounding contract: the model sees ONLY the customer message, the predicted
intent, the retrieved precedents, and the policy constraints. It is instructed to
cite which precedents it used. It has no other knowledge source to draw a policy
from, which is the cheapest available defence against inventing one.

Both backends return the same structured object, so everything downstream
(critic, API, evaluation) is backend-agnostic:
    {reply, grounded, evidence_used[], uncertainty, should_escalate, backend}
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import llm
from risk_engine import POLICY_CONSTRAINTS

SYSTEM = (
    f"You draft public Twitter replies for {C.BRAND}, the customer support account. "
    "You reply ONLY in the way this brand has historically replied, and you may use "
    "nothing except the precedent replies supplied to you as a source of policy or fact. "
    "You are drafting for a human reviewer, not sending directly."
)

TEMPLATE = """CUSTOMER MESSAGE
{message}

CLASSIFIER OUTPUT
intent: {intent} (confidence {conf:.2f})
risk level: {risk}
evidence quality: {eq:.2f}

HISTORICAL PRECEDENTS — how this brand actually handled similar messages
{cases}

POLICY CONSTRAINTS (hard rules, violating any one makes the draft unusable)
{policy}

INSTRUCTIONS
1. Write one reply, under 280 characters, in the voice of the precedents.
2. Ground every commitment in a precedent. If the precedents only ask the customer
   for more information, then ask for the same information — that is the correct reply,
   not a weakness.
3. If the precedents do not cover this situation, say so in "uncertainty" and set
   should_escalate to true rather than inventing a resolution.
4. List the precedent numbers you actually used in evidence_used.

Return ONLY a JSON object:
{{"reply": "...", "grounded": true, "evidence_used": [1,2],
  "uncertainty": "...", "should_escalate": false}}"""


def _clean_draft(reply: str) -> str:
    """Strip handles the model copied out of the precedents.

    Found by human review of live output: drafts came back starting with
    "@user ...", "@328829 ..." or even "@SpotifyCares ..." — the model imitates
    the precedent replies, which all begin with the customer handle, and our
    own preprocessing placeholder leaks in too. A public draft addressed to
    "@user", or to the brand's own account, is unusable as-is.

    The reply is posted in-thread, so no leading handle is needed at all.
    """
    r = (reply or "").strip()
    r = re.sub(r"^\s*(?:@[\w<>]+\s+)+", "", r)          # leading handles
    r = re.sub(r"@" + re.escape(C.BRAND) + r"\b", "", r, flags=re.I)  # brand self-mention
    r = re.sub(r"@?<user>|@user\b", "", r, flags=re.I)  # placeholder leakage
    r = re.sub(r"\s{2,}", " ", r).strip()
    # A stray leading "@" survives when the model glues the handle to the first
    # word ("@Hey!") or when a substitution above removes the handle but not the
    # sigil. Seen live from llama3.
    r = re.sub(r"^@(?=[A-Za-z])", "", r).strip()
    return r


def _format_cases(cases: list[dict]) -> str:
    if not cases:
        return "(none retrieved)"
    out = []
    for c in cases:
        out.append(
            f"[{c['rank']}] similarity={c['similarity']:.2f} intent={c['historical_intent']}\n"
            f"    customer said: {c['customer_message'][:220]}\n"
            f"    brand replied: {c['historical_reply'][:220]}"
        )
    return "\n".join(out)


def _fallback(message: str, intent: str, cases: list[dict], evidence_quality: float) -> dict:
    """Offline backend: reuse the best substantive precedent verbatim.

    This is intentionally extractive, not generative. It cannot hallucinate a
    policy because it writes no new claims — it surfaces the brand's own prior
    wording and says where it came from. Lower fluency, zero fabrication risk.
    """
    usable = [c for c in cases if not c["is_boilerplate"]] or cases
    if not usable:
        return {"reply": "", "grounded": False, "evidence_used": [],
                "uncertainty": "No historical precedent was retrieved for this message.",
                "should_escalate": True, "backend": "template"}
    best = usable[0]
    return {
        "reply": _clean_draft(best["historical_reply"]),
        "grounded": True,
        "evidence_used": [best["rank"]],
        "uncertainty": ("Extractive fallback: this is the brand's own reply to the closest "
                        f"historical case (similarity {best['similarity']:.2f}), reused verbatim "
                        "rather than rewritten, because no LLM backend is configured."),
        "should_escalate": bool(evidence_quality < 0.35),
        "backend": "template",
    }


def generate_reply(message: str, intent: str, confidence: float, cases: list[dict],
                   evidence_quality: float, risk_level: str) -> dict:
    if not llm.available():
        return _fallback(message, intent, cases, evidence_quality)
    prompt = TEMPLATE.format(
        message=message, intent=intent, conf=confidence, risk=risk_level,
        eq=evidence_quality, cases=_format_cases(cases),
        policy="\n".join(f"- {p}" for p in POLICY_CONSTRAINTS),
    )
    # CRITICAL call: the customer-facing draft. If every provider fails we do not
    # raise — we degrade to the extractive backend and say so, because a support
    # queue with no draft is worse than a draft in the brand's own prior words.
    try:
        obj, served_by = llm.chat_json_with_provider(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=2500)
    except llm.LLMError as e:
        out = _fallback(message, intent, cases, evidence_quality)
        out["uncertainty"] += f" (all LLM providers failed: {e})"
        out["provider_failed"] = True
        return out
    return {
        "reply": _clean_draft(str(obj.get("reply", "")))[:600],
        "grounded": bool(obj.get("grounded", False)),
        "evidence_used": list(obj.get("evidence_used", []) or []),
        "uncertainty": str(obj.get("uncertainty", "")),
        "should_escalate": bool(obj.get("should_escalate", False)),
        "backend": served_by,
        "provider_failed": False,
    }
