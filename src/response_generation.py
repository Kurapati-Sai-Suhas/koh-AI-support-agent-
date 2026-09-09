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
        "reply": best["historical_reply"],
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
    try:
        obj = llm.chat_json([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
                            temperature=0.2, max_tokens=500)
    except llm.LLMError as e:
        out = _fallback(message, intent, cases, evidence_quality)
        out["uncertainty"] += f" (LLM call failed: {e})"
        return out
    return {
        "reply": str(obj.get("reply", ""))[:600],
        "grounded": bool(obj.get("grounded", False)),
        "evidence_used": list(obj.get("evidence_used", []) or []),
        "uncertainty": str(obj.get("uncertainty", "")),
        "should_escalate": bool(obj.get("should_escalate", False)),
        "backend": llm.backend_name(),
    }
