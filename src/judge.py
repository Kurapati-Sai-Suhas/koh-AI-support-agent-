"""LLM-as-judge for reply quality, plus a deterministic fallback judge.

Rubric design follows the practice established by Zheng et al., "Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023): single-answer
grading against an explicit rubric, with a written justification before the
scores. Known biases from that paper and from Panickssery et al. (NeurIPS 2024)
are handled explicitly rather than ignored:

  position bias      not applicable — we grade one reply at a time, no pairwise ordering
  verbosity bias     the rubric scores grounding against evidence, and replies are
                     length-capped at 280 chars, so padding cannot buy a higher score
  self-enhancement   REAL AND UNMITIGATED when one key drafts and judges. Set
                     JUDGE_MODEL to a different model than LLM_MODEL to avoid it.
                     The output records both model ids so the report can say which
                     configuration produced the numbers.

The fallback judge is NOT an LLM and never pretends to be. It scores the same six
dimensions from computable quantities (evidence overlap, policy-violation regexes,
length, question-answering shape). Every record carries judge_backend, so no
number in the report can be mistaken for an LLM judgment when it is not one.
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import llm
from critic import lexical_grounding
from risk_engine import check_forbidden

JUDGE_MODEL = os.getenv("JUDGE_MODEL", C.LLM_MODEL)

DIMENSIONS = ["relevance", "correctness", "grounding", "helpfulness", "tone",
              "hallucination_risk"]

RUBRIC = """You are grading a customer support reply drafted for {brand}'s Twitter account.

Score each dimension 1-5. Judge ONLY against the historical precedents shown —
they are the sole source of truth for what this brand does.

  relevance          5 = directly addresses what the customer asked; 1 = off-topic
  correctness        5 = consistent with how the brand handled the precedents; 1 = contradicts them
  grounding          5 = every claim traceable to a precedent; 1 = claims with no support
  helpfulness        5 = moves the customer forward; 1 = wastes their time
  tone               5 = matches the brand voice in the precedents; 1 = rude or robotic
  hallucination_risk 5 = invents nothing; 1 = invents policy, refunds, timelines or account facts

Asking the customer for more detail IS a good reply when the precedents do that.
Do not reward length. Do not reward politeness that is not in the precedents.

CUSTOMER MESSAGE
{message}

PREDICTED INTENT: {intent}

HISTORICAL PRECEDENTS
{cases}

REPLY UNDER EVALUATION
{reply}

Return ONLY JSON:
{{"reason": "one or two sentences first", "relevance": 1-5, "correctness": 1-5,
  "grounding": 1-5, "helpfulness": 1-5, "tone": 1-5, "hallucination_risk": 1-5,
  "overall": 1-5}}"""


def _fmt(cases: list[dict]) -> str:
    if not cases:
        return "(no precedents retrieved)"
    return "\n".join(
        f"[{c['rank']}] sim={c['similarity']:.2f}\n"
        f"    customer: {c['customer_message'][:200]}\n"
        f"    brand:    {c['historical_reply'][:200]}"
        for c in cases)


def _clip(x, lo=1, hi=5) -> int:
    try:
        return int(max(lo, min(hi, round(float(x)))))
    except (TypeError, ValueError):
        return 3


def heuristic_judge(message: str, reply: str, cases: list[dict], intent: str) -> dict:
    """Deterministic stand-in. Transparent, cheap, and clearly labelled."""
    if not (reply or "").strip():
        return {**{d: 1 for d in DIMENSIONS}, "overall": 1,
                "reason": "empty reply", "judge_backend": "heuristic"}

    g = lexical_grounding(reply, cases)
    violations = check_forbidden(reply)
    top_sim = max((c["similarity"] for c in cases), default=0.0)
    substantive = any(not c["is_boilerplate"] for c in cases)

    grounding = _clip(1 + 4 * min(g / 0.45, 1.0))
    hallucination = _clip(5 - 2 * len(violations) - (2 if g < 0.15 else 0))
    relevance = _clip(1 + 4 * min(top_sim / 0.45, 1.0))
    correctness = _clip((grounding + relevance) / 2)
    helpfulness = _clip(relevance - (1 if not substantive else 0))
    # tone: brand replies in this dataset are warm and short
    polite = bool(re.search(r"\b(sorry|thanks|thank you|happy to|we can|let'?s|hey|hi)\b",
                            reply, re.I))
    tone = _clip(4 + (1 if polite else -1) - (1 if len(reply) > 280 else 0))
    overall = _clip((relevance + correctness + grounding + helpfulness + tone + hallucination) / 6)
    return {"relevance": relevance, "correctness": correctness, "grounding": grounding,
            "helpfulness": helpfulness, "tone": tone, "hallucination_risk": hallucination,
            "overall": overall,
            "reason": (f"lexical grounding {g:.2f}, top similarity {top_sim:.2f}, "
                       f"{len(violations)} policy violation(s)"),
            "judge_backend": "heuristic"}


def judge_reply(message: str, reply: str, cases: list[dict], intent: str) -> dict:
    if not llm.available():
        return heuristic_judge(message, reply, cases, intent)
    prompt = RUBRIC.format(brand=C.BRAND, message=message, intent=intent,
                           cases=_fmt(cases), reply=reply)
    try:
        saved = C.LLM_MODEL
        C.LLM_MODEL = JUDGE_MODEL           # judge may differ from the drafter
        try:
            obj = llm.chat_json([{"role": "user", "content": prompt}],
                                temperature=0.0, max_tokens=400)
        finally:
            C.LLM_MODEL = saved
    except llm.LLMError as e:
        out = heuristic_judge(message, reply, cases, intent)
        out["reason"] += f" | LLM judge failed: {e}"
        return out
    out = {d: _clip(obj.get(d, 3)) for d in DIMENSIONS}
    out["overall"] = _clip(obj.get("overall", sum(out.values()) / len(out)))
    out["reason"] = str(obj.get("reason", ""))[:400]
    out["judge_backend"] = f"llm:{JUDGE_MODEL}"
    out["self_graded"] = (JUDGE_MODEL == C.LLM_MODEL)
    return out
