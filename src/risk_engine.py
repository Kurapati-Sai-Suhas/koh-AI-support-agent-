"""Tool: assess_risk() — deterministic policy layer. The LLM never overrides this.

Design position: τ-bench (Yao et al., 2024) showed that frontier function-calling
agents follow domain policy inconsistently — pass^8 under 25% on retail tasks,
i.e. the same task solved once is often failed on a retry. The conclusion we draw
is not "agents are useless" but "policy must not live inside the prompt". Risk
classification and the hard blocks below are ordinary Python, so they behave
identically on every run and can be unit-tested (tests/test_risk_engine.py).

Two independent things are computed here:
  1. risk_level  — how costly is a wrong autonomous answer for this intent?
  2. blocks      — is the customer asking for something the agent may never do
                   autonomously, regardless of how confident it is?
"""
from __future__ import annotations
import re

RISK_BY_INTENT = {
    # money moves or account loss -> a wrong autonomous reply is expensive and
    # publicly visible on Twitter.
    "billing_payment": "HIGH",
    "cancellation_request": "HIGH",
    "account_login_access": "HIGH",      # account takeover / credential handling
    # service faults: a wrong answer wastes the customer's time but is recoverable
    "subscription_plan": "MEDIUM",
    "playback_streaming_issue": "MEDIUM",
    "app_device_bug": "MEDIUM",
    "content_availability": "LOW",
    "feature_how_to": "LOW",
    "general_complaint_feedback": "LOW",
}
DEFAULT_RISK = "MEDIUM"

# Requests the agent must never satisfy on its own. These are matched on the raw
# customer message, so they fire regardless of what the classifier decided.
HARD_BLOCKS = [
    ("legal_or_regulatory",
     r"\b(lawyer|solicitor|legal action|sue|suing|court|gdpr|data protection|ombudsman|"
     r"consumer rights|police|fraud department)\b"),
    ("payment_credentials_present",
     r"\b(\d[ -]?){13,16}\b|\bcvv\b|\bsort code\b|\biban\b|\bpaypal (email|password)\b"),
    ("account_takeover_or_security",
     r"\b(hacked|compromised|someone (else )?(is )?(using|on) my account|stolen|"
     r"unauthorised access|unauthorized access|identity theft)\b"),
    ("explicit_human_request",
     r"\b(speak|talk|transfer|escalate)\b.{0,20}\b(human|person|agent|manager|supervisor|"
     r"real (person|human))\b|\bnot a bot\b"),
    ("vulnerability_or_distress",
     r"\b(suicide|kill myself|self harm|depressed|dying|terminal|bereave|funeral|"
     r"passed away)\b"),
]
HARD_BLOCKS = [(name, re.compile(p, re.I)) for name, p in HARD_BLOCKS]

# Things a grounded reply is never allowed to promise. Enforced twice: stated as
# a constraint in the generation prompt, then re-checked in critic.py, because a
# constraint that is only in the prompt is a suggestion, not a control.
FORBIDDEN_COMMITMENTS = [
    ("promises_refund", r"\b(we (will|'ll)|i (will|'ll)|you (will|'ll) (get|receive))\b.{0,30}\brefund"),
    ("promises_compensation", r"\b(compensat|credit your account|free month|voucher|goodwill payment)\w*\b"),
    ("promises_timeline", r"\b(within|in) \d+[ -](hour|day|week|business day)s?\b"),
    ("claims_account_action", r"\b(i have|i've|we have|we've) (cancelled|canceled|refunded|reset|"
                              r"deleted|upgraded|changed) your\b"),
    ("requests_credentials", r"\b(send|share|dm|give) (us |me )?(your )?(password|card number|cvv|pin|"
                             r"full card|bank details)\b"),
]
FORBIDDEN_COMMITMENTS = [(n, re.compile(p, re.I)) for n, p in FORBIDDEN_COMMITMENTS]

POLICY_CONSTRAINTS = [
    "Never promise a refund, credit, compensation or free service.",
    "Never claim an action has already been taken on the customer's account.",
    "Never ask for passwords, full card numbers, CVV or PIN.",
    "Never invent a resolution timeline or a policy that is not visible in the precedents.",
    "Never state a fact about this customer's account that the precedents do not support.",
    "If the precedents only show the brand asking for more detail, do the same — that is a valid resolution.",
]


def assess_risk(message: str, intent: str) -> dict:
    """-> {risk_level, blocks[], policy_constraints[]}"""
    blocks = [name for name, rx in HARD_BLOCKS if rx.search(message or "")]
    level = RISK_BY_INTENT.get(intent, DEFAULT_RISK)
    if blocks:
        level = "HIGH"
    return {"risk_level": level, "blocks": blocks,
            "policy_constraints": POLICY_CONSTRAINTS}


def check_forbidden(reply: str) -> list[str]:
    """Post-generation guard: which never-say rules did the draft violate?"""
    return [name for name, rx in FORBIDDEN_COMMITMENTS if rx.search(reply or "")]
