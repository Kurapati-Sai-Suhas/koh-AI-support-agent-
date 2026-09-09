"""The orchestrator: a hybrid agent over a fixed tool graph.

WHY THE CONTROL FLOW IS DETERMINISTIC AND NOT LLM-PLANNED
---------------------------------------------------------
ReAct (Yao et al., ICLR 2023) showed an LLM can interleave reasoning with tool
calls; τ-bench (Yao et al., 2024) then measured what that costs in a customer
support setting — frontier function-calling agents were both weak (<50% task
success) and *inconsistent* (pass^8 under 25% in retail), meaning the same case
handled correctly once is often handled wrongly on a retry. For a support agent
whose main promise is "you can trust the escalation decision", non-determinism in
the control path is the one thing we cannot afford.

So the split is:
  agent (this file)  — fixed tool order, state, error handling, final decision
  ML models          — intent probabilities, calibration, similarity retrieval
  deterministic rules— risk level, hard blocks, forbidden commitments, thresholds
  LLM                — drafting language, and an advisory critique. Nothing else.

The LLM cannot change the decision, cannot lower the risk level, and cannot
bypass a hard block. It writes prose inside a box the rest of the system defines.
Every step appends to state["trace"], so any decision can be replayed.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from feature_engineering import make_frame
from retrieval import HistoricalCaseIndex
from evidence import validate_evidence
from risk_engine import assess_risk
from response_generation import generate_reply
from critic import critique_reply
import llm

THRESHOLDS_PATH = C.MODELS / "thresholds.json"


def _load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        return json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    return {"confidence": C.CONF_THRESHOLD, "margin": C.MARGIN_THRESHOLD,
            "evidence_quality": 0.30, "source": "config defaults (not yet tuned)"}


class SupportAgent:
    """Loads once, serves many. Model artefacts are produced by our own pipeline
    (joblib), never fetched from an untrusted source."""

    def __init__(self):
        bundle = joblib.load(C.MODELS / "intent_model.joblib")
        self.model = bundle["model"]
        self.classes = np.asarray(bundle["classes"])
        self.calibrator = None
        cal_path = C.MODELS / "calibrator.joblib"
        if cal_path.exists():
            self.calibrator = joblib.load(cal_path)
        self.index = HistoricalCaseIndex.load()
        self.thresholds = _load_thresholds()

    # ---- tools ---------------------------------------------------------
    def classify_intent(self, message: str) -> dict:
        X = make_frame([message], C.BRAND)
        if self.calibrator is not None:
            proba = self.calibrator.predict_proba(X)[0]
            classes = np.asarray(self.calibrator.classes_)
            calibrated = True
        else:
            proba = self.model.predict_proba(X)[0]
            classes = np.asarray(self.model.classes_)
            calibrated = False
        order = np.argsort(-proba)
        top, second = order[0], order[1] if len(order) > 1 else order[0]
        return {
            "intent": str(classes[top]),
            "intent_confidence": round(float(proba[top]), 4),
            "margin": round(float(proba[top] - proba[second]), 4),
            "runner_up": str(classes[second]),
            "calibrated": calibrated,
            "distribution": {str(c): round(float(p), 4)
                             for c, p in zip(classes[order][:4], proba[order][:4])},
        }

    def retrieve_similar_cases(self, message: str, k: int = C.RETRIEVAL_K) -> list[dict]:
        try:
            return self.index.search(message, k=k)
        except Exception as e:  # index corrupt / unseen vocabulary
            return []

    # ---- decision ------------------------------------------------------
    def _decide(self, st: dict) -> tuple[str, str, list[str]]:
        """Selective prediction: abstain (escalate) unless every gate passes.

        Framing follows selective classification (Geifman & El-Yaniv, 2017): we
        accept lower coverage to buy lower risk on the answered slice, and we
        report the whole risk-coverage trade-off rather than one flattering point.
        """
        t = self.thresholds
        reasons: list[str] = []

        if st["risk"]["blocks"]:
            return ("ESCALATE",
                    "Hard policy block: " + ", ".join(st["risk"]["blocks"]),
                    ["hard_block"])

        if st["intent_confidence"] < t["confidence"]:
            reasons.append(
                f"intent confidence {st['intent_confidence']:.2f} < {t['confidence']:.2f}")
        if st["margin"] < t["margin"]:
            reasons.append(
                f"top-2 intents are close (margin {st['margin']:.2f} < {t['margin']:.2f}): "
                f"'{st['intent']}' vs '{st['runner_up']}'")
        if st["evidence_quality"] < t["evidence_quality"]:
            reasons.append(
                f"evidence quality {st['evidence_quality']:.2f} < {t['evidence_quality']:.2f}")
        for f in st["evidence_flags"]:
            reasons.append(f"evidence flag: {f}")
        if st["risk"]["risk_level"] == "HIGH":
            reasons.append("high-risk intent requires human sign-off")

        if reasons:
            return "ESCALATE", "; ".join(reasons), ["gate_failed"]
        return ("AUTO-HANDLE",
                f"intent confidence {st['intent_confidence']:.2f} and evidence quality "
                f"{st['evidence_quality']:.2f} both clear threshold on a "
                f"{st['risk']['risk_level'].lower()}-risk intent, with "
                f"{st['n_supporting']} supporting precedent(s)", [])

    # ---- entry point ---------------------------------------------------
    def handle(self, message: str, k: int = C.RETRIEVAL_K) -> dict:
        t0 = time.time()
        st: dict = {"customer_message": message, "trace": []}

        cls = self.classify_intent(message)
        st.update(cls)
        st["trace"].append({"tool": "classify_intent",
                            "out": {k2: cls[k2] for k2 in ("intent", "intent_confidence", "margin")}})

        st["risk"] = assess_risk(message, st["intent"])
        st["risk_level"] = st["risk"]["risk_level"]
        st["trace"].append({"tool": "assess_risk",
                            "out": {"risk_level": st["risk_level"], "blocks": st["risk"]["blocks"]}})

        cases = self.retrieve_similar_cases(message, k=k)
        st["historical_evidence"] = cases
        st["trace"].append({"tool": "retrieve_similar_cases",
                            "out": {"n": len(cases),
                                    "top_similarity": cases[0]["similarity"] if cases else 0.0}})

        ev = validate_evidence(cases, st["intent"])
        st["evidence_quality"] = ev["evidence_quality"]
        st["evidence_signals"] = ev["signals"]
        st["evidence_flags"] = ev["flags"]
        st["n_supporting"] = int(sum(c["similarity"] >= 0.20 for c in cases))
        st["trace"].append({"tool": "validate_evidence", "out": ev})

        decision, reason, _ = self._decide(st)
        st["decision"], st["decision_reason"] = decision, reason
        st["trace"].append({"tool": "determine_action",
                            "out": {"decision": decision, "reason": reason}})

        # Draft in both branches: on ESCALATE the draft becomes a suggestion for
        # the human reviewer, which is the actual Hiver-shaped workflow (assist
        # the agent, do not replace them). It is labelled as such.
        gen = generate_reply(message, st["intent"], st["intent_confidence"], cases,
                             st["evidence_quality"], st["risk_level"])
        st["draft_reply"] = gen["reply"]
        st["generation"] = gen
        st["trace"].append({"tool": "generate_reply",
                            "out": {"backend": gen["backend"], "grounded": gen["grounded"],
                                    "evidence_used": gen["evidence_used"]}})

        crit = critique_reply(gen["reply"], cases, st["risk_level"])
        st["critique"] = crit
        st["trace"].append({"tool": "critique_reply", "out": crit})

        # The critic can only tighten the decision, never loosen it.
        if st["decision"] == "AUTO-HANDLE" and crit["verdict"] != "PASS":
            st["decision"] = "ESCALATE"
            st["decision_reason"] = (
                f"draft failed the response critic ({', '.join(crit['issues'])}); "
                "downgraded from AUTO-HANDLE")
            st["trace"].append({"tool": "critic_override",
                                "out": {"decision": "ESCALATE", "issues": crit["issues"]}})
        if st["decision"] == "AUTO-HANDLE" and gen.get("should_escalate"):
            st["decision"] = "ESCALATE"
            st["decision_reason"] = "the drafter reported it could not ground the reply"

        st["reply"] = st["draft_reply"]
        st["reply_is_suggestion_only"] = st["decision"] == "ESCALATE"
        st["backend"] = gen["backend"]
        st["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return st


_AGENT: SupportAgent | None = None


def get_agent() -> SupportAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = SupportAgent()
    return _AGENT


if __name__ == "__main__":
    a = get_agent()
    demos = [
        "@SpotifyCares you charged me twice for premium this month, refund me now",
        "@SpotifyCares how do I make a playlist collaborative?",
        "@SpotifyCares my downloaded songs won't play offline since the last update",
        "@SpotifyCares someone hacked my account and changed my email",
    ]
    for m in demos:
        r = a.handle(m)
        print("=" * 78)
        print(m)
        print(f"  intent={r['intent']} conf={r['intent_confidence']:.2f} "
              f"risk={r['risk_level']} eq={r['evidence_quality']:.2f}")
        print(f"  {r['decision']}: {r['decision_reason'][:150]}")
        print(f"  reply[{r['backend']}]: {r['reply'][:150]}")
