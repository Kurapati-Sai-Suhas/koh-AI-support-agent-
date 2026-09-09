"""Build reports/human_review.csv from real live-agent output.

WHO DID THIS REVIEW — read this before quoting the numbers
----------------------------------------------------------
The reviewer is an AI assistant (Claude) acting as a customer-support reviewer,
not a human annotator. It is recorded in the CSV as `reviewer=ai_simulated`.

This is a structured qualitative review of REAL output from REAL dataset messages
through the live HTTP API — none of the outputs, evidence or scores below are
invented. But an AI reviewing an AI is not independent evidence of quality, and
it is not what the assignment means by "hand-labelled". It is useful for finding
defects (it found two real ones, both fixed), and it is NOT a substitute for the
human review still pending in `src/review_golden_set.py`.

The judgements themselves were made by inspecting each case's full output —
intent, retrieved precedents with similarity scores, decision reason, draft reply
and critic verdict — recorded in e2e_results.json.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

E2E = C.ROOT / "e2e_results.json"
OUT = C.REPORTS / "human_review.csv"

# id -> reviewer judgements. Keyed to the order in e2e_results.json.
# Scales: *_correct / *_good / sufficient / grounded = yes | partial | no
#         reply_quality = 1-5 ; hallucination = none | low | medium | high
REVIEW = {
1: dict(intent_correct="yes", retrieval_good="yes", evidence_sufficient="yes",
        decision_correct="yes", escalation_appropriate="n/a (auto-handled)",
        reply_grounded="yes", reply_quality=5, hallucination="none",
        notes="Best case in the sample. Four near-identical precedents (sim 0.70/0.69/0.61/0.52), "
              "all the same resolution. The draft reuses the brand's own wording including the real "
              "singles link from evidence; critic grounding 1.00. Auto-handling this is clearly right."),
2: dict(intent_correct="yes", retrieval_good="yes", evidence_sufficient="yes",
        decision_correct="yes", escalation_appropriate="n/a (auto-handled)",
        reply_grounded="partial", reply_quality=3, hallucination="medium",
        notes="Intent and decision right, but on a re-run the draft said 'We've just sent a DM your "
              "way' - claiming an action the system never performed. Copied from a precedent where the "
              "brand really had sent one. Forbidden-commitment regexes do not currently cover "
              "claimed-action-by-us phrasing. Logged as an open issue, not fixed in MVP."),
3: dict(intent_correct="yes", retrieval_good="yes", evidence_sufficient="yes",
        decision_correct="yes", escalation_appropriate="yes",
        reply_grounded="yes", reply_quality=4, hallucination="none",
        notes="Account-compromise language triggers the deterministic hard block, so it escalates "
              "regardless of the 0.98 confidence. Exactly the intended behaviour: the block fires on "
              "the raw message and the classifier cannot talk it out of it."),
4: dict(intent_correct="yes", retrieval_good="no", evidence_sufficient="no",
        decision_correct="yes", escalation_appropriate="yes",
        reply_grounded="yes", reply_quality=3, hallucination="none",
        notes="Intent right (billing_payment, 0.97) but retrieval is weak - top similarity 0.20 and "
              "the precedents are all generic 'DM us' hand-offs. The system detected this itself "
              "(precedent_is_boilerplate_only) and escalated. Right outcome, and reached for the "
              "right reason rather than by luck."),
5: dict(intent_correct="partial", retrieval_good="partial", evidence_sufficient="no",
        decision_correct="yes", escalation_appropriate="yes",
        reply_grounded="yes", reply_quality=3, hallucination="none",
        notes="Genuinely ambiguous: student-discount question sits between subscription_plan and "
              "feature_how_to. Model gave 0.49 with a 0.00 margin - a good calibration signal - and "
              "escalated. The draft asks for account details, which is a safe non-answer."),
6: dict(intent_correct="n/a (no standalone intent)", retrieval_good="partial",
        evidence_sufficient="no", decision_correct="yes", escalation_appropriate="yes",
        reply_grounded="yes", reply_quality=3, hallucination="none",
        notes="Device specs only ('Samsung Galaxy 6, Android 7.0, Spotify 8.4.28.875') - a follow-up "
              "whose intent lives in the previous turn. No single-turn label is correct. Escalated on "
              "low confidence and zero margin, which is the right outcome."),
7: dict(intent_correct="n/a (no standalone intent)", retrieval_good="no",
        evidence_sufficient="no", decision_correct="no", escalation_appropriate="no",
        reply_grounded="no", reply_quality=1, hallucination="high",
        notes="THE WORST FAILURE FOUND, and it drove a code fix. 'No it did not' was AUTO-HANDLED at "
              "0.91 confidence and 0.94 evidence quality because TF-IDF matched other short generic "
              "messages at similarity 1.00. The draft asked about a video ad - a total non-sequitur. "
              "Fixed by the low_information_message guard; this case now escalates."),
8: dict(intent_correct="no", retrieval_good="partial", evidence_sufficient="partial",
        decision_correct="yes", escalation_appropriate="yes",
        reply_grounded="yes", reply_quality=4, hallucination="none",
        notes="'My Spotify isn't loading any playlists' should be playback_streaming_issue but was "
              "predicted general_complaint_feedback. The weak-supervision rules do not cover 'isn't "
              "loading'. Wrong intent, right decision - low confidence (0.49) caused an escalation "
              "that masked the classification error. The draft sensibly asks for device and version."),
9: dict(intent_correct="partial", retrieval_good="no", evidence_sufficient="no",
        decision_correct="yes", escalation_appropriate="yes",
        reply_grounded="yes", reply_quality=3, hallucination="none",
        notes="Long multi-topic message about an Apple ID country change. Predicted "
              "cancellation_request (0.96) - defensible but the message is really about region/billing. "
              "Retrieval weak (0.22). High-risk intent forced escalation, which was right."),
10: dict(intent_correct="partial", retrieval_good="no", evidence_sufficient="no",
         decision_correct="yes", escalation_appropriate="yes",
         reply_grounded="yes", reply_quality=3, hallucination="none",
         notes="Sudden logout mid-session. account_login_access is arguable but confidence was only "
               "0.47 with a 0.00 margin against the runner-up. Retrieval poor (0.23). Escalated "
               "correctly. Shows the margin gate doing real work on a genuinely uncertain case."),
}

FIELDS = ["id", "customer_message", "predicted_intent", "intent_confidence", "risk_level",
          "evidence_quality", "top_similarity", "decision", "intent_correct", "retrieval_good",
          "evidence_sufficient", "decision_correct", "escalation_appropriate", "reply_grounded",
          "reply_quality", "hallucination", "generated_reply", "human_notes", "reviewer"]


def main():
    if not E2E.exists():
        raise SystemExit(f"{E2E} missing — run the end-to-end demo script first")
    rows = json.loads(E2E.read_text(encoding="utf-8"))
    out = []
    for i, row in enumerate(rows, 1):
        r = row["result"]
        j = REVIEW.get(i)
        if not j:
            continue
        out.append({
            "id": i,
            "customer_message": row["message"],
            "predicted_intent": r["intent"],
            "intent_confidence": round(r["intent_confidence"], 3),
            "risk_level": r["risk_level"],
            "evidence_quality": round(r["evidence_quality"], 3),
            "top_similarity": round(r["evidence"][0]["similarity"], 3) if r["evidence"] else 0.0,
            "decision": r["decision"],
            "generated_reply": r["reply"],
            "reviewer": "ai_simulated",
            **j, "human_notes": j["notes"],
        })
    for o in out:
        o.pop("notes", None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)

    n = len(out)
    def share(k, v):
        return sum(1 for o in out if o[k] == v)
    print(f"wrote {OUT} ({n} reviewed cases)")
    print(f"  decision correct      : {share('decision_correct','yes')}/{n}")
    print(f"  intent correct (yes)  : {share('intent_correct','yes')}/{n}")
    print(f"  retrieval good (yes)  : {share('retrieval_good','yes')}/{n}")
    print(f"  evidence sufficient   : {share('evidence_sufficient','yes')}/{n}")
    print(f"  hallucination none    : {share('hallucination','none')}/{n}")
    print(f"  mean reply quality    : {sum(o['reply_quality'] for o in out)/n:.1f}/5")


if __name__ == "__main__":
    main()
