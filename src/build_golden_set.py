"""Build the 200-example golden evaluation set.

SAMPLING (transparent and reproducible with SEED=42)
----------------------------------------------------
Drawn ONLY from the held-out test split, so nothing here was seen by the model,
the retrieval index, the calibrator or the threshold sweep.

A uniform random sample would be ~95% easy cases and would tell us nothing about
where the agent breaks, so we sample by stratum on purpose and record which
stratum each row came from. The strata deliberately over-sample hard cases, which
means the golden set is HARDER than production traffic — the resulting score is a
conservative estimate, and the report says so rather than quietly benefiting.

  per_intent      120  proportional across the 9 intents, floor of 6 for rare ones
  ambiguous        25  no labelling rule fired, or two rules tied within 20%
  low_confidence   20  model's top probability in the bottom decile
  short            15  under 60 characters after cleaning
  long             10  over 220 characters
  high_risk        10  matches a risk_engine hard-block pattern
                  ---
                  200

LABELLING
---------
Proposals come from a source INDEPENDENT of the weak-supervision rules that
produced the training labels — an LLM if a key is configured, otherwise the rules
are used as a starting point and flagged as such. Either way the file is not a
golden set until a human has reviewed it: run `python -m src.review_golden_set`.
Nothing in the evaluation reports a row as human-labelled unless reviewed == True.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from feature_engineering import make_frame
from intents import INTENTS, TAXONOMY, label_message
from risk_engine import HARD_BLOCKS
import llm

TARGET = 200
OUT = C.GOLDEN / "golden_set.csv"

PROPOSAL_SYSTEM = (
    "You are labelling customer support tweets sent to Spotify's support account "
    "for an intent-classification evaluation set. Choose exactly one intent."
)


def _proposal_prompt(msg: str) -> str:
    lines = ["INTENT OPTIONS", ""]
    for name, spec in TAXONOMY.items():
        lines.append(f"- {name}: {spec['description']}")
    lines += [
        "", "CUSTOMER MESSAGE", msg, "",
        "Pick the single intent that best matches what the customer actually wants.",
        "If the message is vague venting with no specific fault, use general_complaint_feedback.",
        "",
        'Return ONLY JSON: {"intent": "...", "confidence": 0.0-1.0, "note": "short reason"}',
    ]
    return "\n".join(lines)


def main():
    test = pd.read_csv(C.PROCESSED / "test.csv")
    bundle = joblib.load(C.MODELS / "intent_model.joblib")
    model = bundle["model"]
    X = make_frame(test.customer_text, C.BRAND)
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    order = np.argsort(-proba, axis=1)
    test = test.assign(
        model_pred=classes[order[:, 0]],
        model_confidence=proba[np.arange(len(proba)), order[:, 0]].round(4),
    )

    scores = [label_message(t) for t in test.customer_clean.astype(str)]
    test["rule_fired_any"] = [bool(s[2]) for s in scores]
    tie = []
    for _, _, sc in scores:
        vals = sorted(sc.values(), reverse=True)
        tie.append(len(vals) >= 2 and vals[1] >= 0.8 * vals[0])
    test["rule_tie"] = tie
    test["is_high_risk"] = [any(rx.search(str(t)) for _, rx in HARD_BLOCKS)
                            for t in test.customer_text]

    rng = np.random.default_rng(C.SEED)
    picked: dict[int, str] = {}

    def take(pool: pd.DataFrame, n: int, stratum: str):
        pool = pool[~pool.index.isin(picked)]
        if len(pool) == 0 or n <= 0:
            return
        idx = rng.choice(pool.index.to_numpy(), size=min(n, len(pool)), replace=False)
        for i in idx:
            picked[int(i)] = stratum

    # 1. proportional per-intent core, with a floor so rare intents are represented
    counts = test.intent.value_counts()
    for intent in INTENTS:
        pool = test[test.intent == intent]
        if not len(pool):
            continue
        share = counts.get(intent, 0) / counts.sum()
        take(pool, max(6, int(round(120 * share))), f"per_intent:{intent}")

    take(test[(~test.rule_fired_any) | (test.rule_tie)], 25, "ambiguous")
    lo = test.model_confidence.quantile(0.10)
    take(test[test.model_confidence <= lo], 20, "low_confidence")
    take(test[test.customer_clean.str.len() < 60], 15, "short")
    take(test[test.customer_clean.str.len() > 220], 10, "long")
    take(test[test.is_high_risk], 10, "high_risk")
    take(test, max(0, TARGET - len(picked)), "random_fill")     # top up to exactly 200

    rows = test.loc[sorted(picked.keys())].copy()
    rows["stratum"] = [picked[i] for i in rows.index]
    rows = rows.head(TARGET)

    # difficulty: agreement between the two independent-ish signals + length
    def difficulty(r):
        if r.stratum in ("ambiguous", "low_confidence") or not r.rule_fired_any:
            return "hard"
        if r.model_pred != r.intent or len(str(r.customer_clean)) < 60:
            return "medium"
        return "easy"
    rows["difficulty"] = rows.apply(difficulty, axis=1)

    # --- label proposals ---------------------------------------------------
    use_llm = llm.available()
    proposals, notes, sources = [], [], []
    if use_llm:
        print(f"proposing labels with {C.LLM_MODEL} for {len(rows)} rows ...")
        for n, msg in enumerate(rows.customer_text.astype(str), 1):
            try:
                obj = llm.chat_json(
                    [{"role": "system", "content": PROPOSAL_SYSTEM},
                     {"role": "user", "content": _proposal_prompt(msg)}],
                    temperature=0.0, max_tokens=200)
                pi = str(obj.get("intent", "")).strip()
                proposals.append(pi if pi in INTENTS else "general_complaint_feedback")
                notes.append(str(obj.get("note", ""))[:200])
                sources.append("llm_proposed")
            except llm.LLMError as e:
                proposals.append(rows.intent.iloc[n - 1]); notes.append(f"llm failed: {e}")
                sources.append("rule_proposed")
            if n % 25 == 0:
                print(f"  {n}/{len(rows)}")
    else:
        proposals = rows.intent.tolist()
        notes = ["no LLM configured: seeded from the weak-supervision rule label; "
                 "MUST be human-reviewed before use"] * len(rows)
        sources = ["rule_proposed"] * len(rows)

    out = pd.DataFrame({
        "id": range(1, len(rows) + 1),
        "message": rows.customer_text.astype(str).values,
        "gold_intent": proposals,
        "proposed_intent": proposals,
        "weak_rule_label": rows.intent.values,
        "model_pred": rows.model_pred.values,
        "model_confidence": rows.model_confidence.values,
        "stratum": rows.stratum.values,
        "difficulty": rows.difficulty.values,
        "label_source": sources,
        "reviewed": False,
        "notes": notes,
        "conversation_id": rows.conversation_id.values,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")

    agree = (out.proposed_intent == out.weak_rule_label).mean()
    print(f"\nwrote {OUT} ({len(out)} rows)")
    print(f"proposal source: {'LLM' if use_llm else 'rules (needs review)'}")
    print(f"proposal vs weak-rule agreement: {agree:.1%}")
    print(out.stratum.value_counts().to_string())
    print(out.difficulty.value_counts().to_string())
    print("\nNEXT: python -m src.review_golden_set   <- required before evaluation")


if __name__ == "__main__":
    main()
