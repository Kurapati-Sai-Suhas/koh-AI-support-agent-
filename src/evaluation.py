"""The evaluation harness. Three levels, kept separate on purpose.

  MODEL quality     intent classification vs labels
  AGENT quality     was the auto-handle / escalate decision right?
  RESPONSE quality  was the drafted reply any good? (judge.py)

Collapsing these into one number is how support agents get shipped that look
good and behave badly, so nothing here averages across levels.

Run: python -m src.evaluation
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from feature_engineering import make_frame
from agent import get_agent
from judge import judge_reply
import llm

GOLDEN = C.GOLDEN / "golden_set.csv"
JUDGE_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def core_metrics(y_true, y_pred) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def per_class_table(y_true, y_pred) -> str:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 zero_division=0)
    lines = ["| intent | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for i, lab in enumerate(labels):
        lines.append(f"| `{lab}` | {p[i]:.2f} | {r[i]:.2f} | {f[i]:.2f} | {int(s[i])} |")
    return "\n".join(lines)


def confusion_md(y_true, y_pred) -> str:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    short = [l[:14] for l in labels]
    lines = ["| true \\ pred | " + " | ".join(short) + " |",
             "|" + "---|" * (len(labels) + 1)]
    for i, lab in enumerate(labels):
        lines.append(f"| **{lab[:22]}** | " + " | ".join(str(v) for v in cm[i]) + " |")
    return "\n".join(lines)


def main():
    out: dict = {}
    agent = get_agent()

    # ================= LEVEL 1: model quality =============================
    test = pd.read_csv(C.PROCESSED / "test.csv")
    Xte = make_frame(test.customer_text, C.BRAND)
    yte = test.intent.astype(str).to_numpy()
    pred_weak = agent.model.predict(Xte)
    out["test_weak_labels"] = core_metrics(yte, pred_weak)

    # ---- golden set (human labels) --------------------------------------
    if not GOLDEN.exists():
        raise SystemExit("golden set missing — run python -m src.build_golden_set")
    gold = pd.read_csv(GOLDEN)
    reviewed = gold[gold.reviewed.astype(bool)].copy()
    out["golden_total_rows"] = int(len(gold))
    out["golden_reviewed_rows"] = int(len(reviewed))
    out["golden_reviewed_pct"] = round(float(gold.reviewed.astype(bool).mean()), 4)

    gold_eval = reviewed if len(reviewed) >= 30 else gold
    used_unreviewed = len(reviewed) < 30
    Xg = make_frame(gold_eval.message, C.BRAND)
    yg = gold_eval.gold_intent.astype(str).to_numpy()
    pg = agent.model.predict(Xg)
    out["golden"] = core_metrics(yg, pg)
    out["golden_labels_are_human_reviewed"] = not used_unreviewed
    out["weak_rule_vs_gold_agreement"] = round(
        float((gold_eval.weak_rule_label.astype(str) == yg).mean()), 4)

    # ================= LEVEL 2: agent quality =============================
    states = [agent.handle(str(m)) for m in gold_eval.message]
    dec = pd.DataFrame({
        "message": gold_eval.message.values,
        "gold": yg,
        "pred": [s["intent"] for s in states],
        "conf": [s["intent_confidence"] for s in states],
        "risk": [s["risk_level"] for s in states],
        "ev_quality": [s["evidence_quality"] for s in states],
        "decision": [s["decision"] for s in states],
        "reason": [s["decision_reason"] for s in states],
        "difficulty": gold_eval.difficulty.values,
        "top_sim": [s["historical_evidence"][0]["similarity"] if s["historical_evidence"] else 0.0
                    for s in states],
    })
    dec["correct"] = dec.gold == dec.pred
    auto = dec[dec.decision == "AUTO-HANDLE"]
    esc = dec[dec.decision == "ESCALATE"]

    out["agent"] = {
        # Coverage is the only label-independent number here: it is just how
        # often the agent chose to answer. Everything below depends on labels.
        "coverage_auto_handled": round(float(len(auto) / len(dec)), 4),
        "selective_accuracy_on_auto": round(float(auto.correct.mean()), 4) if len(auto) else None,
        "accuracy_on_escalated": round(float(esc.correct.mean()), 4) if len(esc) else None,
        "accuracy_overall": round(float(dec.correct.mean()), 4),
        # The number that matters for trust: of everything we answered alone,
        # how much was wrong? This is what a customer would experience.
        "error_rate_on_auto_handled": round(float(1 - auto.correct.mean()), 4) if len(auto) else None,
        "measured_against": "human-reviewed labels" if not used_unreviewed
                            else "UNREVIEWED proposals — not a quality measure",
    }

    # When the golden set is not yet human-reviewed, report the SAME agent
    # metrics under both independent label sources. They disagree wildly, and
    # publishing one of them alone would be the exact error this project warns
    # about. The spread is the honest statement of what we do not yet know.
    if used_unreviewed and "weak_rule_label" in gold_eval.columns:
        alt = gold_eval.weak_rule_label.astype(str).to_numpy()
        alt_correct = dec.pred.to_numpy() == alt
        am = dec.decision.to_numpy() == "AUTO-HANDLE"
        out["agent_under_alternative_labels"] = {
            "label_source": "weak-supervision rules (what the model trained on)",
            "selective_accuracy_on_auto": round(float(alt_correct[am].mean()), 4) if am.any() else None,
            "accuracy_overall": round(float(alt_correct.mean()), 4),
            "note": ("The primary figures above use the independent proposals; these use "
                     "the training rules. Neither is ground truth. The true value lies "
                     "somewhere between and requires the human review."),
        }
    dec.to_csv(C.REPORTS / "agent_decisions_golden.csv", index=False)

    # ================= LEVEL 3: response quality ==========================
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(states), size=min(JUDGE_SAMPLE, len(states)), replace=False)
    judged = []
    for i in idx:
        s = states[int(i)]
        # Pass the drafter's backend so the judge can record whether it graded
        # its own output (Panickssery et al., 2024). Without this the
        # self_graded flag silently reports False even for same-model runs.
        j = judge_reply(s["customer_message"], s["reply"], s["historical_evidence"],
                        s["intent"], drafted_by=s.get("backend"))
        judged.append({"message": s["customer_message"], "reply": s["reply"],
                       "decision": s["decision"], "intent": s["intent"],
                       "evidence_quality": s["evidence_quality"],
                       "grounding_score": s["critique"]["grounding_score"], **j})
    jdf = pd.DataFrame(judged)
    jdf.to_csv(C.REPORTS / "judge_scores.csv", index=False)
    dims = ["relevance", "correctness", "grounding", "helpfulness", "tone",
            "hallucination_risk", "overall"]
    out["judge"] = {
        "backend": jdf.judge_backend.iloc[0] if len(jdf) else "n/a",
        "n_judged": int(len(jdf)),
        "means": {d: round(float(jdf[d].mean()), 2) for d in dims if d in jdf},
    }

    # ================= failure analysis (real rows only) ==================
    fails = dec[~dec.correct].copy()
    fails["bucket"] = np.where(
        fails.gold == "general_complaint_feedback", "over-triggered on vague message",
        np.where(fails.pred == "general_complaint_feedback", "missed a specific intent",
                 np.where(fails.conf < 0.5, "low-confidence confusion",
                          np.where(fails.message.str.len() < 60, "too short to disambiguate",
                                   "specific-intent confusion"))))
    top_buckets = fails.bucket.value_counts().head(5)

    # ================= write the report ===================================
    L = []
    A = L.append
    A("# Evaluation")
    A("")
    A(f"Brand `{C.BRAND}` · seed {C.SEED} · generated by `python -m src.evaluation`")
    A("")
    A("Three levels are reported separately. A system can score well on the first")
    A("and still be untrustworthy, which is the point of the other two.")
    A("")
    A("## Level 1 — model quality (intent classification)")
    A("")
    A("| evaluation set | labels | n | accuracy | macro F1 | weighted F1 |")
    A("|---|---|---|---|---|---|")
    t = out["test_weak_labels"]
    A(f"| held-out test | weak-supervision rules | {t['n']} | {t['accuracy']:.3f} | "
      f"{t['macro_f1']:.3f} | {t['weighted_f1']:.3f} |")
    g = out["golden"]
    lab = "human-reviewed" if not used_unreviewed else "UNREVIEWED (rule-seeded)"
    A(f"| golden set | {lab} | {g['n']} | {g['accuracy']:.3f} | {g['macro_f1']:.3f} | "
      f"{g['weighted_f1']:.3f} |")
    A("")
    if used_unreviewed:
        A("> **The golden row above is not yet a real measurement.** "
          f"Only {out['golden_reviewed_rows']}/{out['golden_total_rows']} rows have been "
          "human-reviewed. The labels are independent *proposals*, and they agree with the "
          f"weak-supervision rules the model trained on only "
          f"**{out['weak_rule_vs_gold_agreement']:.0%}** of the time. Two independent "
          "sources disagreeing this much means neither can be treated as ground truth: "
          "measured against the proposals the agent looks terrible, against the rules it "
          "looks strong, and the truth is somewhere between. "
          "Run `python -m src.review_golden_set` and re-run this script. "
          "Until then, treat the *test* row as the only defensible model number, and see "
          "the dual-label agent table below.")
        if "agent_under_alternative_labels" in out:
            a2 = out["agent_under_alternative_labels"]
            A("")
            A("| agent metric | vs independent proposals | vs training rules |")
            A("|---|---|---|")
            sa1 = out["agent"]["selective_accuracy_on_auto"]
            sa2 = a2["selective_accuracy_on_auto"]
            A(f"| selective accuracy on auto-handled | {sa1:.1%} | {sa2:.1%} |")
            A(f"| accuracy overall | {out['agent']['accuracy_overall']:.1%} | "
              f"{a2['accuracy_overall']:.1%} |")
            A("")
            A("Coverage is unaffected by labels and is a real measurement.")
    else:
        A(f"Weak rules agree with the human labels on "
          f"**{out['weak_rule_vs_gold_agreement']:.1%}** of golden rows — that gap is the "
          "label noise the training set was built on.")
    A("")
    A("### Per-class performance (golden set)")
    A("")
    A(per_class_table(yg, pg))
    A("")
    A("### Confusion matrix (golden set)")
    A("")
    A(confusion_md(yg, pg))
    A("")
    A("## Level 2 — agent quality (was the decision right?)")
    A("")
    a = out["agent"]
    A("| metric | value |")
    A("|---|---|")
    A(f"| coverage (share auto-handled) | {a['coverage_auto_handled']:.1%} |")
    A(f"| selective accuracy on auto-handled | "
      f"{a['selective_accuracy_on_auto']:.1%} |" if a['selective_accuracy_on_auto'] is not None
      else "| selective accuracy on auto-handled | n/a (nothing auto-handled) |")
    A(f"| error rate on auto-handled | {a['error_rate_on_auto_handled']:.1%} |"
      if a['error_rate_on_auto_handled'] is not None else "| error rate | n/a |")
    A(f"| accuracy on escalated slice | "
      f"{a['accuracy_on_escalated']:.1%} |" if a['accuracy_on_escalated'] is not None
      else "| accuracy on escalated | n/a |")
    A(f"| accuracy over everything | {a['accuracy_overall']:.1%} |")
    A("")
    A("The agent is *supposed* to be less accurate on the escalated slice — that is what")
    A("abstention buys. If the two slices had equal accuracy the gate would be sorting noise.")
    A("")
    A("### Decisions by difficulty stratum")
    A("")
    ct = pd.crosstab(dec.difficulty, dec.decision)
    A("| difficulty | " + " | ".join(ct.columns) + " |")
    A("|" + "---|" * (len(ct.columns) + 1))
    for i, r in ct.iterrows():
        A(f"| {i} | " + " | ".join(str(v) for v in r.values) + " |")
    A("")
    A("## Level 3 — response quality (LLM-as-judge)")
    A("")
    A(f"Backend: `{out['judge']['backend']}` · {out['judge']['n_judged']} replies judged")
    A("")
    if str(out["judge"]["backend"]).startswith("heuristic"):
        A("> **This is not an LLM judgment.** No LLM key was configured, so the rubric was")
        A("> scored by the deterministic fallback in `src/judge.py`, which computes the same")
        A("> six dimensions from lexical grounding, policy-violation regexes and reply shape.")
        A("> It is a sanity check, not a quality measure. Set `LLM_API_KEY` in `.env` and")
        A("> re-run to get real judge scores.")
        A("")
    A("| dimension | mean (1-5) |")
    A("|---|---|")
    for d, v in out["judge"]["means"].items():
        A(f"| {d} | {v:.2f} |")
    A("")
    A("## Failure analysis — top buckets (real rows, no invented examples)")
    A("")
    A(f"{len(fails)} of {len(dec)} golden messages were classified wrongly.")
    A("")
    A("| failure mode | n |")
    A("|---|---|")
    for b, n in top_buckets.items():
        A(f"| {b} | {n} |")
    A("")
    A("### Worked examples")
    A("")
    for b in top_buckets.index[:5]:
        ex = fails[fails.bucket == b].sort_values("conf", ascending=False).head(1)
        if not len(ex):
            continue
        e = ex.iloc[0]
        A(f"**{b}**")
        A("")
        A(f"- message: `{str(e.message)[:200]}`")
        A(f"- expected: `{e.gold}` · predicted: `{e.pred}` (confidence {e.conf:.2f})")
        A(f"- top retrieved similarity: {e.top_sim:.2f} · evidence quality: {e.ev_quality:.2f}")
        A(f"- agent decision: **{e.decision}**")
        A("")
    (C.REPORTS / "evaluation.md").write_text("\n".join(L), encoding="utf-8")
    (C.REPORTS / "evaluation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("\n-> reports/evaluation.md")


if __name__ == "__main__":
    main()
