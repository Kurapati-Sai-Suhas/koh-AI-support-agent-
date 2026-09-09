"""Train the intent classifier: 2 baselines + a tuned model.

Run:  python -m src.train
Writes models/intent_model.joblib, models/best_params.json, reports/training.md
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from feature_engineering import build_feature_union, make_frame

SCORES = ["accuracy", "macro_f1", "weighted_f1"]


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def load_splits():
    tr = pd.read_csv(C.PROCESSED / "train.csv")
    va = pd.read_csv(C.PROCESSED / "val.csv")
    te = pd.read_csv(C.PROCESSED / "test.csv")
    return tr, va, te


def main():
    tr, va, te = load_splits()
    Xtr, ytr = make_frame(tr.customer_text, C.BRAND), tr.intent.astype(str).to_numpy()
    Xva, yva = make_frame(va.customer_text, C.BRAND), va.intent.astype(str).to_numpy()
    Xte, yte = make_frame(te.customer_text, C.BRAND), te.intent.astype(str).to_numpy()
    print(f"train={len(tr)} val={len(va)} test={len(te)} classes={tr.intent.nunique()}")

    results = {}

    # ---- Baseline 1: majority class -------------------------------------
    # The floor. On imbalanced data its accuracy looks respectable while its
    # macro-F1 is near zero, which is the whole argument for using macro-F1.
    b1 = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    results["baseline_1_majority"] = metrics(yte, b1.predict(Xte))

    # ---- Baseline 2: plain TF-IDF words + logistic regression -------------
    b2 = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, random_state=C.SEED)),
    ]).fit(Xtr["clean"], ytr)
    results["baseline_2_tfidf_lr"] = metrics(yte, b2.predict(Xte["clean"]))

    # ---- Baseline 3: Complement Naive Bayes (strong, cheap, imbalance-aware)
    b3 = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", ComplementNB()),
    ]).fit(Xtr["clean"], ytr)
    results["baseline_3_complement_nb"] = metrics(yte, b3.predict(Xte["clean"]))

    # ---- Tuned model ------------------------------------------------------
    # LogisticRegression rather than LinearSVC: the decision layer needs a
    # calibrated-ish probability to threshold on, and SVC would need an extra
    # CalibratedClassifierCV wrapper (3x the fits) for the same purpose.
    pipe = Pipeline([
        ("features", build_feature_union()),
        ("clf", LogisticRegression(max_iter=2000, random_state=C.SEED)),
    ])
    # Compact on purpose. The char-ngram block is expensive to refit, so its
    # parameters are fixed and only the cheap, high-leverage axes are searched:
    # 2 x 3 x 2 = 12 configs x 3 folds = 36 fits.
    grid = {
        "features__word__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [1.0, 4.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }
    # GroupKFold on conversation_id: even inside cross-validation two turns of the
    # same thread never straddle the fold boundary. A plain KFold here would
    # leak and would pick hyper-parameters tuned on that leak.
    cv = GroupKFold(n_splits=3)
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=3, verbose=2,
                      refit=True)
    t0 = time.time()
    gs.fit(Xtr, ytr, groups=tr.conversation_id.to_numpy())
    took = time.time() - t0
    print(f"grid search: {len(gs.cv_results_['params'])} configs in {took:.0f}s")
    print("best params:", gs.best_params_)

    best = gs.best_estimator_
    results["tuned_full_model"] = metrics(yte, best.predict(Xte))
    results["tuned_full_model_val"] = metrics(yva, best.predict(Xva))

    # Ablation: does the char n-gram + numeric block actually earn its place?
    abl = Pipeline([
        ("features", build_feature_union(use_char=False, use_numeric=False)),
        ("clf", LogisticRegression(max_iter=2000, random_state=C.SEED,
                                   C=gs.best_params_["clf__C"],
                                   class_weight=gs.best_params_["clf__class_weight"])),
    ]).fit(Xtr, ytr)
    results["ablation_word_tfidf_only"] = metrics(yte, abl.predict(Xte))

    joblib.dump({"model": best, "classes": list(best.classes_), "brand": C.BRAND},
                C.MODELS / "intent_model.joblib")

    # Re-load the artefact we just wrote and recompute the headline row from it.
    # Every number in the report must be reproducible from the shipped model, not
    # from an in-memory object that no one else can inspect. If these disagree,
    # the reported figure is the one from disk.
    reloaded = joblib.load(C.MODELS / "intent_model.joblib")["model"]
    from_disk = metrics(yte, reloaded.predict(Xte))
    if from_disk != results["tuned_full_model"]:
        print(f"WARNING: in-memory {results['tuned_full_model']} != on-disk {from_disk}; "
              "reporting the on-disk figures")
    results["tuned_full_model"] = from_disk
    (C.MODELS / "best_params.json").write_text(
        json.dumps({"best_params": {k: str(v) for k, v in gs.best_params_.items()},
                    "cv_best_macro_f1": round(gs.best_score_, 4),
                    "n_configs": len(gs.cv_results_["params"]),
                    "search_seconds": round(took, 1)}, indent=2), encoding="utf-8")
    (C.REPORTS / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- markdown table ---
    order = ["baseline_1_majority", "baseline_2_tfidf_lr", "baseline_3_complement_nb",
             "ablation_word_tfidf_only", "tuned_full_model"]
    lines = ["# Training results", "",
             f"Brand `{C.BRAND}` · {len(tr):,} train / {len(va):,} val / {len(te):,} test rows",
             "(conversation-disjoint splits, labels from weak supervision)", "",
             "| model | accuracy | macro F1 | weighted F1 |", "|---|---|---|---|"]
    for k in order:
        m = results[k]
        lines.append(f"| {k} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} |")
    base = results["baseline_2_tfidf_lr"]["macro_f1"]
    tuned = results["tuned_full_model"]["macro_f1"]
    lines += ["",
              f"**Improvement over the simple baseline: macro-F1 {base:.3f} -> {tuned:.3f} "
              f"({(tuned-base)/base:+.1%})**", "",
              "## Best hyper-parameters", "",
              "```json", json.dumps({k: str(v) for k, v in gs.best_params_.items()}, indent=2), "```",
              "",
              f"Searched {len(gs.cv_results_['params'])} configurations with 3-fold "
              f"GroupKFold cross-validation on the training split only "
              f"({took:.0f}s). Best CV macro-F1 {gs.best_score_:.3f}.",
              "",
              "The test numbers above are measured against *weak-supervision* labels and are",
              "optimistic by construction. The honest headline is the golden-set score in",
              "`reports/evaluation.md`."]
    (C.REPORTS / "training.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
