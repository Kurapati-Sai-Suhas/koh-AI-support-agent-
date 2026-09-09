"""Does the judge agree with a human? Measure it, do not assert it.

    python -m src.human_agreement --rate     # you score 25 replies blind
    python -m src.human_agreement            # compute agreement

Deliverable 3 asks for *evidence* that the judge tracks human opinion. The
protocol here is deliberately small but real:

  1. Sample 25 judged replies (seeded, reproducible).
  2. You rate each one 1-5 for overall quality WITHOUT seeing the judge's score.
     The rating prompt hides it — that is the whole point, an unblinded rater
     anchors and the agreement number becomes meaningless.
  3. Compare on four statistics, because each hides a different failure:
       exact agreement      strict, punishes 1-point drift
       within-1 agreement   the practical bar for a 5-point rubric
       Spearman rho         does the judge RANK replies like the human?
       Cohen's kappa        on "acceptable" (>=4), chance-corrected

Zheng et al. (NeurIPS 2023) report ~80% agreement between GPT-4 and humans, and
note that human-human agreement sits at a similar level — so ~80% is the target,
not 100%. If our number lands far above that, the likely explanation is a
degenerate rating distribution (everything rated 4), not a superb judge, and the
script prints the distribution so that is visible.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

SCORES = C.REPORTS / "judge_scores.csv"
RATINGS = C.REPORTS / "human_ratings.csv"
N_RATE = 25


def build_sheet():
    if not SCORES.exists():
        raise SystemExit("run python -m src.evaluation first")
    df = pd.read_csv(SCORES)
    n = min(N_RATE, len(df))
    sample = df.sample(n=n, random_state=C.SEED).reset_index(drop=True)
    sheet = sample[["message", "reply", "intent", "decision"]].copy()
    sheet["human_overall"] = np.nan
    sheet["judge_overall"] = sample["overall"]      # kept but never shown while rating
    sheet["human_note"] = ""
    sheet.to_csv(RATINGS, index=False, encoding="utf-8")
    return sheet


def rate():
    sheet = pd.read_csv(RATINGS) if RATINGS.exists() else build_sheet()
    todo = sheet[sheet.human_overall.isna()]
    if not len(todo):
        print("all rated. run without --rate to compute agreement.")
        return
    print(f"\nRating {len(todo)} replies. Score OVERALL quality 1-5.")
    print("  5 = I would send this as-is   3 = usable after an edit   1 = unusable\n")
    print("The judge's score is hidden until you are done.\n")
    for i, row in todo.iterrows():
        print("=" * 76)
        print(f"CUSTOMER: {row.message}")
        print(f"\nDRAFT REPLY ({row.decision}):\n  {row.reply}\n")
        try:
            v = input("  your score 1-5 (ENTER to skip, q to stop) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if v == "q":
            break
        if v in {"1", "2", "3", "4", "5"}:
            sheet.at[i, "human_overall"] = int(v)
            note = input("  optional note > ").strip()
            sheet.at[i, "human_note"] = note
            sheet.to_csv(RATINGS, index=False, encoding="utf-8")
    sheet.to_csv(RATINGS, index=False, encoding="utf-8")
    done = int(sheet.human_overall.notna().sum())
    print(f"\nsaved {RATINGS}: {done}/{len(sheet)} rated")


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    obs = float((a == b).mean())
    exp = sum((np.mean(a == c) * np.mean(b == c)) for c in cats)
    return float((obs - exp) / (1 - exp)) if exp < 1 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", action="store_true")
    args = ap.parse_args()
    if args.rate:
        rate(); return

    if not RATINGS.exists():
        build_sheet()
        raise SystemExit(f"created {RATINGS}\nnow run: python -m src.human_agreement --rate")

    df = pd.read_csv(RATINGS)
    rated = df[df.human_overall.notna()].copy()
    lines = ["# Judge vs human agreement", ""]
    if len(rated) < 10:
        lines += [
            f"**NOT COMPLETE — {len(rated)} of {len(df)} replies rated.**", "",
            "This deliverable requires a human in the loop and cannot be produced by the",
            "pipeline alone. Run:", "", "```bash",
            "python -m src.human_agreement --rate", "```", "",
            "and re-run this script. No agreement figure is reported until at least 10",
            "replies are rated, because a number computed on fewer is noise dressed as evidence.",
        ]
        (C.REPORTS / "human_agreement.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return

    h = rated.human_overall.to_numpy().astype(int)
    j = rated.judge_overall.to_numpy().astype(int)
    exact = float((h == j).mean())
    within1 = float((np.abs(h - j) <= 1).mean())
    rho = spearman(h, j)
    kappa = cohens_kappa((h >= 4).astype(int), (j >= 4).astype(int))
    bias = float((j - h).mean())

    lines += [
        f"n = {len(rated)} replies, rated blind (judge score hidden during rating).", "",
        "| statistic | value | reading |", "|---|---|---|",
        f"| exact agreement | {exact:.1%} | identical 1-5 score |",
        f"| within-1 agreement | {within1:.1%} | the practical bar for a 5-point rubric |",
        f"| Spearman rho | {rho:.2f} | does the judge rank replies like the human? |",
        f"| Cohen's kappa (acceptable >=4) | {kappa:.2f} | chance-corrected pass/fail agreement |",
        f"| mean judge - human | {bias:+.2f} | positive = judge is more generous |",
        "",
        "## Score distributions", "",
        "| score | human | judge |", "|---|---|---|",
    ]
    for s in range(1, 6):
        lines.append(f"| {s} | {int((h == s).sum())} | {int((j == s).sum())} |")
    lines += [
        "",
        "## Disagreements", "",
    ]
    rated["gap"] = rated.judge_overall - rated.human_overall
    worst = rated.reindex(rated.gap.abs().sort_values(ascending=False).index).head(5)
    for _, r in worst.iterrows():
        if r.gap == 0:
            continue
        lines += [
            f"**judge {int(r.judge_overall)} vs human {int(r.human_overall)}**",
            f"- customer: `{str(r.message)[:160]}`",
            f"- reply: `{str(r.reply)[:160]}`",
            f"- human note: {r.human_note if isinstance(r.human_note, str) else ''}",
            "",
        ]
    lines += [
        "## Interpretation", "",
        f"Zheng et al. (NeurIPS 2023) found GPT-4 judges agree with humans ~80% of the "
        f"time, which is also roughly how often two humans agree with each other. Our "
        f"within-1 agreement of {within1:.1%} on n={len(rated)} should be read against that "
        "bar, and with the obvious caveat that n is small and one rater is not an "
        "inter-annotator study.",
    ]
    (C.REPORTS / "human_agreement.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:20]))
    print("\n-> reports/human_agreement.md")


if __name__ == "__main__":
    main()
