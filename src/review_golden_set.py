"""Terminal tool for hand-reviewing the golden set. This is the manual step.

    python -m src.review_golden_set            # review everything unreviewed
    python -m src.review_golden_set --only hard
    python -m src.review_golden_set --stats

The assignment asks for hand-labelled examples. This tool exists so that means
about 15 minutes of key presses rather than an afternoon in a spreadsheet: it
shows the message and the proposed label, ENTER accepts, a digit overrides.
Progress is saved after every answer, so it is safe to stop and resume.

Nothing downstream counts a row as human-labelled until reviewed == True here.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from intents import INTENTS, TAXONOMY

PATH = C.GOLDEN / "golden_set.csv"


def show_menu():
    print("\nINTENTS")
    for i, name in enumerate(INTENTS, 1):
        print(f"  {i}. {name:<28} {TAXONOMY[name]['description'][:60]}")
    print("  s. skip   q. save and quit\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["hard", "medium", "easy"], default=None)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if not PATH.exists():
        raise SystemExit(f"{PATH} not found — run: python -m src.build_golden_set")
    df = pd.read_csv(PATH)

    if args.stats:
        print(f"rows: {len(df)}   reviewed: {int(df.reviewed.sum())} "
              f"({df.reviewed.mean():.0%})")
        print("\nby difficulty:\n" + df.groupby("difficulty").reviewed.agg(["count", "sum"]).to_string())
        print("\ngold label distribution:\n" + df.gold_intent.value_counts().to_string())
        if df.reviewed.any():
            r = df[df.reviewed]
            print(f"\nhuman vs proposal agreement on reviewed rows: "
                  f"{(r.gold_intent == r.proposed_intent).mean():.1%}")
            print(f"human vs weak-rule agreement on reviewed rows: "
                  f"{(r.gold_intent == r.weak_rule_label).mean():.1%}")
        return

    todo = df[~df.reviewed.astype(bool)]
    if args.only:
        todo = todo[todo.difficulty == args.only]
    if not len(todo):
        print("nothing left to review.")
        return

    print(f"{len(todo)} rows to review. ENTER accepts the proposal, a number overrides it.")
    show_menu()
    done = 0
    for pos, (i, row) in enumerate(todo.iterrows(), 1):
        print("=" * 76)
        print(f"[{pos}/{len(todo)}]  difficulty={row.difficulty}  stratum={row.stratum}")
        print(f"\n  {row.message}\n")
        print(f"  proposed : {row.proposed_intent}")
        print(f"  weak rule: {row.weak_rule_label}    model: {row.model_pred} "
              f"({row.model_confidence:.2f})")
        try:
            ans = input("  accept [ENTER] / 1-9 / s / q > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\ninterrupted — saving.")
            break
        if ans == "q":
            break
        if ans == "s":
            continue
        if ans == "":
            df.at[i, "gold_intent"] = row.proposed_intent
        elif ans.isdigit() and 1 <= int(ans) <= len(INTENTS):
            df.at[i, "gold_intent"] = INTENTS[int(ans) - 1]
        else:
            print("  ? not understood, skipping")
            continue
        df.at[i, "reviewed"] = True
        df.at[i, "label_source"] = "human_reviewed"
        done += 1
        if done % 10 == 0:
            df.to_csv(PATH, index=False, encoding="utf-8")
            print(f"  ...saved ({int(df.reviewed.sum())} reviewed)")

    df.to_csv(PATH, index=False, encoding="utf-8")
    print(f"\nsaved {PATH}: {int(df.reviewed.sum())}/{len(df)} reviewed "
          f"({df.reviewed.mean():.0%})")


if __name__ == "__main__":
    main()
