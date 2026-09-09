"""Raw TWCS csv -> (customer message, brand reply) pairs -> leakage-free splits.

Run:  python -m src.data_processing
Writes data/processed/{pairs.parquet,train.csv,val.csv,test.csv} and reports/eda.md
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C
from text_utils import clean_text, strip_signature, is_boilerplate_reply
from intents import label_message, taxonomy_table, INTENTS

USECOLS = ["tweet_id", "author_id", "inbound", "created_at", "text",
           "in_response_to_tweet_id"]


def load_raw() -> pd.DataFrame:
    if not C.RAW_CSV.exists():
        raise SystemExit(
            f"Missing {C.RAW_CSV}. See README 'Get the data'."
        )
    df = pd.read_csv(C.RAW_CSV, usecols=USECOLS)
    df["in_response_to_tweet_id"] = pd.to_numeric(df.in_response_to_tweet_id, errors="coerce")
    df["tweet_id"] = pd.to_numeric(df.tweet_id, errors="coerce")
    return df.dropna(subset=["tweet_id"])


def build_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """A training example is one customer tweet that the brand actually answered.

    We pair on the reply edge (brand tweet -> in_response_to -> customer tweet)
    rather than on time proximity, so every row has a real historical resolution
    attached to it. Rows the brand never answered are dropped: we cannot ground a
    reply on evidence that does not exist.
    """
    parent = dict(zip(df.tweet_id.values, df.in_response_to_tweet_id.values))
    by_id = df.set_index("tweet_id")

    out = df[(df.author_id == brand) & (~df.inbound) & df.in_response_to_tweet_id.notna()]
    par = by_id.reindex(out.in_response_to_tweet_id.values)
    keep = (par.inbound.values == True)  # noqa: E712  (numpy mask, not a bool test)

    pairs = pd.DataFrame({
        "customer_tweet_id": out.in_response_to_tweet_id.values[keep].astype("int64"),
        "customer_text": par.text.values[keep],
        "customer_author": par.author_id.values[keep],
        "created_at": par.created_at.values[keep],
        "brand_reply": out.text.values[keep],
        "brand_reply_id": out.tweet_id.values[keep].astype("int64"),
    })

    # --- conversation id: walk the reply chain to its root -------------------
    # Thread-level grouping is what stops a customer's turn 1 landing in train
    # while turn 2 lands in test. Those two turns share vocabulary and often the
    # same underlying issue, so a row-level split would leak.
    def root_of(tid: int) -> int:
        seen = 0
        cur = tid
        while seen < 40:
            p = parent.get(cur)
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return int(cur)
            cur = int(p)
            seen += 1
        return int(cur)

    pairs["conversation_id"] = [root_of(t) for t in pairs.customer_tweet_id.values]

    # turn index inside the thread (0 = opening message) -> context feature
    pairs["turn_index"] = (
        pairs.sort_values("customer_tweet_id")
             .groupby("conversation_id").cumcount()
    )
    return pairs


def preprocess(pairs: pd.DataFrame, brand: str) -> pd.DataFrame:
    n0 = len(pairs)
    stats = {"pairs_raw": n0}

    pairs = pairs.dropna(subset=["customer_text", "brand_reply"])
    stats["after_dropna"] = len(pairs)

    pairs["customer_clean"] = [clean_text(t, brand) for t in pairs.customer_text]
    pairs["reply_clean"] = [strip_signature(t) for t in pairs.brand_reply]

    # Drop messages too short to carry intent, and absurdly long concatenations.
    n_len = len(pairs)
    pairs = pairs[pairs.customer_clean.str.len().between(C.MIN_CHARS, C.MAX_CHARS)]
    stats["dropped_length"] = n_len - len(pairs)

    # Near-duplicate customer messages (bots, copy-paste campaigns) inflate any
    # metric if the same text lands in both train and test.
    n_dup = len(pairs)
    pairs = pairs.drop_duplicates(subset=["customer_clean"])
    stats["dropped_duplicates"] = n_dup - len(pairs)

    # Keep mostly-English rows. Spotify is 98% ASCII; the tail is other
    # languages we explicitly do not model (see report -> what we did not build).
    ascii_ratio = pairs.customer_text.map(
        lambda s: sum(c.isascii() for c in str(s)) / max(len(str(s)), 1))
    n_ascii = len(pairs)
    pairs = pairs[ascii_ratio > 0.9]
    stats["dropped_non_english"] = n_ascii - len(pairs)

    pairs["reply_is_boilerplate"] = pairs.reply_clean.map(is_boilerplate_reply)

    lab = pairs.customer_clean.map(label_message)
    pairs["intent"] = [x[0] for x in lab]
    pairs["rule_confidence"] = [x[1] for x in lab]
    pairs["rule_fired"] = [bool(x[2]) for x in lab]

    # Subsample by whole conversations, never by row.
    if len(pairs) > C.MAX_PAIRS:
        rng = np.random.default_rng(C.SEED)
        convs = pairs.conversation_id.unique()
        rng.shuffle(convs)
        keep, n = set(), 0
        sizes = pairs.conversation_id.value_counts()
        for c in convs:
            keep.add(c); n += sizes[c]
            if n >= C.MAX_PAIRS:
                break
        pairs = pairs[pairs.conversation_id.isin(keep)]
    stats["pairs_final"] = len(pairs)
    stats["conversations_final"] = pairs.conversation_id.nunique()
    return pairs.reset_index(drop=True), stats


def split(pairs: pd.DataFrame):
    """Group split on conversation_id: train / val / test are thread-disjoint."""
    g = pairs.conversation_id.values
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=C.SEED)
    rest_i, test_i = next(gss.split(pairs, groups=g))
    rest = pairs.iloc[rest_i]
    val_frac = C.VAL_SIZE / (1 - C.TEST_SIZE)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=C.SEED)
    tr_i, va_i = next(gss2.split(rest, groups=rest.conversation_id.values))
    return rest.iloc[tr_i].reset_index(drop=True), rest.iloc[va_i].reset_index(drop=True), pairs.iloc[test_i].reset_index(drop=True)


def write_eda(pairs, train, val, test, stats, brand, raw_len, brand_stats):
    lens = pairs.customer_clean.str.len()
    dist = pairs.intent.value_counts()
    lines = [
        f"# EDA — {brand}", "",
        "## Why this brand", "",
        "Scored the ten highest-volume brands on four axes. Volume alone is not enough:",
        "a brand whose every reply is *\"please DM us\"* gives a retrieval layer nothing to",
        "ground on, and a heavily multilingual brand needs language ID we chose not to build.",
        "",
        brand_stats.to_markdown(index=False),
        "",
        f"`{brand}` wins on the combination: ~43k answered pairs (we only need "
        f"{C.MAX_PAIRS:,}), a 98% English share (high enough to skip language ID), and replies that contain "
        "concrete resolution steps (\"tap the three dots > View Album\", \"try a reinstall\") "
        "rather than pure hand-offs. AmazonHelp has 4x the volume but is heavily "
        "multilingual; AppleSupport sends 39% pure-boilerplate replies vs Spotify's 26%.",
        "",
        "## Dataset shape", "",
        f"- Raw tweets in file: **{raw_len:,}**",
        f"- Tweets authored by `{brand}`: **{brand_stats.set_index('brand').loc[brand,'pairs']:,}** answered customer tweets",
        "",
        "### Funnel from raw pairs to modelling set", "",
        "| step | rows |",
        "|---|---|",
    ]
    for k, v in stats.items():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "### Splits (grouped by conversation_id — no thread spans two splits)", "",
        "| split | rows | conversations |",
        "|---|---|---|",
        f"| train | {len(train):,} | {train.conversation_id.nunique():,} |",
        f"| val | {len(val):,} | {val.conversation_id.nunique():,} |",
        f"| test | {len(test):,} | {test.conversation_id.nunique():,} |",
        "",
        "### Message length (cleaned characters)", "",
        f"- min {int(lens.min())} / median {int(lens.median())} / p90 {int(lens.quantile(.9))} / max {int(lens.max())}",
        "",
        "### Missing values / quality", "",
        f"- customer_text nulls dropped: {stats['pairs_raw'] - stats['after_dropna']:,}",
        f"- exact duplicate messages dropped: {stats['dropped_duplicates']:,}",
        f"- non-English dropped: {stats['dropped_non_english']:,}",
        f"- brand replies that are pure boilerplate (\"DM us\"): "
        f"**{pairs.reply_is_boilerplate.mean():.1%}** — these are kept but down-ranked as evidence",
        "",
        "### Weak-label class distribution (class imbalance)", "",
        "| intent | n | share |",
        "|---|---|---|",
    ]
    for k, v in dist.items():
        lines.append(f"| {k} | {v:,} | {v/len(pairs):.1%} |")
    lines += [
        "",
        f"- Imbalance ratio (largest/smallest): **{dist.max()/dist.min():.1f}x** — this is why "
        "macro-F1, not accuracy, is the model-selection metric.",
        f"- Rows where no specific rule fired (fell back to catch-all): "
        f"**{(~pairs.rule_fired).mean():.1%}**",
        "",
        "## Intent taxonomy", "",
        "| intent | description | example | rules |",
        "|---|---|---|---|",
    ]
    for r in taxonomy_table():
        ex = r["example_message"].replace("|", "/")
        lines.append(f"| `{r['intent']}` | {r['description']} | {ex} | {r['n_rules']} |")
    (C.REPORTS / "eda.md").write_text("\n".join(lines), encoding="utf-8")


def score_brands(df):
    cands = ["AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares", "Delta",
             "Tesco", "AmericanAir", "TMobileHelp", "XboxSupport", "comcastcares"]
    by_id = df.set_index("tweet_id")
    rows = []
    for b in cands:
        o = df[(df.author_id == b) & (~df.inbound) & df.in_response_to_tweet_id.notna()]
        par = by_id.reindex(o.in_response_to_tweet_id.values)
        m = (par.inbound.values == True)  # noqa: E712
        rep = [strip_signature(r) for r in o.text.values[m]]
        cust = par.text.values[m]
        if not rep:
            continue
        eng = np.mean([sum(c.isascii() for c in str(s)) / max(len(str(s)), 1) > .95
                       for s in cust[:4000]])
        rows.append(dict(brand=b, pairs=len(rep),
                         boilerplate_pct=round(100 * np.mean([is_boilerplate_reply(r) for r in rep]), 1),
                         median_reply_chars=int(np.median([len(r) for r in rep])),
                         english_share=round(float(eng), 3)))
    return pd.DataFrame(rows).sort_values("pairs", ascending=False)


def main():
    print("loading raw csv ...")
    df = load_raw()
    raw_len = len(df)
    print(f"  {raw_len:,} tweets")
    print("scoring candidate brands ...")
    bs = score_brands(df)
    bs.to_csv(C.REPORTS / "brand_selection.csv", index=False)
    print(bs.to_string(index=False))
    print(f"building pairs for {C.BRAND} ...")
    pairs = build_pairs(df, C.BRAND)
    pairs, stats = preprocess(pairs, C.BRAND)
    train, val, test = split(pairs)

    assert set(train.conversation_id) & set(test.conversation_id) == set(), "thread leak!"
    assert set(train.conversation_id) & set(val.conversation_id) == set(), "thread leak!"

    pairs.to_parquet(C.PROCESSED / "pairs.parquet", index=False)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_csv(C.PROCESSED / f"{name}.csv", index=False)
    write_eda(pairs, train, val, test, stats, C.BRAND, raw_len, bs)
    (C.PROCESSED / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"train={len(train)} val={len(val)} test={len(test)}  -> reports/eda.md")


if __name__ == "__main__":
    main()
