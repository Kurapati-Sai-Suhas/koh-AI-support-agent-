"""Tool: retrieve_similar_cases() — historical resolution retrieval.

Method: TF-IDF + cosine similarity over the TRAINING customer messages only.

Why not embeddings: a sentence-transformer download is ~90MB and a cold encode
of 14k messages costs minutes we do not have; TF-IDF retrieves in <5ms with no
model download, and the whole point of the retrieval layer here is provenance,
not the last few points of recall@k. Embedding retrieval is the first item on
the one-week roadmap, with the harness already in place to measure whether it
actually helps (reports/retrieval_eval.md).

LEAKAGE DISCIPLINE: the index is built from the training split only. If val/test
messages were indexed, a test message would retrieve *itself* with similarity
1.0 and every downstream evidence-quality number would be fiction.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from text_utils import clean_text, is_boilerplate_reply

INDEX_PATH = C.MODELS / "retrieval_index.joblib"


class HistoricalCaseIndex:
    def __init__(self, vectorizer, matrix, frame: pd.DataFrame):
        self.vectorizer = vectorizer
        self.matrix = matrix          # L2-normalised tf-idf -> dot product == cosine
        self.frame = frame.reset_index(drop=True)

    @classmethod
    def build(cls, train: pd.DataFrame) -> "HistoricalCaseIndex":
        texts = train.customer_clean.astype(str).tolist()
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                              strip_accents="unicode")
        mat = vec.fit_transform(texts)
        keep = ["customer_text", "customer_clean", "brand_reply", "reply_clean",
                "intent", "reply_is_boilerplate", "conversation_id"]
        return cls(vec, mat, train[keep])

    def search(self, message: str, k: int = C.RETRIEVAL_K) -> list[dict]:
        q = self.vectorizer.transform([clean_text(message, C.BRAND)])
        sims = linear_kernel(q, self.matrix).ravel()
        if sims.size == 0:
            return []
        top = np.argsort(-sims)[:k]
        out = []
        for rank, i in enumerate(top, 1):
            row = self.frame.iloc[int(i)]
            out.append({
                "rank": rank,
                "similarity": round(float(sims[i]), 4),
                "customer_message": str(row.customer_text),
                "historical_reply": str(row.reply_clean),
                "historical_intent": str(row.intent),
                "is_boilerplate": bool(row.reply_is_boilerplate),
                "conversation_id": int(row.conversation_id),
            })
        return out

    # Persist the *components*, not the object. Pickling `self` records the class
    # as `__main__.HistoricalCaseIndex` when this file is run as a script, which
    # then fails to unpickle from any other entry point. Artefacts are produced
    # by this repo's own pipeline, never loaded from an untrusted source.
    def save(self, path=INDEX_PATH):
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix,
                     "frame": self.frame}, path)

    @staticmethod
    def load(path=INDEX_PATH) -> "HistoricalCaseIndex":
        d = joblib.load(path)
        return HistoricalCaseIndex(d["vectorizer"], d["matrix"], d["frame"])


def main():
    train = pd.read_csv(C.PROCESSED / "train.csv")
    idx = HistoricalCaseIndex.build(train)
    idx.save()
    print(f"indexed {idx.matrix.shape[0]:,} historical cases "
          f"({idx.matrix.shape[1]:,} tf-idf features) -> {INDEX_PATH.name}")
    demo = idx.search("I was charged twice for premium this month")
    for d in demo:
        print(f"  {d['similarity']:.3f} [{d['historical_intent']}] {d['customer_message'][:70]}")


if __name__ == "__main__":
    main()
