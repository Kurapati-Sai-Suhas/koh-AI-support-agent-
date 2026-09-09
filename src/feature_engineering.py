"""Feature construction: TF-IDF (word + char) on cleaned text, hand-built
numeric features on the RAW text, fused in one ColumnTransformer.

Design notes worth defending in an interview
--------------------------------------------
* Word 1-2 grams catch the topical signal ("log in", "charged twice").
* Char 3-4 grams (char_wb) are the Twitter insurance policy: they still match
  "cancell", "cancle", "playlst", emoji-glued tokens and hashtag mashups that a
  word vectoriser turns into unseen vocabulary.
* Numeric features run on RAW text because case and punctuation are the signal
  ("WTF IS GOING ON" is a different message from "wtf is going on"); cleaning
  would destroy exactly what they measure.
* StandardScaler is applied ONLY to the 11 dense numeric columns. Standardising a
  sparse TF-IDF matrix would subtract the mean from every zero, densify a
  20k x 200k matrix and blow up memory for no accuracy gain. TF-IDF is already
  L2-normalised per row, so the two blocks arrive at the classifier on
  comparable scales.
* Everything lives inside one sklearn Pipeline, so the vectoriser vocabulary and
  the scaler statistics are fitted on the training fold only — inside
  cross-validation too. That is the mechanical guarantee against preprocessing
  leakage.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from text_utils import clean_text, URL_RE, MENTION_RE, HASHTAG_RE

# A tiny hand-built polarity lexicon. Deliberately not VADER/TextBlob: those pull
# an extra dependency and an NLTK download, and for a 9-word-per-tweet signal the
# gain does not pay for the install. Limitation is stated in the report.
NEG = set("""not no never cant cannot wont dont doesnt broken bug crash crashes fail failed
failing error issue issues problem problems wrong bad worst terrible awful hate sucks stupid
useless annoying angry frustrated disappointed unacceptable ridiculous garbage trash scam
charged stolen missing lost stuck refuse unable""".split())
POS = set("""thanks thank thankyou great love loving awesome amazing perfect good nice happy
best excellent works working fixed solved appreciate please help""".split())

FEATURE_NAMES = ["n_chars", "n_words", "n_sentences", "n_question", "n_exclaim",
                 "upper_ratio", "digit_ratio", "has_url", "n_mentions",
                 "n_hashtags", "sentiment"]


def _row_features(text: str) -> list[float]:
    t = str(text) if text is not None else ""
    words = t.split()
    n_chars = len(t)
    n_words = len(words)
    letters = [c for c in t if c.isalpha()]
    toks = re.findall(r"[a-z']+", t.lower())
    pos = sum(w in POS for w in toks)
    neg = sum(w in NEG for w in toks)
    return [
        n_chars,                                              # long rants vs one-liners
        n_words,
        max(1, len(re.findall(r"[.!?]+", t))),                # sentence count
        t.count("?"),                                         # questions -> how-to intents
        t.count("!"),                                         # urgency / anger
        (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,  # SHOUTING
        (sum(c.isdigit() for c in t) / n_chars) if n_chars else 0.0,  # order/amounts
        1.0 if URL_RE.search(t) else 0.0,                     # screenshots/links
        len(MENTION_RE.findall(t)),                           # public pile-on vs 1:1
        len(HASHTAG_RE.findall(t)),
        (pos - neg) / max(1, pos + neg),                      # crude polarity in [-1, 1]
    ]


class NumericFeatures(BaseEstimator, TransformerMixin):
    """Raw text -> dense (n, 11) matrix. Stateless, so nothing can leak here."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return np.asarray([_row_features(t) for t in np.asarray(X).ravel()], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)


def make_frame(texts, brand: str) -> pd.DataFrame:
    """The single entry point used by training AND by the API, so a live request
    is featurised by identical code to a training row."""
    texts = [t if isinstance(t, str) else "" for t in texts]
    return pd.DataFrame({"raw": texts, "clean": [clean_text(t, brand) for t in texts]})


def build_feature_union(word_ngram=(1, 2), word_min_df=2, word_max_df=0.9,
                        sublinear=True, char_ngram=(3, 4), char_min_df=3,
                        use_char=True, use_numeric=True) -> ColumnTransformer:
    parts = [
        ("word", TfidfVectorizer(ngram_range=word_ngram, min_df=word_min_df,
                                 max_df=word_max_df, sublinear_tf=sublinear,
                                 strip_accents="unicode"), "clean"),
    ]
    if use_char:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram,
                                              min_df=char_min_df, sublinear_tf=sublinear,
                                              max_features=30000), "clean"))
    if use_numeric:
        parts.append(("num", Pipeline([("feat", NumericFeatures()),
                                       ("scale", StandardScaler())]), ["raw"]))
    return ColumnTransformer(parts, sparse_threshold=1.0)
