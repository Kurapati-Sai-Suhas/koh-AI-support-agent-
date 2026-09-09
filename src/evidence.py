"""Tool: validate_evidence() — is the retrieved history strong enough to answer on?

Research anchor: RAGAS (Es et al., EACL 2024 demo) evaluates a RAG answer along
faithfulness / answer-relevance / *context relevance*. Context relevance is a
property of the retrieval, knowable BEFORE any text is generated. We take that
idea and use it as a gate rather than only as a post-hoc metric: if the context
is weak, the correct move is to abstain, not to generate fluently over nothing.

The score is a transparent weighted sum of five signals, not a learned model.
That is deliberate — an interviewer can recompute it by hand from the panel, and
each term is independently inspectable. Weights are set by reasoning about what
makes an answer defensible, then the ESCALATE cut-point is chosen on validation
data (see calibration.py / reports/thresholds.md), so the *decision boundary* is
empirical even though the score is hand-built.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# weight, meaning
W = {
    "top_similarity": 0.35,     # is the single best precedent actually close?
    "support": 0.20,            # do several precedents agree it is this situation?
    "intent_agreement": 0.20,   # do precedents share the predicted intent?
    "substantive": 0.15,        # do precedents contain a resolution, or just "DM us"?
    "consistency": 0.10,        # do the precedents resolve it the *same* way?
}
SUPPORT_SIM = 0.20  # a case counts as "supporting" above this cosine similarity


def _reply_consistency(replies: list[str]) -> float:
    """1.0 = precedents say the same thing, 0.0 = they contradict each other.

    Conflicting precedent is a real failure mode: if history shows both "we
    refunded" and "we cannot refund" for the same complaint, no single grounded
    reply is defensible and the case belongs with a human.
    """
    replies = [r for r in replies if r and len(r.split()) >= 3]
    if len(replies) < 2:
        return 0.5  # unknown, not "good"
    try:
        v = TfidfVectorizer(min_df=1).fit_transform(replies)
        sim = cosine_similarity(v)
        iu = np.triu_indices_from(sim, k=1)
        return float(np.clip(sim[iu].mean(), 0, 1))
    except ValueError:
        return 0.5


def _is_low_information(message: str, predicted_intent: str) -> bool:
    """Context-free conversational follow-up with no standalone intent.

    Found by human review of live agent output: "No it did not" was auto-handled
    at 0.91 confidence with 0.94 evidence quality, because TF-IDF cosine over a
    3-token message matches other short generic messages at similarity 1.0. The
    similarity is real; the *meaning* is not. On the test split this pattern is
    10% of everything the agent auto-handled.

    The rule is deliberately conjunctive. Length alone would wrongly escalate
    legitimate short questions ("hey how do I get verified??" has 4 content
    words), so we require BOTH few content words AND that the classifier could
    only reach the catch-all intent — which is precisely the signature of a
    message whose real intent lives in the previous turn.

    Proper fix is thread context at inference time; this is the honest MVP proxy
    and it is listed as such in the report.
    """
    from critic import _content_words   # local import: avoids a circular import
    return (len(_content_words(message or "")) < 4
            and predicted_intent == "general_complaint_feedback")


def validate_evidence(cases: list[dict], predicted_intent: str,
                      message: str | None = None) -> dict:
    """-> {evidence_quality, signals{...}, flags[...]}"""
    if not cases:
        return {"evidence_quality": 0.0,
                "signals": {k: 0.0 for k in W},
                "flags": ["no_evidence_retrieved"]}

    sims = np.array([c["similarity"] for c in cases], dtype=float)
    top_sim = float(sims.max())
    n_support = int((sims >= SUPPORT_SIM).sum())
    support = min(n_support / 3.0, 1.0)              # 3+ supporting cases saturates
    intent_agreement = float(np.mean([c["historical_intent"] == predicted_intent
                                      for c in cases]))
    substantive = float(np.mean([not c["is_boilerplate"] for c in cases]))
    consistency = _reply_consistency([c["historical_reply"] for c in cases])

    signals = {
        "top_similarity": round(top_sim, 4),
        "support": round(support, 4),
        "intent_agreement": round(intent_agreement, 4),
        "substantive": round(substantive, 4),
        "consistency": round(consistency, 4),
    }
    score = sum(W[k] * signals[k] for k in W)

    flags = []
    if top_sim < 0.20:
        flags.append("weak_similarity")
    if n_support == 0:
        flags.append("no_supporting_case")
    if intent_agreement < 0.34:
        flags.append("intent_mismatch_with_precedent")
    if substantive < 0.34:
        flags.append("precedent_is_boilerplate_only")
    if consistency < 0.10 and len(cases) > 1:
        flags.append("conflicting_precedent")
    if message is not None and _is_low_information(message, predicted_intent):
        flags.append("low_information_message")

    return {"evidence_quality": round(float(np.clip(score, 0, 1)), 4),
            "signals": signals, "flags": flags}
