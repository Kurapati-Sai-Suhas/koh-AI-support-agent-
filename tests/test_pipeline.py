"""Guard-rail tests. These check the claims the report makes, not the happy path.

    pytest -q
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C                       # noqa: E402
from text_utils import clean_text        # noqa: E402
from intents import label_message, INTENTS  # noqa: E402
from risk_engine import assess_risk, check_forbidden  # noqa: E402
from evidence import validate_evidence   # noqa: E402
from critic import critique_reply, lexical_grounding  # noqa: E402


# ---------- the leakage claim ------------------------------------------------
@pytest.mark.skipif(not (C.PROCESSED / "train.csv").exists(), reason="run data_processing")
def test_no_conversation_spans_two_splits():
    tr = set(pd.read_csv(C.PROCESSED / "train.csv").conversation_id)
    va = set(pd.read_csv(C.PROCESSED / "val.csv").conversation_id)
    te = set(pd.read_csv(C.PROCESSED / "test.csv").conversation_id)
    assert not (tr & te), "conversation leaked between train and test"
    assert not (tr & va), "conversation leaked between train and val"
    assert not (va & te), "conversation leaked between val and test"


@pytest.mark.skipif(not (C.PROCESSED / "train.csv").exists(), reason="run data_processing")
def test_no_duplicate_messages_across_splits():
    tr = set(pd.read_csv(C.PROCESSED / "train.csv").customer_clean)
    te = set(pd.read_csv(C.PROCESSED / "test.csv").customer_clean)
    assert not (tr & te), "identical message text in train and test"


@pytest.mark.skipif(not (C.MODELS / "retrieval_index.joblib").exists(), reason="run retrieval")
def test_retrieval_index_contains_only_training_rows():
    """A test message must not be able to retrieve itself."""
    from retrieval import HistoricalCaseIndex
    idx = HistoricalCaseIndex.load()
    te = set(pd.read_csv(C.PROCESSED / "test.csv").conversation_id)
    assert not (set(idx.frame.conversation_id) & te), "test conversations are in the index"


# ---------- preprocessing ----------------------------------------------------
def test_clean_text_normalises_noise():
    out = clean_text("@SpotifyCares WHY?! http://t.co/x @115712 #Premium", "SpotifyCares")
    assert "<brand>" in out and "<url>" in out and "<user>" in out
    assert "premium" in out          # hashtag word survives
    assert "http" not in out


def test_clean_text_handles_none_and_empty():
    assert clean_text(None, "SpotifyCares") == ""
    assert clean_text("", "SpotifyCares") == ""


# ---------- intent rules -----------------------------------------------------
@pytest.mark.parametrize("msg,expected", [
    ("I can't log in to my account", "account_login_access"),
    ("you charged me twice, refund please", "billing_payment"),
    ("how do I cancel my premium subscription", "cancellation_request"),
    ("the app crashes on my xbox every time", "app_device_bug"),
])
def test_labelling_functions(msg, expected):
    assert label_message(msg)[0] == expected


def test_every_message_gets_a_valid_label():
    for m in ["", "?", "hi", "asdkjhasd", "🎵🎵"]:
        assert label_message(m)[0] in INTENTS


# ---------- risk policy ------------------------------------------------------
def test_hard_block_fires_regardless_of_intent():
    r = assess_risk("someone hacked my account", "feature_how_to")
    assert r["risk_level"] == "HIGH"
    assert "account_takeover_or_security" in r["blocks"]


def test_explicit_human_request_is_blocked():
    assert assess_risk("let me speak to a real human", "feature_how_to")["blocks"]


def test_forbidden_commitments_detected():
    assert "promises_refund" in check_forbidden("We will issue a refund today")
    assert "requests_credentials" in check_forbidden("Please DM us your password")
    assert check_forbidden("Can you tell us your device and OS version?") == []


# ---------- evidence + critic ------------------------------------------------
def test_evidence_quality_zero_without_cases():
    ev = validate_evidence([], "billing_payment")
    assert ev["evidence_quality"] == 0.0
    assert "no_evidence_retrieved" in ev["flags"]


def _case(sim, reply, intent="billing_payment", boiler=False, rank=1):
    return {"rank": rank, "similarity": sim, "customer_message": "x",
            "historical_reply": reply, "historical_intent": intent,
            "is_boilerplate": boiler, "conversation_id": 1}


def test_evidence_quality_rises_with_good_precedent():
    weak = validate_evidence([_case(0.05, "DM us", boiler=True)], "billing_payment")
    strong = validate_evidence(
        [_case(0.8, "Sorry about that, we can check the charge, send us your email", rank=1),
         _case(0.7, "We can look at the charge if you send your account email", rank=2),
         _case(0.6, "Send us the email on the account and we will check the charge", rank=3)],
        "billing_payment")
    assert strong["evidence_quality"] > weak["evidence_quality"]


def test_critic_escalates_on_forbidden_promise():
    cases = [_case(0.9, "We can look into the charge if you DM your email")]
    v = critique_reply("We will refund you within 3 business days", cases, "HIGH")
    assert v["verdict"] == "ESCALATE"
    assert any("policy_violation" in i for i in v["issues"])


def test_critic_escalates_on_invented_link():
    cases = [_case(0.9, "Please check your settings")]
    v = critique_reply("Go to https://evil.example.com to fix your settings", cases, "LOW")
    assert "invented_link" in v["issues"]


# ---- fixes driven by the human review (reports/human_review.csv) -----------
def test_low_information_followup_is_flagged():
    """"No it did not" was auto-handled at 0.91 confidence before this guard."""
    cases = [_case(1.0, "Thanks for confirming!", intent="general_complaint_feedback")]
    ev = validate_evidence(cases, "general_complaint_feedback", "@SpotifyCares No it did not")
    assert "low_information_message" in ev["flags"]


def test_short_but_specific_question_is_not_flagged():
    """The guard must not escalate legitimate short questions."""
    cases = [_case(0.7, "Check the steps here", intent="feature_how_to")]
    ev = validate_evidence(cases, "feature_how_to",
                           "@SpotifyCares hey how do I get verified?? Thanks!")
    assert "low_information_message" not in ev["flags"]


def test_evidence_validation_is_backwards_compatible():
    """message is optional: omitting it must not raise or add the flag."""
    ev = validate_evidence([_case(0.5, "some reply")], "billing_payment")
    assert "low_information_message" not in ev["flags"]


@pytest.mark.parametrize("draft,banned", [
    ("@user Hey there, can you DM us?", "@user"),
    ("@328829 Thanks for confirming.", "@328829"),
    ("@SpotifyCares Hey! Check the steps.", "@SpotifyCares"),
])
def test_draft_strips_leaked_handles(draft, banned):
    from response_generation import _clean_draft
    out = _clean_draft(draft)
    assert banned.lower() not in out.lower()
    assert out and not out.startswith("@")


def test_lexical_grounding_bounds():
    cases = [_case(0.9, "send us your account email and we will check the charge")]
    assert lexical_grounding("", cases) == 0.0
    assert 0.0 <= lexical_grounding("send us your account email", cases) <= 1.0
