"""Text normalisation shared by preprocessing, features, retrieval and the API.

One module so that a message hitting POST /predict is cleaned with exactly the
same code that cleaned the training data. Divergence here is a classic source of
silent train/serve skew.
"""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
# In this dataset customers are anonymised to numeric handles (@115712) while
# brands keep their real handle (@AppleSupport). We keep the brand token because
# "who was addressed" is signal, and collapse customer handles to one token.
NUM_MENTION_RE = re.compile(r"@\d+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "]+"
)
WS_RE = re.compile(r"\s+")
SIGNATURE_RE = re.compile(r"\s[-^~]{1,2}\s?[A-Z]{2,3}\s*$")  # agent sign-offs: "^SG", "-JL"


def clean_text(text: str, brand: str | None = None) -> str:
    """Lowercase, normalise noise, keep intent-bearing structure."""
    if not isinstance(text, str):
        return ""
    t = text
    t = URL_RE.sub(" <url> ", t)
    t = NUM_MENTION_RE.sub(" <user> ", t)
    if brand:
        t = re.sub(rf"@{re.escape(brand)}\b", " <brand> ", t, flags=re.I)
    t = MENTION_RE.sub(" <user> ", t)
    t = HASHTAG_RE.sub(r" \1 ", t)          # hashtags carry topic words: #iOS11 -> iOS11
    t = EMOJI_RE.sub(" <emoji> ", t)
    t = t.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    t = WS_RE.sub(" ", t).strip().lower()
    return t


def strip_signature(text: str) -> str:
    """Remove trailing agent initials from a brand reply before showing it as evidence."""
    if not isinstance(text, str):
        return ""
    return SIGNATURE_RE.sub("", text).strip()


def is_boilerplate_reply(text: str) -> bool:
    """True for replies that contain no resolution content ('DM us and we'll help')."""
    t = (text or "").lower()
    t_no_url = URL_RE.sub("", t)
    has_dm = bool(re.search(r"\b(dm|direct message|private message|send us a message)\b", t_no_url))
    return has_dm and len(t_no_url.split()) < 25
