"""Intent taxonomy for SpotifyCares + the weak-supervision labelling functions.

WHY WEAK SUPERVISION
--------------------
The dataset ships with no intent labels. Hand-labelling 20k training rows is not
possible in the time budget, so we use Snorkel-style labelling functions: a set
of high-precision regex rules per intent, combined by a weighted vote.

This is a deliberate trade and it has a cost we measure rather than hide: a model
trained on rule labels partly re-learns the rules, so its score on rule-labelled
test data is optimistic. That is exactly why the 200-example golden set is
labelled independently of these rules and is the number we report as headline.
See reports/report.md -> "What is misleading about my headline number?".
"""
import re
from collections import OrderedDict

# name -> {description, patterns [(weight, regex)], example}
TAXONOMY = OrderedDict()


def _add(name, description, patterns, example):
    TAXONOMY[name] = dict(
        description=description,
        example=example,
        patterns=[(w, re.compile(p, re.I)) for w, p in patterns],
    )


_add(
    "playback_streaming_issue",
    "Music/podcast will not play, stops, skips, buffers, or offline mode fails.",
    [
        (2.0, r"\b(won'?t|will not|can'?t|cannot|doesn'?t|does not|not)\s+(play|stream|load|start)"),
        (2.0, r"\b(keeps?|keep)\s+(skipping|stopping|pausing|buffering|cutting out|crashing)"),
        (1.5, r"\b(buffer(ing)?|stutter|skipping|no sound|no audio|silent|glitch)\b"),
        (1.5, r"\b(offline (mode|listening)|download(ed)? songs? (won'?t|not))\b"),
        (1.2, r"\bshuffle\b.*\b(broken|not work|weird|same songs)"),
        (1.2, r"\b(songs?|music|playlist|podcast|track)\b.*\b(not playing|won'?t play|stopped)"),
        # tier 2: topical triggers that do not require a full fault phrase
        (1.3, r"\b(play|pause|skip|shuffle|repeat|volume|next track|previous)\b.{0,25}\b(no longer|not work\w*|doesn'?t work|broken|stopped|won'?t)\b"),
        (1.2, r"\b(keeps? (pausing|stopping)|cuts? out|randomly (stops|pauses)|stops playing)\b"),
        (1.2, r"\b(offline|downloaded?)\b.{0,20}\b(gone|disappear\w*|not there|missing)\b"),
    ],
    "@SpotifyCares my downloaded songs won't play offline since the update",
)
_add(
    "account_login_access",
    "Cannot sign in, password/email problems, locked or compromised account.",
    [
        (2.5, r"\b(can'?t|cannot|unable to|won'?t let me)\s+(log ?in|login|sign ?in|access)"),
        (2.0, r"\b(log ?in|login|sign ?in|password|reset link)\b.*\b(not work|doesn'?t work|fail|error|issue|problem)"),
        (2.0, r"\b(hacked|compromised|someone (else )?(is )?using|stolen account)\b"),
        (1.8, r"\b(locked out|account (locked|disabled|suspended))\b"),
        (1.5, r"\b(forgot|reset|change)\b.*\bpassword\b"),
        (1.2, r"\b(facebook login|log in with facebook)\b"),
        (1.5, r"\b(email|username|account)\b.{0,30}\b(forgot|don'?t remember|do not remember|recover|lost|no longer have)\b"),
        (1.5, r"\b(forgot|don'?t remember|lost)\b.{0,30}\b(email|username|password|login|account)\b"),
        (1.2, r"\b(verification code|2fa|two.factor|confirm my email)\b"),
    ],
    "@SpotifyCares I can't log in to my account, the password reset email never arrives",
)
_add(
    "billing_payment",
    "Charged incorrectly, double charge, payment declined, invoice or refund of a charge.",
    [
        (2.5, r"\b(charg(ed|e|ing)|billed|debited)\b.*\b(twice|again|two|double|wrong|still|without|extra)"),
        (2.5, r"\b(double|duplicate|unauthorised|unauthorized|fraudulent)\s+(charge|payment|billing)"),
        (2.0, r"\b(payment|card|billing)\b.*\b(fail(ed|ing)?|declin(e|ed)|not work|issue|problem|error)"),
        (2.0, r"\brefund\b"),
        (1.8, r"\b(money|charge|payment)\b.*\b(taken|took|back|returned)\b"),
        (1.5, r"\b(invoice|receipt|billing date|payment method)\b"),
        (1.3, r"\b(charge|charged|payment|billing|subscription fee)\b.{0,30}\b(why|still|again|wrong|didn'?t|haven'?t)\b"),
        (1.2, r"\bmoney\b.{0,20}\b(taken|gone|missing)\b"),
    ],
    "@SpotifyCares you charged me twice this month, I want the second payment refunded",
)
_add(
    "subscription_plan",
    "Questions about Premium/Free tiers, trials, family or student plans, upgrades.",
    [
        (2.0, r"\b(family|duo|student|premium|free trial|trial)\s+(plan|account|subscription|offer)?\b.*\b(how|can|why|not|issue|problem|work)"),
        (2.0, r"\b(upgrade|downgrade|switch)\b.*\b(plan|premium|account|family)"),
        (1.8, r"\b(student discount|family plan|premium duo|3 month|free month)\b"),
        (1.5, r"\b(premium)\b.*\b(not (working|active)|still (free|showing)|didn'?t activate)"),
        (1.2, r"\b(subscription)\b.*\b(status|active|renew)"),
        (1.3, r"(\$|£|€)\s?\d+([.,]\d+)?"),
        (1.2, r"\b(price|pricing|cost|deal|offer|discount|promo|months? for|per month)\b"),
        (1.2, r"\b(premium|free|trial|family|duo|student)\b.{0,25}\b(work|active|expire|renew|start|end|switch|change)\w*\b"),
    ],
    "@SpotifyCares I paid for Premium but my account still shows as Free",
)
_add(
    "cancellation_request",
    "Wants to cancel, unsubscribe, or delete the account.",
    [
        (3.0, r"\b(cancel|cancelling|canceling)\b.*\b(subscription|premium|account|membership|plan)"),
        (2.5, r"\b(how (do|can) i|want to|trying to|need to)\s+(cancel|unsubscribe|delete my account)"),
        (2.5, r"\b(unsubscribe|delete my account|close my account|end my subscription)\b"),
        (1.5, r"\bcancel\b"),
    ],
    "@SpotifyCares how do I cancel my premium subscription? I can't find the option",
)
_add(
    "content_availability",
    "A song/album/artist/podcast is missing, removed, or not available in a country.",
    [
        (2.5, r"\b(not available|unavailable|isn'?t available|not in my country|region)\b"),
        (2.2, r"\b(missing|removed|disappeared|gone|taken (off|down))\b.*\b(song|album|artist|podcast|episode|playlist|track)"),
        (2.2, r"\b(song|album|artist|podcast|episode|track)\b.*\b(missing|removed|disappeared|gone|not (there|on|available))"),
        (2.0, r"\b(when|why|will)\b.*\b(add|available|release|come to)\b.*\b(spotify|the app|your platform)"),
        (1.5, r"\b(greyed out|grayed out|can'?t find)\b.*\b(song|album|artist|podcast)"),
        (1.5, r"\blaunch(ing)?\b.*\b(country|nigeria|india|region)"),
        (1.6, r"\b(not|isn'?t|aren'?t)\b.{0,25}\b(on|available (on|in)|in)\b.{0,18}\b(spotify|<brand>|your (app|platform)|my country|india|africa|nigeria)\b"),
        (1.5, r"\b(bring back|put .{1,30} back|add .{1,40} to (spotify|your app)|upload .{1,30} to spotify)\b"),
        (1.5, r"\b(removed|taken off|deleted)\b.{0,30}\b(song|album|artist|track|episode|podcast|from spotify)\b"),
        (1.4, r"\bwish\b.{0,30}\b(was|were)\b.{0,20}\b(on|available)\b"),
        (1.3, r"\b(why|when)\b.{0,25}\b(isn'?t|is|will)\b.{0,30}\b(available|on spotify|added)\b"),
    ],
    "@SpotifyCares why has the whole album disappeared from my library? It is not available anymore",
)
_add(
    "app_device_bug",
    "App crashes, will not install/update, or misbehaves on a specific device.",
    [
        (2.5, r"\b(app|spotify)\b.*\b(crash(es|ing|ed)?|freez(e|es|ing)|keeps closing|won'?t open|force clos)"),
        (2.0, r"\b(update|install|reinstall|download the app)\b.*\b(fail|error|won'?t|can'?t|broke|issue)"),
        (2.0, r"\b(since|after)\b.*\bupdate\b.*\b(broke|not work|worse|bug)"),
        (1.8, r"\b(xbox|playstation|ps4|alexa|echo|google home|car ?play|android auto|apple watch|chromecast|sonos|smart ?tv)\b"),
        (1.5, r"\b(bug|glitch|error code|not responding)\b"),
        (1.4, r"\b(android|ios|iphone|ipad|samsung|galaxy|pixel|windows|mac|desktop|web ?player|tablet)\b.{0,35}\b(update|version|not work\w*|broken|bug|crash\w*|no longer|issue|problem)\b"),
        (1.3, r"\bno longer work\w*\b"),
        (1.2, r"\b(when|whens|when's)\b.{0,25}\b(update|new (app|version)|fix)\b"),
    ],
    "@SpotifyCares the app crashes every time I open it on my Xbox One since the last update",
)
_add(
    "feature_how_to",
    "Asking how to do something or whether a feature exists. No fault reported.",
    [
        (2.2, r"^\s*(hi |hey |hello )?(how (do|can|would) i|is there a way|can i|is it possible|where (is|do i find))\b"),
        (1.8, r"\b(how (do|can) i)\b"),
        (1.5, r"\b(is there (a way|an option)|any plans to|will you add|feature request)\b"),
        (1.2, r"\bhow to\b"),
        (1.3, r"\b(please (add|make)|can you (add|make)|any plans to|would be (nice|great|cool) if|feature request)\b"),
    ],
    "@SpotifyCares is there a way to see my listening history from last year?",
)
_add(
    "general_complaint_feedback",
    "Catch-all: venting, praise, or feedback with no specific actionable fault.",
    [
        (1.0, r"\b(sucks?|terrible|awful|worst|hate|garbage|trash|fix your|useless|ridiculous)\b"),
        (0.8, r"\b(love|great|amazing|thank you|thanks)\b"),
        (0.6, r"\b(please|why|seriously)\b"),
    ],
    "@SpotifyCares your app is genuinely the worst thing on my phone right now",
)

INTENTS = list(TAXONOMY.keys())
FALLBACK_INTENT = "general_complaint_feedback"
MIN_SCORE = 1.0  # below this, no rule really fired -> fallback


def score_intents(text: str) -> dict:
    """Return {intent: summed weight of matching rules}."""
    t = text or ""
    scores = {}
    for name, spec in TAXONOMY.items():
        s = 0.0
        for w, rx in spec["patterns"]:
            if rx.search(t):
                s += w
        if s:
            scores[name] = round(s, 2)
    return scores


def label_message(text: str):
    """Weak label -> (intent, rule_vote_confidence, all_scores).

    Ties break by taxonomy order, which puts specific fault intents ahead of the
    catch-all, so a message that trips both a specific rule and the vague
    complaint rule lands on the specific one.
    """
    scores = score_intents(text)
    if not scores:
        return FALLBACK_INTENT, 0.0, scores
    best = max(scores.items(), key=lambda kv: (kv[1], -INTENTS.index(kv[0])))
    if best[1] < MIN_SCORE:
        return FALLBACK_INTENT, 0.0, scores
    total = sum(scores.values())
    return best[0], round(best[1] / total, 3), scores


def taxonomy_table():
    return [
        dict(intent=k, description=v["description"], example_message=v["example"],
             n_rules=len(v["patterns"]))
        for k, v in TAXONOMY.items()
    ]
