# EDA — SpotifyCares

## Why this brand

Scored the ten highest-volume brands on four axes. Volume alone is not enough:
a brand whose every reply is *"please DM us"* gives a retrieval layer nothing to
ground on, and a heavily multilingual brand needs language ID we chose not to build.

| brand        |   pairs |   boilerplate_pct |   median_reply_chars |   english_share |
|:-------------|--------:|------------------:|---------------------:|----------------:|
| AmazonHelp   |  168814 |               0.6 |                  120 |           0.926 |
| AppleSupport |  106646 |              38.9 |                  129 |           0.969 |
| Uber_Support |   56160 |              31.3 |                  104 |           0.995 |
| SpotifyCares |   43092 |              25.7 |                  131 |           0.983 |
| Delta        |   42114 |              14   |                  102 |           0.985 |
| Tesco        |   38468 |               9   |                  134 |           0.985 |
| AmericanAir  |   36531 |              13.6 |                  107 |           0.99  |
| TMobileHelp  |   34215 |              60.6 |                  126 |           0.987 |
| comcastcares |   32921 |              40.8 |                  124 |           0.992 |
| XboxSupport  |   23235 |              19.2 |                  110 |           0.993 |

`SpotifyCares` wins on the combination: ~43k answered pairs (we only need 20,000), a 98% English share (high enough to skip language ID), and replies that contain concrete resolution steps ("tap the three dots > View Album", "try a reinstall") rather than pure hand-offs. AmazonHelp has 4x the volume but is heavily multilingual; AppleSupport sends 39% pure-boilerplate replies vs Spotify's 26%.

## Dataset shape

- Raw tweets in file: **2,811,774**
- Tweets authored by `SpotifyCares`: **43,092** answered customer tweets

### Funnel from raw pairs to modelling set

| step | rows |
|---|---|
| pairs_raw | 43,092 |
| after_dropna | 43,092 |
| dropped_length | 791 |
| dropped_duplicates | 1,964 |
| dropped_non_english | 145 |
| pairs_final | 20,000 |
| conversations_final | 13,637 |

### Splits (grouped by conversation_id — no thread spans two splits)

| split | rows | conversations |
|---|---|---|
| train | 13,891 | 9,545 |
| val | 3,033 | 2,046 |
| test | 3,076 | 2,046 |

### Message length (cleaned characters)

- min 15 / median 96 / p90 153 / max 303

### Missing values / quality

- customer_text nulls dropped: 0
- exact duplicate messages dropped: 1,964
- non-English dropped: 145
- brand replies that are pure boilerplate ("DM us"): **26.5%** — these are kept but down-ranked as evidence

### Weak-label class distribution (class imbalance)

| intent | n | share |
|---|---|---|
| general_complaint_feedback | 14,432 | 72.2% |
| subscription_plan | 1,703 | 8.5% |
| app_device_bug | 867 | 4.3% |
| feature_how_to | 724 | 3.6% |
| content_availability | 641 | 3.2% |
| account_login_access | 614 | 3.1% |
| playback_streaming_issue | 415 | 2.1% |
| billing_payment | 357 | 1.8% |
| cancellation_request | 247 | 1.2% |

- Imbalance ratio (largest/smallest): **58.4x** — this is why macro-F1, not accuracy, is the model-selection metric.
- Rows where no specific rule fired (fell back to catch-all): **50.5%**

## Intent taxonomy

| intent | description | example | rules |
|---|---|---|---|
| `playback_streaming_issue` | Music/podcast will not play, stops, skips, buffers, or offline mode fails. | @SpotifyCares my downloaded songs won't play offline since the update | 9 |
| `account_login_access` | Cannot sign in, password/email problems, locked or compromised account. | @SpotifyCares I can't log in to my account, the password reset email never arrives | 9 |
| `billing_payment` | Charged incorrectly, double charge, payment declined, invoice or refund of a charge. | @SpotifyCares you charged me twice this month, I want the second payment refunded | 8 |
| `subscription_plan` | Questions about Premium/Free tiers, trials, family or student plans, upgrades. | @SpotifyCares I paid for Premium but my account still shows as Free | 8 |
| `cancellation_request` | Wants to cancel, unsubscribe, or delete the account. | @SpotifyCares how do I cancel my premium subscription? I can't find the option | 4 |
| `content_availability` | A song/album/artist/podcast is missing, removed, or not available in a country. | @SpotifyCares why has the whole album disappeared from my library? It is not available anymore | 11 |
| `app_device_bug` | App crashes, will not install/update, or misbehaves on a specific device. | @SpotifyCares the app crashes every time I open it on my Xbox One since the last update | 8 |
| `feature_how_to` | Asking how to do something or whether a feature exists. No fault reported. | @SpotifyCares is there a way to see my listening history from last year? | 5 |
| `general_complaint_feedback` | Catch-all: venting, praise, or feedback with no specific actionable fault. | @SpotifyCares your app is genuinely the worst thing on my phone right now | 3 |