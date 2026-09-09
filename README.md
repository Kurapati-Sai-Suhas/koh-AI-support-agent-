# Evidence-Backed AI Support Agent — @SpotifyCares

A hybrid agentic support system for the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. It classifies an incoming customer message, retrieves how the brand
actually resolved similar cases, drafts a grounded reply, and decides whether to
**auto-handle** or **escalate to a human** — with a stated reason and the evidence
attached.

The design bias is **trust over autonomy**: the LLM writes prose inside a box that
deterministic code defines. It cannot change a decision, lower a risk level, or
bypass a policy block.

📄 **[Full report](reports/report.md)** · 🔬 **[Research & gap analysis](reports/research.md)** · 📋 **[Decision log](reports/decision_log.md)** · 📊 **[Evaluation](reports/evaluation.md)**

---

## Headline numbers

| | |
|---|---|
| Brand | `SpotifyCares` — chosen by a 4-axis score, not by volume ([why](reports/report.md#2-selected-brand-spotifycares)) |
| Data | 2.81M raw tweets → 43,092 answered pairs → **20,000 modelled** across 13,637 conversations |
| Intents | 9, derived from the brand's own traffic |
| Intent model | **84.2% accuracy · 0.711 macro F1** (test, 3,076 rows) |
| vs majority baseline | 0.093 macro F1 |
| vs simple TF-IDF baseline | 0.544 macro F1 → **+31%** |
| Agent behaviour | answers **46%** of messages alone at **85.9%** intent accuracy; escalates the rest with a reason |
| Calibration | ECE 0.0290 → 0.0290 (Platt scaling changed nothing — [reported as a negative result](reports/calibration.md)) |

⚠️ **Read [§10 "What is misleading about my headline number"](reports/report.md#10-what-is-misleading-about-my-headline-number) before trusting 84.2%.** Training labels come from weak supervision, so the model is partly graded by its own teacher.

---

## Architecture

```
CUSTOMER MESSAGE
      |
      v
 [1] classify_intent      ML    calibrated probabilities + top-2 margin
 [2] assess_risk          RULE  risk level + hard blocks (on the RAW message)
 [3] retrieve_similar     ML    TF-IDF cosine over 13,891 historical cases
 [4] validate_evidence    RULE  5-signal evidence-quality score + flags
      |
      v
 [5] determine_action     RULE  selective-prediction gate
      |
   AUTO-HANDLE                        ESCALATE
      |                                  |
 [6] generate_reply  LLM          same draft, marked "suggestion only"
 [7] critique_reply  RULE          + written escalation reason
      |                                  |
      +---------------+------------------+
                      v
        EVIDENCE-BACKED SUPPORT DECISION
        intent · confidence · risk · evidence quality · decision ·
        reason · precedents + similarities · critic verdict · full trace
```

| Layer | Owns | Cannot do |
|---|---|---|
| **Agent** | tool order, state, error handling, final decision | — |
| **ML** | intent probabilities, calibration, similarity | set policy |
| **Rules** | risk, hard blocks, thresholds, forbidden commitments | be overridden by the LLM |
| **LLM** | reply wording, advisory critique | change the decision or bypass a block |

---

## Reproduce in under 15 minutes

Verified end-to-end: **~6 minutes** on a laptop (excluding the dataset download).

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Get the data

Download `twcs.csv` (516MB) from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
and place it at `data/raw/twcs.csv`. Kaggle needs an account, so if you have none:

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv','data/raw/twcs.csv')"
```

That mirror is the byte-identical Kaggle CSV (7 columns: `tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`).

### 3. Configure (optional)

```bash
cp .env.example .env
```

**Everything runs without an API key.** With no key, replies are *extractive*
(the brand's own closest precedent, reused verbatim — zero fabrication risk) and
the judge runs a deterministic rubric. Both are stamped `backend: template` /
`heuristic` so no number can be mistaken for an LLM result. Add an
OpenAI-compatible key (NVIDIA NIM's free tier works) to enable the LLM paths.

### 4. Run the pipeline

```bash
python -m src.data_processing    # ~90s  brand scoring, pairs, cleaning, grouped splits
python -m src.train              # ~150s 3 baselines + 36-fit GroupKFold grid search
python -m src.retrieval          # ~5s   build the historical case index
python -m src.calibration        # ~60s  ECE + risk-coverage threshold sweep
python -m src.build_golden_set   # ~10s  200 stratified evaluation examples
python -m src.evaluation         # ~90s  all three evaluation levels
pytest -q                        # 18 tests, incl. 3 leakage guards
```

### 5. Serve

```bash
uvicorn src.api:app --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"message":"@SpotifyCares I was charged twice for premium, please refund"}'
```

```json
{"intent": "billing_payment", "intent_confidence": 0.972, "risk_level": "HIGH",
 "evidence_quality": 0.556, "decision": "ESCALATE",
 "decision_reason": "evidence flag: precedent_is_boilerplate_only; high-risk intent requires human sign-off",
 "reply": "@... Can you DM us your account's email address or username? We'll take a look backstage",
 "reply_is_suggestion_only": true, "evidence": [ ... 4 precedents with similarity ... ]}
```

Endpoints: `POST /predict` · `GET /health` · `GET /intents`

---

## The two manual steps — required, and not fakeable

These are the only things the pipeline cannot produce alone.

### Golden set review (~15 min) — **required before the golden number means anything**

`build_golden_set` samples 200 stratified examples and *proposes* labels. Without
an LLM key the proposals are seeded from the same weak-supervision rules the model
trained on, which makes any score against them **circular**. `evaluation.py`
detects this and refuses to call them human-reviewed.

```bash
python -m src.review_golden_set     # ENTER accepts, 1-9 overrides, q saves and quits
python -m src.evaluation            # re-run; the golden row is now real
```

### Judge–human agreement (~10 min)

```bash
python -m src.human_agreement --rate   # rate 25 replies blind (judge score hidden)
python -m src.human_agreement          # exact / within-1 / Spearman / Cohen's kappa
```

Target is ~80% agreement, not 100% — that is where GPT-4-vs-human and
human-vs-human both sit (Zheng et al., 2023). Much higher would suggest a
degenerate rating distribution, not a great judge.

---

## Repository layout

```
src/
  config.py               paths, seed, thresholds, LLM settings
  text_utils.py           shared normalisation (used by training AND the API)
  intents.py              9-intent taxonomy + weak-supervision labelling functions
  data_processing.py      brand scoring, pair building, cleaning, grouped splits, EDA
  feature_engineering.py  TF-IDF (word+char) + 11 numeric features, correct scaling
  train.py                baselines, GroupKFold grid search, self-verifying artefacts
  calibration.py          ECE + Platt scaling + risk-coverage threshold sweep
  retrieval.py            TF-IDF historical case index (training rows only)
  evidence.py             5-signal evidence-quality score  [validate_evidence]
  risk_engine.py          risk levels, hard blocks, forbidden commitments  [assess_risk]
  response_generation.py  grounded LLM draft + extractive fallback  [generate_reply]
  critic.py               deterministic external checks  [critique_reply]
  agent.py                the orchestrator: tool graph, state, decision
  judge.py                LLM-as-judge rubric + heuristic fallback
  evaluation.py           3-level harness + failure analysis
  build_golden_set.py     stratified 200-example sampler
  review_golden_set.py    human review CLI
  human_agreement.py      blind rating protocol + agreement statistics
  api.py                  FastAPI
reports/                  report, research, decision log, EDA, evaluation, calibration
data/golden_set/          golden_set.csv
tests/                    18 tests, incl. conversation-leakage guards
```

---

## Honest implementation status

| Deliverable | Status |
|---|---|
| Runnable repo, <15 min | ✅ ~6 min |
| Intent classification, 2+ baselines, real HPO | ✅ 3 baselines + ablation, 36 fits |
| Historical retrieval + grounded replies | ✅ extractive path verified · ⚠️ LLM path implemented, **never executed** (no key) |
| Auto-handle / escalate + reason | ✅ thresholds empirically tuned |
| Golden set 150–250 | ⚠️ 200 built · **0 human-reviewed** |
| Automated metrics | ✅ 3 levels |
| LLM-as-judge | ⚠️ implemented · ran on **heuristic backend only** |
| Judge–human agreement | ❌ **NOT IMPLEMENTED** — harness ready, needs your ratings |
| Failure analysis (real examples) | ✅ |
| Misleading-headline section | ✅ |
| Decision log | ✅ 14 entries |
| Tests | ✅ 18 passing |

Nothing above is marked ✅ unless it actually ran.

---

## Known limitations

- **Single-turn.** 50.5% of messages trip no intent rule; many are context-dependent follow-ups (*"ok will try"*). This causes the largest failure mode (65% of errors).
- **Lexical grounding proxy.** Word overlap, not entailment — over-credits verbatim copying, which is what the extractive fallback does.
- **Weak-label ceiling.** Every metric measured against rule labels is optimistic.
- **Hand-built sentiment lexicon**, not VADER — avoids an NLTK download.
- **Non-English dropped** (145 rows), not routed.
- **Ablation beats the shipped model on macro F1** (0.721 vs 0.711): the char-ngram and numeric blocks did not earn their place, and the feature-block choice should have been inside the CV grid rather than tested afterwards.
