# Demo script — 4 minutes

A walkthrough of the working system. Everything shown comes from the live API.

---

## 0 · Setup (before the call)

```bash
uvicorn src.api:app --port 8000
```

One process serves both the API and the UI. Open <http://localhost:8000>.

Check the badge under the title reads something like
`SpotifyCares · 13,891 historical cases · llm:…` — that is the live `/health`
response, so if it renders, the model, retrieval index and backends all loaded.

> **If you have an LLM key configured**, a single `/predict` takes 10–25s on a
> free tier because of rate limiting. Say so up front, or unset `LLM_API_KEY` for
> a snappier demo — replies then come from the extractive fallback and the UI
> labels them `generation_backend: template`.

---

## 1 · Frame the problem (20s)

> "Brands answer complaints publicly on Twitter. A wrong public answer is
> quotable and permanent; routing to a human costs a few minutes. So the goal
> isn't to answer everything — it's to answer what we can answer **with
> evidence**, and hand over the rest **with a reason**."

---

## 2 · An auto-handled case (60s)

Click the **Easy auto-handle** chip →
*"@SpotifyCares when will reputation be available to stream??"* → **Analyze Message**.

Walk the page top to bottom — it is ordered as the argument:

1. **Green AUTO-HANDLE banner** — "Agent can safely handle this request", with the reason spelled out beneath it.
2. **Four stat cards** — intent `content_availability`, confidence ~66%, risk LOW, evidence quality ~78%.
3. **Suggested response** — the drafted reply.
4. **Why the agent trusts this** — confidence and margin, closest precedent similarity, what the evidence score is made of, risk level.
5. **Historical evidence** — four real past cases with similarity badges.

The line to land:

> "The top precedent is a 70% match, and Spotify's own reply to it is right there.
> The draft reuses that wording — including the real link from the evidence. It
> isn't inventing an answer, it's reusing a proven one."

Open **Agent trace & raw signals** to show the 7 tools and the five evidence
sub-signals.

---

## 3 · A hard policy block (45s)

Click **Clear escalation (hard block)** — an account-compromise message.

> "The classifier is 98% confident here. It still escalates — because
> `assess_risk` matched account-takeover language on the **raw message**, before
> the classifier ran. Policy is Python, not a line in a prompt. High confidence
> cannot talk it out of this."

Point at the amber banner reason: `Hard policy block: account_takeover_or_security`.

---

## 4 · Uncertainty, not just rules (45s)

Click **Ambiguous** — *"if i pay 10$ /m and im a student how do i get the discount?"*

> "Confidence 0.49, and the margin between the top two intents is 0.00 —
> `subscription_plan` versus `feature_how_to`. It genuinely is both. The gate
> escalates on the margin, and says so."

Then click **Noisy / context-free follow-up** — the device-specs message.

> "This is a follow-up. The intent lives in the previous turn, so no single-turn
> label is correct. Escalating is the only right answer."

---

## 5 · The bug the review caught (45s)

Type manually: `@SpotifyCares No it did not`

> "This is a real dataset message, and it used to be **auto-handled at 0.91
> confidence with 0.94 evidence quality** — because TF-IDF matches short generic
> messages at similarity 1.00. The similarity was real; the meaning wasn't. The
> draft it produced was a non-sequitur about a video ad.
>
> Human review caught it, and it now escalates on a `low_information_message`
> flag. On the test split that pattern was 10% of everything the agent
> auto-handled."

Honest add-on:

> "The trade is visible: coverage dropped 46% → 38.5%, and weak-label selective
> accuracy actually *fell*, because the rule labels think 'general complaint' was
> the right answer. The metric and the right product decision disagreed. That's
> in the README."

---

## 6 · Evaluation, briefly (45s)

Leave the UI. Show two files:

**`reports/evaluation.md`** — three levels kept separate:

| | |
|---|---|
| Model | 84.2% accuracy, 0.711 macro F1 vs 0.093 majority / 0.544 simple baseline |
| Agent | answers 38.5% alone at 83.1% intent accuracy |
| Response | 6-dimension judge rubric |

**`reports/human_review.csv`** — 10 real cases end-to-end:

> "Decisions were right 9 out of 10, but intents only 4 out of 10. That gap *is*
> the architecture — the gate escalates the cases the classifier gets wrong, so
> classifier errors become escalations instead of wrong public replies."

Then the closing honesty note:

> "The golden set is built but not yet human-reviewed, so its number is still
> circular with the training labels — the script refuses to call it
> human-reviewed. Judge–human agreement isn't implemented; the harness is there
> and needs my ratings. Both are marked ❌/⚠️ in the README rather than papered over."

---

## 7 · Architecture (30s)

Open the README and scroll to the two Mermaid diagrams.

> "Blue is ML, green is deterministic rules, amber is the LLM. The LLM touches
> exactly one box, and it's not the one that decides anything. τ-bench found
> frontier agents fail the same customer-service task on retry a quarter of the
> time — so control flow is code, and the LLM only writes prose inside it."

---

## Fallbacks if something breaks

| symptom | cause | say / do |
|---|---|---|
| "Backend unreachable" | uvicorn not running | `uvicorn src.api:app --port 8000` |
| `/health` shows `degraded` | model artefacts missing | run the pipeline in README → How to run |
| Request takes ~40s | LLM rate-limit retries | it falls back to extractive and labels itself — point at `generation_backend` |
| Reply looks generic | precedents are all "DM us" | that's the `precedent_is_boilerplate_only` flag — the system detected it and escalated |
