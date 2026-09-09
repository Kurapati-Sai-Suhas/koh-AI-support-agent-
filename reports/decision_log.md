# Decision log

Fourteen non-obvious calls, with what each one cost.

---

**1. Brand = SpotifyCares, not AmazonHelp (4× larger)**
*Why:* the system grounds replies in historical resolutions, so reply *substance* beats volume. Amazon is heavily multilingual; Apple sends 39% pure "DM us"; T-Mobile 61%. Spotify: 43k pairs, 98% English, 26% boilerplate, replies with real steps.
*Trade-off:* gave up 4× the data. At 20k modelled pairs we were never data-bound, so the trade was free.

**2. Weak supervision for training labels instead of hand-labelling**
*Why:* no intent labels ship with the dataset, and hand-labelling 14k rows is impossible in 2 hours. Snorkel-style labelling functions (Ratner et al., 2017).
*Trade-off:* the model partly re-learns its own teacher, which makes the test score optimistic. Accepted *because* it is measurable — that is what the independent golden set is for, and it drives §10 of the report.

**3. Broadened the labelling functions mid-build after inspecting the catch-all**
*Why:* the first taxonomy dumped 79% into `general_complaint_feedback`. Sampling 25 of those rows showed real misses (*"y isnt spotify available in south africa"*). Broadening dropped it to 72.2%.
*Trade-off:* cost a full reprocess + retrain (~10 min of a 120-min budget) and slightly lowers rule precision. Worth it — a 79% catch-all is not a taxonomy.

**4. Conversation-level grouping in four places, not one**
*Why:* Kapoor & Narayanan (2023) put leakage behind most irreproducible ML results. Grouping applies to the train/val/test split, `GroupKFold` inside HPO, the calibration/threshold halves, **and the retrieval index**.
*Trade-off:* fewer effective independent samples and slightly worse-looking numbers. The retrieval one matters most: index test rows and a message retrieves itself at similarity 1.0.

**5. Deterministic tool order instead of an LLM planner**
*Why:* τ-bench (2024) — frontier agents scored `pass^8 < 25%` on retail support. Every case needs the same four facts before a decision is possible, so there is nothing for a planner to decide.
*Trade-off:* cannot handle workflows needing genuinely novel tool sequences. For triage-and-draft, that is not a real limitation.

**6. Policy as code, never as prompt text**
*Why:* a prompt constraint is a suggestion. Risk levels, hard blocks and forbidden commitments are Python and unit-tested; the LLM cannot lower a risk level or bypass a block.
*Trade-off:* rigid — regexes miss paraphrases ("I'd like my money back" may evade one). A classifier would generalise better but could not be audited line-by-line.

**7. Logistic regression over LinearSVC**
*Why:* the decision layer thresholds on probabilities. SVC needs `CalibratedClassifierCV` for the same thing at 3× the fits.
*Trade-off:* SVC is often a point or two better on sparse text. Bought decision-layer simplicity with it.

**8. TF-IDF retrieval over sentence embeddings**
*Why:* ~90MB model download and a minutes-long cold encode against a 15-minute reproduction budget. 13.9k documents retrieve in <5ms sparse.
*Trade-off:* misses paraphrase with no lexical overlap. First item on the roadmap — with the harness already in place to *prove* dense wins rather than assume it.

**9. StandardScaler on the 11 numeric features only**
*Why:* standardising sparse TF-IDF subtracts the mean from every zero and densifies a 20k × 200k matrix. TF-IDF rows are already L2-normalised.
*Trade-off:* none. This is just the correct thing.

**10. Numeric features computed on RAW text, TF-IDF on cleaned text**
*Why:* uppercase ratio and punctuation counts *are* the signal — cleaning would erase what they measure.
*Trade-off:* two text columns to keep in sync. Handled by one `make_frame()` used by both training and the API, so train/serve skew is structurally impossible.

**11. Thresholds from a risk-coverage sweep, not chosen by hand**
*Why:* escalation is selective prediction (Geifman & El-Yaniv, 2017). Swept (confidence × evidence quality) on validation, took max coverage subject to ≥90% selective accuracy.
*Trade-off:* tuned against weak labels, so "90%" means 90% agreement with rules. Stated wherever the number appears.

**12. Kept the calibrator despite it doing nothing**
*Why:* ECE went 0.0290 → 0.0290. Reported as a **negative result** rather than quietly dropping the experiment or implying it helped.
*Trade-off:* a slightly odd-looking report section. Better than an unaudited "calibrated confidence" claim — Guo et al. studied deep nets; a regularised linear model is a different regime.

**13. The critic is deterministic, and can only tighten a decision**
*Why:* Huang et al. (ICLR 2024) — intrinsic self-correction does not reliably help. So forbidden-commitment regexes, lexical grounding and URL provenance decide verdicts; an LLM critique is advisory and may only downgrade AUTO-HANDLE → ESCALATE.
*Trade-off:* lexical grounding is a blunt proxy that over-credits verbatim copying — precisely what the extractive fallback does. Called out in the report rather than papered over.

**14. Shipped with two deliverables explicitly incomplete**
*Why:* the golden set needs human review and the judge needs an API key. Both are one command away, but neither can be faked. The evaluation script *refuses* to label rule-seeded golden rows as human-reviewed and stamps every judge record with its backend.
*Trade-off:* the status table shows two ⚠️ and one ❌. An interviewer can verify every green tick — which is worth more than a table of ticks they cannot.
