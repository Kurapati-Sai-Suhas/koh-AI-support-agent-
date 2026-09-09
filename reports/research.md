# Research review, gap analysis, and what it changed

Scope note, stated up front: this was a **focused** review done inside a two-hour
build, not a systematic survey. Ten primary sources were read at abstract/method
level and each one is cited below only where it actually changed a line of code or
a claim in the report. Papers I could not verify are not cited.

---

## 1. What the literature says, and what we took from it

| # | Work | Problem it addresses | Method | What we took | What we rejected, and why |
|---|---|---|---|---|---|
| 1 | **τ-bench** — Yao, Shinn, Razavi, Narasimhan (2024), [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) | Can LLM agents follow domain policy in real customer-service workflows? | Simulated user + tool APIs + written policy; database-state comparison; `pass^k` over repeated trials | The single most decision-changing finding: frontier function-calling agents scored **<50% task success and pass^8 < 25% in retail** — i.e. the same task solved once is often failed on retry. So **policy and control flow are Python, not prompt text** (`risk_engine.py`, `agent.py`) | An LLM-planned control loop. Non-determinism in the escalation path is the one thing a trust-first support agent cannot afford |
| 2 | **ReAct** — Yao et al. (2023), ICLR, [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) | Interleaving reasoning with tool use | Thought/Action/Observation loop | The decomposition into named tools with structured outputs, and the idea that a trace is what makes an agent auditable (`state["trace"]`) | Free-form iterative tool selection. Our tool order is fixed because every case needs the same four facts before a decision is possible |
| 3 | **Selective Classification for DNNs** — Geifman & El-Yaniv (2017), NeurIPS, [arXiv:1705.08500](https://arxiv.org/abs/1705.08500) | Trading coverage for lower error via abstention | Risk-coverage curve; selective risk at a confidence threshold | The entire framing of escalation as **abstention**, and the evaluation format: report the risk-coverage curve, not one flattering point (`calibration.py`, `reports/risk_coverage_sweep.csv`) | Learned selection functions (SelectiveNet-style). No labelled escalation data exists here to train one |
| 4 | **On Calibration of Modern Neural Networks** — Guo et al. (2017), ICML, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) | Confidence ≠ accuracy in modern nets | Temperature/Platt scaling on held-out data; ECE | Ran Platt scaling + ECE properly on a conversation-disjoint half of validation. **Result was negative** and is reported as such: ECE 0.029 → 0.029 | Assuming miscalibration. A regularised linear model is a different regime from a deep net and was already near-calibrated |
| 5 | **Snorkel** — Ratner et al. (2017), VLDB, [arXiv:1711.10160](https://arxiv.org/abs/1711.10160) | No labels, no budget to hand-label | Labelling functions + denoising label model | Weak supervision as the training-label strategy (`intents.py`), and the explicit expectation that rule labels are *noisy* | Snorkel's generative label model. With 9 mostly-disjoint rule groups a weighted vote is close enough, and the dependency is not worth the install |
| 6 | **RAGAS** — Es et al. (2024), EACL demo, [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) | Reference-free RAG evaluation | Faithfulness / answer relevance / **context relevance** | Turned context relevance from a *post-hoc metric* into a **pre-generation gate** (`evidence.py`): if the retrieved context is weak, abstain rather than generate fluently over nothing | LLM-decomposed statement-level faithfulness. Needs an LLM per statement per reply; we approximate with lexical overlap and say so |
| 7 | **LLMs Cannot Self-Correct Reasoning Yet** — Huang et al. (2024), ICLR, [arXiv:2310.01798](https://arxiv.org/abs/2310.01798) | Does self-critique actually help? | Intrinsic self-correction without external feedback | Shaped the critic decisively: **every verdict-changing check in `critic.py` is deterministic and external** (regex policy checks, evidence overlap, URL provenance). The optional LLM critique may only lower a verdict, never raise it | "Ask the LLM if its answer was good" as the reliability mechanism |
| 8 | **Judging LLM-as-a-Judge** — Zheng et al. (2023), NeurIPS, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) | Are LLM judges trustworthy? | MT-Bench/Chatbot Arena; single-answer grading; bias taxonomy | The rubric format (reason first, then per-dimension scores) and the **~80% target**: GPT-4-vs-human agreement is ~80%, which is also human-vs-human agreement. So >95% agreement would be a red flag, not a triumph (`human_agreement.py`) | Pairwise comparison. We need an absolute quality score per reply, not a preference ordering |
| 9 | **LLM Evaluators Favor Their Own Generations** — Panickssery, Bowman, Feng (2024), NeurIPS | Self-preference bias in LLM judges | Self-recognition correlates with self-preference | `JUDGE_MODEL` is a separate env var from `LLM_MODEL`, and every judge record stores `self_graded` so the report can disclose the configuration | Silently using one model for both roles |
| 10 | **Leakage and the Reproducibility Crisis** — Kapoor & Narayanan (2023), *Patterns* 4(9) | Leakage invalidates published ML results | Taxonomy of 8 leakage types across 294 papers | Grouped splitting by `conversation_id` **everywhere** — train/val/test, `GroupKFold` inside HPO, the calibration/threshold halves, and the retrieval index. Enforced by tests, not by intention | Random row-level splitting, which would have inflated every number here |

**Dataset**: Customer Support on Twitter, Kaggle `thoughtvector/customer-support-on-twitter` (~2.81M tweets). Obtained via a HuggingFace mirror of the identical CSV (`SunidhiSriram/twcs`) because Kaggle download requires credentials; schema verified column-for-column against the Kaggle description.

---

## 2. Gap analysis

Assessed *before* implementation, against what the research above would consider adequate.

| Current design | Research practice | Gap | Severity | Fix | Done in 2h? |
|---|---|---|---|---|---|
| Escalate on a hand-set confidence number | Risk-coverage curve, threshold chosen for a target selective risk [3] | Thresholds were arbitrary | **High** | Sweep (confidence × evidence) on validation, publish the whole curve | ✅ `calibration.py` |
| Probabilities used as if meaningful | ECE measured, calibration fitted on held-out data [4] | "Confidence 0.9" was an unaudited claim | **High** | Fit Platt scaling, measure ECE before/after, report honestly even when negative | ✅ (negative result reported) |
| Retrieve top-k, generate | Context relevance gates the answer [6] | System would answer confidently over irrelevant precedent | **High** | 5-signal evidence score + flags, gates the decision | ✅ `evidence.py` |
| Prompt says "don't promise refunds" | Policy as code, not as prompt text [1] | A prompt constraint is a suggestion | **Critical** | Deterministic hard blocks + post-generation forbidden-commitment regexes | ✅ `risk_engine.py` + `critic.py` |
| LLM self-checks its draft | Intrinsic self-correction unreliable [7] | Critic would be theatre | **High** | All verdict-changing checks external and deterministic | ✅ `critic.py` |
| Judge asserted to be good | Agreement measured against humans, biases named [8][9] | Unsupported trust claim | **High** | Blind rating protocol, 4 agreement statistics, separate judge model | ⚠️ Harness ✅, human ratings **pending user** |
| Random train/test split | Group splitting by conversation [10] | Thread leakage inflates everything | **Critical** | `conversation_id` grouping everywhere + tests | ✅ `test_pipeline.py` |
| No conflicting-evidence handling | Contradiction detection in RAG eval [6] | Two opposite precedents → confident wrong reply | Medium | Pairwise TF-IDF consistency across retrieved replies | ✅ `evidence.py::_reply_consistency` |
| No OOD / unsupported-request detection | Abstention on distribution shift [3] | Unknown requests answered anyway | Medium | Low max-similarity + `no_supporting_case` flag → escalate | ✅ (proxy only) |
| Single-turn only | Conversation-level modelling | Follow-up turns lack context | Medium | Out of scope; measured instead — 50.5% of messages trip no intent rule, many being context-dependent follow-ups | ❌ documented, one-week item |
| Lexical grounding proxy | NLI entailment for faithfulness [6] | Word overlap over-credits copying | Medium | Would need an NLI model download | ❌ one-week item |
| No embedding retrieval | Dense retrieval standard | TF-IDF misses paraphrase | Low-Med | Sentence-transformer index | ❌ one-week item, harness ready |
| No human feedback loop | Human-in-the-loop learning | No path from reviewer edits to model | Low (for MVP) | Log reviewer overrides as training signal | ❌ one-week item |

---

## 3. Architecture classification

Every major choice, labelled by *why* it exists:

**RESEARCH-SUPPORTED**
- Deterministic control flow and policy-as-code [1]
- Escalation modelled as selective prediction; risk-coverage reporting [3]
- Evidence quality gating generation [6]
- Externally-grounded, non-LLM critic [7]
- Conversation-level grouping everywhere [10]
- Weak supervision for training labels [5]
- Judge rubric shape and the ~80% agreement expectation [8][9]

**ENGINEERING TRADE-OFF**
- TF-IDF + logistic regression over a transformer — explainability, 5ms inference, no GPU, and the intent signal here is largely lexical
- TF-IDF retrieval over dense embeddings — no model download inside a 15-minute reproduction budget
- Logistic regression over LinearSVC — the decision layer needs probabilities; SVC would need an extra calibration wrapper
- Hand-weighted evidence score over a learned one — no labelled evidence-quality data exists

**MVP SIMPLIFICATION** (all disclosed as limitations)
- Single-turn classification; thread context unused at inference
- Lexical overlap as the grounding proxy instead of NLI entailment
- Hand-built sentiment lexicon instead of VADER/TextBlob
- Non-English messages dropped (145 rows) rather than routed
- One rater for judge agreement, not an inter-annotator study
