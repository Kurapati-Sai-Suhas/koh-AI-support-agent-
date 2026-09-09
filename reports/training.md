# Training results

Brand `SpotifyCares` · 13,891 train / 3,033 val / 3,076 test rows
(conversation-disjoint splits, labels from weak supervision)

| model | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| baseline_1_majority | 0.720 | 0.093 | 0.603 |
| baseline_2_tfidf_lr | 0.813 | 0.544 | 0.790 |
| baseline_3_complement_nb | 0.786 | 0.528 | 0.767 |
| ablation_word_tfidf_only | 0.841 | 0.721 | 0.847 |
| tuned_full_model | 0.842 | 0.711 | 0.848 |

**Improvement over the simple baseline: macro-F1 0.544 -> 0.711 (+30.7%)**

## Best hyper-parameters

```json
{
  "clf__C": "10.0",
  "clf__class_weight": "balanced",
  "features__word__ngram_range": "(1, 2)"
}
```

Searched 12 configurations with 3-fold GroupKFold cross-validation on the training split only (109s). Best CV macro-F1 0.718.

The test numbers above are measured against *weak-supervision* labels and are
optimistic by construction. The honest headline is the golden-set score in
`reports/evaluation.md`.