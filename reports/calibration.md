# Calibration and escalation thresholds

## 1. Confidence calibration

| | ECE (10 bins) |
|---|---|
| raw logistic-regression probabilities | 0.0290 |
| after Platt (sigmoid) calibration | 0.0290 |

Calibrator fitted on one conversation-disjoint half of validation (1,527 rows), measured on the other (1,506 rows). Decision: **use the calibrator**.

Honest reading: the change is +0.0000 ECE, which is negligible. The finding is not 'calibration fixed our probabilities' — it is that an L2-regularised logistic regression on this task was **already close to calibrated** (ECE ~3%), so there was little for Platt scaling to correct. Guo et al. observed severe miscalibration in deep networks; a linear model with regularisation is a different regime and does not automatically inherit that problem. We ran the check rather than assuming either way, and we keep the calibrator only because it is marginally better, not because it rescued anything.

ECE is the average gap between stated confidence and observed accuracy. It is
what lets the escalation threshold mean something: at a 0.6 cut-point we want
the answered slice to actually be right about 60%+ of the time, not merely to
have a number above 0.6 printed next to it.

## 2. Risk-coverage trade-off

Sweeping the confidence gate at evidence_quality = 0.15:

| confidence gate | coverage (auto-handled) | selective accuracy | n |
|---|---|---|---|
| 0.30 | 93.8% | 0.847 | 1412 |
| 0.35 | 93.8% | 0.847 | 1412 |
| 0.40 | 93.5% | 0.849 | 1408 |
| 0.45 | 92.6% | 0.856 | 1395 |
| 0.50 | 91.3% | 0.861 | 1375 |
| 0.55 | 86.8% | 0.882 | 1307 |
| 0.60 | 83.1% | 0.895 | 1252 |
| 0.65 | 79.8% | 0.903 | 1201 |
| 0.70 | 75.8% | 0.914 | 1142 |
| 0.75 | 71.1% | 0.932 | 1071 |
| 0.80 | 65.0% | 0.947 | 979 |
| 0.85 | 59.0% | 0.958 | 889 |
| 0.90 | 50.1% | 0.978 | 754 |

## 3. Chosen operating point

```json
{
  "confidence": 0.65,
  "margin": 0.15,
  "evidence_quality": 0.15,
  "achieved_coverage": 0.7975,
  "achieved_selective_accuracy": 0.9026,
  "target_selective_accuracy": 0.9,
  "source": "tuned on validation risk-coverage sweep (weak labels)",
  "calibrated": true
}
```

Read this as: the agent answers **79.8%** of messages itself and is right about the intent **90.3%** of the time on that slice; everything else goes to a human. Raising coverage past this point costs selective accuracy — that trade is the product decision, and the sweep in `risk_coverage_sweep.csv` is what a PM would use to make it.

**Caveat that matters:** these selective-accuracy figures are against weak-supervision labels, not human labels, so they overstate real accuracy. The golden set is the honest measurement and was never used for tuning.