# 07 — Evaluation Protocol

Owner: Ankit (Evaluation + Baselines + Dashboard/QA)

Status: **Part A only — core evaluation foundation.** Baselines, lead time,
early warning rate, unseen-attack evaluation, and ablation studies are
intentionally not implemented yet (see "Not yet implemented" below).

## 1. What this evaluates

`src/eval/metrics.py` provides the core evaluation foundation for the
project's forecasting output. It scores a single set of predictions
(ground-truth label vs. predicted label, or ground-truth label vs. a risk
score) using standard binary classification metrics. It is deliberately
independent of the model, feature pipeline, and dashboard, so it can be
reused unchanged once real predictions replace mock predictions.

## 2. Metrics implemented

| Metric | Definition |
| --- | --- |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | `2 * Precision * Recall / (Precision + Recall)` |
| False Positive Rate (FPR) | `FP / (FP + TN)` |
| Confusion Matrix | counts of TN, FP, FN, TP |

Where, for binary labels `y_true`/`y_pred` (0 = benign, 1 = malicious):

- **TP** — predicted malicious, actually malicious
- **TN** — predicted benign, actually benign
- **FP** — predicted malicious, actually benign
- **FN** — predicted benign, actually malicious

Confusion matrix ordering is fixed and documented in code
(`ConfusionMatrix`, `matrix()` returns `[[TN, FP], [FN, TP]]`).

Not implemented yet: Lead Time, Early Warning Rate, Unseen Attack
evaluation, Ablation evaluation. These will be added as separate modules
(`lead_time.py`, `unseen_attack.py`, `ablation.py`) that consume the same
prediction contract described below.

## 3. Threshold handling

Model output may be a continuous risk/probability score rather than an
already-binarized label. `risk_to_label(risk, threshold)` converts a risk
score into a binary prediction:

```
risk >= threshold -> 1 (malicious)
risk <  threshold -> 0 (benign)
```

`threshold` defaults to `0.70` (the project's experimental warning
threshold) but is a normal function argument, not a hard-coded constant —
any value in `[0, 1]` may be passed. A threshold outside `[0, 1]` raises
`EvaluationError`.

## 4. Zero-division behaviour

The evaluator must never crash on skewed or degenerate batches (e.g. no
positive predictions, or no positive ground truth) — these are expected
when testing against small mock prediction sets. The policy:

- Precision: `0.0` if `TP + FP == 0` (no positive predictions)
- Recall: `0.0` if `TP + FN == 0` (no positive ground truth)
- F1: `0.0` if precision and recall are both `0.0`
- FPR: `0.0` if `FP + TN == 0` (no negative ground truth)

This is documented explicitly so results are never silently misleading —
a `0.0` from an empty denominator means "undefined, defaulted to zero,"
not "the model failed every case."

## 5. Expected prediction input

Two modes are supported:

**Mode 1 — already-binarized predictions**

```python
evaluate_predictions(y_true=[0, 1, 0, 1], y_pred=[0, 1, 1, 1])
```

**Mode 2 — risk/probability predictions**

```python
evaluate_risk(y_true=[0, 1, 0, 1], risk=[0.20, 0.85, 0.75, 0.90], threshold=0.70)
```

Both return an `EvaluationResult` (precision, recall, f1, fpr,
confusion_matrix), with a `.to_dict()` method for structured/JSON output.

The full forecasting prediction schema (timestamp, host_id, per-horizon
risk at +10s/+20s/+30s) is not yet frozen elsewhere in the repository. This
module only requires flat, per-sample `y_true`/`y_pred` (or `y_true`/`risk`)
arrays, so it can be called once per forecast horizon, or on a flattened
view across horizons, without changes to the metric engine itself.

## 6. Input validation

The evaluator validates and raises `EvaluationError` (a `ValueError`
subclass) on:

- mismatched lengths between `y_true` and `y_pred`/`risk`
- empty input
- non-binary labels in `y_true`/`y_pred`
- an invalid threshold (outside `[0, 1]`)
- NaN/infinite risk values
- risk values outside `[0, 1]`

No input is silently truncated, coerced, or dropped.

## 7. Why temporal data must not be randomly shuffled

This is a time-series forecasting problem: the model predicts future
network state from past state. A random train/test split leaks future
behaviour into training and produces optimistic, unrealistic metrics. The
project protocol requires a **global timestamp cutoff** across all hosts
instead. This evaluation module does not perform any splitting itself —
it only scores predictions it is given — but its API and documentation are
written so as not to encourage random shuffling, and no
`sklearn.train_test_split`-style utility is included here.

## 8. Why preprocessing must be fitted on training data only

Scalers, encoders, and thresholds fitted on test data leak test-set
statistics into evaluation, inflating apparent performance. This module
never fits anything — it consumes already-prepared predictions and labels.
Any preprocessing artifacts (e.g. `artifacts/scaler.pkl`) must be fitted
upstream, on training data only.

## 9. How this will consume real model output

Today, `evaluate_predictions` / `evaluate_risk` are exercised with small,
deterministic mock arrays (see `tests/test_metrics.py`) and with
`mock_predictions.csv`-style data once available. When Sohini's world
model produces real predictions, the same functions are called with the
real `y_true`/`risk` arrays — the evaluator itself does not change.

## 10. Intentionally not implemented (belongs to later parts of Ankit's role)

- Logistic Regression / Random Forest / Persistence baselines
- Lead-time evaluation
- Early-warning-rate evaluation
- Unseen-attack generalization evaluation
- Ablation studies (flow-only vs packet-only vs fused; static vs temporal)
- Streamlit dashboard
- SHAP, MITRE mapping (owned by Srijani)
- Any feature extraction (owned by Shaurya/Aman) or model training (owned
  by Sohini)
