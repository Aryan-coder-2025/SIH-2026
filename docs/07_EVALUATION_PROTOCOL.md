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

## 10. Status update

Sections 1-9 above describe the original core metric engine
(`src/eval/metrics.py`) and remain accurate as written. The items that
were listed as "intentionally not implemented" in the original version of
this document — baselines, lead time, early warning rate, unseen-attack
evaluation, and ablation studies — have since been implemented as a
forecast-aware evaluation layer built on top of that same metric engine
(`precision`/`recall`/`f1_score`/`false_positive_rate`/`confusion_matrix`
in `src/eval/metrics.py` are unchanged; two additive secondary metrics,
`accuracy` and `specificity`, were added — see §11.7). See §11 below for
what is now implemented, and §12 for what still is not.

## 11. Forecast-aware evaluation layer

This section documents everything built on top of the core metric engine
to address forecast-specific evaluation — timestamps, horizons, host
identity, temporal splitting, threshold selection, lead time, early
warning, unseen-attack generalization, ablations, and reproducibility.

### 11.1 Prediction record contract (`src/eval/records.py`)

The canonical evaluation sample is a `PredictionRecord`:

```python
PredictionRecord(
    source_host: str,      # canonical forecasting entity
    prediction_time: float,  # when the forecast was issued
    forecast_horizon: int,   # seconds ahead: one of SUPPORTED_HORIZONS (10, 20, 30)
    target_time: float,      # must equal prediction_time + forecast_horizon
    y_true: int,              # binary ground-truth label at target_time
    risk: float,               # predicted risk/probability in [0, 1]
)
```

`validate_prediction_records(records, expected_keys=None)` rejects (with
every problem listed, not just the first):
- a non-string/empty `source_host`
- non-finite `prediction_time`/`target_time`
- `forecast_horizon` outside `{10, 20, 30}`
- `target_time != prediction_time + forecast_horizon` (float-tolerant)
- non-binary `y_true`, or `risk` outside `[0, 1]` / non-finite
- **duplicate** `(source_host, prediction_time, forecast_horizon)` keys
  — the unique key for one prediction. Duplicates are reported, never
  silently deduplicated (`find_duplicate_keys`).
- **missing** predictions, when `expected_keys` (the full set of
  host/time/horizon combinations that should have a prediction) is
  supplied — missing entries are reported, never silently dropped
  (`find_missing_prediction_keys`).

### 11.2 Timestamp/host-aware alignment (`src/eval/forecast.py`)

Predictions and ground truth almost never arrive pre-joined. `RawPrediction`
(`source_host`, `prediction_time`, `forecast_horizon`) and
`GroundTruthLabel` (`source_host`, `target_time`, `y_true`) are joined by
`align_predictions_with_ground_truth()` using the
**`(source_host, target_time)` key — never row position**. A prediction
for host A is matched only against ground truth also keyed on host A;
this makes host isolation a property of the join key itself, not an
assumption. Ambiguous ground truth (two labels for the same key) and
unmatched predictions both raise `AlignmentError` rather than silently
guessing or dropping rows.

### 11.3 Per-horizon evaluation (`evaluate_forecast`)

`evaluate_forecast(records, threshold=0.70)` groups already-validated,
already-aligned records by `forecast_horizon` and computes a full
`EvaluationResult` (via `evaluate_risk`, so it reuses the same metric
engine and zero-division policy as §4 above) **separately for each
horizon** — a +10s record is never averaged into the +20s confusion
matrix. `threshold` may be a single float or a `{horizon: threshold}`
dict (e.g. horizon-specific thresholds selected via §11.5). The result
also exposes a simple unweighted-mean `aggregate` across horizons, which
is explicitly documented (in code and here) as a convenience summary,
**not** a recomputed pooled confusion matrix — different horizons can
have different sample counts.

### 11.4 Temporal train/validation/test split (`src/eval/temporal_split.py`)

`temporal_train_validation_test_split(timestamps, train_end, validation_end)`
splits by a **single global timestamp cutoff** applied across every host
— never a fraction, never per-host, never `shuffle=True`:

```
TRAIN:      timestamp <  train_end
VALIDATION: train_end <= timestamp < validation_end
FINAL TEST: timestamp >= validation_end
```

Returns index arrays (`apply_split` indexes parallel `X`/`y` arrays by
them); row order within each split is preserved exactly as given, since
the function only filters by value. `assert_no_temporal_leakage` is a
QA/test helper that verifies, from the actual timestamp values, that
train < validation < test with zero overlap.

### 11.5 Threshold selection (`src/eval/threshold.py`)

`select_threshold(y_val, risk_val, metric="f1")` scans a deterministic
threshold grid and returns the one maximizing the chosen metric —
**evaluated only on the data passed in**. This function has no way to
enforce that its caller passes validation (not test) data; that
discipline is procedural, and is why the intended flow is documented
explicitly:

```
train model -> validation predictions -> select_threshold(...)
    -> freeze the returned threshold -> evaluate ONCE on the final test set
```

`0.70` (`DEFAULT_EXPERIMENTAL_THRESHOLD`) remains only the module's
default starting point for ad-hoc evaluation, not a claim of a
calibrated value — see §11.11 for the risk-score-vs-probability
distinction this default depends on.

### 11.6 Persistence baseline — forecast-horizon behavior

Unchanged from Part B (`docs/BASELINE_PROTOCOL.md` §2): `PersistenceBaseline`
persists a designated "current known risk/label" column forward as the
forecast for whichever horizon it is called for. It naturally supports
+10s/+20s/+30s because each call is scoped to one horizon already (see
§11.3); no horizon-specific code exists inside `PersistenceBaseline`
itself, by design — its behavior does not depend on which horizon it is
being used for; the horizon lives in the `PredictionRecord`/experiment
metadata wrapping it, not in the baseline.

### 11.7 Accuracy and specificity (`src/eval/metrics.py`)

Two secondary metrics were added to the core engine, alongside
Precision/Recall/F1/FPR (unchanged):

- `accuracy(cm)` = (TP+TN)/(TP+TN+FP+FN) — **secondary only**; class
  imbalance can make it look high even when Recall/F1/FPR reveal a model
  that misses most attacks. Never report accuracy alone as evidence of
  success.
- `specificity(cm)` = TN/(TN+FP) = 1 − FPR — provided for readability
  alongside FPR; same zero-division policy (`0.0` if `TN+FP == 0`).

`EvaluationResult`/`.to_dict()` now include `accuracy` and `specificity`
alongside the original four fields. This is an **additive, documented**
change to the public result shape (two existing test assertions that
enumerated the exact key set were updated accordingly); every other
function signature and the confusion-matrix ordering are unchanged.

### 11.8 Lead time (`src/eval/lead_time.py`, `src/eval/episodes.py`)

`group_into_episodes(observations, max_gap_seconds=10.0)` turns raw
per-window `(source_host, time, y_true)` ground truth into
`AttackEpisode(source_host, onset_time, offset_time)` objects — merging
consecutive malicious windows for the same host (gap ≤ one project
window, 10s, by default) into **one** episode, so one ongoing attack is
never double-counted as several.

`compute_lead_time(episode, predictions, threshold=0.70)`:
a prediction is a **valid warning** for an episode if it is for the same
host, its `prediction_time <= episode.onset_time`, and its
`risk >= threshold`. Lead time is
`episode.onset_time - earliest_valid_warning.prediction_time`, in
seconds; `None` if never warned about (never silently coerced to 0).
`summarize_lead_times` aggregates mean/median across the episodes that
were warned about (episodes never warned about are excluded from the
average, not averaged in as zero).

### 11.9 Early Warning Rate (`src/eval/early_warning.py`)

`early_warning_rate(episodes, predictions, threshold=0.70)` = (episodes
with a valid warning) / (total episodes), reusing the exact same
episode/warning definitions as §11.8 (it calls `summarize_lead_times`
rather than reimplementing the rule, so the two metrics cannot silently
diverge). `0.0` if there are zero episodes (documented zero-episode
policy). False warnings (threshold crossed with no corresponding attack)
do **not** affect this metric — that is an FPR concern, covered
separately by `false_positive_rate`.

### 11.10 Unseen-attack evaluation (`src/eval/unseen_attack.py`)

`unseen_attack_split(attack_category, held_out_category)` splits strictly
by **attack category label**, never by random row sampling: train = every
row except the held-out category (including benign and every other
attack); held-out test = only that category. `assert_no_category_leakage`
verifies the held-out category never appears in the train mask.

**Scientific framing (enforced in the docstring, not just prose here):**
a model scoring well on the held-out set demonstrates *"future-risk
detection/generalization on an attack category not seen during
training,"* not *"unseen attack classification"* — the model was never
asked to name the category.

### 11.11 Risk score vs. calibrated probability

A `risk` value in `[0, 1]` is called a **risk score** throughout this
codebase, not a "probability," unless a calibration step has actually
been run and validated (e.g. a reliability/calibration curve showing
predicted risk tracks observed frequency). None of Part A, B, or this
layer perform or claim calibration. Describing an uncalibrated
`risk=0.8` as "80% probability" is a claim this codebase does not
support — do not make it in reports generated from these results.

### 11.12 Ablation framework (`src/eval/ablation.py`)

`AblationConfig` is a validated, JSON-serializable description of one
comparison (`feature_group`, `temporal_mode`, `history_length`,
`model_name`, `forecast_horizon`) — it configures an experiment, it does
not run one. `standard_ablation_grid(model_name, forecast_horizon)`
builds the project's five required comparisons: flow-only, packet-only,
flow+packet (all temporal), plus static-vs-temporal (holding
feature_group=flow+packet fixed so only the temporal dimension varies).
This lets the team agree on the comparison grid before the LSTM/world
model exists, and lets it plug into the same structure later by adding
`model_name="lstm_world_model"` configs — no other code changes needed.

### 11.13 Feature schema awareness (`src/eval/feature_schema.py`)

`FeatureSchema(feature_names, version)` is an ordered feature manifest.
`validate_array` checks column *count*; `validate_dataframe_columns`
checks column *names and order* exactly, catching the exact failure mode
the project plan warns about (training with `[bytes, packets, ttl, syn]`,
predicting with `[packets, bytes, syn, ttl]`). `LogisticRegressionBaseline`
and `RandomForestBaseline` accept an optional `feature_schema` constructor
argument (default `None`, fully backward compatible); when set, a
DataFrame with mismatched columns raises `BaselineError` at both `fit()`
and `predict()`/`predict_risk()` (schema errors are wrapped into
`BaselineError` at the baseline layer, so callers only ever handle one
exception type from the baseline API).

### 11.14 Fair temporal history for LR/RF (`src/baseline/temporal_features.py`)

The future LSTM/world model sees the full 10-window history; comparing it
against LR/RF fit only on the current window would be an unfair baseline.
`flatten_temporal_history(history)` takes a
`(n_samples, n_windows, n_features)` array — window 0 = oldest (t-9),
last window = most recent (t) — and reshapes it (a pure, deterministic
reshape; never reorders) into `(n_samples, n_windows * n_features)`:
`[features_t-9, features_t-8, ..., features_t]`, ready for
`LogisticRegressionBaseline`/`RandomForestBaseline`. This gives LR/RF
access to the *same underlying information* as the LSTM will have (just
not its temporal structure) — the correct comparison is then "does
modeling the sequence help, beyond just having the same raw history?"
`build_flattened_feature_names` produces matching column names for
documentation/`FeatureSchema` use.

### 11.15 Reproducible experiment artifact (`src/eval/experiment.py`)

`ExperimentArtifact` is a validated, JSON-serializable
(`to_dict()`/`from_dict()`) record of one run: experiment/model name,
dataset version, train/validation/test periods (validated temporally
ordered and non-overlapping), forecast horizon, feature set + schema
version, history length, random seed, hyperparameters, frozen threshold,
final metrics, and a creation timestamp. This is the standard shape other
components (dashboard, Aryan's integration, QA) can consume without a
bespoke per-experiment schema, per the project's rule that "no component
is complete until another member can consume its output without manual
schema edits."

## 12. Intentionally not implemented

- Real feature-matrix integration (depends on Shaurya/Aman's flow+packet
  fusion output and the frozen data/model contracts)
- Actually training or running any model (persistence/LR/RF are
  implemented as reusable APIs — §11 and `docs/BASELINE_PROTOCOL.md` —
  but this codebase does not itself produce or claim a trained result
  on real network data)
- Serialized baseline artifacts (`artifacts/baseline_lr.pkl`,
  `artifacts/baseline_rf.pkl`)
- Probability calibration (see §11.11 — risk scores are explicitly not
  claimed to be calibrated probabilities)
- Streamlit dashboard, SHAP, MITRE mapping (owned by Srijani), any
  feature extraction (owned by Shaurya/Aman), model training (owned by
  Sohini)
