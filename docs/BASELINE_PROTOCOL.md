# Baseline Protocol (Part B)

Owner: Ankit (Evaluation + Baselines + Dashboard/QA)

Status: **Part B — baseline framework, now integrated with the
forecast-aware evaluation layer** (`docs/07_EVALUATION_PROTOCOL.md` §11).
Lead time, early warning rate, unseen-attack evaluation, and ablation
studies are now implemented as part of that layer — see §11.8-§11.12
there for what they do and how they relate to these baselines. The
Streamlit dashboard is still not implemented.

## 1. Purpose

Simple, defensible baselines that the future LSTM/world model must beat
to justify its added complexity, per the project's own evaluation
philosophy: *"if our model cannot beat 'same as now,' our temporal
intelligence isn't doing much."* Three baselines are provided:

- **Persistence** — no learning at all; forecasts that the current risk
  continues unchanged.
- **Logistic Regression** — simple linear classifier.
- **Random Forest** — non-linear ensemble, still far simpler than the
  future world model.

All three share one interface (`fit(X_train, y_train)`, `predict(X)`,
`predict_risk(X)`) and produce output that plugs directly into
`src/eval/metrics.py` (`evaluate_predictions` / `evaluate_risk`) with no
schema translation, per the project's "no component is done until
another member can consume its output unmodified" rule.

## 2. Persistence definition

Persistence means: *whatever the risk/label is right now is forecast to
still be true at t+10s / t+20s / t+30s.* It has no learnable parameters.
`PersistenceBaseline.fit()` is a documented no-op (it only validates
inputs for interface parity — it never uses `y_train`). This is stated
explicitly in code and here so it is never mistaken for, or described as,
a trained model.

`PersistenceBaseline` reads one designated column of `X` — the
most-recently-known risk/label for that row's entity — via
`current_state_column` (position or, for a pandas DataFrame, a column
name), and returns it unchanged as the forecast risk.

## 3. Logistic Regression / Random Forest

Both follow the same `fit` / `predict` / `predict_risk` interface:

```python
baseline = LogisticRegressionBaseline(random_state=42, threshold=0.70)
baseline.fit(X_train, y_train)
risk = baseline.predict_risk(X_test)     # continuous, in [0, 1]
labels = baseline.predict(X_test)        # binary, via risk_to_label()
```

- **Logistic Regression** standardizes features with `StandardScaler`,
  fit only on `X_train` inside `fit()`; `predict`/`predict_risk` only
  ever call `.transform()`, never `.fit()`/`.fit_transform()`, so no
  test-set statistics leak into the fitted scaler.
- **Random Forest** does not require scaling and uses `X` as given.
- Both default `random_state=42` (the project's seed convention) and
  expose the usual scikit-learn hyperparameters as constructor arguments
  (`C`/`max_iter` for Logistic Regression; `n_estimators`, `max_depth`,
  `min_samples_leaf` for Random Forest) rather than hard-coding them.
- The `threshold=0.70` constructor default is a starting point for
  ad-hoc use, not a calibrated value. For a real run, select it on
  validation data via `src.eval.threshold.select_threshold` (see
  `docs/07_EVALUATION_PROTOCOL.md` §11.5) and pass the frozen result to
  `predict(X, threshold=selected_threshold)`.
- Binary conversion reuses `src/eval/metrics.risk_to_label`, so a
  baseline's own `.predict()` and a caller manually thresholding
  `.predict_risk()` output always agree.

## 4. Expected input / output

- `X_train`, `X_test`: 2D numeric array-like or pandas DataFrame, shape
  `(n_samples, n_features)`. Finite values only (no NaN/inf).
- `y_train`: 1D binary (0/1) array-like, one label per row of `X_train`,
  for **one specific forecast horizon** (+10s, or +20s, or +30s). Each
  baseline instance is fit and evaluated per horizon — call `fit`/
  `predict` again with that horizon's own `y` to score a different
  horizon. This mirrors how `src/eval/metrics.py` is itself
  horizon-agnostic (Part A doc, §5): the metric engine and these
  baselines both work on flat per-sample arrays, one horizon at a time.
- `predict_risk(X)` → 1D float array, shape `(n_samples,)`, values in
  `[0, 1]`.
- `predict(X)` → 1D int array, shape `(n_samples,)`, values in `{0, 1}`.

The real forecasting feature-matrix schema (columns produced by
Shaurya/Aman's flow+packet fusion) is not yet frozen elsewhere in the
repository. These baselines only require a plain numeric 2D `X` and 1D
binary `y`, so no change is needed here once the real schema lands —
only the arrays passed in change.

## 5. Temporal split expectations

No baseline in this package performs a train/test split itself. That
split logic now lives in one place —
`src.eval.temporal_split.temporal_train_validation_test_split` (see
`docs/07_EVALUATION_PROTOCOL.md` §11.4) — which splits by a **global
timestamp cutoff** across every host into TRAIN / VALIDATION / FINAL
TEST, never `shuffle=True` and never a per-host split. Callers build
`X_train`/`y_train`, `X_val`/`y_val`, `X_test`/`y_test` via that utility
(or an equivalent already-split source) and pass them to a baseline's
`fit`/`predict` as usual.

## 6. Preprocessing expectations

- Logistic Regression's `StandardScaler` is fit exclusively on
  `X_train`, inside `fit()`. It is never re-fit on `X_test`/`X_val`.
- Random Forest performs no preprocessing.
- Neither baseline reads, splits, or otherwise touches `data/raw`.

## 7. Seed / reproducibility

Both learned baselines default `random_state=42`. Fitting either one
twice on identical `(X_train, y_train)` produces identical
`predict`/`predict_risk` output (covered by
`test_logistic_deterministic_with_fixed_seed` and
`test_random_forest_deterministic_with_fixed_seed` in
`tests/test_baselines.py`).

## 8. Relationship to evaluation metrics

Every baseline's `predict_risk()` output can be passed directly to
`evaluate_risk(y_true, risk, threshold=...)`, and every `predict()`
output can be passed directly to `evaluate_predictions(y_true, y_pred)` —
both from `src/eval/metrics.py`, unchanged from Part A. This is exercised
end-to-end in `tests/test_baselines.py::test_full_pipeline_features_to_baseline_to_evaluation`
(features → baseline → prediction → metrics → Precision/Recall/F1/FPR).

## 9. Artifacts

No trained model artifacts (`baseline_lr.pkl`, `baseline_rf.pkl`) are
committed by this work. The project's artifact contract expects those
filenames once real training data is available; producing them now, from
synthetic test data, would misrepresent them as real results. When the
real feature matrix and temporal split exist, the intended flow is:

```python
baseline = LogisticRegressionBaseline(random_state=42)
baseline.fit(X_train, y_train)
# then, upstream: joblib.dump(baseline, "artifacts/baseline_lr.pkl")
```

(and equivalently for `RandomForestBaseline` → `artifacts/baseline_rf.pkl`).
That serialization step is intentionally not implemented here.

## 10. What tests use

All tests in `tests/test_baselines.py` use small, deterministic synthetic
data generated with `numpy.random.default_rng(seed=42)` — not
CSE-CIC-IDS2018 or any other real dataset. Any metric values referenced
in those tests (e.g. an F1 sanity bound) describe test-fixture behaviour
only and are **not** experimental results on real network traffic; no
benchmark numbers are claimed here.

## 11. Feature-history fairness for LR/RF (implemented)

The future LSTM/world model sees the full 10-window history per
prediction. Comparing it against LR/RF fit only on the current window
would give the baselines a strictly smaller information budget — an
unfair comparison. `src/baseline/temporal_features.py::flatten_temporal_history`
takes a `(n_samples, n_windows, n_features)` history array (oldest window
first) and deterministically reshapes it into
`(n_samples, n_windows * n_features)` — `[features_t-9, ..., features_t]`
— which `LogisticRegressionBaseline`/`RandomForestBaseline` then consume
exactly like any other `X`. `build_flattened_feature_names` produces the
matching column names. See `docs/07_EVALUATION_PROTOCOL.md` §11.14 for
the full rationale and `tests/test_temporal_features.py` for the
determinism/ordering tests.

## 12. Feature schema (implemented)

Both `LogisticRegressionBaseline` and `RandomForestBaseline` accept an
optional `feature_schema` constructor argument
(`src.eval.feature_schema.FeatureSchema`, default `None`, fully backward
compatible). When set, a pandas DataFrame passed to `fit()` or
`predict()`/`predict_risk()` with the wrong column names or order raises
`BaselineError` — the schema-mismatch failure mode described in the
project plan (training on `[bytes, packets, ttl, syn]`, predicting on
`[packets, bytes, syn, ttl]`) fails loudly instead of silently producing
garbage. See `docs/07_EVALUATION_PROTOCOL.md` §11.13 and
`tests/test_feature_schema.py`.

## 13. What tests use — implemented, no dataset dependency

Confirmed: `tests/test_temporal_features.py` and `tests/test_feature_schema.py`
follow the same synthetic-data convention as §10 — deterministic
`numpy.random.default_rng(seed=42)` fixtures, no real dataset, no claimed
benchmark numbers.

## 14. Not yet implemented

- Real feature-matrix integration (depends on Shaurya/Aman's flow+packet
  fusion output and the frozen data/model contracts)
- Serialized baseline artifacts (`artifacts/baseline_lr.pkl`,
  `artifacts/baseline_rf.pkl`)
- Streamlit dashboard
- Probability calibration for `predict_risk()` output — see
  `docs/07_EVALUATION_PROTOCOL.md` §11.11; risk scores are explicitly
  not claimed to be calibrated probabilities
