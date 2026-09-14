import numpy as np
import pandas as pd
import pytest

from src.baseline._validation import BaselineError
from src.baseline.logistic import LogisticRegressionBaseline
from src.baseline.persistence import PersistenceBaseline
from src.baseline.random_forest import RandomForestBaseline
from src.eval.metrics import EvaluationError, evaluate_predictions, evaluate_risk


# ----------------------------------------------------------------------
# Deterministic synthetic dataset (no real/downloaded data)
# ----------------------------------------------------------------------
def _make_synthetic_dataset(n_samples: int = 200, n_features: int = 5, seed: int = 42):
    """
    Small, deterministic, linearly-separable-ish synthetic dataset.
    Standing in for a (feature_matrix, future_malicious_label) pair for
    one forecast horizon. Not derived from CSE-CIC-IDS2018 or any real
    capture; used only to exercise the baseline API.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    true_weights = np.array([2.0, -1.5, 1.0, 0.0, 0.5])
    logits = X @ true_weights
    probs = 1 / (1 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)

    # keep it a genuine temporal-style split: first 70% "earlier", last
    # 30% "later" -- no shuffling.
    split = int(n_samples * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# Persistence baseline
# ----------------------------------------------------------------------
def test_persistence_basic_behaviour():
    # last column holds the "current known risk"
    X = np.array(
        [
            [0.0, 0.0, 0.10],
            [0.0, 0.0, 0.90],
            [0.0, 0.0, 0.50],
        ]
    )
    baseline = PersistenceBaseline(current_state_column=-1, threshold=0.70)

    risk = baseline.predict_risk(X)
    assert np.allclose(risk, [0.10, 0.90, 0.50])

    labels = baseline.predict(X)
    assert list(labels) == [0, 1, 0]


def test_persistence_prediction_shape():
    X = np.random.default_rng(0).uniform(0, 1, size=(15, 4))
    baseline = PersistenceBaseline(current_state_column=-1)

    risk = baseline.predict_risk(X)
    labels = baseline.predict(X)

    assert risk.shape == (15,)
    assert labels.shape == (15,)


def test_persistence_does_not_pretend_to_train():
    """fit() must be a documented no-op: different y_train never changes predict()."""
    X = np.array([[0.2], [0.8], [0.5]])

    baseline_a = PersistenceBaseline(current_state_column=-1)
    baseline_a.fit(X, y_train=[0, 0, 0])

    baseline_b = PersistenceBaseline(current_state_column=-1)
    baseline_b.fit(X, y_train=[1, 1, 1])

    assert np.array_equal(baseline_a.predict_risk(X), baseline_b.predict_risk(X))
    assert np.array_equal(baseline_a.predict(X), baseline_b.predict(X))

    # fit() is optional entirely -- predicting without fitting first works.
    baseline_c = PersistenceBaseline(current_state_column=-1)
    assert np.array_equal(baseline_c.predict_risk(X), baseline_a.predict_risk(X))


def test_persistence_column_selection_by_name_with_dataframe():
    df = pd.DataFrame(
        {
            "packet_count": [10, 20, 30],
            "current_risk": [0.2, 0.9, 0.4],
        }
    )
    baseline = PersistenceBaseline(current_state_column="current_risk", threshold=0.5)

    risk = baseline.predict_risk(df)
    assert np.allclose(risk, [0.2, 0.9, 0.4])
    assert list(baseline.predict(df)) == [0, 1, 0]


def test_persistence_rejects_out_of_range_state_values():
    X = np.array([[1.5], [0.5]])
    baseline = PersistenceBaseline(current_state_column=-1)

    with pytest.raises(ValueError):
        baseline.predict_risk(X)


# ----------------------------------------------------------------------
# Logistic Regression baseline
# ----------------------------------------------------------------------
def test_logistic_fit_predict_shapes_and_ranges():
    X_train, X_test, y_train, y_test = _make_synthetic_dataset()

    baseline = LogisticRegressionBaseline(random_state=42)
    baseline.fit(X_train, y_train)

    risk = baseline.predict_risk(X_test)
    labels = baseline.predict(X_test)

    assert risk.shape == (X_test.shape[0],)
    assert labels.shape == (X_test.shape[0],)
    assert np.all((risk >= 0.0) & (risk <= 1.0))
    assert set(np.unique(labels)).issubset({0, 1})


def test_logistic_predict_matches_manual_thresholding():
    X_train, X_test, y_train, _ = _make_synthetic_dataset()

    baseline = LogisticRegressionBaseline(random_state=42, threshold=0.5)
    baseline.fit(X_train, y_train)

    risk = baseline.predict_risk(X_test)
    labels = baseline.predict(X_test)

    manual_labels = (risk >= 0.5).astype(int)
    assert np.array_equal(labels, manual_labels)


def test_logistic_deterministic_with_fixed_seed():
    X_train, X_test, y_train, _ = _make_synthetic_dataset()

    baseline_a = LogisticRegressionBaseline(random_state=42)
    baseline_a.fit(X_train, y_train)

    baseline_b = LogisticRegressionBaseline(random_state=42)
    baseline_b.fit(X_train, y_train)

    assert np.allclose(baseline_a.predict_risk(X_test), baseline_b.predict_risk(X_test))


def test_logistic_scaler_fit_only_on_training_data():
    """No leakage: the scaler's fitted mean must reflect X_train only."""
    X_train, X_test, y_train, _ = _make_synthetic_dataset()

    baseline = LogisticRegressionBaseline(random_state=42)
    baseline.fit(X_train, y_train)

    assert np.allclose(baseline._scaler.mean_, X_train.mean(axis=0))
    # calling predict_risk on test data must not perturb the fitted scaler
    baseline.predict_risk(X_test)
    assert np.allclose(baseline._scaler.mean_, X_train.mean(axis=0))


def test_logistic_predict_before_fit_raises():
    baseline = LogisticRegressionBaseline()
    with pytest.raises(BaselineError):
        baseline.predict_risk([[0.1, 0.2]])


def test_logistic_single_class_training_data_raises():
    X_train = np.array([[0.1], [0.2], [0.3]])
    y_train = [0, 0, 0]

    baseline = LogisticRegressionBaseline()
    with pytest.raises(BaselineError):
        baseline.fit(X_train, y_train)


def test_logistic_mismatched_lengths_raise():
    X_train = np.array([[0.1], [0.2], [0.3]])
    y_train = [0, 1]

    baseline = LogisticRegressionBaseline()
    with pytest.raises(BaselineError):
        baseline.fit(X_train, y_train)


# ----------------------------------------------------------------------
# Random Forest baseline
# ----------------------------------------------------------------------
def test_random_forest_fit_predict_shapes_and_ranges():
    X_train, X_test, y_train, y_test = _make_synthetic_dataset()

    baseline = RandomForestBaseline(random_state=42, n_estimators=50)
    baseline.fit(X_train, y_train)

    risk = baseline.predict_risk(X_test)
    labels = baseline.predict(X_test)

    assert risk.shape == (X_test.shape[0],)
    assert labels.shape == (X_test.shape[0],)
    assert np.all((risk >= 0.0) & (risk <= 1.0))
    assert set(np.unique(labels)).issubset({0, 1})


def test_random_forest_deterministic_with_fixed_seed():
    X_train, X_test, y_train, _ = _make_synthetic_dataset()

    baseline_a = RandomForestBaseline(random_state=42, n_estimators=50)
    baseline_a.fit(X_train, y_train)

    baseline_b = RandomForestBaseline(random_state=42, n_estimators=50)
    baseline_b.fit(X_train, y_train)

    assert np.array_equal(baseline_a.predict_risk(X_test), baseline_b.predict_risk(X_test))


def test_random_forest_predict_before_fit_raises():
    baseline = RandomForestBaseline()
    with pytest.raises(BaselineError):
        baseline.predict_risk([[0.1, 0.2]])


def test_random_forest_single_class_training_data_raises():
    X_train = np.array([[0.1], [0.2], [0.3]])
    y_train = [1, 1, 1]

    baseline = RandomForestBaseline()
    with pytest.raises(BaselineError):
        baseline.fit(X_train, y_train)


def test_random_forest_feature_count_mismatch_at_predict_raises():
    X_train, _, y_train, _ = _make_synthetic_dataset(n_features=5)

    baseline = RandomForestBaseline(random_state=42)
    baseline.fit(X_train, y_train)

    wrong_shape_X = np.zeros((3, 3))
    with pytest.raises(BaselineError):
        baseline.predict_risk(wrong_shape_X)


# ----------------------------------------------------------------------
# Shared validation: empty / mismatched / invalid input (all baselines)
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "baseline_factory",
    [
        lambda: PersistenceBaseline(current_state_column=-1),
        lambda: LogisticRegressionBaseline(),
        lambda: RandomForestBaseline(),
    ],
)
def test_fit_rejects_empty_input(baseline_factory):
    baseline = baseline_factory()
    with pytest.raises(BaselineError):
        baseline.fit(np.empty((0, 3)), [])


@pytest.mark.parametrize(
    "baseline_factory",
    [
        lambda: PersistenceBaseline(current_state_column=-1),
        lambda: LogisticRegressionBaseline(),
        lambda: RandomForestBaseline(),
    ],
)
def test_fit_rejects_non_binary_labels(baseline_factory):
    X = np.array([[0.1], [0.2], [0.3]])
    y = [0, 1, 2]  # invalid: not binary

    baseline = baseline_factory()
    with pytest.raises(BaselineError):
        baseline.fit(X, y)


@pytest.mark.parametrize(
    "baseline_factory",
    [
        lambda: PersistenceBaseline(current_state_column=-1),
        lambda: LogisticRegressionBaseline(),
        lambda: RandomForestBaseline(),
    ],
)
def test_predict_rejects_nan_input(baseline_factory):
    baseline = baseline_factory()
    X_bad = np.array([[np.nan, 0.2]])
    with pytest.raises(BaselineError):
        baseline.predict_risk(X_bad)


# ----------------------------------------------------------------------
# Temporal ordering preserved
# ----------------------------------------------------------------------
def test_persistence_preserves_row_order():
    # Distinct, order-sensitive values -- output must align index-for-index.
    X = np.array([[0.9], [0.1], [0.5], [0.2]])
    baseline = PersistenceBaseline(current_state_column=-1)

    risk = baseline.predict_risk(X)
    assert list(risk) == [0.9, 0.1, 0.5, 0.2]


def test_random_forest_preserves_row_order():
    X_train, _, y_train, _ = _make_synthetic_dataset()
    baseline = RandomForestBaseline(random_state=42, n_estimators=50)
    baseline.fit(X_train, y_train)

    X_probe = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0],  # strongly positive-weighted -> high risk
            [-5.0, 5.0, -5.0, -5.0, -5.0],  # strongly negative-weighted -> low risk
        ]
    )
    risk = baseline.predict_risk(X_probe)
    # row 0 (favourable weights) must score higher than row 1, in that order
    assert risk[0] > risk[1]


# ----------------------------------------------------------------------
# Integration: baseline output -> src/eval/metrics.py
# ----------------------------------------------------------------------
def test_persistence_output_is_consumable_by_evaluate_risk():
    y_true = [0, 1, 0, 1]
    current_risk_column = np.array([0.10, 0.85, 0.20, 0.90]).reshape(-1, 1)

    baseline = PersistenceBaseline(current_state_column=-1, threshold=0.70)
    risk = baseline.predict_risk(current_risk_column)

    result = evaluate_risk(y_true, risk, threshold=0.70)

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.confusion_matrix.tp == 2
    assert result.confusion_matrix.tn == 2


def test_logistic_and_random_forest_outputs_are_consumable_by_evaluators():
    X_train, X_test, y_train, y_test = _make_synthetic_dataset()

    for baseline in (
        LogisticRegressionBaseline(random_state=42),
        RandomForestBaseline(random_state=42, n_estimators=50),
    ):
        baseline.fit(X_train, y_train)

        risk = baseline.predict_risk(X_test)
        labels = baseline.predict(X_test)

        # Mode 2: risk scores straight into evaluate_risk
        risk_result = evaluate_risk(y_test, risk, threshold=baseline.threshold)
        assert 0.0 <= risk_result.precision <= 1.0
        assert 0.0 <= risk_result.recall <= 1.0
        assert 0.0 <= risk_result.f1 <= 1.0
        assert 0.0 <= risk_result.fpr <= 1.0

        # Mode 1: already-binarized predictions straight into evaluate_predictions
        label_result = evaluate_predictions(y_test, labels)
        assert label_result.confusion_matrix.tp + label_result.confusion_matrix.fn == int(
            np.sum(y_test == 1)
        )


def test_full_pipeline_features_to_baseline_to_evaluation():
    """
    input features -> baseline -> predicted label/risk -> src/eval/metrics.py
    -> Precision/Recall/F1/FPR

    Demonstrates the project's "golden rule": Ankit's evaluator consumes
    a baseline's output with no manual schema changes.
    """
    X_train, X_test, y_train, y_test = _make_synthetic_dataset(n_samples=300)

    baseline = RandomForestBaseline(random_state=42, n_estimators=100)
    baseline.fit(X_train, y_train)

    risk = baseline.predict_risk(X_test)
    result = evaluate_risk(y_test, risk, threshold=0.5)

    # This synthetic dataset is close to linearly separable, so a
    # reasonably-fit Random Forest should score well above chance --
    # this is a sanity bound on a test fixture, NOT a claimed
    # experimental benchmark result on real network data.
    assert result.f1 > 0.7
    result_dict = result.to_dict()
    assert set(result_dict.keys()) == {
        "precision",
        "recall",
        "f1",
        "fpr",
        "confusion_matrix",
    }


def test_baseline_predictions_reject_bad_inputs_via_metrics_layer():
    """A baseline producing mismatched-length output should still surface
    as a clear EvaluationError from the metrics layer, not a silent bug."""
    y_true = [0, 1, 0]
    risk = [0.9, 0.1]  # wrong length, as if a baseline mis-batched predict()

    with pytest.raises(EvaluationError):
        evaluate_risk(y_true, risk, threshold=0.5)
