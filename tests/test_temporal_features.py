import numpy as np
import pytest

from src.baseline._validation import BaselineError
from src.baseline.logistic import LogisticRegressionBaseline
from src.baseline.random_forest import RandomForestBaseline
from src.baseline.temporal_features import (
    build_flattened_feature_names,
    flatten_temporal_history,
)


def test_flatten_preserves_order_oldest_to_newest():
    # 1 sample, 3 windows, 2 features each.
    # window t-2 = [1, 2], window t-1 = [3, 4], window t = [5, 6]
    history = np.array([[[1, 2], [3, 4], [5, 6]]])

    flattened = flatten_temporal_history(history)

    assert flattened.shape == (1, 6)
    assert list(flattened[0]) == [1, 2, 3, 4, 5, 6]


def test_flatten_is_deterministic():
    rng = np.random.default_rng(42)
    history = rng.normal(size=(5, 10, 4))

    flat_a = flatten_temporal_history(history)
    flat_b = flatten_temporal_history(history)

    assert np.array_equal(flat_a, flat_b)


def test_flatten_matches_manual_reshape_construction():
    rng = np.random.default_rng(0)
    n_samples, n_windows, n_features = 4, 3, 5
    history = rng.normal(size=(n_samples, n_windows, n_features))

    flattened = flatten_temporal_history(history)

    # Manually build the expected flattened row for sample 0.
    expected_row0 = np.concatenate([history[0, w, :] for w in range(n_windows)])
    assert np.allclose(flattened[0], expected_row0)


def test_flatten_shape_matches_windows_times_features():
    history = np.zeros((7, 10, 6))  # project's canonical 10-window history
    flattened = flatten_temporal_history(history)
    assert flattened.shape == (7, 60)


def test_flatten_rejects_non_3d_input():
    with pytest.raises(BaselineError):
        flatten_temporal_history(np.zeros((5, 4)))  # only 2D


def test_flatten_rejects_nan():
    history = np.zeros((2, 3, 2))
    history[0, 0, 0] = np.nan
    with pytest.raises(BaselineError):
        flatten_temporal_history(history)


def test_flatten_rejects_empty():
    with pytest.raises(BaselineError):
        flatten_temporal_history(np.zeros((0, 3, 2)))


# ----------------------------------------------------------------------
# Flattened feature names
# ----------------------------------------------------------------------
def test_flattened_feature_names_order_and_labels():
    names = build_flattened_feature_names(["syn_count", "bytes_total"], n_windows=3)

    assert names == [
        "t-2::syn_count",
        "t-2::bytes_total",
        "t-1::syn_count",
        "t-1::bytes_total",
        "t::syn_count",
        "t::bytes_total",
    ]


def test_flattened_feature_names_length_matches_flattened_array_width():
    feature_names = ["a", "b", "c"]
    n_windows = 4
    names = build_flattened_feature_names(feature_names, n_windows=n_windows)

    history = np.zeros((2, n_windows, len(feature_names)))
    flattened = flatten_temporal_history(history)

    assert len(names) == flattened.shape[1]


def test_flattened_feature_names_rejects_wrong_window_label_count():
    with pytest.raises(BaselineError):
        build_flattened_feature_names(["a"], n_windows=3, window_labels=["t-1", "t"])  # only 2 labels


# ----------------------------------------------------------------------
# Fairness integration: LR/RF can consume the same flattened history
# ----------------------------------------------------------------------
def test_lr_and_rf_can_fit_and_predict_on_flattened_history():
    rng = np.random.default_rng(42)
    n_samples, n_windows, n_features = 60, 10, 3
    history = rng.normal(size=(n_samples, n_windows, n_features))
    weights = rng.normal(size=n_windows * n_features)
    flattened = flatten_temporal_history(history)
    y = (flattened @ weights >= 0).astype(int)

    split = 40
    X_train, X_test = flattened[:split], flattened[split:]
    y_train, y_test = y[:split], y[split:]

    for baseline in (
        LogisticRegressionBaseline(random_state=42),
        RandomForestBaseline(random_state=42, n_estimators=30),
    ):
        baseline.fit(X_train, y_train)
        risk = baseline.predict_risk(X_test)
        assert risk.shape == (X_test.shape[0],)
        assert risk.shape == y_test.shape
        assert np.all((risk >= 0.0) & (risk <= 1.0))
