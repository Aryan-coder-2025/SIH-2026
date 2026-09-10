from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[Sequence[Sequence[float]], np.ndarray]
LabelLike = Union[Sequence[float], np.ndarray]


class BaselineError(ValueError):
    """Raised when baseline inputs are missing, malformed, or invalid."""


def to_feature_array(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Convert X (array-like or DataFrame) to a validated 2D float array."""
    try:
        array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"'{name}' must be a numeric 2D array-like.") from exc

    if array.ndim != 2:
        raise BaselineError(
            f"'{name}' must be 2-dimensional (n_samples, n_features); "
            f"got shape {array.shape}."
        )

    if array.shape[0] == 0:
        raise BaselineError(f"'{name}' is empty. At least one sample is required.")

    if not np.all(np.isfinite(array)):
        raise BaselineError(f"'{name}' contains NaN or infinite values.")

    return array


def to_label_array(y: LabelLike, name: str = "y") -> np.ndarray:
    """Convert y to a validated 1D binary (0/1) int array."""
    try:
        array = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"'{name}' must be a numeric 1D array-like.") from exc

    if array.ndim != 1:
        raise BaselineError(f"'{name}' must be a one-dimensional sequence.")

    if array.shape[0] == 0:
        raise BaselineError(f"'{name}' is empty. At least one sample is required.")

    if not np.all(np.isfinite(array)):
        raise BaselineError(f"'{name}' contains NaN or infinite values.")

    valid_mask = np.isin(array, [0.0, 1.0])
    if not np.all(valid_mask):
        invalid_values = np.unique(array[~valid_mask])
        raise BaselineError(
            f"'{name}' must contain only binary labels (0 or 1). "
            f"Found invalid value(s): {invalid_values.tolist()}."
        )

    return array.astype(int)


def validate_fit_inputs(X: ArrayLike, y: LabelLike) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert (X_train, y_train) for a baseline's fit()."""
    X_arr = to_feature_array(X, "X_train")
    y_arr = to_label_array(y, "y_train")

    if X_arr.shape[0] != y_arr.shape[0]:
        raise BaselineError(
            "X_train and y_train must have the same number of samples "
            f"(got {X_arr.shape[0]} and {y_arr.shape[0]})."
        )

    return X_arr, y_arr


def validate_predict_input(X: ArrayLike, expected_n_features: int | None = None) -> np.ndarray:
    """Validate and convert X for a baseline's predict()/predict_risk()."""
    X_arr = to_feature_array(X, "X")

    if expected_n_features is not None and X_arr.shape[1] != expected_n_features:
        raise BaselineError(
            "X has a different number of features than the data used to fit "
            f"this baseline (expected {expected_n_features}, got {X_arr.shape[1]})."
        )

    return X_arr


def validate_threshold(threshold: float) -> float:
    """Validate a risk-to-label threshold lies within [0, 1]."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise BaselineError("'threshold' must be a number.")

    if not np.isfinite(threshold):
        raise BaselineError("'threshold' must be a finite number.")

    if threshold < 0.0 or threshold > 1.0:
        raise BaselineError(f"'threshold' must be within [0, 1] (got {threshold}).")

    return float(threshold)
