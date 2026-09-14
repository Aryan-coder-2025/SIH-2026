from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from src.baseline._validation import BaselineError

ArrayLike = Union[Sequence, np.ndarray]


def flatten_temporal_history(history: ArrayLike, name: str = "history") -> np.ndarray:
    """
    Flatten a (n_samples, n_windows, n_features) temporal-history array
    into the (n_samples, n_windows * n_features) representation
    Logistic Regression and Random Forest consume.

    FAIRNESS RATIONALE
    -------------------
    The future LSTM/world model sees the full history
    (10 windows x features). If Logistic Regression / Random Forest were
    only given the current window, they would be compared against the
    LSTM on a fundamentally smaller information budget -- an unfair
    baseline comparison. Flattening the same history window-by-window
    into one long feature vector gives LR/RF access to the identical
    underlying information (just not its temporal structure), which is
    the correct baseline comparison: "does modeling the *sequence*
    help, beyond just having access to the same raw history?"

    ORDERING (must match `build_flattened_feature_names` exactly)
    ----------------------------------------------------------------
    `history[:, 0, :]` is assumed to be the OLDEST window (t-9) and
    `history[:, -1, :]` the MOST RECENT window (t). The output is the
    row-major flattening of that array, i.e. for one sample:

        [features_t-9, features_t-8, ..., features_t]

    (each block is that window's full feature vector, in its original
    per-window order). This is a pure reshape -- deterministic, and it
    never reorders windows or features relative to the input.

    Raises BaselineError if `history` is not 3-dimensional or contains
    non-finite values.
    """
    array = np.asarray(history, dtype=float)

    if array.ndim != 3:
        raise BaselineError(
            f"'{name}' must be 3-dimensional (n_samples, n_windows, n_features); "
            f"got shape {array.shape}."
        )

    if array.shape[0] == 0:
        raise BaselineError(f"'{name}' is empty. At least one sample is required.")

    if not np.all(np.isfinite(array)):
        raise BaselineError(f"'{name}' contains NaN or infinite values.")

    n_samples, n_windows, n_features = array.shape
    return array.reshape(n_samples, n_windows * n_features)


def build_flattened_feature_names(
    feature_names: Sequence[str],
    n_windows: int,
    window_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Build the flattened column names matching `flatten_temporal_history`'s
    output order, for documentation / a FeatureSchema
    (src/eval/feature_schema.py).

    `window_labels` defaults to ["t-(n_windows-1)", ..., "t-1", "t"],
    oldest first -- matching the ordering assumption in
    `flatten_temporal_history`. Output format: "{window_label}::{feature_name}".
    """
    if window_labels is None:
        window_labels = [f"t-{n_windows - 1 - i}" if i < n_windows - 1 else "t" for i in range(n_windows)]

    if len(window_labels) != n_windows:
        raise BaselineError(
            f"'window_labels' has {len(window_labels)} entries but n_windows={n_windows}."
        )

    return [f"{window_label}::{feature_name}" for window_label in window_labels for feature_name in feature_names]
