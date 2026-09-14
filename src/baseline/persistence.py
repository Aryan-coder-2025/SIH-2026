from __future__ import annotations

from typing import Optional, Union

import numpy as np

from src.baseline._validation import (
    ArrayLike,
    LabelLike,
    validate_fit_inputs,
    validate_predict_input,
    validate_threshold,
)
from src.eval.metrics import risk_to_label


class PersistenceBaseline:
    """
    Persistence forecasting baseline.

    DEFINITION
    ----------
    "Persistence" means: whatever the network's malicious/benign risk is
    *right now* is forecast to still be true at every future horizon
    (+10s, +20s, +30s, ...). It answers "what if nothing about the
    forecast changes from the current state?" — the minimum bar any
    genuinely temporal model (baselines or the future LSTM/world model)
    must beat to justify its added complexity.

    This is NOT a trained machine-learning model. It has no learnable
    parameters, no loss function, and no optimization step. `fit()` is
    provided only so this class matches the shared baseline interface
    (`fit` / `predict` / `predict_risk`) used by the Logistic Regression
    and Random Forest baselines and is a documented no-op: it validates
    its inputs (so a bad call fails fast, the same as for the learned
    baselines) but does not use `y_train` to compute or store anything.
    Calling `fit()` before `predict()` is optional for this baseline.

    EXPECTED INPUT
    ---------------
    `X` is a 2D array/DataFrame of shape (n_samples, n_features) where
    one designated column holds the most-recently-known risk or binary
    label for that row's entity (source host) at forecast time t — i.e.
    the value being "persisted forward" to t+10/t+20/t+30. Which column
    that is is controlled by `current_state_column` (an integer position,
    or a column name when `X` is a pandas DataFrame). This baseline does
    not interpret any other column.

    OUTPUT
    ------
    `predict_risk(X)` returns that column's values directly, as a 1D risk
    array in [0, 1] — one value per row of `X`, for a single forecast
    horizon per call (call again with the appropriate `current_state`
    column for a different horizon; the metric engine in
    `src/eval/metrics.py` is horizon-agnostic and works the same way for
    every horizon).
    `predict(X)` thresholds that risk into a binary label using the same
    `risk_to_label` helper the evaluator itself uses, so behaviour stays
    identical between this baseline and any other prediction source.
    """

    def __init__(
        self,
        current_state_column: Union[int, str] = -1,
        threshold: float = 0.70,
    ) -> None:
        self.current_state_column = current_state_column
        self.threshold = validate_threshold(threshold)
        self._fitted = False

    def fit(self, X_train: ArrayLike, y_train: LabelLike) -> "PersistenceBaseline":
        """
        No-op by design (see class docstring). Validates inputs for
        interface parity with the learned baselines, then returns self.
        """
        validate_fit_inputs(X_train, y_train)
        self._fitted = True
        return self

    def _resolve_column(self, X: ArrayLike, X_arr: np.ndarray) -> int:
        if isinstance(self.current_state_column, str):
            try:
                import pandas as pd  # local import: optional dependency

                if isinstance(X, pd.DataFrame):
                    return X.columns.get_loc(self.current_state_column)
            except ImportError:  # pragma: no cover - pandas is present in this env
                pass
            raise ValueError(
                "current_state_column was given as a column name "
                f"('{self.current_state_column}') but X is not a pandas DataFrame."
            )

        n_features = X_arr.shape[1]
        column = self.current_state_column
        if column < 0:
            column += n_features
        if not (0 <= column < n_features):
            raise ValueError(
                f"current_state_column={self.current_state_column} is out of "
                f"range for X with {n_features} feature columns."
            )
        return column

    def predict_risk(self, X: ArrayLike) -> np.ndarray:
        """Return the current-state column as the forecast risk, unchanged."""
        X_arr = validate_predict_input(X)
        column = self._resolve_column(X, X_arr)
        risk = X_arr[:, column]

        if np.any((risk < 0.0) | (risk > 1.0)):
            raise ValueError(
                "The current-state column must contain values in [0, 1] "
                "(a risk score or a binary label); persistence forecasts "
                "that value unchanged."
            )

        return risk

    def predict(self, X: ArrayLike, threshold: Optional[float] = None) -> np.ndarray:
        """Threshold the persisted risk into a binary forecast label."""
        t = self.threshold if threshold is None else validate_threshold(threshold)
        risk = self.predict_risk(X)
        return risk_to_label(risk, threshold=t)
