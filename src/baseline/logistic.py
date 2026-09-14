from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.baseline._validation import (
    ArrayLike,
    BaselineError,
    LabelLike,
    validate_fit_inputs,
    validate_predict_input,
    validate_threshold,
)
from src.eval.metrics import risk_to_label


class LogisticRegressionBaseline:
    """
    Logistic Regression forecasting baseline.

    A simple, well-understood linear classifier used as a defensible
    baseline that the future LSTM/world model must outperform to justify
    its added complexity. Uses the same (X, y) convention as every other
    baseline: `X` is the feature matrix at forecast time t, `y` is the
    binary future-malicious label for one specific forecast horizon
    (call `fit`/`predict` once per horizon — +10s, +20s, +30s — with
    that horizon's own y, exactly as for the other baselines).

    PREPROCESSING
    --------------
    Features are standardized (zero mean, unit variance) before fitting.
    The scaler is fit ONLY on `X_train` inside `fit()`; `predict()` and
    `predict_risk()` only ever call `.transform()` on the already-fitted
    scaler, never `.fit()` or `.fit_transform()`. This prevents any
    statistics from validation/test data leaking into training, per the
    project's evaluation protocol.

    DETERMINISM
    ------------
    `random_state` (default 42, matching the project's seed convention)
    is passed through to scikit-learn's `LogisticRegression`, so fitting
    twice on identical data produces identical predictions.

    TEMPORAL SPLIT
    ---------------
    This class does not split data itself. Callers must pass an
    already-temporally-split `X_train`/`y_train` (earlier time) and
    `X_test`/`y_test` (later time) built via a global timestamp cutoff —
    never `sklearn.model_selection.train_test_split(..., shuffle=True)`.
    """

    def __init__(
        self,
        random_state: int = 42,
        threshold: float = 0.70,
        max_iter: int = 1000,
        C: float = 1.0,
    ) -> None:
        self.random_state = random_state
        self.threshold = validate_threshold(threshold)
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            random_state=random_state, max_iter=max_iter, C=C
        )
        self._n_features: Optional[int] = None
        self._fitted = False

    def fit(self, X_train: ArrayLike, y_train: LabelLike) -> "LogisticRegressionBaseline":
        X_arr, y_arr = validate_fit_inputs(X_train, y_train)

        if len(np.unique(y_arr)) < 2:
            raise BaselineError(
                "y_train must contain both classes (0 and 1) to fit a "
                "Logistic Regression baseline."
            )

        self._scaler.fit(X_arr)  # train-only: no test data is ever seen here
        X_scaled = self._scaler.transform(X_arr)
        self._model.fit(X_scaled, y_arr)

        self._n_features = X_arr.shape[1]
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise BaselineError("This baseline is not fitted yet. Call fit() first.")

    def predict_risk(self, X: ArrayLike) -> np.ndarray:
        """Return predicted probability of the malicious/future-risk class."""
        self._check_fitted()
        X_arr = validate_predict_input(X, expected_n_features=self._n_features)
        X_scaled = self._scaler.transform(X_arr)  # transform only, never fit
        return self._model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: ArrayLike, threshold: Optional[float] = None) -> np.ndarray:
        """Threshold the predicted risk into a binary forecast label."""
        t = self.threshold if threshold is None else validate_threshold(threshold)
        risk = self.predict_risk(X)
        return risk_to_label(risk, threshold=t)
