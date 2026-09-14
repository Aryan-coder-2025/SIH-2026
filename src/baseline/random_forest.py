from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.baseline._validation import (
    ArrayLike,
    BaselineError,
    LabelLike,
    validate_fit_inputs,
    validate_predict_input,
    validate_threshold,
)
from src.eval.feature_schema import FeatureSchema, FeatureSchemaError
from src.eval.metrics import risk_to_label


class RandomForestBaseline:
    """
    Random Forest forecasting baseline.

    A stronger non-linear baseline than Logistic Regression, still far
    simpler than the future LSTM/world model. Uses the same (X, y)
    convention as the other baselines: `X` is the feature matrix at
    forecast time t, `y` is the binary future-malicious label for one
    specific forecast horizon (fit/predict once per horizon).

    PREPROCESSING
    --------------
    Tree ensembles do not require feature scaling, so no scaler is fit
    or applied here. `X` is used as given (already validated to be
    finite/numeric).

    DETERMINISM
    ------------
    `random_state` (default 42) is passed through to scikit-learn's
    `RandomForestClassifier`, so fitting twice on identical data produces
    identical predictions. Hyperparameters (`n_estimators`, `max_depth`,
    `min_samples_leaf`) are configurable constructor arguments rather
    than hard-coded.

    TEMPORAL SPLIT
    ---------------
    This class does not split data itself. Callers must pass an
    already-temporally-split `X_train`/`y_train` and `X_test`/`y_test`
    built via a global timestamp cutoff — never a random/shuffled split.

    FEATURE SCHEMA (optional)
    ---------------------------
    Passing `feature_schema` (a `src.eval.feature_schema.FeatureSchema`)
    additionally validates, at both `fit()` and `predict()`/
    `predict_risk()`, that a pandas DataFrame's columns match the
    expected feature names AND order exactly. Optional and backward
    compatible — omitting it preserves the original count-only check.
    """

    def __init__(
        self,
        random_state: int = 42,
        threshold: float = 0.70,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        feature_schema: Optional[FeatureSchema] = None,
    ) -> None:
        self.random_state = random_state
        self.threshold = validate_threshold(threshold)
        self.feature_schema = feature_schema
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self._n_features: Optional[int] = None
        self._fitted = False

    def _check_schema(self, X: ArrayLike) -> None:
        if self.feature_schema is not None and hasattr(X, "columns"):
            try:
                self.feature_schema.validate_dataframe_columns(X)
            except FeatureSchemaError as exc:
                raise BaselineError(str(exc)) from exc

    def fit(self, X_train: ArrayLike, y_train: LabelLike) -> "RandomForestBaseline":
        self._check_schema(X_train)
        X_arr, y_arr = validate_fit_inputs(X_train, y_train)

        if self.feature_schema is not None:
            try:
                self.feature_schema.validate_array(X_arr, "X_train")
            except FeatureSchemaError as exc:
                raise BaselineError(str(exc)) from exc

        if len(np.unique(y_arr)) < 2:
            raise BaselineError(
                "y_train must contain both classes (0 and 1) to fit a "
                "Random Forest baseline."
            )

        self._model.fit(X_arr, y_arr)

        self._n_features = X_arr.shape[1]
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise BaselineError("This baseline is not fitted yet. Call fit() first.")

    def predict_risk(self, X: ArrayLike) -> np.ndarray:
        """Return predicted probability of the malicious/future-risk class."""
        self._check_fitted()
        self._check_schema(X)
        X_arr = validate_predict_input(X, expected_n_features=self._n_features)
        return self._model.predict_proba(X_arr)[:, 1]

    def predict(self, X: ArrayLike, threshold: Optional[float] = None) -> np.ndarray:
        """Threshold the predicted risk into a binary forecast label."""
        t = self.threshold if threshold is None else validate_threshold(threshold)
        risk = self.predict_risk(X)
        return risk_to_label(risk, threshold=t)
