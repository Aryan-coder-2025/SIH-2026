from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[Sequence[float]], np.ndarray]


class FeatureSchemaError(ValueError):
    """Raised when input features do not match the expected schema."""


@dataclass(frozen=True)
class FeatureSchema:
    """
    A named, ordered feature manifest a baseline or evaluator can be
    validated against, so mismatched feature ordering between what a
    model was trained on and what it is later given fails loudly instead
    of silently producing garbage predictions (the exact failure mode
    the project plan warns about: training with
    [bytes, packets, ttl, syn] but predicting with
    [packets, bytes, syn, ttl]).

    `feature_names` is the frozen, ordered list of feature names this
    schema describes. `version` is a free-form tag (e.g. "v1",
    "flow_packet_fusion_2026-09-10") for recording in an
    ExperimentArtifact (src/eval/experiment.py).
    """

    feature_names: Sequence[str]
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise FeatureSchemaError("'feature_names' must be a non-empty sequence.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise FeatureSchemaError("'feature_names' must not contain duplicate names.")

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def validate_array(self, X: ArrayLike, name: str = "X") -> None:
        """Validate that a plain array's column COUNT matches this schema.

        Plain arrays carry no column names, so only the count is checked
        here; use `validate_dataframe_columns` when `X` is a DataFrame
        to also check names and order.
        """
        array = np.asarray(X)
        if array.ndim != 2:
            raise FeatureSchemaError(f"'{name}' must be 2-dimensional; got shape {array.shape}.")
        if array.shape[1] != self.n_features:
            raise FeatureSchemaError(
                f"'{name}' has {array.shape[1]} feature column(s); schema "
                f"'{self.version}' expects {self.n_features}: {list(self.feature_names)}."
            )

    def validate_dataframe_columns(self, df) -> None:
        """Validate that a pandas DataFrame's columns match this schema
        exactly, in the same order -- not just the same count."""
        columns = list(df.columns)
        if columns != list(self.feature_names):
            raise FeatureSchemaError(
                "DataFrame columns do not match the expected feature schema "
                f"'{self.version}'.\n  expected: {list(self.feature_names)}\n  got:      {columns}"
            )
