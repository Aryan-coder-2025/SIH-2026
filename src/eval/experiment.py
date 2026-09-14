from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.eval.records import SUPPORTED_HORIZONS


class ExperimentArtifactError(ValueError):
    """Raised when an experiment artifact is invalid."""


@dataclass(frozen=True)
class ExperimentArtifact:
    """
    A structured, JSON-serializable record of one evaluation run.

    This is the standard shape other team members (dashboard, QA,
    Aryan's integration) can consume without agreeing on a bespoke
    schema per experiment -- see `to_dict()` / `from_dict()`.

    Fields:
        experiment_name:      short unique label, e.g. "rf_flow_packet_h10_v1"
        model_name:           e.g. "persistence", "logistic_regression",
                               "random_forest", "lstm_world_model"
        dataset_version:      e.g. "CSE-CIC-IDS2018" or a versioned tag
        train_period:         (start_timestamp, end_timestamp), exclusive end
        validation_period:    (start_timestamp, end_timestamp), exclusive end
        test_period:          (start_timestamp, end_timestamp), exclusive end
        forecast_horizon:     one of SUPPORTED_HORIZONS (10, 20, 30)
        feature_set:          ordered list of feature names used
        feature_schema_version: version tag for that feature set, e.g. "v1"
        history_length:       number of 10-second windows of history used
                               (1 for a static/current-window model)
        random_seed:          the seed used for this run
        hyperparameters:      JSON-serializable dict of model hyperparameters
        threshold:            the frozen decision threshold used for the
                               final test evaluation (selected on
                               validation data -- see src/eval/threshold.py)
        metrics:               JSON-serializable dict of the final test
                               metrics, e.g. {"precision": ..., "recall": ...}
        created_at:            ISO-8601 timestamp string for when this
                               artifact was produced

    Periods must be temporally ordered and non-overlapping:
    train_period.end <= validation_period.start <=
    validation_period.end <= test_period.start.
    """

    experiment_name: str
    model_name: str
    dataset_version: str
    train_period: Tuple[float, float]
    validation_period: Tuple[float, float]
    test_period: Tuple[float, float]
    forecast_horizon: int
    feature_set: List[str]
    feature_schema_version: str
    history_length: int
    random_seed: int
    hyperparameters: Dict[str, Any]
    threshold: float
    metrics: Dict[str, float]
    created_at: str

    def __post_init__(self) -> None:
        _require_non_empty_str(self.experiment_name, "experiment_name")
        _require_non_empty_str(self.model_name, "model_name")
        _require_non_empty_str(self.dataset_version, "dataset_version")
        _require_non_empty_str(self.feature_schema_version, "feature_schema_version")
        _require_non_empty_str(self.created_at, "created_at")

        for name, period in (
            ("train_period", self.train_period),
            ("validation_period", self.validation_period),
            ("test_period", self.test_period),
        ):
            if len(period) != 2 or period[1] <= period[0]:
                raise ExperimentArtifactError(
                    f"'{name}' must be a (start, end) tuple with end > start; got {period!r}."
                )

        if self.train_period[1] > self.validation_period[0]:
            raise ExperimentArtifactError(
                "'train_period' must end at or before 'validation_period' begins "
                f"(train_period={self.train_period}, validation_period={self.validation_period})."
            )

        if self.validation_period[1] > self.test_period[0]:
            raise ExperimentArtifactError(
                "'validation_period' must end at or before 'test_period' begins "
                f"(validation_period={self.validation_period}, test_period={self.test_period})."
            )

        if self.forecast_horizon not in SUPPORTED_HORIZONS:
            raise ExperimentArtifactError(
                f"forecast_horizon={self.forecast_horizon!r} must be one of {SUPPORTED_HORIZONS}."
            )

        if not self.feature_set:
            raise ExperimentArtifactError("'feature_set' must be a non-empty list of feature names.")

        if not isinstance(self.history_length, int) or self.history_length < 1:
            raise ExperimentArtifactError("'history_length' must be a positive integer.")

        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ExperimentArtifactError(f"'threshold' must be within [0, 1] (got {self.threshold}).")

        if not self.metrics:
            raise ExperimentArtifactError("'metrics' must be a non-empty dict of recorded metric values.")

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "dataset_version": self.dataset_version,
            "train_period": list(self.train_period),
            "validation_period": list(self.validation_period),
            "test_period": list(self.test_period),
            "forecast_horizon": self.forecast_horizon,
            "feature_set": list(self.feature_set),
            "feature_schema_version": self.feature_schema_version,
            "history_length": self.history_length,
            "random_seed": self.random_seed,
            "hyperparameters": dict(self.hyperparameters),
            "threshold": self.threshold,
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentArtifact":
        required_keys = {
            "experiment_name",
            "model_name",
            "dataset_version",
            "train_period",
            "validation_period",
            "test_period",
            "forecast_horizon",
            "feature_set",
            "feature_schema_version",
            "history_length",
            "random_seed",
            "hyperparameters",
            "threshold",
            "metrics",
            "created_at",
        }
        missing = required_keys - data.keys()
        if missing:
            raise ExperimentArtifactError(f"Missing required key(s): {sorted(missing)}.")

        return cls(
            experiment_name=data["experiment_name"],
            model_name=data["model_name"],
            dataset_version=data["dataset_version"],
            train_period=tuple(data["train_period"]),
            validation_period=tuple(data["validation_period"]),
            test_period=tuple(data["test_period"]),
            forecast_horizon=data["forecast_horizon"],
            feature_set=list(data["feature_set"]),
            feature_schema_version=data["feature_schema_version"],
            history_length=data["history_length"],
            random_seed=data["random_seed"],
            hyperparameters=dict(data["hyperparameters"]),
            threshold=data["threshold"],
            metrics=dict(data["metrics"]),
            created_at=data["created_at"],
        )


def _require_non_empty_str(value: Any, name: str) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise ExperimentArtifactError(f"'{name}' must be a non-empty string.")
