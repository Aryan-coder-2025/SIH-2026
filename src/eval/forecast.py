from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.eval.metrics import EvaluationResult, evaluate_risk
from src.eval.records import (
    SUPPORTED_HORIZONS,
    PredictionRecord,
    RecordValidationError,
    validate_prediction_records,
)


class AlignmentError(ValueError):
    """Raised when predictions cannot be unambiguously joined to ground truth."""


# ----------------------------------------------------------------------
# Raw inputs, before they are joined into PredictionRecords
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RawPrediction:
    """A model's forecast, before it has been matched to a ground-truth label."""

    source_host: str
    prediction_time: float
    forecast_horizon: int
    risk: float

    @property
    def target_time(self) -> float:
        return self.prediction_time + self.forecast_horizon


@dataclass(frozen=True)
class GroundTruthLabel:
    """The actual observed label for one (source_host, target_time)."""

    source_host: str
    target_time: float
    y_true: int


# ----------------------------------------------------------------------
# Key-based (never positional) alignment
# ----------------------------------------------------------------------
def align_predictions_with_ground_truth(
    predictions: Sequence[RawPrediction],
    ground_truth: Sequence[GroundTruthLabel],
) -> List[PredictionRecord]:
    """
    Join predictions to ground truth by (source_host, target_time) -- NEVER
    by list position. Predictions and ground truth may arrive in any
    order, from any source, and this produces the same result either way.

    Raises AlignmentError if:
      - two ground-truth labels share the same (source_host, target_time)
        key (ambiguous join target), or
      - any prediction's (source_host, target_time) has no matching
        ground-truth label.

    A prediction for host A is matched only against ground truth also
    keyed on host A -- host isolation is enforced by construction, since
    the join key always includes source_host.
    """
    truth_index: Dict[Tuple[str, float], int] = {}
    for label in ground_truth:
        key = (label.source_host, label.target_time)
        if key in truth_index:
            raise AlignmentError(
                f"Ambiguous ground truth: multiple labels found for "
                f"source_host={label.source_host!r}, target_time={label.target_time!r}."
            )
        truth_index[key] = label.y_true

    records: List[PredictionRecord] = []
    unmatched: List[Tuple[str, float]] = []
    for prediction in predictions:
        key = (prediction.source_host, prediction.target_time)
        if key not in truth_index:
            unmatched.append(key)
            continue
        records.append(
            PredictionRecord(
                source_host=prediction.source_host,
                prediction_time=prediction.prediction_time,
                forecast_horizon=prediction.forecast_horizon,
                target_time=prediction.target_time,
                y_true=truth_index[key],
                risk=prediction.risk,
            )
        )

    if unmatched:
        raise AlignmentError(
            f"{len(unmatched)} prediction(s) have no matching ground truth "
            f"(source_host, target_time) key: {unmatched}."
        )

    return records


# ----------------------------------------------------------------------
# Per-horizon forecast evaluation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastEvaluationResult:
    """
    Per-horizon evaluation results, plus a simple cross-horizon summary.

    `per_horizon` maps each forecast horizon (10, 20, 30, ...) present in
    the input records to its own EvaluationResult -- computed only from
    that horizon's own records, never mixed with another horizon's.

    `aggregate`, if present, is the unweighted arithmetic mean of each
    metric across the horizons in `per_horizon`. It is a convenience
    summary, NOT a pooled/weighted recomputation over all samples --
    different horizons may have different sample counts, so this is
    documented explicitly to avoid being read as a single overall
    confusion matrix.
    """

    per_horizon: Dict[int, EvaluationResult]
    aggregate: Dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "per_horizon": {
                str(horizon): result.to_dict()
                for horizon, result in sorted(self.per_horizon.items())
            },
            "aggregate": dict(self.aggregate),
        }


def evaluate_forecast(
    records: Sequence[PredictionRecord],
    threshold: float | Dict[int, float] = 0.70,
) -> ForecastEvaluationResult:
    """
    Evaluate a batch of already-aligned PredictionRecords, separately for
    every forecast horizon present.

    `threshold` may be a single float (applied to every horizon) or a
    dict mapping horizon -> threshold (e.g. the frozen, per-horizon
    thresholds selected via src/eval/threshold.py on validation data).

    Validates every record first (see src/eval/records.py) -- malformed
    records, duplicate keys, and non-binary/out-of-range values are all
    rejected before any metric is computed.
    """
    validate_prediction_records(records)

    by_horizon: Dict[int, List[PredictionRecord]] = {}
    for record in records:
        by_horizon.setdefault(record.forecast_horizon, []).append(record)

    per_horizon: Dict[int, EvaluationResult] = {}
    for horizon, horizon_records in by_horizon.items():
        horizon_threshold = (
            threshold[horizon] if isinstance(threshold, dict) else threshold
        )
        y_true = [r.y_true for r in horizon_records]
        risk = [r.risk for r in horizon_records]
        per_horizon[horizon] = evaluate_risk(y_true, risk, threshold=horizon_threshold)

    aggregate: Dict[str, float] = {}
    if per_horizon:
        for metric_name in ("precision", "recall", "f1", "fpr"):
            values = [getattr(result, metric_name) for result in per_horizon.values()]
            aggregate[metric_name] = float(np.mean(values))

    return ForecastEvaluationResult(per_horizon=per_horizon, aggregate=aggregate)


__all__ = [
    "AlignmentError",
    "RawPrediction",
    "GroundTruthLabel",
    "ForecastEvaluationResult",
    "align_predictions_with_ground_truth",
    "evaluate_forecast",
    "SUPPORTED_HORIZONS",
    "PredictionRecord",
    "RecordValidationError",
]
