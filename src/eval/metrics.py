from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


class EvaluationError(ValueError):
    """Raised when evaluation inputs are missing, malformed, or invalid."""


# ----------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConfusionMatrix:
    """
    Binary confusion matrix counts.

    Ordering (matches the standard [[TN, FP], [FN, TP]] layout used by
    `matrix()` below):

        tn: true negatives  (y_true=0, y_pred=0)
        fp: false positives (y_true=0, y_pred=1)
        fn: false negatives (y_true=1, y_pred=0)
        tp: true positives  (y_true=1, y_pred=1)
    """

    tn: int
    fp: int
    fn: int
    tp: int

    def matrix(self) -> np.ndarray:
        """Return the matrix as [[TN, FP], [FN, TP]]."""
        return np.array(
            [[self.tn, self.fp], [self.fn, self.tp]],
            dtype=int,
        )

    def to_dict(self) -> dict[str, int]:
        return {"tn": self.tn, "fp": self.fp, "fn": self.fn, "tp": self.tp}


# ----------------------------------------------------------------------
# Structured evaluation result
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate evaluation output for one set of predictions."""

    precision: float
    recall: float
    f1: float
    fpr: float
    confusion_matrix: ConfusionMatrix

    def to_dict(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "confusion_matrix": self.confusion_matrix.to_dict(),
        }


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
def _to_array(values: ArrayLike, name: str) -> np.ndarray:
    try:
        array = np.asarray(values)
    except Exception as exc:  # pragma: no cover - defensive
        raise EvaluationError(f"'{name}' could not be converted to an array.") from exc

    if array.ndim != 1:
        raise EvaluationError(f"'{name}' must be a one-dimensional sequence.")

    return array


def _validate_non_empty(array: np.ndarray, name: str) -> None:
    if array.size == 0:
        raise EvaluationError(
            f"'{name}' is empty. At least one sample is required for evaluation."
        )


def _validate_same_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape[0] != y_pred.shape[0]:
        raise EvaluationError(
            "y_true and y_pred must have the same number of samples "
            f"(got {y_true.shape[0]} and {y_pred.shape[0]})."
        )


def _validate_binary_labels(array: np.ndarray, name: str) -> np.ndarray:
    """
    Validate that every value in `array` is a binary label (0 or 1) and
    return it as an int array.

    Booleans and float-valued 0.0/1.0 entries are accepted; any other
    value raises EvaluationError.
    """
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise EvaluationError(f"'{name}' contains NaN or infinite values.")

    try:
        as_float = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"'{name}' must contain numeric binary labels.") from exc

    valid_mask = np.isin(as_float, [0.0, 1.0])
    if not np.all(valid_mask):
        invalid_values = np.unique(as_float[~valid_mask])
        raise EvaluationError(
            f"'{name}' must contain only binary labels (0 or 1). "
            f"Found invalid value(s): {invalid_values.tolist()}."
        )

    return as_float.astype(int)


def _validate_risk(array: np.ndarray, name: str = "risk") -> np.ndarray:
    if array.dtype == object:
        raise EvaluationError(f"'{name}' must contain numeric probability values.")

    as_float = array.astype(float)

    if not np.all(np.isfinite(as_float)):
        raise EvaluationError(f"'{name}' contains NaN or infinite values.")

    if np.any(as_float < 0.0) or np.any(as_float > 1.0):
        raise EvaluationError(
            f"'{name}' values must lie within [0, 1] (probabilities/risk scores)."
        )

    return as_float


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise EvaluationError("'threshold' must be a number.")

    if not np.isfinite(threshold):
        raise EvaluationError("'threshold' must be a finite number.")

    if threshold < 0.0 or threshold > 1.0:
        raise EvaluationError(
            f"'threshold' must be within [0, 1] (got {threshold})."
        )

    return float(threshold)


# ----------------------------------------------------------------------
# Threshold conversion (risk score -> binary label)
# ----------------------------------------------------------------------
def risk_to_label(risk: ArrayLike, threshold: float = 0.70) -> np.ndarray:
    """
    Convert continuous risk/probability scores into binary predictions.

    risk >= threshold -> 1 (malicious)
    risk <  threshold -> 0 (benign)

    `threshold` is configurable and must lie in [0, 1]. The project plan's
    0.70 experimental warning threshold is only the *default* here, not a
    hard-coded value — callers evaluating other operating points should
    pass their own threshold explicitly.
    """
    threshold = _validate_threshold(threshold)

    risk_array = _to_array(risk, "risk")
    _validate_non_empty(risk_array, "risk")
    risk_array = _validate_risk(risk_array, "risk")

    return (risk_array >= threshold).astype(int)


# ----------------------------------------------------------------------
# Confusion matrix computation
# ----------------------------------------------------------------------
def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> ConfusionMatrix:
    """
    Compute the binary confusion matrix for already-binarized predictions.

    Raises EvaluationError on empty input, mismatched lengths, or
    non-binary label values. Never raises due to class imbalance (e.g.
    all-positive or all-negative inputs are valid).
    """
    y_true_arr = _to_array(y_true, "y_true")
    y_pred_arr = _to_array(y_pred, "y_pred")

    _validate_non_empty(y_true_arr, "y_true")
    _validate_non_empty(y_pred_arr, "y_pred")
    _validate_same_length(y_true_arr, y_pred_arr)

    y_true_arr = _validate_binary_labels(y_true_arr, "y_true")
    y_pred_arr = _validate_binary_labels(y_pred_arr, "y_pred")

    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))

    return ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp)


# ----------------------------------------------------------------------
# Core metrics
#
# Zero-division policy: a metric with an undefined denominator (e.g. no
# positive predictions for precision, no positive ground truth for
# recall/FPR) returns 0.0 rather than raising or returning NaN. This
# keeps the evaluator usable on skewed/empty-class batches, which are
# expected during early development against mock predictions. Callers
# who need to distinguish "0.0 because it failed" from "0.0 because the
# denominator was empty" should inspect the ConfusionMatrix directly.
# ----------------------------------------------------------------------
def precision(cm: ConfusionMatrix) -> float:
    """TP / (TP + FP). Returns 0.0 if there are no positive predictions."""
    denominator = cm.tp + cm.fp
    if denominator == 0:
        return 0.0
    return cm.tp / denominator


def recall(cm: ConfusionMatrix) -> float:
    """TP / (TP + FN). Returns 0.0 if there is no positive ground truth."""
    denominator = cm.tp + cm.fn
    if denominator == 0:
        return 0.0
    return cm.tp / denominator


def f1_score(cm: ConfusionMatrix) -> float:
    """2 * P * R / (P + R). Returns 0.0 if precision and recall are both 0."""
    p = precision(cm)
    r = recall(cm)
    denominator = p + r
    if denominator == 0:
        return 0.0
    return 2 * p * r / denominator


def false_positive_rate(cm: ConfusionMatrix) -> float:
    """FP / (FP + TN). Returns 0.0 if there is no negative ground truth."""
    denominator = cm.fp + cm.tn
    if denominator == 0:
        return 0.0
    return cm.fp / denominator


# ----------------------------------------------------------------------
# Aggregate evaluators
# ----------------------------------------------------------------------
def evaluate_predictions(y_true: ArrayLike, y_pred: ArrayLike) -> EvaluationResult:
    """
    Evaluate already-binarized predictions (MODE 1).

    y_true, y_pred: binary (0/1) sequences of equal length.
    """
    cm = confusion_matrix(y_true, y_pred)

    return EvaluationResult(
        precision=precision(cm),
        recall=recall(cm),
        f1=f1_score(cm),
        fpr=false_positive_rate(cm),
        confusion_matrix=cm,
    )


def evaluate_risk(
    y_true: ArrayLike,
    risk: ArrayLike,
    threshold: float = 0.70,
) -> EvaluationResult:
    """
    Evaluate continuous risk/probability predictions (MODE 2).

    risk is converted to binary predictions via `risk_to_label` before
    the same metrics used by `evaluate_predictions` are computed.
    """
    y_pred = risk_to_label(risk, threshold=threshold)
    return evaluate_predictions(y_true, y_pred)


def evaluate_prediction_records(
    records: Sequence[Any],
    threshold: float = 0.50,
) -> dict[int, EvaluationResult]:
    """
    Evaluate multi-horizon PredictionRecords (+10s, +20s, +30s).

    Groups records by forecast_horizon (in seconds) and computes
    individual EvaluationResult metrics for each discrete horizon.
    """
    if not records:
        raise EvaluationError("records cannot be empty.")

    from collections import defaultdict
    by_horizon: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))

    for rec in records:
        h = int(getattr(rec, "forecast_horizon"))
        y_t = float(getattr(rec, "y_true"))
        p_r = float(getattr(rec, "predicted_risk"))
        by_horizon[h][0].append(y_t)
        by_horizon[h][1].append(p_r)

    results: dict[int, EvaluationResult] = {}
    for h in sorted(by_horizon.keys()):
        y_true_list, risk_list = by_horizon[h]
        results[h] = evaluate_risk(y_true_list, risk_list, threshold=threshold)

    return results


# ----------------------------------------------------------------------
# Threshold-independent metrics (ROC-AUC and PR-AUC)
# ----------------------------------------------------------------------
def compute_roc_auc(y_true: ArrayLike, risk: ArrayLike) -> float | None:
    """
    Compute Area Under the Receiver Operating Characteristic (ROC-AUC).
    Returns None if y_true contains only a single class (AUC is undefined).
    """
    from sklearn.metrics import roc_auc_score

    y_true_arr = _validate_binary_labels(_to_array(y_true, "y_true"), "y_true")
    risk_arr = _validate_risk(_to_array(risk, "risk"), "risk")
    _validate_same_length(y_true_arr, risk_arr)

    if len(np.unique(y_true_arr)) < 2:
        return None

    return float(roc_auc_score(y_true_arr, risk_arr))


def compute_pr_auc(y_true: ArrayLike, risk: ArrayLike) -> float | None:
    """
    Compute Area Under the Precision-Recall Curve (PR-AUC / Average Precision).
    Returns None if y_true contains only a single class.
    """
    from sklearn.metrics import average_precision_score

    y_true_arr = _validate_binary_labels(_to_array(y_true, "y_true"), "y_true")
    risk_arr = _validate_risk(_to_array(risk, "risk"), "risk")
    _validate_same_length(y_true_arr, risk_arr)

    if len(np.unique(y_true_arr)) < 2:
        return None

    return float(average_precision_score(y_true_arr, risk_arr))


# ----------------------------------------------------------------------
# Scientific threshold optimization (validation freeze protocol)
# ----------------------------------------------------------------------
def optimize_threshold(
    y_true: ArrayLike,
    risk: ArrayLike,
    metric: str = "f1",
    candidate_thresholds: Sequence[float] | None = None,
) -> float:
    """
    Search candidate decision thresholds on validation data to maximize a target metric.

    SCIENTIFIC VALIDATION PROTOCOL:
    Threshold selection must occur on the validation split and be frozen
    prior to final test set evaluation.

    Args:
        y_true: Binary ground-truth labels (validation set).
        risk: Continuous risk/probability predictions in [0, 1].
        metric: Optimization target ("f1", "precision", or "recall").
        candidate_thresholds: Optional sequence of candidate thresholds in (0, 1).

    Returns:
        Best threshold value (float) to be frozen for test evaluation.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]

    best_threshold = 0.50
    best_score = -1.0

    for t in candidate_thresholds:
        res = evaluate_risk(y_true, risk, threshold=t)
        score = getattr(res, metric, res.f1)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold


# ----------------------------------------------------------------------
# Early-warning and lead-time analysis
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LeadTimeResult:
    """Early-warning and lead-time analysis output."""
    total_events: int
    detected_events: int
    detection_rate: float
    mean_lead_time_seconds: float
    earliest_detections_by_horizon: dict[int, int]
    lead_times_seconds: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "detected_events": self.detected_events,
            "detection_rate": self.detection_rate,
            "mean_lead_time_seconds": self.mean_lead_time_seconds,
            "earliest_detections_by_horizon": self.earliest_detections_by_horizon,
            "lead_times_seconds": self.lead_times_seconds,
        }


def evaluate_lead_time(
    records: Sequence[Any],
    threshold: float = 0.50,
) -> LeadTimeResult:
    """
    Evaluate early-warning lead time for attack forecast records.

    For each unique target attack moment (source_host, target_time) where y_true == 1:
    Determines the earliest advance horizon (+30s, +20s, or +10s) at which
    predicted_risk met or exceeded the frozen decision threshold.
    """
    if not records:
        raise EvaluationError("records cannot be empty.")

    from collections import defaultdict
    events: dict[tuple[str, Any], list[Any]] = defaultdict(list)
    for rec in records:
        if getattr(rec, "y_true", 0.0) == 1.0:
            key = (str(getattr(rec, "source_host")), getattr(rec, "target_time"))
            events[key].append(rec)

    total_events = len(events)
    if total_events == 0:
        return LeadTimeResult(
            total_events=0,
            detected_events=0,
            detection_rate=0.0,
            mean_lead_time_seconds=0.0,
            earliest_detections_by_horizon={10: 0, 20: 0, 30: 0},
            lead_times_seconds=[],
        )

    detected_count = 0
    earliest_by_h: dict[int, int] = defaultdict(int)
    lead_times: list[int] = []

    for event_key, event_records in events.items():
        sorted_recs = sorted(
            event_records,
            key=lambda r: int(getattr(r, "forecast_horizon")),
            reverse=True,
        )
        detected_horizon = None
        for rec in sorted_recs:
            if float(getattr(rec, "predicted_risk")) >= threshold:
                detected_horizon = int(getattr(rec, "forecast_horizon"))
                break

        if detected_horizon is not None:
            detected_count += 1
            earliest_by_h[detected_horizon] += 1
            lead_times.append(detected_horizon)

    rate = (detected_count / total_events) if total_events > 0 else 0.0
    mean_lt = (float(np.mean(lead_times))) if lead_times else 0.0

    return LeadTimeResult(
        total_events=total_events,
        detected_events=detected_count,
        detection_rate=round(rate, 4),
        mean_lead_time_seconds=round(mean_lt, 2),
        earliest_detections_by_horizon=dict(earliest_by_h),
        lead_times_seconds=lead_times,
    )


# ----------------------------------------------------------------------
# Unseen attack evaluation
# ----------------------------------------------------------------------
def evaluate_unseen_attacks(
    records: Sequence[Any],
    attack_labels: Sequence[str],
    seen_attack_types: set[str],
    threshold: float = 0.50,
) -> dict[str, EvaluationResult]:
    """
    Partition forecast evaluation between seen attack types and unseen / zero-day attacks.
    """
    if len(records) != len(attack_labels):
        raise EvaluationError("records and attack_labels must have the same length.")

    seen_recs = []
    unseen_recs = []

    for rec, attack_type in zip(records, attack_labels):
        if str(attack_type) in seen_attack_types:
            seen_recs.append(rec)
        else:
            unseen_recs.append(rec)

    results: dict[str, EvaluationResult] = {}
    if seen_recs:
        y_t = [float(getattr(r, "y_true")) for r in seen_recs]
        r_k = [float(getattr(r, "predicted_risk")) for r in seen_recs]
        results["seen"] = evaluate_risk(y_t, r_k, threshold=threshold)
    if unseen_recs:
        y_t = [float(getattr(r, "y_true")) for r in unseen_recs]
        r_k = [float(getattr(r, "predicted_risk")) for r in unseen_recs]
        results["unseen"] = evaluate_risk(y_t, r_k, threshold=threshold)

    return results


# ----------------------------------------------------------------------
# Smoke test (optional manual run: python src/eval/metrics.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_y_true = [0, 1, 0, 1, 1, 0]
    demo_risk = [0.10, 0.85, 0.40, 0.90, 0.60, 0.05]

    result = evaluate_risk(demo_y_true, demo_risk, threshold=0.70)

    print("=== Evaluation Smoke Test ===")
    print(f"Precision: {result.precision:.2f}")
    print(f"Recall:    {result.recall:.2f}")
    print(f"F1:        {result.f1:.2f}")
    print(f"FPR:       {result.fpr:.2f}")
    print()
    print(f"TN: {result.confusion_matrix.tn}")
    print(f"FP: {result.confusion_matrix.fp}")
    print(f"FN: {result.confusion_matrix.fn}")
    print(f"TP: {result.confusion_matrix.tp}")
