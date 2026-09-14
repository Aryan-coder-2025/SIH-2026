from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from src.eval.metrics import EvaluationError, EvaluationResult, evaluate_risk

ArrayLike = Union[Sequence[float], np.ndarray]

# The project's experimental default. This is NOT a scientifically
# calibrated probability threshold -- see the module docstring below and
# docs/07_EVALUATION_PROTOCOL.md, section on risk score vs. calibrated
# probability.
DEFAULT_EXPERIMENTAL_THRESHOLD = 0.70

SUPPORTED_SELECTION_METRICS = ("f1", "precision", "recall")


class ThresholdSelectionError(ValueError):
    """Raised when threshold selection inputs or configuration are invalid."""


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """
    The outcome of selecting a threshold on validation data.

    `threshold` is the selected/frozen value; `selection_metric` names
    which metric drove the choice; `validation_score` is that metric's
    value on the validation set AT the selected threshold;
    `candidates_evaluated` is exposed for transparency/QA (e.g. plotting
    the metric curve), not required for normal use.
    """

    threshold: float
    selection_metric: str
    validation_score: float
    candidates_evaluated: int


def select_threshold(
    y_val: ArrayLike,
    risk_val: ArrayLike,
    candidate_thresholds: Optional[ArrayLike] = None,
    metric: str = "f1",
) -> ThresholdSelectionResult:
    """
    Select a decision threshold by scanning `candidate_thresholds` and
    picking the one that maximizes `metric`, evaluated ONLY on the data
    passed in here.

    CRITICAL: this function has no way to know whether the arrays you
    pass are validation or final-test data -- that discipline is the
    caller's responsibility. Per the project's evaluation protocol, this
    must be called with VALIDATION data only. The frozen output threshold
    is then applied, unchanged, to the held-out final test set. Never
    call this function again on the test set, and never let the test
    set's own metric value influence which threshold you pick.

        train model
            -> validation predictions          (this function's input)
            -> select_threshold(...)           (this function)
            -> freeze the returned threshold
            -> evaluate ONCE on the final test set at that frozen value

    `candidate_thresholds` defaults to a deterministic grid of 99 values
    (0.01 .. 0.99 in steps of 0.01) if not given. Ties are broken by
    preferring the lowest threshold that achieves the maximum score, for
    determinism.
    """
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ThresholdSelectionError(
            f"metric={metric!r} is not supported; choose one of {SUPPORTED_SELECTION_METRICS}."
        )

    if candidate_thresholds is None:
        candidates = np.round(np.arange(0.01, 1.00, 0.01), 2)
    else:
        candidates = np.asarray(candidate_thresholds, dtype=float)
        if candidates.ndim != 1 or candidates.size == 0:
            raise ThresholdSelectionError(
                "'candidate_thresholds' must be a non-empty one-dimensional sequence."
            )

    best_threshold: Optional[float] = None
    best_score = -np.inf
    evaluated = 0

    for candidate in candidates:
        try:
            result: EvaluationResult = evaluate_risk(y_val, risk_val, threshold=float(candidate))
        except EvaluationError as exc:
            raise ThresholdSelectionError(
                f"Validation data is invalid for threshold selection: {exc}"
            ) from exc

        evaluated += 1
        score = getattr(result, metric)
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)

    if best_threshold is None:  # pragma: no cover - candidates is never empty here
        raise ThresholdSelectionError("No candidate thresholds were evaluated.")

    return ThresholdSelectionResult(
        threshold=best_threshold,
        selection_metric=metric,
        validation_score=float(best_score),
        candidates_evaluated=evaluated,
    )
