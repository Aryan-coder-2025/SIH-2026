from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.schemas.traffic import normalize_to_utc


@dataclass(frozen=True)
class PredictionRecord:
    """
    Canonical evaluation record for model predictions and ground-truth comparison.
    Guarantees internal consistency between prediction time, horizon, and target time.

    Invariants Enforced:
    1. Alignment: target_time MUST equal prediction_time + timedelta(seconds=forecast_horizon).
    2. Probability Range: predicted_risk MUST be a finite float in [0.0, 1.0].
    3. Ground Truth: y_true MUST be a finite float (e.g. 0.0 or 1.0).
    4. Timezone: prediction_time and target_time are normalized to UTC.
    5. Identity: source_host must be a non-empty string.
    """

    source_host: str
    prediction_time: datetime
    forecast_horizon: int  # in seconds (e.g. 10, 20, 30)
    target_time: datetime
    y_true: float
    predicted_risk: float

    def __post_init__(self) -> None:
        # Validate and normalize source_host
        if not isinstance(self.source_host, str) or not self.source_host.strip():
            raise ValueError("source_host must be a non-empty string")
        object.__setattr__(self, "source_host", self.source_host.strip())

        # Validate and normalize timestamps to UTC
        norm_pred = normalize_to_utc(self.prediction_time)
        norm_targ = normalize_to_utc(self.target_time)
        object.__setattr__(self, "prediction_time", norm_pred)
        object.__setattr__(self, "target_time", norm_targ)

        # Validate forecast horizon
        if isinstance(self.forecast_horizon, bool) or not isinstance(self.forecast_horizon, int):
            raise ValueError("forecast_horizon must be an integer")
        if self.forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be a positive integer, got {self.forecast_horizon}")

        # Temporal Invariant: target_time MUST equal prediction_time + timedelta(seconds=forecast_horizon)
        expected_target_time = norm_pred + timedelta(seconds=self.forecast_horizon)
        if norm_targ != expected_target_time:
            raise ValueError(
                f"PredictionRecord alignment error: prediction_time ({norm_pred.isoformat()}) "
                f"+ {self.forecast_horizon}s does not equal target_time ({norm_targ.isoformat()}). "
                f"Expected: {expected_target_time.isoformat()}"
            )

        # Validate y_true
        if isinstance(self.y_true, bool) or not isinstance(self.y_true, (int, float)):
            raise ValueError("y_true must be a numeric float or int")
        if not math.isfinite(self.y_true):
            raise ValueError("y_true must be a finite number")

        # Validate predicted_risk
        if isinstance(self.predicted_risk, bool) or not isinstance(self.predicted_risk, (int, float)):
            raise ValueError("predicted_risk must be a numeric float or int")
        if not math.isfinite(self.predicted_risk):
            raise ValueError("predicted_risk must be a finite number")
        if not (0.0 <= float(self.predicted_risk) <= 1.0):
            raise ValueError(
                f"predicted_risk must be a probability score in [0.0, 1.0], got {self.predicted_risk}"
            )
