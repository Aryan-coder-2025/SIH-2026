from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.eval.records import SUPPORTED_HORIZONS

SUPPORTED_FEATURE_GROUPS = ("flow", "packet", "flow+packet")
SUPPORTED_TEMPORAL_MODES = ("static", "temporal")


class AblationConfigError(ValueError):
    """Raised when an ablation configuration is invalid."""


@dataclass(frozen=True)
class AblationConfig:
    """
    One ablation experiment's configuration.

    This does NOT run anything -- it is a validated, JSON-serializable
    description of a comparison to run once the corresponding model
    (baseline or the future LSTM/world model) is available. Building the
    grid of configurations now lets the team agree on exactly which
    comparisons matter before any of them are trained, and lets whoever
    trains the LSTM plug into the same structure later.

    Fields:
        name:             short identifier, e.g. "flow_only_static_rf_h10"
        feature_group:     one of SUPPORTED_FEATURE_GROUPS
                           ("flow", "packet", "flow+packet")
        temporal_mode:     "static" (current-window features only) or
                           "temporal" (full history window)
        history_length:    number of 10-second windows of history used.
                           Must be 1 when temporal_mode="static" (a
                           static model only ever sees the current
                           window) and >= 2 when temporal_mode="temporal".
        model_name:        e.g. "persistence", "logistic_regression",
                           "random_forest", "lstm_world_model" -- a label
                           only; this module does not import or run models.
        forecast_horizon:  one of SUPPORTED_HORIZONS (10, 20, 30)
    """

    name: str
    feature_group: str
    temporal_mode: str
    history_length: int
    model_name: str
    forecast_horizon: int

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise AblationConfigError("'name' must be a non-empty string.")

        if self.feature_group not in SUPPORTED_FEATURE_GROUPS:
            raise AblationConfigError(
                f"feature_group={self.feature_group!r} must be one of {SUPPORTED_FEATURE_GROUPS}."
            )

        if self.temporal_mode not in SUPPORTED_TEMPORAL_MODES:
            raise AblationConfigError(
                f"temporal_mode={self.temporal_mode!r} must be one of {SUPPORTED_TEMPORAL_MODES}."
            )

        if not isinstance(self.history_length, int) or self.history_length < 1:
            raise AblationConfigError("'history_length' must be a positive integer.")

        if self.temporal_mode == "static" and self.history_length != 1:
            raise AblationConfigError(
                "temporal_mode='static' requires history_length=1 "
                f"(got {self.history_length})."
            )

        if self.temporal_mode == "temporal" and self.history_length < 2:
            raise AblationConfigError(
                "temporal_mode='temporal' requires history_length >= 2 "
                f"(got {self.history_length}); a single window is not a history."
            )

        if not self.model_name or not isinstance(self.model_name, str):
            raise AblationConfigError("'model_name' must be a non-empty string.")

        if self.forecast_horizon not in SUPPORTED_HORIZONS:
            raise AblationConfigError(
                f"forecast_horizon={self.forecast_horizon!r} must be one of {SUPPORTED_HORIZONS}."
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "feature_group": self.feature_group,
            "temporal_mode": self.temporal_mode,
            "history_length": self.history_length,
            "model_name": self.model_name,
            "forecast_horizon": self.forecast_horizon,
        }


def standard_ablation_grid(
    model_name: str,
    forecast_horizon: int,
    history_length: int = 10,
) -> List[AblationConfig]:
    """
    Build the project's standard required comparisons for one model and
    horizon:

        A. flow-only
        B. packet-only
        C. flow + packet
        D. static (current-window) features
        E. temporal-history features (using `history_length` windows)

    Feature-group ablations (A-C) use temporal_mode="temporal" with the
    full `history_length`, matching how the model would actually be
    compared in the project's final benchmark table. Modes D and E hold
    feature_group="flow+packet" fixed and vary only the temporal mode,
    isolating the effect of history length on its own. Returns 5
    AblationConfig entries; callers needing a different combination
    should construct AblationConfig directly.
    """
    return [
        AblationConfig(
            name=f"{model_name}_flow_only_h{forecast_horizon}",
            feature_group="flow",
            temporal_mode="temporal",
            history_length=history_length,
            model_name=model_name,
            forecast_horizon=forecast_horizon,
        ),
        AblationConfig(
            name=f"{model_name}_packet_only_h{forecast_horizon}",
            feature_group="packet",
            temporal_mode="temporal",
            history_length=history_length,
            model_name=model_name,
            forecast_horizon=forecast_horizon,
        ),
        AblationConfig(
            name=f"{model_name}_flow_packet_h{forecast_horizon}",
            feature_group="flow+packet",
            temporal_mode="temporal",
            history_length=history_length,
            model_name=model_name,
            forecast_horizon=forecast_horizon,
        ),
        AblationConfig(
            name=f"{model_name}_static_h{forecast_horizon}",
            feature_group="flow+packet",
            temporal_mode="static",
            history_length=1,
            model_name=model_name,
            forecast_horizon=forecast_horizon,
        ),
        AblationConfig(
            name=f"{model_name}_temporal_h{forecast_horizon}",
            feature_group="flow+packet",
            temporal_mode="temporal",
            history_length=history_length,
            model_name=model_name,
            forecast_horizon=forecast_horizon,
        ),
    ]
