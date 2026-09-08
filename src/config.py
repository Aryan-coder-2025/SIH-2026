from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"


class ConfigError(ValueError):
    """Raised when the project configuration is invalid."""


def _require_mapping(config: Any, name: str) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigError(f"'{name}' must be a mapping/object.")
    return config


def _require_positive_int(
    section: dict[str, Any],
    key: str,
    *,
    minimum: int = 1,
) -> int:
    value = section.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer.")

    if value < minimum:
        raise ConfigError(f"'{key}' must be >= {minimum}.")

    return value


def load_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Load and validate the SIH26153 project configuration.

    Uses yaml.safe_load() so arbitrary Python objects cannot be
    constructed from the configuration file.
    """
    path = Path(config_path).resolve()

    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML syntax: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Unable to read configuration: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a mapping/object.")

    project = _require_mapping(config.get("project"), "project")
    data = _require_mapping(config.get("data"), "data")
    entity = _require_mapping(config.get("entity"), "entity")
    temporal = _require_mapping(config.get("temporal"), "temporal")
    prediction = _require_mapping(config.get("prediction"), "prediction")
    features = _require_mapping(config.get("features"), "features")
    evaluation = _require_mapping(config.get("evaluation"), "evaluation")

    # ------------------------------------------------------------------
    # Project identity
    # ------------------------------------------------------------------
    if project.get("id") != "SIH26153":
        raise ConfigError("project.id must be 'SIH26153'.")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if data.get("dataset") != "CSE-CIC-IDS2018":
        raise ConfigError(
            "data.dataset must be 'CSE-CIC-IDS2018'."
        )

    # ------------------------------------------------------------------
    # Entity
    # ------------------------------------------------------------------
    if entity.get("type") != "source_host":
        raise ConfigError("entity.type must be 'source_host'.")

    if entity.get("key") != "src_ip":
        raise ConfigError("entity.key must be 'src_ip'.")

    # ------------------------------------------------------------------
    # Temporal contract
    # ------------------------------------------------------------------
    window_seconds = _require_positive_int(
        temporal,
        "window_seconds",
    )

    sequence_length = _require_positive_int(
        temporal,
        "sequence_length_windows",
    )

    forecast_horizon = _require_positive_int(
        temporal,
        "forecast_horizon_windows",
    )

    if window_seconds != 10:
        raise ConfigError(
            "window_seconds must be exactly 10."
        )

    if sequence_length != 10:
        raise ConfigError(
            "sequence_length_windows must be exactly 10."
        )

    if forecast_horizon != 3:
        raise ConfigError(
            "forecast_horizon_windows must be exactly 3."
        )

    # ------------------------------------------------------------------
    # Forecast offset contract
    # ------------------------------------------------------------------
    offsets = prediction.get("forecast_offsets_seconds")

    if not isinstance(offsets, list):
        raise ConfigError(
            "prediction.forecast_offsets_seconds must be a list."
        )

    if not all(
        isinstance(offset, int) and not isinstance(offset, bool)
        for offset in offsets
    ):
        raise ConfigError(
            "prediction.forecast_offsets_seconds must contain "
            "only integers."
        )

    expected_offsets = [
        window_seconds * index
        for index in range(1, forecast_horizon + 1)
    ]

    if offsets != expected_offsets:
        raise ConfigError(
            "forecast_offsets_seconds must match the configured "
            f"forecast horizon: {expected_offsets}."
        )

    # ------------------------------------------------------------------
    # Feature-level contract
    # ------------------------------------------------------------------
    required_levels = features.get("required_levels")

    if not isinstance(required_levels, list):
        raise ConfigError(
            "features.required_levels must be a list."
        )

    if not all(
        isinstance(level, str)
        for level in required_levels
    ):
        raise ConfigError(
            "features.required_levels must contain only strings."
        )

    required_levels = set(required_levels)

    if not {"flow", "packet"}.issubset(required_levels):
        raise ConfigError(
            "Both 'flow' and 'packet' feature levels are required."
        )

    # ------------------------------------------------------------------
    # Evaluation safety contract
    # ------------------------------------------------------------------
    for key in (
        "temporal_split",
        "split_before_sequence_construction",
        "prevent_future_information_leakage",
    ):
        if evaluation.get(key) is not True:
            raise ConfigError(
                f"evaluation.{key} must be true."
            )

    return config


if __name__ == "__main__":
    loaded_config = load_config()

    print("Configuration validation: PASS")
    print(
        f"Project: {loaded_config['project']['id']}"
    )
    print(
        f"Dataset: {loaded_config['data']['dataset']}"
    )
    print(
        "Temporal contract: "
        f"{loaded_config['temporal']['window_seconds']}s window, "
        f"{loaded_config['temporal']['sequence_length_windows']} "
        "windows, "
        f"{loaded_config['temporal']['forecast_horizon_windows']}"
        "-step forecast"
    )
    print(
        "Feature levels: "
        + ", ".join(
            loaded_config["features"]["required_levels"]
        )
    )