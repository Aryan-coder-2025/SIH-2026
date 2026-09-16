from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "project.yaml"
EXPECTED_PROJECT_ID = "SIH26153"


def _require(config: dict, path: str) -> Any:
    """Read a required nested configuration value with a clear error."""
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing configuration value: {path}")
        current = current[key]
    return current


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load and strictly validate the single project configuration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    required_sections = ("project", "temporal", "prediction", "model", "training")
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")

    if config["project"].get("id") != EXPECTED_PROJECT_ID:
        raise ValueError(
            f"Expected {EXPECTED_PROJECT_ID} configuration, "
            f"but found {config['project'].get('id')!r}."
        )

    temporal = config["temporal"]
    prediction = config["prediction"]
    model = config["model"]
    training = config["training"]

    window_seconds = _require(config, "temporal.window_seconds")
    sequence_length = _require(config, "temporal.sequence_length_windows")
    horizon = _require(config, "temporal.forecast_horizon_windows")
    offsets = _require(config, "prediction.forecast_offsets_seconds")

    if not isinstance(window_seconds, int) or window_seconds <= 0:
        raise ValueError("temporal.window_seconds must be a positive integer.")
    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError("temporal.sequence_length_windows must be a positive integer.")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("temporal.forecast_horizon_windows must be a positive integer.")

    if not isinstance(offsets, list) or len(offsets) != horizon:
        raise ValueError("forecast_offsets_seconds length must equal forecast_horizon_windows.")
    if any(not isinstance(x, int) or x <= 0 for x in offsets):
        raise ValueError("forecast_offsets_seconds must contain positive integers.")
    if offsets != sorted(set(offsets)):
        raise ValueError("forecast_offsets_seconds must be unique and strictly increasing.")
    if any(x % window_seconds != 0 for x in offsets):
        raise ValueError("Every forecast offset must be an exact multiple of window_seconds.")

    # The SIH forecasting contract currently requires these exact horizons.
    if window_seconds != 10 or sequence_length != 10 or horizon != 3 or offsets != [10, 20, 30]:
        raise ValueError(
            "SIH26153 currently requires window=10s, sequence=10 windows, "
            "horizon=3 and offsets=[10,20,30]."
        )

    for name in ("train_ratio", "validation_ratio"):
        value = training.get(name)
        if not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(f"training.{name} must be between 0 and 1.")

    if training["train_ratio"] + training["validation_ratio"] >= 1:
        raise ValueError("Train + validation ratio must be less than 1.")

    # Validate model/training values when present in project.yaml.
    numeric_positive = ("hidden_size", "num_layers", "batch_size", "epochs")
    for name in numeric_positive:
        if name in model and name in ("hidden_size", "num_layers") and (
            not isinstance(model[name], int) or model[name] <= 0
        ):
            raise ValueError(f"model.{name} must be a positive integer.")
        if name in training and name in ("batch_size", "epochs") and (
            not isinstance(training[name], int) or training[name] <= 0
        ):
            raise ValueError(f"training.{name} must be a positive integer.")

    if "dropout" in model and not 0 <= float(model["dropout"]) < 1:
        raise ValueError("model.dropout must be in [0, 1).")
    if "learning_rate" in training and float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive.")
    if "seed" in training and not isinstance(training["seed"], int):
        raise ValueError("training.seed must be an integer.")

    return config


def get_temporal_config(config: dict) -> dict:
    """Return all temporal settings used by sequence/model/inference code."""
    temporal = config["temporal"]
    prediction = config["prediction"]
    return {
        "window_seconds": int(temporal["window_seconds"]),
        "sequence_length": int(temporal["sequence_length_windows"]),
        "horizon": int(temporal["forecast_horizon_windows"]),
        "forecast_offsets_seconds": list(prediction["forecast_offsets_seconds"]),
    }


def get_model_config(config: dict) -> dict:
    """Return model settings without duplicating them in Python code."""
    model = config["model"]
    temporal = get_temporal_config(config)
    return {
        "input_size": int(model.get("input_size", 0)),
        "hidden_size": int(model.get("hidden_size", 128)),
        "num_layers": int(model.get("num_layers", 2)),
        "dropout": float(model.get("dropout", 0.2)),
        "horizon": temporal["horizon"],
    }


def get_training_config(config: dict) -> dict:
    """Return training/reproducibility settings."""
    training = config["training"]
    return {
        "train_ratio": float(training["train_ratio"]),
        "validation_ratio": float(training["validation_ratio"]),
        "seed": int(training.get("seed", 42)),
        "batch_size": int(training.get("batch_size", 64)),
        "epochs": int(training.get("epochs", 50)),
        "learning_rate": float(training.get("learning_rate", 1e-3)),
    }
