from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .config import DEFAULT_CONFIG_PATH, get_temporal_config, load_config
    from .model import RiskLSTM
    from .schema import CANONICAL_FEATURES, SCHEMA_VERSION, validate_feature_order
except ImportError:
    from config import DEFAULT_CONFIG_PATH, get_temporal_config, load_config
    from model import RiskLSTM
    from schema import CANONICAL_FEATURES, SCHEMA_VERSION, validate_feature_order


def _temporal_contract(config_path=DEFAULT_CONFIG_PATH) -> dict:
    return get_temporal_config(load_config(config_path))


def load_model(model_path: str, device, config_path=DEFAULT_CONFIG_PATH) -> RiskLSTM:
    """Load and validate a trusted checkpoint against the current schema/config."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid model checkpoint: expected a dictionary.")

    required = {
        "model_state_dict", "input_size", "hidden_size",
        "num_layers", "dropout", "horizon", "schema_version",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Invalid model checkpoint. Missing keys: {sorted(missing)}")

    config = _temporal_contract(config_path)
    if checkpoint["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Model schema version does not match current schema.")
    if checkpoint["input_size"] != len(CANONICAL_FEATURES):
        raise ValueError("Model input feature count does not match current schema.")
    if checkpoint["horizon"] != config["horizon"]:
        raise ValueError("Model horizon does not match project configuration.")

    checkpoint_sequence_length = int(checkpoint.get("sequence_length", config["sequence_length"]))
    if checkpoint_sequence_length != config["sequence_length"]:
        raise ValueError("Model sequence length does not match project configuration.")

    saved_order = checkpoint.get("feature_order")
    if saved_order is not None:
        validate_feature_order(list(saved_order))

    saved_offsets = checkpoint.get("forecast_offsets_seconds")
    if saved_offsets is not None and list(saved_offsets) != config["forecast_offsets_seconds"]:
        raise ValueError("Checkpoint forecast offsets do not match project configuration.")

    model = RiskLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        horizon=int(checkpoint["horizon"]),
        sequence_length=checkpoint_sequence_length,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_scaler(scaler_path: str):
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    with open(path, "rb") as file:
        scaler = pickle.load(file)

    if not hasattr(scaler, "transform") or not hasattr(scaler, "n_features_in_"):
        raise ValueError("Invalid scaler artifact.")
    if scaler.n_features_in_ != len(CANONICAL_FEATURES):
        raise ValueError("Scaler feature count does not match current schema.")
    return scaler


def predict_risk(
    model,
    scaler,
    X,
    device,
    sequence_length: int | None = None,
    horizon: int | None = None,
    config_path=DEFAULT_CONFIG_PATH,
):
    """Validate, scale and predict exactly the configured forecast horizons."""
    contract = _temporal_contract(config_path)
    sequence_length = contract["sequence_length"] if sequence_length is None else sequence_length
    horizon = contract["horizon"] if horizon is None else horizon

    if horizon != contract["horizon"] or sequence_length != contract["sequence_length"]:
        raise ValueError("Inference sequence/horizon does not match project configuration.")

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Inference input must be 2D, received {X.shape}.")
    if X.shape != (sequence_length, len(CANONICAL_FEATURES)):
        raise ValueError(
            f"Inference input must have shape "
            f"({sequence_length}, {len(CANONICAL_FEATURES)}), received {X.shape}."
        )
    if not np.isfinite(X).all():
        raise ValueError("Inference input contains NaN or Inf.")

    X_scaled = np.asarray(scaler.transform(X), dtype=np.float32)
    if X_scaled.shape != X.shape or not np.isfinite(X_scaled).all():
        raise ValueError("Scaled inference input is malformed or contains NaN/Inf.")

    tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)

    if not isinstance(logits, torch.Tensor) or logits.shape != (1, horizon):
        raise ValueError(f"Malformed model output; expected (1, {horizon}).")
    if not torch.isfinite(logits).all():
        raise ValueError("Model produced NaN or Inf logits.")

    probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    if probabilities.shape != (horizon,) or not np.isfinite(probabilities).all():
        raise ValueError("Malformed probability output.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Model produced probabilities outside [0,1].")
    return probabilities


def format_prediction(
    source_host: str,
    prediction_time,
    probabilities,
    forecast_offsets_seconds=None,
    config_path=DEFAULT_CONFIG_PATH,
):
    """Create the dashboard/API prediction contract."""
    contract = _temporal_contract(config_path)
    offsets = contract["forecast_offsets_seconds"] if forecast_offsets_seconds is None else list(forecast_offsets_seconds)
    probabilities = np.asarray(probabilities, dtype=np.float32)

    if len(offsets) != contract["horizon"] or offsets != contract["forecast_offsets_seconds"]:
        raise ValueError("Forecast offsets do not match project configuration.")
    if probabilities.shape != (contract["horizon"],) or not np.isfinite(probabilities).all():
        raise ValueError("Prediction must contain exactly the configured finite probabilities.")
    if not isinstance(source_host, str) or not source_host.strip():
        raise ValueError("source_host must be a non-empty string.")

    return {
        "source_host": source_host,
        "prediction_time": str(pd.Timestamp(prediction_time)),
        **{f"risk_{offset}s": float(prob) for offset, prob in zip(offsets, probabilities)},
    }


def create_prediction_records(
    source_host: str,
    prediction_time,
    probabilities,
    config_path=DEFAULT_CONFIG_PATH,
):
    """Create one evaluation record per configured forecast horizon."""
    contract = _temporal_contract(config_path)
    offsets = contract["forecast_offsets_seconds"]
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (contract["horizon"],):
        raise ValueError("Exactly the configured number of probabilities is required.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction contains NaN or Inf.")

    prediction_time = pd.Timestamp(prediction_time)
    records = []
    for offset, probability in zip(offsets, probabilities):
        records.append({
            "source_host": source_host,
            "prediction_time": str(prediction_time),
            "target_time": str(prediction_time + pd.Timedelta(seconds=offset)),
            "horizon_seconds": int(offset),
            "predicted_risk": float(probability),
        })
    return records
