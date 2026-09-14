"""
Inference Pipeline Module for WorldModel Cyber Risk Forecasting.

Loads trained model artifacts and generates multi-horizon (+10s, +20s, +30s)
risk forecasts and auxiliary stage predictions for active source hosts.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from src.model import WorldModel
except ImportError:
    from model import WorldModel  # type: ignore

# ============================================================
# CONFIG
# ============================================================

ARTIFACT_DIR = Path("artifacts")
SEQUENCE_LENGTH = 10
HORIZON = 3

ENTITY_COLUMN = "source_host"
TIMESTAMP_COLUMN = "timestamp"


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts(artifact_dir: Path | str = ARTIFACT_DIR):
    """
    Load trained WorldModel checkpoint, scaler, feature ordering, and classes.
    """
    art_path = Path(artifact_dir)

    checkpoint_path = art_path / "world_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    with open(art_path / "feature_order.json", "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    with open(art_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    stage_classes = ["benign", "malicious"]
    stage_classes_path = art_path / "stage_classes.json"
    if stage_classes_path.is_file():
        with open(stage_classes_path, "r", encoding="utf-8") as f:
            stage_classes = json.load(f)

    model = WorldModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint.get("hidden_size", 128),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.2),
        horizon=checkpoint.get("horizon", 3),
        num_stages=checkpoint.get("num_stages", len(stage_classes)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler, feature_order, stage_classes


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(
    df: pd.DataFrame,
    source_host: str,
    feature_order: list[str],
    scaler: Any,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract the most recent sequence_length windows for source_host and scale.
    """
    if ENTITY_COLUMN not in df.columns:
        # Fallback to src_ip if source_host is not present
        if "src_ip" in df.columns:
            entity_col = "src_ip"
        else:
            raise ValueError(f"Missing entity column: {ENTITY_COLUMN}")
    else:
        entity_col = ENTITY_COLUMN

    host_df = df[df[entity_col] == source_host].copy()

    if len(host_df) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} windows for host '{source_host}'. "
            f"Found {len(host_df)}."
        )

    if TIMESTAMP_COLUMN in host_df.columns:
        host_df[TIMESTAMP_COLUMN] = pd.to_datetime(host_df[TIMESTAMP_COLUMN])
        host_df = host_df.sort_values(TIMESTAMP_COLUMN)

    missing_features = [f for f in feature_order if f not in host_df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    recent = host_df.tail(sequence_length)
    X = recent[feature_order].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X)
    X_batch = np.expand_dims(X_scaled, axis=0)

    return X_batch, recent


# ============================================================
# FORECAST
# ============================================================

def forecast(
    model: WorldModel,
    X: np.ndarray,
    stage_classes: Sequence[str],
) -> tuple[np.ndarray, str, float]:
    """
    Generate calibrated future risk probabilities and auxiliary stage prediction.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        risk_logits, stage_logits = model(X_tensor)
        risk_probabilities = torch.sigmoid(risk_logits)
        stage_probabilities = torch.softmax(stage_logits, dim=1)

    risk = risk_probabilities[0].numpy()
    stage_idx = int(torch.argmax(stage_probabilities[0]).item())
    stage = stage_classes[stage_idx] if stage_idx < len(stage_classes) else "unknown"
    stage_conf = float(stage_probabilities[0][stage_idx].item())

    return risk, stage, stage_conf
