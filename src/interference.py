from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import WorldModel


# ============================================================
# CONFIG
# ============================================================

ARTIFACT_DIR = Path(
    "artifacts"
)

SEQUENCE_LENGTH = 10
HORIZON = 3

ENTITY_COLUMN = "source_host"
TIMESTAMP_COLUMN = "timestamp"


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():

    # Model
    checkpoint = torch.load(
        ARTIFACT_DIR
        / "world_model.pt",
        map_location="cpu"
    )

    # Feature order
    with open(
        ARTIFACT_DIR
        / "feature_order.json",
        "r"
    ) as f:

        feature_order = json.load(f)

    # Scaler
    with open(
        ARTIFACT_DIR
        / "scaler.pkl",
        "rb"
    ) as f:

        scaler = pickle.load(f)

    # Stage classes
    with open(
        ARTIFACT_DIR
        / "stage_classes.json",
        "r"
    ) as f:

        stage_classes = json.load(f)

    # Build model
    model = WorldModel(
        input_size=
            checkpoint["input_size"],

        hidden_size=
            checkpoint["hidden_size"],

        num_layers=
            checkpoint["num_layers"],

        dropout=
            checkpoint["dropout"],

        horizon=
            checkpoint["horizon"],

        num_stages=
            checkpoint["num_stages"],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        scaler,
        feature_order,
        stage_classes,
    )


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(
    df: pd.DataFrame,
    source_host: str,
    feature_order,
    scaler,
):

    if ENTITY_COLUMN not in df.columns:
        raise ValueError(
            f"Missing column: {ENTITY_COLUMN}"
        )

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing column: {TIMESTAMP_COLUMN}"
        )

    # Select host
    host_df = df[
        df[ENTITY_COLUMN]
        == source_host
    ].copy()

    if len(host_df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least "
            f"{SEQUENCE_LENGTH} windows. "
            f"Found {len(host_df)}."
        )

    host_df[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        host_df[TIMESTAMP_COLUMN]
    )

    host_df = host_df.sort_values(
        TIMESTAMP_COLUMN
    )

    # Check features
    missing_features = [
        feature
        for feature in feature_order
        if feature not in host_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features: "
            + str(missing_features)
        )

    # Last 10 windows
    recent = host_df.tail(
        SEQUENCE_LENGTH
    )

    X = recent[
        feature_order
    ].to_numpy(
        dtype=np.float32
    )

    # Scale using training scaler
    X = scaler.transform(
        X
    )

    # Add batch dimension
    X = np.expand_dims(
        X,
        axis=0
    )

    return (
        X,
        recent
    )


# ============================================================
# FORECAST
# ============================================================

def forecast(
    model,
    X,
    stage_classes,
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    with torch.no_grad():

        risk_logits, stage_logits = (
            model(X_tensor)
        )

        risk_probabilities = (
            torch.sigmoid(
                risk_logits
            )
        )

        stage_probabilities = (
            torch.softmax(
                stage_logits,
                dim=1
            )
        )

    risk = (
        risk_probabilities[0]
        .numpy()
    )

    stage_index = int(
        torch.argmax(
            stage_probabilities[0]
        ).item()
    )

    stage = stage_classes[
        stage_index
    ]

    stage_confidence = float(
        stage_probabilities[0]
        [stage_index]
        .item()
    )

    return (
        risk,
        stage,
        stage_confidence
    )


# ============================================================
# CREATE JSON
# ============================================================

def create_prediction(
    source_host,
    recent_df,
    risk,
    stage,
    stage_confidence,
):

    last_timestamp = pd.to_datetime(
        recent_df[
            TIMESTAMP_COLUMN
        ].iloc[-1]
    )

    risk_timeline = {}

    for i, probability in enumerate(
        risk,
        start=1
    ):

        seconds = i * 10

        risk_timeline[
            f"+{seconds}s"
        ] = round(
            float(probability),
            4
        )

    current_risk = float(
        risk[0]
    )

    if current_risk >= 0.5:
        current_label = "HIGH_RISK"
    else:
        current_label = "LOW_RISK"

    prediction = {

        "source_host":
            source_host,

        "prediction_time":
            last_timestamp.isoformat(),

        "risk_timeline":
            risk_timeline,

        "current_risk_label":
            current_label,

        "predicted_stage":
            stage,

        "stage_confidence":
            round(
                stage_confidence,
                4
            ),

        "forecast_horizon_seconds":
            30,
    }

    return prediction


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python "
            "src/model/inference.py "
            "data/processed/"
            "feature_matrix.parquet "
            "[source_host]"
        )

        return

    data_path = Path(
        sys.argv[1]
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_parquet(
        data_path
    )

    # --------------------------------------------------------
    # DETERMINE HOST
    # --------------------------------------------------------

    if len(sys.argv) >= 3:

        source_host = sys.argv[2]

    else:

        source_host = (
            df[ENTITY_COLUMN]
            .dropna()
            .iloc[0]
        )

    print(
        "Forecasting host:",
        source_host
    )

    # --------------------------------------------------------
    # LOAD ARTIFACTS
    # --------------------------------------------------------

    (
        model,
        scaler,
        feature_order,
        stage_classes,
    ) = load_artifacts()

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    X, recent_df = prepare_input(
        df,
        source_host,
        feature_order,
        scaler,
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    (
        risk,
        stage,
        stage_confidence,
    ) = forecast(
        model,
        X,
        stage_classes,
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    prediction = create_prediction(
        source_host,
        recent_df,
        risk,
        stage,
        stage_confidence,
    )

    output_dir = Path(
        "outputs"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output_path = (
        output_dir
        / "prediction.json"
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            prediction,
            f,
            indent=2
        )

    print()
    print(
        json.dumps(
            prediction,
            indent=2
        )
    )

    print()
    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()