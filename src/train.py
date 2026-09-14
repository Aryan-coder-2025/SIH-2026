from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from sequence_builder import build_sequences
from model import WorldModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/feature_matrix.parquet"
)

ARTIFACT_DIR = Path("artifacts")

SEQUENCE_LENGTH = 10
HORIZON = 3

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

SEED = 42

ENTITY_COLUMN = "source_host"
TIMESTAMP_COLUMN = "timestamp"
RISK_COLUMN = "is_malicious"
STAGE_COLUMN = "stage"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
):

    df = df.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    n = len(df)

    train_end = int(
        n * train_ratio
    )

    validation_end = int(
        n * (
            train_ratio
            + validation_ratio
        )
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# SCALE SEQUENCES
# ============================================================

def scale_sequences(
    X_train,
    X_validation,
    X_test,
):

    n_features = X_train.shape[-1]

    scaler = StandardScaler()

    # Flatten temporal dimension temporarily
    train_2d = X_train.reshape(
        -1,
        n_features
    )

    validation_2d = X_validation.reshape(
        -1,
        n_features
    )

    test_2d = X_test.reshape(
        -1,
        n_features
    )

    # IMPORTANT:
    # fit ONLY on training data
    scaler.fit(train_2d)

    train_scaled = scaler.transform(
        train_2d
    ).reshape(
        X_train.shape
    )

    validation_scaled = scaler.transform(
        validation_2d
    ).reshape(
        X_validation.shape
    )

    test_scaled = scaler.transform(
        test_2d
    ).reshape(
        X_test.shape
    )

    return (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler
    )


# ============================================================
# PREPARE STAGE LABELS
# ============================================================

def encode_stages(
    y_train_stage,
    y_validation_stage,
    y_test_stage,
):

    encoder = LabelEncoder()

    encoder.fit(
        y_train_stage
    )

    y_train = encoder.transform(
        y_train_stage
    )

    y_validation = encoder.transform(
        y_validation_stage
    )

    y_test = encoder.transform(
        y_test_stage
    )

    return (
        y_train,
        y_validation,
        y_test,
        encoder
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    risk_loss_fn,
    stage_loss_fn,
    device,
):

    model.train()

    total_loss = 0.0

    for X, y_risk, y_stage in loader:

        X = X.to(device)

        y_risk = y_risk.to(device)

        y_stage = y_stage.to(device)

        optimizer.zero_grad()

        risk_logits, stage_logits = model(X)

        # Binary risk forecasting
        risk_loss = risk_loss_fn(
            risk_logits,
            y_risk
        )

        # Stage classification
        stage_loss = stage_loss_fn(
            stage_logits,
            y_stage
        )

        # Combined objective
        loss = (
            risk_loss
            + stage_loss
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================

def evaluate_loss(
    model,
    loader,
    risk_loss_fn,
    stage_loss_fn,
    device,
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for X, y_risk, y_stage in loader:

            X = X.to(device)

            y_risk = y_risk.to(device)

            y_stage = y_stage.to(device)

            risk_logits, stage_logits = model(X)

            risk_loss = risk_loss_fn(
                risk_logits,
                y_risk
            )

            stage_loss = stage_loss_fn(
                stage_logits,
                y_stage
            )

            loss = (
                risk_loss
                + stage_loss
            )

            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()

    print("Device:", device)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        f"Loading: {DATA_PATH}"
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN]
    )

    df = df.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    excluded_columns = {
        ENTITY_COLUMN,
        TIMESTAMP_COLUMN,
        RISK_COLUMN,
        STAGE_COLUMN,
    }

    feature_columns = [
        col
        for col in df.columns
        if col not in excluded_columns
    ]

    # Keep only numeric features
    feature_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    print(
        "Number of features:",
        len(feature_columns)
    )

    print(
        "Features:",
        feature_columns
    )

    # --------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------

    train_df, validation_df, test_df = (
        temporal_split(df)
    )

    print(
        "Train rows:",
        len(train_df)
    )

    print(
        "Validation rows:",
        len(validation_df)
    )

    print(
        "Test rows:",
        len(test_df)
    )

    # --------------------------------------------------------
    # BUILD SEQUENCES
    # --------------------------------------------------------

    print(
        "Building training sequences..."
    )

    X_train, y_train_risk, y_train_stage = (
        build_sequences(
            train_df,
            feature_columns,
            sequence_length=SEQUENCE_LENGTH,
            horizon=HORIZON,
            entity_column=ENTITY_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN,
            risk_column=RISK_COLUMN,
            stage_column=STAGE_COLUMN,
        )
    )

    X_validation, y_validation_risk, y_validation_stage = (
        build_sequences(
            validation_df,
            feature_columns,
            sequence_length=SEQUENCE_LENGTH,
            horizon=HORIZON,
            entity_column=ENTITY_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN,
            risk_column=RISK_COLUMN,
            stage_column=STAGE_COLUMN,
        )
    )

    X_test, y_test_risk, y_test_stage = (
        build_sequences(
            test_df,
            feature_columns,
            sequence_length=SEQUENCE_LENGTH,
            horizon=HORIZON,
            entity_column=ENTITY_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN,
            risk_column=RISK_COLUMN,
            stage_column=STAGE_COLUMN,
        )
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "Risk target:",
        y_train_risk.shape
    )

    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        scaler,
    ) = scale_sequences(
        X_train,
        X_validation,
        X_test,
    )

    # --------------------------------------------------------
    # ENCODE STAGES
    # --------------------------------------------------------

    (
        y_train_stage,
        y_validation_stage,
        y_test_stage,
        stage_encoder,
    ) = encode_stages(
        y_train_stage,
        y_validation_stage,
        y_test_stage,
    )

    num_stages = len(
        stage_encoder.classes_
    )

    print(
        "Stage classes:",
        list(
            stage_encoder.classes_
        )
    )

    # --------------------------------------------------------
    # PYTORCH DATA
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train_risk,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train_stage,
            dtype=torch.long
        ),
    )

    validation_dataset = TensorDataset(
        torch.tensor(
            X_validation,
            dtype=torch.float32
        ),
        torch.tensor(
            y_validation_risk,
            dtype=torch.float32
        ),
        torch.tensor(
            y_validation_stage,
            dtype=torch.long
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = WorldModel(
        input_size=len(
            feature_columns
        ),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
        num_stages=num_stages,
    ).to(device)

    print(model)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    risk_loss_fn = nn.BCEWithLogitsLoss()

    stage_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_state = None

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            risk_loss_fn,
            stage_loss_fn,
            device,
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            risk_loss_fn,
            stage_loss_fn,
            device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    model_path = (
        ARTIFACT_DIR
        / "world_model.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_size":
                len(feature_columns),

            "hidden_size":
                HIDDEN_SIZE,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT,

            "horizon":
                HORIZON,

            "num_stages":
                num_stages,
        },
        model_path,
    )

    # --------------------------------------------------------
    # SAVE SCALER
    # --------------------------------------------------------

    with open(
        ARTIFACT_DIR / "scaler.pkl",
        "wb"
    ) as f:

        pickle.dump(
            scaler,
            f
        )

    # --------------------------------------------------------
    # SAVE FEATURE ORDER
    # --------------------------------------------------------

    with open(
        ARTIFACT_DIR
        / "feature_order.json",
        "w"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # SAVE STAGE CLASSES
    # --------------------------------------------------------

    with open(
        ARTIFACT_DIR
        / "stage_classes.json",
        "w"
    ) as f:

        json.dump(
            stage_encoder.classes_.tolist(),
            f,
            indent=2
        )

    # --------------------------------------------------------
    # SAVE LABEL ENCODER
    # --------------------------------------------------------

    with open(
        ARTIFACT_DIR
        / "label_encoder.pkl",
        "wb"
    ) as f:

        pickle.dump(
            stage_encoder,
            f
        )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {
        "dataset":
            "CSE-CIC-IDS2018",

        "window_seconds":
            10,

        "sequence_length":
            SEQUENCE_LENGTH,

        "forecast_horizon":
            HORIZON,

        "entity":
            ENTITY_COLUMN,

        "feature_count":
            len(feature_columns),

        "seed":
            SEED,
    }

    with open(
        ARTIFACT_DIR
        / "metadata.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print(
        "Training complete."
    )

    print(
        "Saved:",
        model_path
    )


if __name__ == "__main__":
    main()