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

try:
    from src.config import get_temporal_config
    from src.model import WorldModel
    from src.schemas.features import CANONICAL_MODEL_FEATURE_NAMES, validate_feature_names
    from src.temporal.sequences import build_sequences_from_dataframe
    from src.temporal.split import temporal_split_by_time
except ImportError:
    from config import get_temporal_config  # type: ignore
    from model import WorldModel  # type: ignore
    from schemas.features import CANONICAL_MODEL_FEATURE_NAMES, validate_feature_names  # type: ignore
    from temporal.sequences import build_sequences_from_dataframe  # type: ignore
    from temporal.split import temporal_split_by_time  # type: ignore


# ============================================================
# CONFIGURATION
# ============================================================

LAMBDA_STAGE: float = 0.3  # Documented auxiliary loss weight for stage classification head

DATA_PATH = Path(
    "data/processed/feature_matrix.parquet"
)

ARTIFACT_DIR = Path("artifacts")

TEMPORAL_CONFIG = get_temporal_config()
SEQUENCE_LENGTH = TEMPORAL_CONFIG.history_length
HORIZON = TEMPORAL_CONFIG.forecast_horizon_windows

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


def select_canonical_model_features(df: pd.DataFrame) -> list[str]:
    """
    Return the authoritative 41-feature model order without dataframe inference.
    """
    feature_columns = list(CANONICAL_MODEL_FEATURE_NAMES)
    validate_feature_names(
        feature_columns,
        expected_order=CANONICAL_MODEL_FEATURE_NAMES,
    )
    if len(feature_columns) != 41:
        raise ValueError(
            f"Canonical model feature contract must contain 41 features, got {len(feature_columns)}"
        )

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical model feature column(s): {missing}")

    return feature_columns


def split_dataframe_temporally(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
):
    """
    DataFrame adapter for src.temporal.split.temporal_split_by_time.

    Training must split raw windows before sequence construction and must not
    maintain an independent temporal split implementation.
    """
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Missing timestamp column: {TIMESTAMP_COLUMN}")

    split = temporal_split_by_time(
        df.to_dict("records"),
        train_ratio=train_ratio,
        val_ratio=validation_ratio,
        timestamp_extractor=lambda row: pd.Timestamp(row[TIMESTAMP_COLUMN]).to_pydatetime(),
    )

    return (
        pd.DataFrame(split.train),
        pd.DataFrame(split.val),
        pd.DataFrame(split.test),
        split,
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

        # Combined objective (primary risk forecasting + auxiliary stage classification)
        loss = (
            risk_loss
            + LAMBDA_STAGE * stage_loss
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
                + LAMBDA_STAGE * stage_loss
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

    feature_columns = select_canonical_model_features(df)

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

    train_df, validation_df, test_df, split_info = (
        split_dataframe_temporally(df)
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
        build_sequences_from_dataframe(
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
        build_sequences_from_dataframe(
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
        build_sequences_from_dataframe(
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

            "project_id":
                "SIH26153",

            "feature_order":
                feature_columns,

            "feature_count":
                len(feature_columns),

            "input_size":
                len(feature_columns),

            "hidden_size":
                HIDDEN_SIZE,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT,

            "sequence_length":
                SEQUENCE_LENGTH,

            "horizon":
                HORIZON,

            "forecast_offsets_seconds":
                list(TEMPORAL_CONFIG.forecast_offsets_seconds),

            "dataset":
                "CSE-CIC-IDS2018",

            "split_cutoffs": {
                "train_end_time": split_info.train_end_time.isoformat()
                if split_info.train_end_time is not None else None,
                "val_start_time": split_info.val_start_time.isoformat()
                if split_info.val_start_time is not None else None,
                "val_end_time": split_info.val_end_time.isoformat()
                if split_info.val_end_time is not None else None,
                "test_start_time": split_info.test_start_time.isoformat()
                if split_info.test_start_time is not None else None,
            },

            "seed":
                SEED,

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
        "project_id":
            "SIH26153",

        "dataset":
            "CSE-CIC-IDS2018",

        "dataset_status":
            "Configured dataset identity only; predictive performance is not established until a real training/evaluation run is completed.",

        "window_seconds":
            TEMPORAL_CONFIG.window_seconds,

        "sequence_length":
            SEQUENCE_LENGTH,

        "forecast_horizon":
            HORIZON,

        "forecast_offsets_seconds":
            list(TEMPORAL_CONFIG.forecast_offsets_seconds),

        "entity":
            ENTITY_COLUMN,

        "feature_count":
            len(feature_columns),

        "feature_order":
            feature_columns,

        "split_cutoffs": {
            "train_end_time": split_info.train_end_time.isoformat()
            if split_info.train_end_time is not None else None,
            "val_start_time": split_info.val_start_time.isoformat()
            if split_info.val_start_time is not None else None,
            "val_end_time": split_info.val_end_time.isoformat()
            if split_info.val_end_time is not None else None,
            "test_start_time": split_info.test_start_time.isoformat()
            if split_info.test_start_time is not None else None,
        },

        "model_architecture": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "num_stages": num_stages,
        },

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
