"""

Expected target:

    y = [
        malicious at t + 10 sec,
        malicious at t + 20 sec,
        malicious at t + 30 sec
    ]

The model therefore predicts exactly 3 future risk values.
"""

from pathlib import Path
import json
import pickle
import random
import platform
import sys
import hashlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ================================================================
# IMPORTS
# ================================================================

try:
    # Package-safe imports
    from .config import load_config, get_temporal_config
    from .schema import (
        get_features,
        CANONICAL_FEATURES,
        SCHEMA_VERSION,
    )
    from .sequence_builder import build_sequences
    from .model import RiskLSTM

except ImportError:
    # Allows:
    # python src/train.py
    from config import load_config, get_temporal_config
    from schema import (
        get_features,
        CANONICAL_FEATURES,
        SCHEMA_VERSION,
    )
    from sequence_builder import build_sequences
    from model import RiskLSTM


# ================================================================
# PATHS
# ================================================================

DATA_PATH = Path(
    "data/processed/feature_matrix.parquet"
)

ARTIFACT_DIR = Path(
    "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR / "risk_lstm.pt"
)

SCALER_PATH = (
    ARTIFACT_DIR / "scaler.pkl"
)

FEATURE_ORDER_PATH = (
    ARTIFACT_DIR / "feature_order.json"
)

METADATA_PATH = (
    ARTIFACT_DIR / "metadata.json"
)

TEST_PREDICTIONS_PATH = (
    ARTIFACT_DIR / "test_predictions.csv"
)


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Try to make CUDA operations deterministic.
    # Some operations may still have limitations depending
    # on the installed CUDA/PyTorch version.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ================================================================
# DEVICE
# ================================================================

def get_device():
    """
    Use GPU when CUDA is available.
    Otherwise use CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ================================================================
# DATA VALIDATION
# ================================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate the processed feature matrix before training.

    The LSTM receives ONLY the canonical traffic features.
    Target/label columns are never allowed to become model features.
    """

    if df.empty:
        raise ValueError("The input feature matrix is empty.")

    required_columns = {"timestamp", "is_malicious"}
    missing_required = required_columns - set(df.columns)

    if missing_required:
        raise ValueError(
            f"Missing required columns: {sorted(missing_required)}"
        )

    # The processed matrix must contain exactly the canonical model
    # features plus timestamp and the target. This prevents accidental
    # leakage from columns such as risk_score, stage, future labels, etc.
    allowed_columns = (
        {"timestamp", "is_malicious"}
        | set(CANONICAL_FEATURES)
    )

    unexpected_columns = sorted(
        set(df.columns) - allowed_columns
    )

    if unexpected_columns:
        raise ValueError(
            "Unexpected columns found in feature matrix. "
            "Remove target/derived columns or update the canonical "
            f"schema explicitly: {unexpected_columns}"
        )

    # Timestamp validation
    timestamp = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_timestamps = int(timestamp.isna().sum())

    if invalid_timestamps > 0:
        raise ValueError(
            f"Found {invalid_timestamps} invalid timestamps."
        )

    # Use normalized datetime values from the validated conversion.
    df["timestamp"] = timestamp

    # Network-level feature matrix: exactly one row per time window.
    duplicate_timestamps = int(
        df["timestamp"].duplicated().sum()
    )

    if duplicate_timestamps > 0:
        raise ValueError(
            "Duplicate timestamp/window rows found: "
            f"{duplicate_timestamps}. "
            "The feature matrix must contain one row per network window."
        )

    # Target validation
    target_values = set(
        pd.Series(df["is_malicious"])
        .dropna()
        .unique()
        .tolist()
    )

    if not target_values.issubset({0, 1, False, True}):
        raise ValueError(
            "is_malicious must contain only 0/1 values. "
            f"Found values: {target_values}"
        )

    if df["is_malicious"].isna().any():
        raise ValueError("is_malicious contains NaN values.")

    # Explicit canonical feature validation.
    missing_features = [
        feature
        for feature in CANONICAL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing canonical features: "
            f"{missing_features}"
        )

    feature_values = df[
        CANONICAL_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if feature_values.isna().any().any():
        bad_columns = (
            feature_values.columns[
                feature_values.isna().any()
            ].tolist()
        )

        raise ValueError(
            "NaN/non-numeric values found in "
            f"features: {bad_columns}"
        )

    if not np.isfinite(
        feature_values.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Inf/-Inf values found in canonical features."
        )

    # Validate chronological order and exact configured window spacing.
    ordered_timestamps = (
        df["timestamp"]
        .sort_values()
        .reset_index(drop=True)
    )

    if not ordered_timestamps.equals(
        df["timestamp"].reset_index(drop=True)
    ):
        raise ValueError(
            "Feature matrix timestamps are not in chronological order."
        )

    # The sequence builder assumes a fixed-width temporal grid.
    # A 10-second value here is checked against the configured value
    # later in main(); this function only verifies monotonicity.
    deltas = ordered_timestamps.diff().dropna()

    if not deltas.empty and (deltas <= pd.Timedelta(0)).any():
        raise ValueError(
            "Timestamps must be strictly increasing."
        )


def validate_temporal_grid(
    df: pd.DataFrame,
    window_seconds: int,
) -> None:
    """
    Verify that the processed network feature matrix is a continuous
    fixed-size temporal grid.

    This prevents sequences from silently crossing missing windows.
    """

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    ).sort_values().reset_index(drop=True)

    if len(timestamps) <= 1:
        return

    deltas = timestamps.diff().dropna()
    expected = pd.Timedelta(seconds=window_seconds)

    invalid = deltas != expected

    if invalid.any():
        bad_count = int(invalid.sum())
        examples = deltas[invalid].head(5).astype(str).tolist()

        raise ValueError(
            f"Temporal grid is not continuous at {bad_count} "
            f"interval(s). Expected exactly {window_seconds}s spacing. "
            f"Example intervals: {examples}"
        )


# ================================================================
# TEMPORAL SPLIT
# ================================================================

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    timestamp_column: str = "timestamp",
):
    """
    Split data chronologically.

    No random split is performed. Each split is an independent temporal
    region, and sequence history is not borrowed across split boundaries.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    df = (
        df.sort_values(timestamp_column)
        .reset_index(drop=True)
    )

    total_rows = len(df)

    train_end = int(total_rows * train_ratio)
    validation_end = int(
        total_rows * (train_ratio + validation_ratio)
    )

    if train_end <= 0:
        raise ValueError("Training split contains no rows.")

    if validation_end <= train_end:
        raise ValueError(
            "Validation split contains no rows."
        )

    if validation_end >= total_rows:
        raise ValueError("Test split contains no rows.")

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    # Explicitly verify temporal ordering and no overlap.
    if train_df[timestamp_column].max() >= validation_df[timestamp_column].min():
        raise ValueError("Train/validation temporal overlap detected.")

    if validation_df[timestamp_column].max() >= test_df[timestamp_column].min():
        raise ValueError("Validation/test temporal overlap detected.")

    return train_df, validation_df, test_df


# ================================================================
# LEAKAGE / SPLIT INTEGRITY VALIDATION
# ================================================================

def validate_split_isolation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> None:
    """Prove that raw rows cannot be shared between temporal partitions."""
    names = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }
    timestamps = {}
    for name, frame in names.items():
        if timestamp_column not in frame.columns:
            raise ValueError(f"{name} split is missing {timestamp_column!r}.")
        values = pd.to_datetime(frame[timestamp_column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{name} split contains invalid timestamps.")
        timestamps[name] = set(values.tolist())

    if timestamps["train"] & timestamps["validation"]:
        raise ValueError("Train/validation raw-window overlap detected.")
    if timestamps["train"] & timestamps["test"]:
        raise ValueError("Train/test raw-window overlap detected.")
    if timestamps["validation"] & timestamps["test"]:
        raise ValueError("Validation/test raw-window overlap detected.")

    if not train_df.empty and not validation_df.empty:
        if pd.Timestamp(train_df[timestamp_column].max()) >= pd.Timestamp(validation_df[timestamp_column].min()):
            raise ValueError("Train/validation temporal boundary is not strictly ordered.")
    if not validation_df.empty and not test_df.empty:
        if pd.Timestamp(validation_df[timestamp_column].max()) >= pd.Timestamp(test_df[timestamp_column].min()):
            raise ValueError("Validation/test temporal boundary is not strictly ordered.")


def validate_no_sequence_overlap(
    split_a: pd.DataFrame,
    split_b: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> None:
    """Verify two sequence source regions share no raw temporal windows."""
    a = set(pd.to_datetime(split_a[timestamp_column], errors="raise").tolist())
    b = set(pd.to_datetime(split_b[timestamp_column], errors="raise").tolist())
    if a & b:
        raise ValueError("Temporal sequence source windows overlap between splits.")


# ================================================================
# SEQUENCE VALIDATION
# ================================================================

def validate_sequences(
    X,
    y,
    sequence_length: int,
    feature_count: int,
    horizon: int,
    split_name: str,
):
    """
    Validate generated temporal sequence arrays.
    """

    if X is None or y is None:
        raise ValueError(
            f"{split_name}: sequence builder returned None."
        )

    X = np.asarray(X)
    y = np.asarray(y)

    expected_x_dimensions = (
        3
    )

    expected_y_dimensions = (
        2
    )

    if X.ndim != expected_x_dimensions:
        raise ValueError(
            f"{split_name}: X must be 3D "
            f"(samples, time, features). "
            f"Got shape {X.shape}."
        )

    if y.ndim != expected_y_dimensions:
        raise ValueError(
            f"{split_name}: y must be 2D "
            f"(samples, horizon). "
            f"Got shape {y.shape}."
        )

    if X.shape[1] != sequence_length:
        raise ValueError(
            f"{split_name}: wrong sequence length. "
            f"Expected {sequence_length}, "
            f"got {X.shape[1]}."
        )

    if X.shape[2] != feature_count:
        raise ValueError(
            f"{split_name}: wrong feature count. "
            f"Expected {feature_count}, "
            f"got {X.shape[2]}."
        )

    if y.shape[1] != horizon:
        raise ValueError(
            f"{split_name}: wrong horizon count. "
            f"Expected {horizon}, "
            f"got {y.shape[1]}."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split_name}: X/y sample count mismatch. "
            f"X={X.shape[0]}, y={y.shape[0]}."
        )

    if X.shape[0] == 0:
        raise ValueError(
            f"{split_name}: no temporal sequences were created."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{split_name}: X contains NaN or Inf."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{split_name}: y contains NaN or Inf."
        )

    if not np.isin(
        y,
        [0, 1]
    ).all():
        raise ValueError(
            f"{split_name}: targets must be binary 0/1."
        )


# ================================================================
# SCALING
# ================================================================
def scale_sequences(
    X_train,
    X_validation,
    X_test,
):
    """
    Scale features using statistics learned ONLY from
    the training set.

    Validation and test data are transformed using
    the same scaler.

    This prevents data leakage.
    """

    # ------------------------------------------------------------
    # 1. Convert to NumPy arrays
    # ------------------------------------------------------------

    X_train = np.asarray(X_train)
    X_validation = np.asarray(X_validation)
    X_test = np.asarray(X_test)

    if X_train.size == 0 or X_validation.size == 0 or X_test.size == 0:
        raise ValueError("Training, validation, and test sequences must not be empty.")

    # ------------------------------------------------------------
    # 2. Validate dimensions
    # ------------------------------------------------------------

    if X_train.ndim != 3:
        raise ValueError(
            "X_train must have shape "
            "(samples, sequence_length, features). "
            f"Got {X_train.shape}"
        )

    if X_validation.ndim != 3:
        raise ValueError(
            "X_validation must have shape "
            "(samples, sequence_length, features). "
            f"Got {X_validation.shape}"
        )

    if X_test.ndim != 3:
        raise ValueError(
            "X_test must have shape "
            "(samples, sequence_length, features). "
            f"Got {X_test.shape}"
        )

    # ------------------------------------------------------------
    # 3. Check feature count
    # ------------------------------------------------------------

    number_of_features = X_train.shape[-1]

    if X_validation.shape[-1] != number_of_features:
        raise ValueError(
            "Validation feature count does not match "
            "training feature count. "
            f"Train={number_of_features}, "
            f"Validation={X_validation.shape[-1]}"
        )

    if X_test.shape[-1] != number_of_features:
        raise ValueError(
            "Test feature count does not match "
            "training feature count. "
            f"Train={number_of_features}, "
            f"Test={X_test.shape[-1]}"
        )

    # ------------------------------------------------------------
    # 4. Check sequence length
    # ------------------------------------------------------------

    if X_validation.shape[1] != X_train.shape[1]:
        raise ValueError(
            "Validation sequence length does not "
            "match training sequence length."
        )

    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError(
            "Test sequence length does not "
            "match training sequence length."
        )

    # ------------------------------------------------------------
    # 5. Check for NaN / Inf BEFORE scaling
    # ------------------------------------------------------------

    if not np.isfinite(X_train).all():
        raise ValueError(
            "X_train contains NaN or Inf values."
        )

    if not np.isfinite(X_validation).all():
        raise ValueError(
            "X_validation contains NaN or Inf values."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "X_test contains NaN or Inf values."
        )

    # ------------------------------------------------------------
    # 6. Create scaler
    # ------------------------------------------------------------

    scaler = StandardScaler()

    # StandardScaler expects:

    #     (rows, features)

    # But our LSTM data is:

    #     (samples, sequence_length, features)

    # Therefore temporarily flatten the first two dimensions.

    train_2d = X_train.reshape(
        -1,
        number_of_features
    )

    validation_2d = X_validation.reshape(
        -1,
        number_of_features
    )

    test_2d = X_test.reshape(
        -1,
        number_of_features
    )

    # ------------------------------------------------------------
    # 7. FIT ONLY ON TRAINING DATA
    # ------------------------------------------------------------

    # IMPORTANT:
    #
    # DO NOT do:
    #
    # scaler.fit(validation_2d)
    # scaler.fit(test_2d)
    #
    # because that would leak future information.
    #

    scaler.fit(
        train_2d
    )

    # ------------------------------------------------------------
    # 8. Transform training data
    # ------------------------------------------------------------

    train_scaled_2d = scaler.transform(
        train_2d
    )

    train_scaled = train_scaled_2d.reshape(
        X_train.shape
    )

    # ------------------------------------------------------------
    # 9. Transform validation data
    # ------------------------------------------------------------

    validation_scaled_2d = scaler.transform(
        validation_2d
    )

    validation_scaled = validation_scaled_2d.reshape(
        X_validation.shape
    )

    # ------------------------------------------------------------
    # 10. Transform test data
    # ------------------------------------------------------------

    test_scaled_2d = scaler.transform(
        test_2d
    )

    test_scaled = test_scaled_2d.reshape(
        X_test.shape
    )

    # ------------------------------------------------------------
    # 11. Check scaled output
    # ------------------------------------------------------------

    if not np.isfinite(
        train_scaled
    ).all():
        raise ValueError(
            "Scaled training data contains NaN or Inf."
        )

    if not np.isfinite(
        validation_scaled
    ).all():
        raise ValueError(
            "Scaled validation data contains NaN or Inf."
        )

    if not np.isfinite(
        test_scaled
    ).all():
        raise ValueError(
            "Scaled test data contains NaN or Inf."
        )

    # ------------------------------------------------------------
    # 12. Final shape verification
    # ------------------------------------------------------------

    if train_scaled.shape != X_train.shape:
        raise ValueError(
            "Training shape changed unexpectedly "
            "during scaling."
        )

    if validation_scaled.shape != X_validation.shape:
        raise ValueError(
            "Validation shape changed unexpectedly "
            "during scaling."
        )

    if test_scaled.shape != X_test.shape:
        raise ValueError(
            "Test shape changed unexpectedly "
            "during scaling."
        )

    return (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler,
    )


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    device,
):
    """
    Train the model for one complete epoch.
    """

    model.train()

    total_loss = 0.0
    batch_count = 0

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        if logits.shape != y.shape:
            raise ValueError(
                "Model output shape does not match target shape. "
                f"logits={logits.shape}, y={y.shape}"
            )

        if not torch.isfinite(
            logits
        ).all():
            raise ValueError(
                "Model produced NaN/Inf logits."
            )

        loss = loss_function(
            logits,
            y
        )

        if not torch.isfinite(loss):
            raise ValueError(
                "Training loss became NaN/Inf."
            )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        batch_count += 1

    if batch_count == 0:
        raise ValueError(
            "Training DataLoader contains no batches."
        )

    return (
        total_loss / batch_count
    )


# ================================================================
# VALIDATION / TEST LOSS
# ================================================================

def evaluate(
    model,
    loader,
    loss_function,
    device,
):
    """
    Calculate loss without updating model weights.
    """

    model.eval()

    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            if logits.shape != y.shape:
                raise ValueError(
                    "Evaluation output shape mismatch. "
                    f"logits={logits.shape}, y={y.shape}"
                )

            loss = loss_function(
                logits,
                y
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    "Evaluation loss became NaN/Inf."
                )

            total_loss += loss.item()

            batch_count += 1

    if batch_count == 0:
        raise ValueError(
            "Evaluation DataLoader contains no batches."
        )

    return (
        total_loss / batch_count
    )


# ================================================================
# PREDICTIONS
# ================================================================

def predict(
    model,
    loader,
    device,
):
    """
    Generate sigmoid probabilities for all samples.
    """

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            logits = model(X)

            if logits.ndim != 2:
                raise ValueError(
                    "Model output must be 2D "
                    "(batch, horizon). "
                    f"Got {logits.shape}."
                )

            if logits.shape[1] != y.shape[1]:
                raise ValueError(
                    "Model output horizon does not match target horizon."
                )

            probabilities = torch.sigmoid(logits)

            if not torch.isfinite(
                probabilities
            ).all():
                raise ValueError(
                    "Model generated NaN/Inf probabilities."
                )

            predictions.append(
                probabilities.cpu().numpy()
            )

            targets.append(
                y.numpy()
            )

    if not predictions:
        raise ValueError(
            "No predictions were generated."
        )

    predictions = np.concatenate(
        predictions,
        axis=0
    )

    targets = np.concatenate(
        targets,
        axis=0
    )

    return (
        predictions,
        targets,
    )


# ================================================================
# TEST METRICS
# ================================================================

def calculate_test_metrics(
    predictions,
    targets,
    forecast_offsets,
):
    """
    Calculate basic metrics for each forecast horizon.

    Threshold 0.5 is used ONLY for reporting demonstration
    metrics here.

    Final scientific threshold selection should be performed
    by the evaluation workflow using validation data.
    """

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError(
            "Predictions and targets must both be 2D arrays."
        )

    if predictions.shape != targets.shape:
        raise ValueError(
            "Prediction/target shape mismatch: "
            f"{predictions.shape} vs {targets.shape}"
        )

    if predictions.shape[1] != len(forecast_offsets):
        raise ValueError(
            "Prediction horizon count does not match "
            "forecast_offsets."
        )

    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN/Inf.")

    if not np.isfinite(targets).all():
        raise ValueError("Targets contain NaN/Inf.")

    metrics = {}

    for index, offset in enumerate(
        forecast_offsets
    ):

        y_true = targets[:, index]

        y_probability = (
            predictions[:, index]
        )

        y_pred = (
            y_probability >= 0.5
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        if (
            tn + fp
        ) > 0:
            fpr = fp / (
                tn + fp
            )
        else:
            fpr = 0.0

        # ROC-AUC requires both classes.
        if len(
            np.unique(y_true)
        ) == 2:
            roc_auc = roc_auc_score(
                y_true,
                y_probability
            )

            pr_auc = average_precision_score(
                y_true,
                y_probability
            )

        else:
            roc_auc = None
            pr_auc = None

        metrics[
            f"{offset}s"
        ] = {
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
            "fpr": float(
                fpr
            ),
            "roc_auc": (
                None
                if roc_auc is None
                else float(roc_auc)
            ),
            "pr_auc": (
                None
                if pr_auc is None
                else float(pr_auc)
            ),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    return metrics


# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

def save_test_predictions(
    predictions,
    targets,
    forecast_offsets,
):
    """
    Save test predictions in a simple CSV format.

    Example columns:

        risk_10s
        target_10s
        risk_20s
        target_20s
        risk_30s
        target_30s
    """

    output = {}

    for index, offset in enumerate(
        forecast_offsets
    ):

        output[
            f"risk_{offset}s"
        ] = predictions[:, index]

        output[
            f"target_{offset}s"
        ] = targets[:, index]

    prediction_df = pd.DataFrame(
        output
    )

    prediction_df.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False
    )

    return prediction_df


# ================================================================
# ARTIFACT INTEGRITY
# ================================================================

def sha256_file(path: Path) -> str:
    """Return SHA-256 checksum of a saved artifact."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ================================================================
# MAIN
# ================================================================

def main():

    # ============================================================
    # 1. LOAD CONFIGURATION
    # ============================================================

    print("=" * 70)
    print("SIH26153 LSTM NETWORK ATTACK FORECASTING")
    print("=" * 70)

    print("\n[1/17] Loading configuration...")

    config = load_config()

    seed = int(
        config.get("training", {}).get("seed", config.get("project", {}).get("seed", 42))
    )

    set_seed(seed)

    # ============================================================
    # 2. TEMPORAL CONFIGURATION
    # ============================================================

    print("[2/17] Reading temporal configuration...")

    temporal_config = get_temporal_config(
        config
    )

    sequence_length = int(
        temporal_config["sequence_length"]
    )

    horizon = int(
        temporal_config["horizon"]
    )

    window_seconds = int(
        temporal_config[
            "window_seconds"
        ]
    )

    forecast_offsets = list(
        temporal_config[
            "forecast_offsets_seconds"
        ]
    )

    if len(forecast_offsets) != horizon:
        raise ValueError(
            "Number of forecast offsets does not "
            "match forecast horizon."
        )

    if len(set(forecast_offsets)) != len(forecast_offsets):
        raise ValueError("Forecast offsets must be unique.")

    if forecast_offsets != sorted(forecast_offsets):
        raise ValueError("Forecast offsets must be chronological.")

    if any(
        int(offset) <= 0
        or int(offset) % window_seconds != 0
        for offset in forecast_offsets
    ):
        raise ValueError(
            "Every forecast offset must be a positive multiple "
            "of window_seconds."
        )

    # ============================================================
    # 3. MODEL / TRAINING SETTINGS
    # ============================================================

    print("[3/17] Reading model/training settings...")

    hidden_size = int(
        config["model"]["hidden_size"]
    )

    num_layers = int(
        config["model"]["num_layers"]
    )

    dropout = float(
        config["model"]["dropout"]
    )

    batch_size = int(
        config["training"]["batch_size"]
    )

    epochs = int(
        config["training"]["epochs"]
    )

    learning_rate = float(
        config["training"]["learning_rate"]
    )

    train_ratio = float(
        config["training"]["train_ratio"]
    )

    validation_ratio = float(
        config["training"]["validation_ratio"]
    )

    # ============================================================
    # 4. ARTIFACT DIRECTORY
    # ============================================================

    print("[4/17] Preparing artifact directory...")

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()

    print(
        "Device:",
        device
    )

    # ============================================================
    # 5. LOAD PROCESSED DATA
    # ============================================================

    print("[5/17] Loading processed feature matrix...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n"
            f"{DATA_PATH}\n\n"
            "Expected upstream pipeline output:\n"
            "data/processed/feature_matrix.parquet"
        )

    print(
        f"Loading: {DATA_PATH}"
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    # ============================================================
    # 6. VALIDATE DATA
    # ============================================================

    print("[6/17] Validating input data...")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    validate_dataframe(df)

    validate_temporal_grid(
        df,
        window_seconds=window_seconds,
    )

    # ============================================================
    # 7. VALIDATE / SELECT CANONICAL FEATURES
    # ============================================================

    print("[7/17] Validating canonical feature schema...")

    # This explicitly uses the project's canonical schema.
    #
    # We do NOT automatically select all numeric columns.
    #
    feature_df = get_features(
        df
    )

    feature_columns = list(
        CANONICAL_FEATURES
    )

    if list(
        feature_df.columns
    ) != feature_columns:
        raise ValueError(
            "Feature order returned by get_features() "
            "does not match CANONICAL_FEATURES."
        )

    print(
        "Schema version:",
        SCHEMA_VERSION
    )

    print(
        "Feature count:",
        len(feature_columns)
    )

    print(
        "Feature order:"
    )

    for index, feature in enumerate(
        feature_columns,
        start=1
    ):
        print(
            f"  {index:02d}. {feature}"
        )

    # ============================================================
    # 8. TEMPORAL TRAIN / VALIDATION / TEST SPLIT
    # ============================================================

    print(
        "[8/17] Creating chronological "
        "train/validation/test split..."
    )

    train_df, validation_df, test_df = (
        temporal_split(
            df,
            train_ratio,
            validation_ratio
        )
    )

    validate_split_isolation(
        train_df, validation_df, test_df
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

    print(
        "Train time:",
        train_df["timestamp"].min(),
        "->",
        train_df["timestamp"].max()
    )

    print(
        "Validation time:",
        validation_df["timestamp"].min(),
        "->",
        validation_df["timestamp"].max()
    )

    print(
        "Test time:",
        test_df["timestamp"].min(),
        "->",
        test_df["timestamp"].max()
    )

    # ============================================================
    # 9. BUILD TEMPORAL SEQUENCES
    # ============================================================

    print(
        "[9/17] Building temporal sequences..."
    )

    print(
        f"History: {sequence_length} windows"
    )

    print(
        f"Window size: {window_seconds} seconds"
    )

    print(
        "Forecast offsets:",
        forecast_offsets
    )

    # ------------------------------------------------------------
    # IMPORTANT TARGET DEFINITION
    #
    # For each sequence ending at time t:
    #
    # y[0] = malicious at t + 10 sec
    # y[1] = malicious at t + 20 sec
    # y[2] = malicious at t + 30 sec
    #
    # The sequence_builder is responsible for exact temporal
    # alignment and continuity checks.
    # ------------------------------------------------------------

    X_train, y_train = build_sequences(
        train_df,
        feature_columns,
        sequence_length=sequence_length,
        horizon=horizon,
    )

    X_validation, y_validation = build_sequences(
        validation_df,
        feature_columns,
        sequence_length=sequence_length,
        horizon=horizon,
    )

    X_test, y_test = build_sequences(
        test_df,
        feature_columns,
        sequence_length=sequence_length,
        horizon=horizon,
    )

    feature_count = len(
        feature_columns
    )

    validate_sequences(
        X_train,
        y_train,
        sequence_length,
        feature_count,
        horizon,
        "TRAIN"
    )

    validate_sequences(
        X_validation,
        y_validation,
        sequence_length,
        feature_count,
        horizon,
        "VALIDATION"
    )

    validate_sequences(
        X_test,
        y_test,
        sequence_length,
        feature_count,
        horizon,
        "TEST"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_validation:",
        X_validation.shape
    )

    print(
        "y_validation:",
        y_validation.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    # Expected:
    #
    # X = (samples, 10, feature_count)
    # y = (samples, 3)
    #

    # ============================================================
    # 10. SCALE FEATURES
    # ============================================================

    print(
        "[10/17] Scaling features..."
    )

    print(
        "Scaler is fitted ONLY on training sequences."
    )

    (
        X_train,
        X_validation,
        X_test,
        scaler,
    ) = scale_sequences(
        X_train,
        X_validation,
        X_test
    )

    # ============================================================
    # 11. CREATE PYTORCH DATASETS
    # ============================================================

    print(
        "[11/17] Creating PyTorch datasets..."
    )

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32
        ),
    )

    validation_dataset = TensorDataset(
        torch.tensor(
            X_validation,
            dtype=torch.float32
        ),
        torch.tensor(
            y_validation,
            dtype=torch.float32
        ),
    )

    test_dataset = TensorDataset(
        torch.tensor(
            X_test,
            dtype=torch.float32
        ),
        torch.tensor(
            y_test,
            dtype=torch.float32
        ),
    )

    # ============================================================
    # 12. CREATE DATALOADERS
    # ============================================================

    print(
        "[12/17] Creating DataLoaders..."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # ============================================================
    # 13. CREATE LSTM MODEL
    # ============================================================

    print(
        "[13/17] Creating RiskLSTM..."
    )

    model = RiskLSTM(
        input_size=feature_count,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
    ).to(device)

    print()
    print(model)
    print()

    # ============================================================
    # 14. LOSS + OPTIMIZER
    # ============================================================

    print(
        "[14/17] Creating loss function and optimizer..."
    )

    # BCEWithLogitsLoss:
    #
    # model output = raw logits
    #
    # sigmoid is applied only during inference/prediction.
    #
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # ============================================================
    # 15. TRAINING
    # ============================================================

    print(
        "[15/17] Training LSTM..."
    )

    print(
        "-" * 70
    )

    best_validation_loss = float(
        "inf"
    )

    best_model_state = None

    best_epoch = None

    training_history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
        )

        validation_loss = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": float(
                train_loss
            ),
            "validation_loss": float(
                validation_loss
            ),
        }

        training_history.append(
            history_row
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {validation_loss:.6f}"
        )

        # --------------------------------------------------------
        # Save best validation model
        # --------------------------------------------------------

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    print(
        "-" * 70
    )

    if best_model_state is None:
        raise RuntimeError(
            "Training did not produce a valid model."
        )

    print(
        f"Best validation epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    # ============================================================
    # RESTORE BEST MODEL
    # ============================================================

    model.load_state_dict(
        best_model_state
    )

    model = model.to(
        device
    )

    # ============================================================
    # 16. FINAL TEST EVALUATION
    # ============================================================

    print(
        "[16/17] Evaluating final model on TEST data..."
    )

    test_loss = evaluate(
        model,
        test_loader,
        loss_function,
        device,
    )

    print(
        f"Test loss: {test_loss:.6f}"
    )

    test_predictions, test_targets = (
        predict(
            model,
            test_loader,
            device,
        )
    )

    print(
        "Test prediction shape:",
        test_predictions.shape
    )

    print(
        "Test target shape:",
        test_targets.shape
    )

    if test_predictions.shape[1] != horizon:
        raise ValueError(
            "Test prediction horizon mismatch."
        )

    # ------------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------------

    test_metrics = calculate_test_metrics(
        test_predictions,
        test_targets,
        forecast_offsets,
    )

    print()
    print(
        "TEST METRICS"
    )
    print(
        "-" * 70
    )

    for horizon_name, values in (
        test_metrics.items()
    ):

        print(
            f"\nForecast {horizon_name}"
        )

        print(
            f"  Precision : "
            f"{values['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{values['recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{values['f1']:.4f}"
        )

        print(
            f"  FPR       : "
            f"{values['fpr']:.4f}"
        )

        if values["roc_auc"] is not None:
            print(
                f"  ROC-AUC   : "
                f"{values['roc_auc']:.4f}"
            )
        else:
            print(
                "  ROC-AUC   : N/A "
                "(only one class present)"
            )

        if values["pr_auc"] is not None:
            print(
                f"  PR-AUC    : "
                f"{values['pr_auc']:.4f}"
            )
        else:
            print(
                "  PR-AUC    : N/A "
                "(only one class present)"
            )

    # ============================================================
    # SAVE TEST PREDICTIONS
    # ============================================================

    prediction_df = save_test_predictions(
        test_predictions,
        test_targets,
        forecast_offsets,
    )

    print()
    print(
        "Test predictions saved to:"
    )

    print(
        TEST_PREDICTIONS_PATH
    )

    # ============================================================
    # SAVE MODEL
    # ============================================================

    print(
        "\n[17/17] Saving model artifacts..."
    )

    # Store CPU tensors so the artifact can be loaded on machines
    # without the training GPU.
    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    model_checkpoint = {
        "model_state_dict":
            cpu_state_dict,

        "project":
            "SIH26153",

        "schema_version":
            SCHEMA_VERSION,

        "feature_order":
            feature_columns,

        "input_size":
            feature_count,

        "sequence_length":
            sequence_length,

        "window_seconds":
            window_seconds,

        "horizon":
            horizon,

        "forecast_offsets_seconds":
            forecast_offsets,

        "hidden_size":
            hidden_size,

        "num_layers":
            num_layers,

        "dropout":
            dropout,

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_validation_loss,

        "test_loss":
            test_loss,

        "seed":
            seed,

        "device_used":
            str(device),

        "pytorch_version":
            torch.__version__,

        "python_version":
            platform.python_version(),

        "target_definition":
            "Binary is_malicious target at future offsets.",

        "artifact_type":
            "risk_lstm_checkpoint",
    }

    torch.save(
        model_checkpoint,
        MODEL_PATH
    )

    print(
        "Model saved:",
        MODEL_PATH
    )

    model_sha256 = sha256_file(MODEL_PATH)

    print(
        "Model SHA-256:",
        model_sha256
    )

    # ============================================================
    # SAVE SCALER
    # ============================================================

    with open(
        SCALER_PATH,
        "wb"
    ) as file:

        pickle.dump(
            scaler,
            file
        )

    print(
        "Scaler saved:",
        SCALER_PATH
    )

    scaler_sha256 = sha256_file(SCALER_PATH)

    print(
        "Scaler SHA-256:",
        scaler_sha256
    )

    # ============================================================
    # SAVE FEATURE ORDER
    # ============================================================

    feature_order_data = {
        "schema_version":
            SCHEMA_VERSION,

        "feature_count":
            feature_count,

        "feature_order":
            feature_columns,
    }

    with open(
        FEATURE_ORDER_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_order_data,
            file,
            indent=2,
        )

    print(
        "Feature order saved:",
        FEATURE_ORDER_PATH
    )

    # ============================================================
    # SAVE METADATA
    # ============================================================

    train_start = str(
        train_df["timestamp"].min()
    )

    train_end = str(
        train_df["timestamp"].max()
    )

    validation_start = str(
        validation_df["timestamp"].min()
    )

    validation_end = str(
        validation_df["timestamp"].max()
    )

    test_start = str(
        test_df["timestamp"].min()
    )

    test_end = str(
        test_df["timestamp"].max()
    )

    metadata = {

        # --------------------------------------------------------
        # Project
        # --------------------------------------------------------

        "project": {
            "id":
                "SIH26153",

            "name":
                "AI based Network Attack Forecasting "
                "from Network Traffic Data",
        },

        # --------------------------------------------------------
        # Dataset
        # --------------------------------------------------------

        "dataset": {
            "name":
                "CSE-CIC-IDS2018",

            "input_file":
                str(DATA_PATH),

            "rows":
                int(len(df)),
        },

        # --------------------------------------------------------
        # Schema
        # --------------------------------------------------------

        "schema": {
            "version":
                SCHEMA_VERSION,

            "feature_count":
                feature_count,

            "feature_order":
                feature_columns,

            "excluded_target_columns": [
                "is_malicious",
                "risk_score",
                "stage",
                "future_label",
                "future_risk",
            ],
        },

        # --------------------------------------------------------
        # Temporal configuration
        # --------------------------------------------------------

        "temporal": {

            "window_seconds":
                window_seconds,

            "sequence_length_windows":
                sequence_length,

            "forecast_horizon_windows":
                horizon,

            "forecast_offsets_seconds":
                forecast_offsets,

            "target_definition":
                "Binary malicious-window ground truth "
                "at future offsets.",
        },

        # --------------------------------------------------------
        # Target definition
        # --------------------------------------------------------

        "target": {

            "name":
                "is_malicious",

            "type":
                "binary",

            "description":
                "For each historical sequence ending at t, "
                "predict whether the future windows at "
                "t+10s, t+20s and t+30s are malicious.",
        },

        # --------------------------------------------------------
        # Model
        # --------------------------------------------------------

        "model": {

            "architecture":
                "RiskLSTM",

            "input_size":
                feature_count,

            "hidden_size":
                hidden_size,

            "num_layers":
                num_layers,

            "dropout":
                dropout,

            "output_horizons":
                horizon,

            "output_type":
                "raw logits during training, "
                "sigmoid risk probabilities during inference",
        },

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        "training": {

            "batch_size":
                batch_size,

            "epochs":
                epochs,

            "learning_rate":
                learning_rate,

            "optimizer":
                "Adam",

            "loss":
                "BCEWithLogitsLoss",

            "best_epoch":
                best_epoch,

            "best_validation_loss":
                float(
                    best_validation_loss
                ),

            "test_loss":
                float(
                    test_loss
                ),
        },

        # --------------------------------------------------------
        # Split
        # --------------------------------------------------------

        "split": {

            "method":
                "chronological",

            "train_ratio":
                train_ratio,

            "validation_ratio":
                validation_ratio,

            "test_ratio":
                1
                - train_ratio
                - validation_ratio,

            "train_start":
                train_start,

            "train_end":
                train_end,

            "validation_start":
                validation_start,

            "validation_end":
                validation_end,

            "test_start":
                test_start,

            "test_end":
                test_end,

            "boundary_history_policy":
                "No cross-split history. "
                "Sequences are independently built within "
                "each split.",
        },

        # --------------------------------------------------------
        # Reproducibility
        # --------------------------------------------------------

        "reproducibility": {

            "seed":
                seed,

            "device":
                str(device),

            "python_version":
                platform.python_version(),

            "platform":
                platform.platform(),

            "pytorch_version":
                torch.__version__,

            "cuda_available":
                torch.cuda.is_available(),

            "cuda_version":
                torch.version.cuda,
        },

        # --------------------------------------------------------
        # Dataset sizes
        # --------------------------------------------------------

        "samples": {

            "train":
                int(len(X_train)),

            "validation":
                int(len(X_validation)),

            "test":
                int(len(X_test)),
        },

        # --------------------------------------------------------
        # Test metrics
        # --------------------------------------------------------

        "test_metrics":
            test_metrics,

        # --------------------------------------------------------
        # Artifact integrity
        # --------------------------------------------------------

        "artifacts": {
            "model_path": str(MODEL_PATH),
            "model_sha256": model_sha256,
            "scaler_path": str(SCALER_PATH),
            "scaler_sha256": scaler_sha256,
            "feature_order_path": str(FEATURE_ORDER_PATH),
        },

        # --------------------------------------------------------
        # Training history
        # --------------------------------------------------------

        "training_history":
            training_history,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            default=str,
        )

    print(
        "Metadata saved:",
        METADATA_PATH
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Scaler:",
        SCALER_PATH
    )

    print(
        "Feature order:",
        FEATURE_ORDER_PATH
    )

    print(
        "Metadata:",
        METADATA_PATH
    )

    print(
        "Test predictions:",
        TEST_PREDICTIONS_PATH
    )

    print()
    print(
        "Input:"
    )

    print(
        f"  {sequence_length} historical "
        f"windows × {feature_count} features"
    )

    print()
    print(
        "Output:"
    )

    for offset in forecast_offsets:
        print(
            f"  risk_{offset}s"
        )

    print()
    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}"
    )

    print(
        "Final test loss:",
        f"{test_loss:.6f}"
    )

    print()
    print(
        "The trained artifact is ready for "
        "the inference/evaluation pipeline."
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()