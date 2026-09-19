from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src._win_torch_fix  # noqa: F401
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler



try:
    from src.config import get_temporal_config
    from src.model import WorldModel
    from src.schemas.features import (
        CANONICAL_MODEL_FEATURE_NAMES,
        CANONICAL_SCHEMA_HASH,
        compute_schema_hash,
        validate_feature_names,
    )
    from src.temporal.sequences import build_sequences_from_dataframe
    from src.temporal.split import temporal_split_by_time
except ImportError:
    from config import get_temporal_config  # type: ignore
    from model import WorldModel  # type: ignore
    from schemas.features import (  # type: ignore
        CANONICAL_MODEL_FEATURE_NAMES,
        CANONICAL_SCHEMA_HASH,
        compute_schema_hash,
        validate_feature_names,
    )
    from temporal.sequences import build_sequences_from_dataframe  # type: ignore
    from temporal.split import temporal_split_by_time  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

LAMBDA_STAGE: float = 0.3  # Documented auxiliary loss weight for stage classification head

DATA_PATH = Path("data/processed/feature_matrix.parquet")
ARTIFACT_DIR = Path("artifacts")

TEMPORAL_CONFIG = get_temporal_config()
SEQUENCE_LENGTH = TEMPORAL_CONFIG.history_length
HORIZON = TEMPORAL_CONFIG.forecast_horizon_windows

INITIAL_BATCH_SIZE = 64
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

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), device_name
    return torch.device("cpu"), "CUDA UNAVAILABLE — CPU EXECUTION"


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

def scale_sequences(X_train: np.ndarray, X_validation: np.ndarray, X_test: np.ndarray):
    n_features = X_train.shape[-1]
    scaler = StandardScaler()

    train_2d = X_train.reshape(-1, n_features)
    validation_2d = X_validation.reshape(-1, n_features)
    test_2d = X_test.reshape(-1, n_features)

    # Invariant: fit ONLY on training data
    scaler.fit(train_2d)

    train_scaled = scaler.transform(train_2d).reshape(X_train.shape)
    validation_scaled = scaler.transform(validation_2d).reshape(X_validation.shape)
    test_scaled = scaler.transform(test_2d).reshape(X_test.shape)

    return train_scaled, validation_scaled, test_scaled, scaler


# ============================================================
# PREPARE STAGE LABELS
# ============================================================

def encode_stages(y_train_stage, y_validation_stage, y_test_stage):
    encoder = LabelEncoder()
    encoder.fit(y_train_stage)

    y_train = encoder.transform(y_train_stage)
    y_validation = encoder.transform(y_validation_stage)
    y_test = encoder.transform(y_test_stage)

    return y_train, y_validation, y_test, encoder


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    risk_loss_fn: nn.Module,
    stage_loss_fn: nn.Module,
    device: torch.device,
    grad_scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = (device.type == "cuda" and grad_scaler is not None)

    for X, y_risk, y_stage in loader:
        X = X.to(device, non_blocking=(device.type == "cuda"))
        y_risk = y_risk.to(device, non_blocking=(device.type == "cuda"))
        y_stage = y_stage.to(device, non_blocking=(device.type == "cuda"))

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                risk_logits, stage_logits = model(X)
                risk_loss = risk_loss_fn(risk_logits, y_risk)
                stage_loss = stage_loss_fn(stage_logits, y_stage)
                loss = risk_loss + LAMBDA_STAGE * stage_loss
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            risk_logits, stage_logits = model(X)
            risk_loss = risk_loss_fn(risk_logits, y_risk)
            stage_loss = stage_loss_fn(stage_logits, y_stage)
            loss = risk_loss + LAMBDA_STAGE * stage_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ============================================================
# VALIDATION
# ============================================================

def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    risk_loss_fn: nn.Module,
    stage_loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y_risk, y_stage in loader:
            X = X.to(device)
            y_risk = y_risk.to(device)
            y_stage = y_stage.to(device)

            risk_logits, stage_logits = model(X)
            risk_loss = risk_loss_fn(risk_logits, y_risk)
            stage_loss = stage_loss_fn(stage_logits, y_stage)
            loss = risk_loss + LAMBDA_STAGE * stage_loss
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    start_time = time.time()
    set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    device, device_name = get_device()
    print(f"Device: {device} ({device_name})")

    # 1. Load Data
    print(f"Loading: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    # 2. Features hard lock (strictly 41 features)
    feature_columns = select_canonical_model_features(df)
    print(f"Canonical feature contract verified: {len(feature_columns)} features.")
    print(f"Authoritative schema hash: {CANONICAL_SCHEMA_HASH}")

    # 3. Temporal Split
    train_df, validation_df, test_df, split_info = split_dataframe_temporally(df)
    print(f"Split counts -> Train: {len(train_df)}, Val: {len(validation_df)}, Test: {len(test_df)}")

    # 4. Build Sequences
    X_train, y_train_risk, y_train_stage = build_sequences_from_dataframe(
        train_df,
        feature_columns,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        entity_column=ENTITY_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        risk_column=RISK_COLUMN,
        stage_column=STAGE_COLUMN,
    )
    X_validation, y_validation_risk, y_validation_stage = build_sequences_from_dataframe(
        validation_df,
        feature_columns,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        entity_column=ENTITY_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        risk_column=RISK_COLUMN,
        stage_column=STAGE_COLUMN,
    )
    X_test, y_test_risk, y_test_stage = build_sequences_from_dataframe(
        test_df,
        feature_columns,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        entity_column=ENTITY_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        risk_column=RISK_COLUMN,
        stage_column=STAGE_COLUMN,
    )

    print(f"Sequence dimensions -> Train: {X_train.shape}, Val: {X_validation.shape}, Test: {X_test.shape}")

    # 5. Scale Sequences
    X_train, X_validation, X_test, scaler = scale_sequences(X_train, X_validation, X_test)

    # 6. Encode Stages
    y_train_stage, y_validation_stage, y_test_stage, stage_encoder = encode_stages(
        y_train_stage, y_validation_stage, y_test_stage
    )
    num_stages = len(stage_encoder.classes_)
    print(f"Stage classes ({num_stages}): {list(stage_encoder.classes_)}")

    # 7. PyTorch Datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_risk, dtype=torch.float32),
        torch.tensor(y_train_stage, dtype=torch.long),
    )
    validation_dataset = TensorDataset(
        torch.tensor(X_validation, dtype=torch.float32),
        torch.tensor(y_validation_risk, dtype=torch.float32),
        torch.tensor(y_validation_stage, dtype=torch.long),
    )

    # 8. Training loop with automatic OOM fallback (64 -> 32 -> 16 -> 8)
    batch_candidates = [64, 32, 16, 8] if device.type == "cuda" else [INITIAL_BATCH_SIZE]
    final_batch_size = INITIAL_BATCH_SIZE
    training_success = False

    for batch_size in batch_candidates:
        try:
            print(f"\nAttempting training with batch_size={batch_size}...")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

            model = WorldModel(
                input_size=len(feature_columns),
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
                horizon=HORIZON,
                num_stages=num_stages,
            ).to(device)

            risk_loss_fn = nn.BCEWithLogitsLoss()
            stage_loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            grad_scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

            best_validation_loss = float("inf")
            best_state = None
            epoch_times = []

            for epoch in range(1, EPOCHS + 1):
                ep_start = time.time()
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, risk_loss_fn, stage_loss_fn, device, grad_scaler
                )
                validation_loss = evaluate_loss(model, validation_loader, risk_loss_fn, stage_loss_fn, device)
                ep_dur = time.time() - ep_start
                epoch_times.append(ep_dur)

                if epoch % 5 == 0 or epoch == EPOCHS:
                    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {validation_loss:.4f} | Time: {ep_dur:.2f}s")

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

            final_batch_size = batch_size
            training_success = True
            break

        except torch.cuda.OutOfMemoryError as oom_err:
            log.warning(f"CUDA OOM encountered with batch_size={batch_size}: {oom_err}")
            torch.cuda.empty_cache()
            if batch_size == batch_candidates[-1]:
                raise RuntimeError("CUDA OOM persisted even at minimum batch size 8!") from oom_err
            continue

    if not training_success or best_state is None:
        raise RuntimeError("Training could not complete successfully.")

    total_train_time = time.time() - start_time
    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0
    throughput = len(X_train) / avg_epoch_time if avg_epoch_time > 0 else 0.0

    print(f"\nTraining completed in {total_train_time:.2f}s (Avg epoch: {avg_epoch_time:.2f}s, Throughput: {throughput:.1f} seq/s)")

    # 9. Save Checkpoint with Provenance Metadata
    model.load_state_dict(best_state)
    model_path = ARTIFACT_DIR / "world_model.pt"

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "project_id": "SIH26153",
        "schema_hash": CANONICAL_SCHEMA_HASH,
        "feature_order": feature_columns,
        "feature_count": len(feature_columns),
        "input_size": len(feature_columns),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "sequence_length": SEQUENCE_LENGTH,
        "horizon": HORIZON,
        "forecast_offsets_seconds": list(TEMPORAL_CONFIG.forecast_offsets_seconds),
        "dataset": "CSE-CIC-IDS2018",
        "dataset_status": "SYNTHETIC_DEMONSTRATION_ONLY",
        "device": str(device),
        "device_name": device_name,
        "batch_size": final_batch_size,
        "seed": SEED,
        "num_stages": num_stages,
        "training_duration_seconds": round(total_train_time, 2),
        "throughput_seq_per_sec": round(throughput, 2),
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "split_cutoffs": {
            "train_end_time": split_info.train_end_time.isoformat() if split_info.train_end_time else None,
            "val_start_time": split_info.val_start_time.isoformat() if split_info.val_start_time else None,
            "val_end_time": split_info.val_end_time.isoformat() if split_info.val_end_time else None,
            "test_start_time": split_info.test_start_time.isoformat() if split_info.test_start_time else None,
        },
    }

    torch.save(checkpoint_payload, model_path)
    print(f"Saved model checkpoint: {model_path}")

    # 10. Save Scaler & Auxiliary Artifacts
    with open(ARTIFACT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(ARTIFACT_DIR / "feature_order.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    with open(ARTIFACT_DIR / "stage_classes.json", "w", encoding="utf-8") as f:
        json.dump(stage_encoder.classes_.tolist(), f, indent=2)

    with open(ARTIFACT_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(stage_encoder, f)

    metadata = {
        "project_id": "SIH26153",
        "dataset": "CSE-CIC-IDS2018",
        "provenance_status": "SYNTHETIC_DEMONSTRATION_ONLY",
        "warning": (
            "REAL DATA EXPERIMENT NOT EXECUTED - RAW CSE-CIC-IDS2018 TELEMETRY PENDING EXTRACTION. "
            "Artifact generated from canonical synthetic integration fixture."
        ),
        "schema_hash": CANONICAL_SCHEMA_HASH,
        "window_seconds": TEMPORAL_CONFIG.window_seconds,
        "sequence_length": SEQUENCE_LENGTH,
        "forecast_horizon": HORIZON,
        "forecast_offsets_seconds": list(TEMPORAL_CONFIG.forecast_offsets_seconds),
        "entity": ENTITY_COLUMN,
        "feature_count": len(feature_columns),
        "feature_order": feature_columns,
        "batch_size": final_batch_size,
        "device": str(device),
        "device_name": device_name,
        "training_duration_seconds": round(total_train_time, 2),
        "throughput_seq_per_sec": round(throughput, 2),
        "split_cutoffs": checkpoint_payload["split_cutoffs"],
        "model_architecture": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "num_stages": num_stages,
        },
        "software_versions": checkpoint_payload["software_versions"],
        "seed": SEED,
    }

    with open(ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved rich metadata: {ARTIFACT_DIR / 'metadata.json'}")
    print("=" * 60)
    print("TRAINING PROCESS COMPLETE: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()
