#!/usr/bin/env python3
"""
train_lstm.py
-------------
Early-warning botnet detection – LSTM baseline (CPU-only).

Usage (from project root or directly):
    python src/models/train_lstm.py
    python src/models/train_lstm.py --epochs 10 --batch-size 32
"""
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from src.features.sequence_builder import (
    build_sequences,
    FEATURE_COLUMNS,
    HISTORY_WINDOWS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------
DATA_PATH = Path("data/processed/ctu13_scenario1_windows.parquet")
MODEL_DIR = Path("models")
SEED = 42

# Always use CPU (CUDA removed for lightweight CPU execution)
DEVICE = torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
class SequenceDataset(Dataset):
    """PyTorch Dataset wrapping sliding window sequence arrays."""
    def __init__(self, sequences: list, target_key: str):
        self.X = np.stack([s["X"] for s in sequences])          # (N, T, F)
        self.y = np.array([s[target_key] for s in sequences], dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """LSTM model for sequential botnet classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)   # raw logits


# -----------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X = X.to(DEVICE)
        if torch.isnan(X).any():
            X = torch.nan_to_num(X)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    preds, truths = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        probs = torch.sigmoid(model(X)).cpu().numpy()
        preds.extend(probs)
        truths.extend(y.numpy())

    preds = np.array(preds)
    truths = np.array(truths)
    binary = (preds >= 0.5).astype(int)

    n_classes = len(np.unique(truths))
    pr_auc  = average_precision_score(truths, preds) if n_classes > 1 else float("nan")
    roc_auc = roc_auc_score(truths, preds)           if n_classes > 1 else float("nan")

    return {
        "precision": float(precision_score(truths, binary, zero_division=0)),
        "recall":    float(recall_score(truths, binary, zero_division=0)),
        "f1":        float(f1_score(truths, binary, zero_division=0)),
        "pr_auc":    pr_auc,
        "roc_auc":   roc_auc,
        "confusion_matrix": confusion_matrix(truths, binary, labels=[0, 1]).tolist(),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main(args) -> None:
    set_seed(SEED)

    # --- load data ---
    if not DATA_PATH.is_file():
        log.error("Processed data not found: %s", DATA_PATH)
        log.error("Run `python src/features/build_dataset.py` first.")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    if not pd.api.types.is_datetime64_any_dtype(df["window_start"]):
        df["window_start"] = pd.to_datetime(df["window_start"])

    # --- build sequences ---
    sequences = build_sequences(df)
    log.info("Generated %d sequences.", len(sequences))
    if not sequences:
        log.error("No sequences generated — check that the processed dataset has sufficient data.")
        sys.exit(1)

    # --- host-wise split (prevents data leakage) ---
    host_ids = list({s["host_id"] for s in sequences})
    random.shuffle(host_ids)
    n = len(host_ids)
    train_hosts = set(host_ids[:int(0.7 * n)])
    val_hosts   = set(host_ids[int(0.7 * n): int(0.85 * n)])
    test_hosts  = set(host_ids[int(0.85 * n):])

    train_seq = [s for s in sequences if s["host_id"] in train_hosts]
    val_seq   = [s for s in sequences if s["host_id"] in val_hosts]
    test_seq  = [s for s in sequences if s["host_id"] in test_hosts]

    log.info("Split — train: %d, val: %d, test: %d", len(train_seq), len(val_seq), len(test_seq))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for horizon, label_key in [(10, "y_10"), (20, "y_20"), (30, "y_30")]:
        log.info("=== Horizon %ds ===", horizon)

        train_ds = SequenceDataset(train_seq, label_key)
        val_ds   = SequenceDataset(val_seq,   label_key)
        test_ds  = SequenceDataset(test_seq,  label_key)

        if len(train_ds) == 0:
            log.warning("No training samples for horizon %ds — skipping.", horizon)
            continue

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

        model = LSTMClassifier(input_dim=len(FEATURE_COLUMNS)).to(DEVICE)

        # Class-weighted loss to handle imbalance
        n_pos = float(train_ds.y.sum())
        n_neg = float(len(train_ds) - n_pos)
        pos_weight = torch.tensor(n_neg / max(n_pos, 1.0)).to(DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_f1   = -1.0
        best_path = MODEL_DIR / f"lstm_{horizon}s_best.pt"

        for epoch in range(1, args.epochs + 1):
            train_loss   = train_one_epoch(model, train_loader, criterion, optimizer)
            val_metrics  = evaluate(model, val_loader) if len(val_ds) > 0 else {"f1": 0.0}
            scheduler.step(train_loss)
            log.info("Epoch %02d | loss %.4f | val F1 %.4f", epoch, train_loss, val_metrics["f1"])
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), best_path)

        # Evaluate best checkpoint on test set
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        test_metrics = evaluate(model, test_loader) if len(test_ds) > 0 else {}
        results[horizon] = test_metrics
        log.info("Test metrics (horizon %ds): %s", horizon, test_metrics)

    # --- persist metrics ---
    metrics_path = MODEL_DIR / "lstm_metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump(results, fp, indent=2)
    log.info("All metrics saved → %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM botnet early-warning model (CPU-only).")
    parser.add_argument("--epochs",     type=int,   default=30,   help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=64,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    main(args)
