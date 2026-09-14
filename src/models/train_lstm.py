#!/usr/bin/env python3
"""
[LEGACY / PROTOTYPE NOTICE: NON-AUTHORITATIVE]
Authoritative training workflow is implemented in src/train.py.
This script represents an earlier CPU baseline and is retained for reference only.
"""
import argparse
from collections import defaultdict
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


def split_sequences(
    sequences: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Split sequences into train, validation, and test sets.

    If there are multiple distinct infected hosts (>= 3), performs a stratified host-wise
    split to avoid host leakage across splits.
    If there are fewer infected hosts (e.g. CTU-13 Scenario 1, where only 1 botnet IP exists),
    a host-wise split would leave validation or test splits with 0 positive samples. In that
    case, it performs a chronological (temporal) split per host based on window_id, which preserves
    positive samples in all splits while avoiding temporal data leakage.
    """
    pos_hosts = {
        s["src_ip"]
        for s in sequences
        if (s.get("y_10", 0) == 1 or s.get("y_20", 0) == 1 or s.get("y_30", 0) == 1)
    }

    if len(pos_hosts) >= 3:
        log.info("Splitting host-wise (stratified across %d infected hosts)...", len(pos_hosts))
        rng = random.Random(seed)
        all_hosts = sorted({s["src_ip"] for s in sequences})
        neg_hosts = [h for h in all_hosts if h not in pos_hosts]
        pos_hosts_list = sorted(pos_hosts)

        rng.shuffle(pos_hosts_list)
        rng.shuffle(neg_hosts)

        def partition(hosts):
            n = len(hosts)
            n_tr = max(1, int(train_ratio * n))
            n_va = max(1, int(val_ratio * n))
            if n_tr + n_va >= n and n > 2:
                n_tr = n - 2
                n_va = 1
            return set(hosts[:n_tr]), set(hosts[n_tr : n_tr + n_va]), set(hosts[n_tr + n_va :])

        tr_pos, va_pos, te_pos = partition(pos_hosts_list)
        tr_neg, va_neg, te_neg = partition(neg_hosts)

        train_hosts = tr_pos | tr_neg
        val_hosts = va_pos | va_neg
        test_hosts = te_pos | te_neg

        train_seq = [s for s in sequences if s["src_ip"] in train_hosts]
        val_seq = [s for s in sequences if s["src_ip"] in val_hosts]
        test_seq = [s for s in sequences if s["src_ip"] in test_hosts]
    else:
        log.info(
            "Found %d infected host(s) (< 3). Performing chronological split per host to prevent 0-positive splits...",
            len(pos_hosts),
        )
        host_to_seqs = defaultdict(list)
        for s in sequences:
            host_to_seqs[s["src_ip"]].append(s)

        train_seq, val_seq, test_seq = [], [], []
        for src_ip, h_seqs in host_to_seqs.items():
            h_seqs.sort(key=lambda s: s["window_id"])
            n = len(h_seqs)
            if n == 1:
                train_seq.append(h_seqs[0])
            elif n == 2:
                train_seq.append(h_seqs[0])
                val_seq.append(h_seqs[1])
            else:
                n_tr = max(1, int(train_ratio * n))
                n_va = max(1, int(val_ratio * n))
                if n_tr + n_va >= n:
                    n_tr = n - 2
                    n_va = 1
                train_seq.extend(h_seqs[:n_tr])
                val_seq.extend(h_seqs[n_tr : n_tr + n_va])
                test_seq.extend(h_seqs[n_tr + n_va :])

    return train_seq, val_seq, test_seq


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
    if "window_start" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["window_start"]):
        df["window_start"] = pd.to_datetime(df["window_start"])

    # --- build sequences ---
    sequences = build_sequences(df)
    log.info("Generated %d sequences.", len(sequences))
    if not sequences:
        log.error("No sequences generated - check that the processed dataset has sufficient data.")
        sys.exit(1)

    # --- split dataset ---
    train_seq, val_seq, test_seq = split_sequences(sequences, seed=SEED)

    log.info("Split - train: %d, val: %d, test: %d", len(train_seq), len(val_seq), len(test_seq))

    assert sum(s["y_10"] for s in val_seq) > 0, "Validation split has 0 positive samples."
    assert sum(s["y_10"] for s in test_seq) > 0, "Test split has 0 positive samples."

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for horizon, label_key in [(10, "y_10"), (20, "y_20"), (30, "y_30")]:
        log.info("=== Horizon %ds ===", horizon)

        train_ds = SequenceDataset(train_seq, label_key)
        val_ds   = SequenceDataset(val_seq,   label_key)
        test_ds  = SequenceDataset(test_seq,  label_key)

        if len(train_ds) == 0:
            log.warning("No training samples for horizon %ds - skipping.", horizon)
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
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    log.info("All metrics saved -> %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM botnet early-warning model (CPU-only).")
    parser.add_argument("--epochs",     type=int,   default=30,   help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=64,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    main(args)