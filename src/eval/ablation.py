"""
Controlled Modality & History Depth Ablation Study for SIH-26153.

ARCHITECTURAL CONTRACT & ISOLATION NOTICE:
- Production WorldModel contract remains strictly locked to CANONICAL_MODEL_FEATURE_NAMES (41 features).
- The modality ablations evaluated here (Flow-only 22 features, Packet-only 19 features)
  and history depth variations (1, 5, 10 windows) are strictly isolated experimental configurations.
- Results are saved to artifacts/ablation_results.json and labeled SYNTHETIC DEMONSTRATION ONLY.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src._win_torch_fix  # noqa: F401
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


from src.config import get_temporal_config
from src.model import WorldModel
from src.schemas.features import (
    CANONICAL_FLOW_FEATURE_NAMES,
    CANONICAL_PACKET_FEATURE_NAMES,
    CANONICAL_MODEL_FEATURE_NAMES,
)
from src.temporal.sequences import build_sequences_from_dataframe
from src.train import split_dataframe_temporally
from src.eval.metrics import evaluate_risk, compute_roc_auc, compute_pr_auc


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/feature_matrix.parquet")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Explicit experimental schemas
FLOW_ABLATION_SCHEMA: Tuple[str, ...] = CANONICAL_FLOW_FEATURE_NAMES      # 22 features
PACKET_ABLATION_SCHEMA: Tuple[str, ...] = CANONICAL_PACKET_FEATURE_NAMES  # 19 features
FUSION_SCHEMA: Tuple[str, ...] = CANONICAL_MODEL_FEATURE_NAMES            # 41 features

TEMPORAL_CONFIG = get_temporal_config()
HORIZON = TEMPORAL_CONFIG.forecast_horizon_windows  # 3 horizons (+10s, +20s, +30s)
SEED = 42


class AblationLSTMModel(nn.Module):
    """
    Isolated experimental LSTM model for modality ablations (22 flow, 19 packet).
    Guarantees production WorldModel remains strictly locked to 41 input features.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 3,
        num_stages: int = 2,
    ):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.risk_head = nn.Linear(hidden_size, horizon)
        self.stage_head = nn.Linear(hidden_size, num_stages)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        last_hidden = self.dropout(out[:, -1, :])
        return self.risk_head(last_hidden), self.stage_head(last_hidden)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train_and_eval_ablation(
    df: pd.DataFrame,
    feature_schema: Tuple[str, ...],
    history_length: int,
    config_name: str,
    epochs: int = 10,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Train and evaluate a controlled ablation configuration.
    """
    log.info(f"--- Running Ablation: {config_name} (features={len(feature_schema)}, history={history_length}) ---")
    set_seed(SEED)

    # 1. Temporal split
    train_df, val_df, test_df, split_info = split_dataframe_temporally(
        df,
        train_ratio=0.7,
        validation_ratio=0.15,
    )

    # 2. Scaler fit on training set only (strict anti-leakage)
    scaler = StandardScaler()
    scaler.fit(train_df[list(feature_schema)])

    stage_encoder = LabelEncoder()
    stage_encoder.fit(df["stage"])
    num_stages = len(stage_encoder.classes_)

    # 3. Build temporal sequences
    X_train, y_train_risk, y_train_stage = build_sequences_from_dataframe(
        train_df,
        feature_columns=list(feature_schema),
        entity_column="source_host",
        timestamp_column="timestamp",
        risk_column="is_malicious",
        stage_column="stage",
        history_length=history_length,
        forecast_horizon=HORIZON,
        scaler=scaler,
        stage_encoder=stage_encoder,
    )

    X_val, y_val_risk, y_val_stage = build_sequences_from_dataframe(
        val_df,
        feature_columns=list(feature_schema),
        entity_column="source_host",
        timestamp_column="timestamp",
        risk_column="is_malicious",
        stage_column="stage",
        history_length=history_length,
        forecast_horizon=HORIZON,
        scaler=scaler,
        stage_encoder=stage_encoder,
    )

    X_test, y_test_risk, y_test_stage = build_sequences_from_dataframe(
        test_df,
        feature_columns=list(feature_schema),
        entity_column="source_host",
        timestamp_column="timestamp",
        risk_column="is_malicious",
        stage_column="stage",
        history_length=history_length,
        forecast_horizon=HORIZON,
        scaler=scaler,
        stage_encoder=stage_encoder,
    )

    # 4. Model instantiation for this specific ablation schema
    if len(feature_schema) == 41:
        model = WorldModel(
            input_size=41,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            horizon=HORIZON,
            num_stages=num_stages,
        ).to(device)
    else:
        model = AblationLSTMModel(
            input_size=len(feature_schema),
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            horizon=HORIZON,
            num_stages=num_stages,
        ).to(device)


    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train_risk, dtype=torch.float32),
            torch.tensor(y_train_stage, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    risk_loss_fn = nn.BCEWithLogitsLoss()
    stage_loss_fn = nn.CrossEntropyLoss()

    # 5. Training loop
    model.train()
    for ep in range(epochs):
        for bx, by_risk, by_stage in train_loader:
            bx, by_risk, by_stage = bx.to(device), by_risk.to(device), by_stage.to(device)
            optimizer.zero_grad()
            out_risk, out_stage = model(bx)
            loss = risk_loss_fn(out_risk, by_risk) + 0.3 * stage_loss_fn(out_stage, by_stage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # 6. Evaluation with validation-only threshold tuning
    model.eval()
    with torch.no_grad():
        val_logits, _ = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

        test_logits, _ = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    # Tune threshold on validation set per horizon
    thresholds = []
    for h in range(HORIZON):
        y_v = y_val_risk[:, h]
        p_v = val_probs[:, h]
        best_t, best_f1 = 0.5, -1.0
        if len(np.unique(y_v)) > 1:
            for cand in np.linspace(0.1, 0.9, 33):
                pred_bin = (p_v >= cand).astype(int)
                from sklearn.metrics import f1_score
                score = f1_score(y_v, pred_bin, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = float(cand)
        thresholds.append(best_t)

    # Compute test set metrics with frozen validation thresholds
    horizons_metrics = {}
    f1_scores = []
    offsets = [10, 20, 30]
    for h_idx, offset_sec in enumerate(offsets):
        thresh = thresholds[h_idx]
        eval_res = evaluate_risk(y_test_risk[:, h_idx], test_probs[:, h_idx], threshold=thresh)
        f1_scores.append(eval_res.f1)
        horizons_metrics[f"+{offset_sec}s"] = {
            "f1": round(eval_res.f1, 4),
            "precision": round(eval_res.precision, 4),
            "recall": round(eval_res.recall, 4),
            "fpr": round(eval_res.fpr, 4),
            "threshold": thresh,
        }

    mean_f1 = float(np.mean(f1_scores))

    return {
        "config_name": config_name,
        "feature_count": len(feature_schema),
        "history_length": history_length,
        "mean_test_f1": round(mean_f1, 4),
        "horizons": horizons_metrics,
        "validation_thresholds": thresholds,
    }



def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Processed feature matrix not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    log.info(f"Loaded dataset from {DATA_PATH}: {df.shape[0]} rows, {df.shape[1]} columns")

    device = torch.device("cpu")

    results = {
        "provenance_status": "SYNTHETIC_DEMONSTRATION_ONLY",
        "notice": (
            "REAL DATA EXPERIMENT NOT EXECUTED - RAW CSE-CIC-IDS2018 TELEMETRY PENDING EXTRACTION. "
            "These controlled ablations were executed on the canonical synthetic integration fixture. "
            "The production WorldModel contract remains strictly locked to 41 input features."
        ),
        "modality_ablations": {},
        "history_depth_ablations": {},
    }

    # 1. Modality Ablations (Flow-only vs Packet-only vs Early Fusion at canonical 10-window depth)
    log.info("Starting Modality Ablations...")
    results["modality_ablations"]["flow_only_22"] = train_and_eval_ablation(
        df, FLOW_ABLATION_SCHEMA, history_length=10, config_name="flow_only_22", epochs=8, device=device
    )
    results["modality_ablations"]["packet_only_19"] = train_and_eval_ablation(
        df, PACKET_ABLATION_SCHEMA, history_length=10, config_name="packet_only_19", epochs=8, device=device
    )
    results["modality_ablations"]["multimodal_fusion_41"] = train_and_eval_ablation(
        df, FUSION_SCHEMA, history_length=10, config_name="multimodal_fusion_41", epochs=8, device=device
    )

    # 2. History Depth Ablations (1 window vs 5 windows vs 10 windows with 41 fused features)
    log.info("Starting History Depth Ablations...")
    results["history_depth_ablations"]["history_1_window"] = train_and_eval_ablation(
        df, FUSION_SCHEMA, history_length=1, config_name="history_1_window", epochs=8, device=device
    )
    results["history_depth_ablations"]["history_5_windows"] = train_and_eval_ablation(
        df, FUSION_SCHEMA, history_length=5, config_name="history_5_windows", epochs=8, device=device
    )
    results["history_depth_ablations"]["history_10_windows"] = train_and_eval_ablation(
        df, FUSION_SCHEMA, history_length=10, config_name="history_10_windows", epochs=8, device=device
    )

    out_file = ARTIFACT_DIR / "ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(f"Saved ablation results to {out_file}")

    print("\n============================================================")
    print("CONTROLLED ABLATION RESULTS SUMMARY (SYNTHETIC DEMONSTRATION ONLY)")
    print("============================================================")
    print("1. Modality Ablation (History = 10 windows):")
    for key, res in results["modality_ablations"].items():
        print(f"   - {key:<24}: Mean F1 = {res['mean_test_f1']:.4f} (features={res['feature_count']})")

    print("\n2. History Depth Ablation (Features = 41):")
    for key, res in results["history_depth_ablations"].items():
        print(f"   - {key:<24}: Mean F1 = {res['mean_test_f1']:.4f} (history={res['history_length']})")
    print("============================================================\n")


if __name__ == "__main__":
    main()
