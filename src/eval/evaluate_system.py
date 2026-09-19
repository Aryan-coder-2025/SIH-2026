"""
Comprehensive System Evaluation & Baseline Benchmarking Module for SIH26153.
Project: SIH26153 - AI-Based Network Attack Forecasting from Network Traffic Data

Evaluates the trained WorldModel against competitive baselines:
1. WorldModel (Authoritative Bi-Head LSTM: Multi-horizon Risk + Stage)
2. Logistic Regression Baseline
3. Random Forest Baseline
4. Persistence Baseline

Enforces:
- Temporal train/validation/test split BEFORE sequence construction.
- Strict anti-leakage: StandardScaler fit ONLY on training set.
- Threshold optimization frozen on validation set before test evaluation.
- Multi-horizon metrics (+10s, +20s, +30s): Precision, Recall, F1, FPR, ROC-AUC, PR-AUC, Confusion Matrix.
- Clear synthetic provenance labeling: SYNTHETIC_DEMONSTRATION_ONLY.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src._win_torch_fix  # noqa: F401

import numpy as np
import pandas as pd
import torch

from src.baseline.logistic import LogisticRegressionBaseline
from src.baseline.persistence import PersistenceBaseline
from src.baseline.random_forest import RandomForestBaseline
from src.config import get_temporal_config
from src.eval.metrics import (
    compute_pr_auc,
    compute_roc_auc,
    evaluate_risk,
    optimize_threshold,
)
from src.inference import load_artifacts
from src.schemas.features import CANONICAL_MODEL_FEATURE_NAMES, validate_feature_names
from src.temporal.sequences import build_sequences_from_dataframe
from src.train import (
    DATA_PATH,
    ENTITY_COLUMN,
    RISK_COLUMN,
    STAGE_COLUMN,
    TIMESTAMP_COLUMN,
    split_dataframe_temporally,
)

ARTIFACT_DIR = Path("artifacts")
TEMPORAL_CONFIG = get_temporal_config()
HORIZONS = [10, 20, 30]


def run_full_evaluation(data_path: Path = DATA_PATH) -> Dict[str, Any]:
    print("=" * 70)
    print("SIH26153 EVALUATION & BASELINE COMPARISON SUITE")
    print("=" * 70)

    # 1. Load Data
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    feature_columns = list(CANONICAL_MODEL_FEATURE_NAMES)
    validate_feature_names(feature_columns, expected_order=CANONICAL_MODEL_FEATURE_NAMES)

    # 2. Temporal Split before Sequence Construction
    train_df, val_df, test_df, split_info = split_dataframe_temporally(df)
    print(f"Split counts -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Build Sequences with Current-Risk Tracking
    seq_len = TEMPORAL_CONFIG.history_length
    horizon_cnt = TEMPORAL_CONFIG.forecast_horizon_windows

    X_train, y_train_risk, _, y_train_curr = build_sequences_from_dataframe(
        train_df, feature_columns, seq_len, horizon_cnt,
        ENTITY_COLUMN, TIMESTAMP_COLUMN, RISK_COLUMN, STAGE_COLUMN,
        return_current_risk=True,
    )
    X_val, y_val_risk, _, y_val_curr = build_sequences_from_dataframe(
        val_df, feature_columns, seq_len, horizon_cnt,
        ENTITY_COLUMN, TIMESTAMP_COLUMN, RISK_COLUMN, STAGE_COLUMN,
        return_current_risk=True,
    )
    X_test, y_test_risk, _, y_test_curr = build_sequences_from_dataframe(
        test_df, feature_columns, seq_len, horizon_cnt,
        ENTITY_COLUMN, TIMESTAMP_COLUMN, RISK_COLUMN, STAGE_COLUMN,
        return_current_risk=True,
    )

    print(f"Sequence shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 4. Load Model & Preprocessing Artifacts
    model, scaler, _, stage_classes = load_artifacts(ARTIFACT_DIR)
    model.eval()

    # Scale 3D Sequences using train-fitted scaler
    n_feats = X_train.shape[-1]
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_feats)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_feats)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_feats)).reshape(X_test.shape)

    # 5. Model Inference on Validation & Test Sets
    with torch.no_grad():
        val_logits, _ = model(torch.tensor(X_val_scaled, dtype=torch.float32))
        val_risk_preds = torch.sigmoid(val_logits).numpy()

        test_logits, _ = model(torch.tensor(X_test_scaled, dtype=torch.float32))
        test_risk_preds = torch.sigmoid(test_logits).numpy()

    # 6. Scientific Validation Freeze Protocol:
    # Optimize decision threshold strictly on validation set (maximizing F1)
    frozen_thresholds: Dict[int, float] = {}
    for h_idx, h_sec in enumerate(HORIZONS):
        opt_thresh = optimize_threshold(
            y_true=y_val_risk[:, h_idx],
            risk=val_risk_preds[:, h_idx],
            metric="f1",
        )
        frozen_thresholds[h_sec] = opt_thresh
    print(f"\nFrozen Decision Thresholds (tuned on validation set): {frozen_thresholds}")

    # 7. Evaluate WorldModel on Held-Out Test Split with Frozen Thresholds
    world_model_results: Dict[str, Any] = {}
    print("\n--- WorldModel Evaluation on Test Split (Frozen Thresholds) ---")
    def compute_acc(er) -> float:
        cm = er.confusion_matrix
        tot = cm.tp + cm.tn + cm.fp + cm.fn
        return float((cm.tp + cm.tn) / max(tot, 1))

    for h_idx, h_sec in enumerate(HORIZONS):
        thresh = frozen_thresholds[h_sec]
        y_true_h = y_test_risk[:, h_idx]
        risk_h = test_risk_preds[:, h_idx]

        eval_res = evaluate_risk(y_true_h, risk_h, threshold=thresh)
        roc_auc = compute_roc_auc(y_true_h, risk_h)
        pr_auc = compute_pr_auc(y_true_h, risk_h)
        all_neg_acc = float((y_true_h == 0).mean())

        h_metrics = {
            "horizon_seconds": h_sec,
            "threshold": thresh,
            "precision": round(eval_res.precision, 4),
            "recall": round(eval_res.recall, 4),
            "f1": round(eval_res.f1, 4),
            "fpr": round(eval_res.fpr, 4),
            "accuracy": round(compute_acc(eval_res), 4),
            "all_negative_baseline_accuracy": round(all_neg_acc, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "confusion_matrix": eval_res.confusion_matrix.to_dict(),
        }
        world_model_results[str(h_sec)] = h_metrics
        print(f"  +{h_sec}s -> F1: {h_metrics['f1']:.4f} | Prec: {h_metrics['precision']:.4f} | "
              f"Rec: {h_metrics['recall']:.4f} | ROC-AUC: {h_metrics['roc_auc']} | PR-AUC: {h_metrics['pr_auc']}")

    # 8. Evaluate Fair Baselines (Full 10-window history = 10 * 41 = 410 features)
    from sklearn.preprocessing import StandardScaler
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # Scaler fit STRICTLY on training split only
    flat_scaler = StandardScaler()
    X_train_flat_scaled = flat_scaler.fit_transform(X_train_flat)
    X_val_flat_scaled = flat_scaler.transform(X_val_flat)
    X_test_flat_scaled = flat_scaler.transform(X_test_flat)

    # 8A. Logistic Regression Baseline (410 Flattened Temporal Features)
    print("\n--- Logistic Regression Baseline (410 features, Fair History) on Test Split ---")
    lr_results: Dict[str, Any] = {}
    for h_idx, h_sec in enumerate(HORIZONS):
        lr = LogisticRegressionBaseline(random_state=42, threshold=0.5)
        lr.fit(X_train_flat_scaled, y_train_risk[:, h_idx])
        val_lr_risk = lr.predict_risk(X_val_flat_scaled)
        lr_thresh = optimize_threshold(y_val_risk[:, h_idx], val_lr_risk, metric="f1")
        lr.threshold = lr_thresh

        lr_pred_risk = lr.predict_risk(X_test_flat_scaled)
        eval_res = evaluate_risk(y_test_risk[:, h_idx], lr_pred_risk, threshold=lr_thresh)
        roc_auc = compute_roc_auc(y_test_risk[:, h_idx], lr_pred_risk)
        pr_auc = compute_pr_auc(y_test_risk[:, h_idx], lr_pred_risk)
        all_neg_acc = float((y_test_risk[:, h_idx] == 0).mean())

        lr_metrics = {
            "horizon_seconds": h_sec,
            "threshold": lr_thresh,
            "precision": round(eval_res.precision, 4),
            "recall": round(eval_res.recall, 4),
            "f1": round(eval_res.f1, 4),
            "fpr": round(eval_res.fpr, 4),
            "accuracy": round(compute_acc(eval_res), 4),
            "all_negative_baseline_accuracy": round(all_neg_acc, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "confusion_matrix": eval_res.confusion_matrix.to_dict(),
        }
        lr_results[str(h_sec)] = lr_metrics
        print(f"  +{h_sec}s -> F1: {lr_metrics['f1']:.4f} | Prec: {lr_metrics['precision']:.4f} | "
              f"Rec: {lr_metrics['recall']:.4f} | ROC-AUC: {lr_metrics['roc_auc']}")

    # 8B. Random Forest Baseline (410 Flattened Temporal Features)
    print("\n--- Random Forest Baseline (410 features, Fair History) on Test Split ---")
    rf_results: Dict[str, Any] = {}
    for h_idx, h_sec in enumerate(HORIZONS):
        rf = RandomForestBaseline(random_state=42, threshold=0.5, n_estimators=50)
        rf.fit(X_train_flat_scaled, y_train_risk[:, h_idx])
        val_rf_risk = rf.predict_risk(X_val_flat_scaled)
        rf_thresh = optimize_threshold(y_val_risk[:, h_idx], val_rf_risk, metric="f1")
        rf.threshold = rf_thresh

        rf_pred_risk = rf.predict_risk(X_test_flat_scaled)
        eval_res = evaluate_risk(y_test_risk[:, h_idx], rf_pred_risk, threshold=rf_thresh)
        roc_auc = compute_roc_auc(y_test_risk[:, h_idx], rf_pred_risk)
        pr_auc = compute_pr_auc(y_test_risk[:, h_idx], rf_pred_risk)
        all_neg_acc = float((y_test_risk[:, h_idx] == 0).mean())

        rf_metrics = {
            "horizon_seconds": h_sec,
            "threshold": rf_thresh,
            "precision": round(eval_res.precision, 4),
            "recall": round(eval_res.recall, 4),
            "f1": round(eval_res.f1, 4),
            "fpr": round(eval_res.fpr, 4),
            "accuracy": round(compute_acc(eval_res), 4),
            "all_negative_baseline_accuracy": round(all_neg_acc, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "confusion_matrix": eval_res.confusion_matrix.to_dict(),
        }
        rf_results[str(h_sec)] = rf_metrics
        print(f"  +{h_sec}s -> F1: {rf_metrics['f1']:.4f} | Prec: {rf_metrics['precision']:.4f} | "
              f"Rec: {rf_metrics['recall']:.4f} | ROC-AUC: {rf_metrics['roc_auc']}")

    # 8C. Persistence Baseline (True ground truth state at time t persisted to future)
    print("\n--- Persistence Baseline (True State Persistence from time t) on Test Split ---")
    persist_results: Dict[str, Any] = {}
    state_train_2d = y_train_curr[:, np.newaxis]
    state_val_2d = y_val_curr[:, np.newaxis]
    state_test_2d = y_test_curr[:, np.newaxis]

    for h_idx, h_sec in enumerate(HORIZONS):
        pb = PersistenceBaseline(current_state_column=0, threshold=0.5)
        pb.fit(state_train_2d, y_train_risk[:, h_idx])
        val_pb_risk = pb.predict_risk(state_val_2d)
        pb_thresh = optimize_threshold(y_val_risk[:, h_idx], val_pb_risk, metric="f1")
        pb.threshold = pb_thresh

        pb_pred_risk = pb.predict_risk(state_test_2d)
        eval_res = evaluate_risk(y_test_risk[:, h_idx], pb_pred_risk, threshold=pb_thresh)
        roc_auc = compute_roc_auc(y_test_risk[:, h_idx], pb_pred_risk)
        pr_auc = compute_pr_auc(y_test_risk[:, h_idx], pb_pred_risk)
        all_neg_acc = float((y_test_risk[:, h_idx] == 0).mean())

        pb_metrics = {
            "horizon_seconds": h_sec,
            "threshold": pb_thresh,
            "precision": round(eval_res.precision, 4),
            "recall": round(eval_res.recall, 4),
            "f1": round(eval_res.f1, 4),
            "fpr": round(eval_res.fpr, 4),
            "accuracy": round(compute_acc(eval_res), 4),
            "all_negative_baseline_accuracy": round(all_neg_acc, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "confusion_matrix": eval_res.confusion_matrix.to_dict(),
        }
        persist_results[str(h_sec)] = pb_metrics
        print(f"  +{h_sec}s -> F1: {pb_metrics['f1']:.4f} | Prec: {pb_metrics['precision']:.4f} | Rec: {pb_metrics['recall']:.4f}")

    # 9. Class Imbalance Breakdown
    train_pos = int((y_train_risk == 1).sum())
    train_neg = int((y_train_risk == 0).sum())
    val_pos = int((y_val_risk == 1).sum())
    val_neg = int((y_val_risk == 0).sum())
    test_pos = int((y_test_risk == 1).sum())
    test_neg = int((y_test_risk == 0).sum())

    imbalance_stats = {
        "train": {"positive": train_pos, "negative": train_neg, "ratio": round(train_pos / max(train_neg, 1), 4)},
        "val": {"positive": val_pos, "negative": val_neg, "ratio": round(val_pos / max(val_neg, 1), 4)},
        "test": {"positive": test_pos, "negative": test_neg, "ratio": round(test_pos / max(test_neg, 1), 4)},
    }

    # 10. Assemble Full Auditable Report with Quarantine Provenance
    evaluation_report = {
        "_quarantine_provenance": {
            "status": "SYNTHETIC_DEMONSTRATION_ONLY",
            "dataset": "CANONICAL_SYNTHETIC_INTEGRATION_DATA",
            "warning": (
                "REAL DATA EXPERIMENT NOT EXECUTED - RAW CSE-CIC-IDS2018 TELEMETRY PENDING EXTRACTION. "
                "This artifact documents end-to-end integration and architectural verification only. "
                "Official benchmark performance on real CSE-CIC-IDS2018 network traffic is pending full raw dataset extraction."
            ),
            "generated_by": "src/eval/evaluate_system.py",
            "num_test_sequences": len(X_test),
            "num_train_sequences": len(X_train),
            "num_val_sequences": len(X_val),
            "canonical_features": len(feature_columns),
            "baseline_input_features": 410,
        },
        "class_imbalance": imbalance_stats,
        "frozen_thresholds": frozen_thresholds,
        "world_model": world_model_results,
        "baselines": {
            "logistic_regression": lr_results,
            "random_forest": rf_results,
            "persistence": persist_results,
        },
    }

    # Save to artifacts/evaluation_metrics.json
    out_path = ARTIFACT_DIR / "evaluation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    print(f"\nSaved comprehensive evaluation report to: {out_path}")
    print("=" * 70)

    return evaluation_report


if __name__ == "__main__":
    run_full_evaluation()
