"""
Single-Command Demonstration Entry Point for SIH26153.
Project: SIH26153 - AI-Based Network Attack Forecasting from Network Traffic Data

Executes the complete end-to-end inference and attribution chain:
1. Loads prepared integrated feature matrix (data/processed/feature_matrix.parquet).
2. Loads trained model artifacts (artifacts/world_model.pt, scaler, metadata).
3. Selects a source host (supports benign and attack phase demonstration).
4. Runs strict 41-feature multi-horizon WorldModel inference (+10s, +20s, +30s).
5. Predicts future risk trajectory and auxiliary cyber attack stage.
6. Computes honest feature attributions and multi-horizon SOC explainability.
7. Evaluates behavioral evidence heuristics.
8. Maps findings to MITRE ATT&CK candidate techniques.
9. Formulates defender recommendations.
10. Prints formatted console summary and saves artifacts/demo_output.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src._win_torch_fix  # noqa: F401

import numpy as np
import pandas as pd
import torch

from src.audit.ledger import TamperEvidentLedger
from src.explain.integrated_gradients import IntegratedGradientsAttributor
from src.inference import forecast, load_artifacts, prepare_input
from src.pipeline import CyberForecastPipeline
from src.schemas.features import CANONICAL_MODEL_FEATURE_NAMES, validate_feature_names
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_matrix.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def run_demo(
    host: Optional[str] = None,
    scenario: str = "attack",
    data_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_data_path = Path(data_path) if data_path is not None else DATA_PATH

    # 1. Load Data with Automatic Fixture Reconstruction if Missing
    if not resolved_data_path.is_file():
        from src.generate_canonical_fixture import build_and_save_canonical_dataset
        print(f"Feature matrix not found at {resolved_data_path}. Reconstructing canonical integration fixture...")
        build_and_save_canonical_dataset()

    df = pd.read_parquet(resolved_data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. Select Host & Target Window
    available_hosts = df["source_host"].unique().tolist()
    if host is None:
        if scenario == "attack":
            malicious_records = df[df["is_malicious"] == 1]
            if not malicious_records.empty:
                selected_host = str(malicious_records["source_host"].iloc[0])
                host_df = df[df["source_host"] == selected_host].sort_values("timestamp")
                attack_idx = host_df[host_df["is_malicious"] == 1].index[0]
                pos = host_df.index.get_loc(attack_idx)
                end_pos = min(len(host_df), max(10, pos + 5))
                slice_df = host_df.iloc[end_pos - 10 : end_pos].copy()
            else:
                selected_host = available_hosts[0]
                slice_df = df[df["source_host"] == selected_host].tail(10).copy()
        else:
            selected_host = available_hosts[0]
            slice_df = df[df["source_host"] == selected_host].head(10).copy()
    else:
        selected_host = host
        slice_df = df[df["source_host"] == selected_host].tail(10).copy()

    # 3. Load Artifacts
    model, scaler, feature_order, stage_classes = load_artifacts(ARTIFACT_DIR)
    validate_feature_names(feature_order, expected_order=CANONICAL_MODEL_FEATURE_NAMES)

    # 4. Prepare Inference Input (Strict 10 Historical Windows x 41 Features)
    if len(slice_df) < 10:
        raise ValueError(f"Selected slice for host {selected_host} has {len(slice_df)} windows, need 10.")

    X_raw = slice_df[feature_order].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X_raw)
    X_batch = np.expand_dims(X_scaled, axis=0)

    # 5. Run Multi-Horizon Forecast (+10s, +20s, +30s)
    risk_preds, pred_stage, stage_conf = forecast(model, X_batch, stage_classes)

    # 6. Compute Authentic 50-Step Integrated Gradients across All 3 Horizons
    # Strict contract: geometry (3, 10, 41), non-SHAP, no silent fallback to gradient*input
    attributor = IntegratedGradientsAttributor(model=model, steps=50)
    horizon_results = [
        attributor.attribute(
            x_input=torch.tensor(X_batch, dtype=torch.float32),
            horizon_idx=h,
            feature_names=feature_order,
        )
        for h in range(3)
    ]
    attribution_tensor = np.array([res.attributions for res in horizon_results])  # Shape (3, 10, 41)
    feature_attributions = [res.feature_importance for res in horizon_results]
    attribution_method = "Integrated Gradients (Path-Integrated Gradients attribution, 50 steps, non-SHAP)"

    # 7. End-to-End CyberForecastPipeline
    pipeline = CyberForecastPipeline()
    window_id = int(slice_df["window_id"].iloc[-1]) if "window_id" in slice_df.columns else int(slice_df.index[-1])
    last_timestamp = str(slice_df["timestamp"].iloc[-1])
    obs_malicious = int(slice_df["is_malicious"].iloc[-1]) if "is_malicious" in slice_df.columns else 0
    observed_state_str = "MALICIOUS" if obs_malicious == 1 else "BENIGN"

    result = pipeline.process_prediction(
        host_id=selected_host,
        window_id=window_id,
        timestamp=last_timestamp,
        forecast_risk_10s=float(risk_preds[0]),
        risk_timeline=risk_preds.tolist(),
        predicted_stage=pred_stage,
        temporal_features=X_raw,
        feature_attributions=feature_attributions,
        feature_names=feature_order,
        temporal_attribution_tensor=attribution_tensor,
        observed_state=observed_state_str,
    )

    # 8. Cryptographically Hash-Chained Tamper-Evident Audit Ledger
    ledger_path = ARTIFACT_DIR / "audit_ledger.json"
    if ledger_path.is_file():
        ledger = TamperEvidentLedger.load_from_file(str(ledger_path))
    else:
        ledger = TamperEvidentLedger()

    base_record_id = f"rec_{selected_host}_w{window_id:03d}_{scenario}"
    existing_ids = {b.record_id for b in ledger.chain}
    record_id = base_record_id
    counter = 1
    while record_id in existing_ids:
        record_id = f"{base_record_id}_{counter}"
        counter += 1

    audit_block = ledger.append_record(
        record_id=record_id,
        payload={
            "host_id": selected_host,
            "window_id": window_id,
            "timestamp": last_timestamp,
            "observed_state": observed_state_str,
            "forecast": [float(r) for r in risk_preds],
            "predicted_stage": pred_stage,
            "mitre_technique": result["mitre_attack"]["candidate_technique_id"],
        }
    )
    is_valid, violations = ledger.verify_ledger_integrity()
    ledger.save_to_file(str(ledger_path))

    # 9. Print Clean Standardized Output
    print("\n" + "=" * 55)
    print(" SIH26153 NETWORK ATTACK FORECAST & CYBER WORLD MODEL")
    print("=" * 55)
    print(f"Host: {result['host_id']}")
    print(f"Observed State: {observed_state_str}")
    print(f"Window Timestamp: {result['timestamp']}")
    print(f"Operational Urgency: {result['forecast']['urgency_level']}")
    print()
    print("Forecast (Multi-Horizon Probabilities):")
    print(f"  +10s : {risk_preds[0]:.4f}  (Decision Threshold: 0.050 -> {'MALICIOUS' if risk_preds[0]>=0.05 else 'BENIGN'})")
    print(f"  +20s : {risk_preds[1]:.4f}  (Decision Threshold: 0.900 -> {'MALICIOUS' if risk_preds[1]>=0.90 else 'BENIGN'})")
    print(f"  +30s : {risk_preds[2]:.4f}  (Decision Threshold: 0.750 -> {'MALICIOUS' if risk_preds[2]>=0.75 else 'BENIGN'})")
    print()
    print("Model Forecast:")
    print(f"  Predicted Stage: {pred_stage} ({stage_conf:.1%} confidence)")
    print()
    print("Evidence-Based ATT&CK Candidate:")
    mitre_tech = result["mitre_attack"]["candidate_technique_id"] or "T0000"
    mitre_name = result["mitre_attack"]["candidate_technique_name"] or "Nominal / Unmapped"
    mitre_conf = float(result["mitre_attack"]["mapping_confidence"] or 0.0)
    print(f"  Candidate Technique: {mitre_tech} — {mitre_name} (Mapping confidence: {mitre_conf:.1%})")
    print()
    print("Top Evidence:")
    evidence_items = result["evidence"]["observed_heuristics"]
    if evidence_items:
        for ev in evidence_items[:3]:
            print(f"  - {ev}")
    else:
        print("  - Nominal traffic baseline observed within expected thresholds.")
    print()
    print("SOC Recommendation:")
    recs = result["recommendations"]
    if recs:
        for rec in recs[:2]:
            print(f"  [{rec.get('priority', 'ADVISORY').upper()} - {rec.get('action_type', '')}] {rec.get('title', '')}")
            print(f"    Action: {rec.get('description', '')}")
            if rec.get('command_example'):
                print(f"    Command: {rec.get('command_example')}")
    else:
        print("  Continue routine passive monitoring.")
    print()
    print("Attribution Provenance:")
    print(f"  {attribution_method}")
    print()
    print("Audit Ledger Provenance:")
    print(f"  Tamper-Evident SHA-256 Chained Ledger: Block #{audit_block.index} [Hash: {audit_block.block_hash[:16]}...]")
    print(f"  Ledger Integrity Verified: {is_valid} (Decentralized blockchain NOT claimed)")
    print()
    print("Data Mode:")
    print("  SYNTHETIC DEMONSTRATION ONLY (Canonical Integration Fixture)")
    print("=" * 55 + "\n")

    # 10. Save demo output JSON
    save_path = output_path or (ARTIFACT_DIR / "demo_output.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved complete prediction record to: {save_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26153 Cyber Attack Forecasting Demo")
    parser.add_argument("--host", type=str, default=None, help="Target host IP (default: auto-select)")
    parser.add_argument("--scenario", type=str, choices=["attack", "benign"], default="attack",
                        help="Traffic scenario to demonstrate ('attack' or 'benign')")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to feature matrix parquet")
    args = parser.parse_args()

    run_demo(
        host=args.host,
        scenario=args.scenario,
        data_path=Path(args.data),
    )


if __name__ == "__main__":
    main()
