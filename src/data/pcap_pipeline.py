"""
PCAP to Canonical 41-Feature Ingestion and Forecasting Pipeline.

Connects Aman's streaming PCAP parsing, packet feature extraction, and flow extraction
directly into Aryan's authoritative 41-feature canonical integration architecture:

    PCAP file
       ↓
    Packet Feature Extractor (19 packet features)
       +
    Flow Feature Extractor & Aggregator (22 flow features)
       ↓
    Traffic Fusion on (src_ip, window_id)
       ↓
    Canonical 41-Feature Matrix Validation
       ↓
    Canonical TrafficWindow Dataclass Construction
       ↓
    Canonical Temporal Sequence Builder (10 windows × 41 features)
       ↓
    Existing Scaler & LSTM WorldModel (+10s, +20s, +30s horizons)
       ↓
    Downstream XAI, MITRE attribution, and Dashboard
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.features.build_packet_features import build_packet_features
from src.flow.flow_extractor import aggregate_flows_to_host_windows, extract_flow_features
from src.fusion.traffic_fusion import (
    JoinValidationReport,
    fuse_flow_and_packet_dfs,
    fused_df_to_traffic_windows,
)
from src.inference import forecast, load_artifacts
from src.schemas.features import (
    CANONICAL_MODEL_FEATURE_NAMES,
    FEATURE_ALIASES,
    FORBIDDEN_FEATURE_NAMES,
    validate_feature_names,
)
from src.schemas.traffic import TrafficWindow
from src.temporal.sequences import TemporalSequenceBatch, build_sequences


def extract_pcap_to_fused_dataframe(
    pcap_path: str | Path,
    output_parquet: Optional[str | Path] = None,
    max_packets: Optional[int] = None,
    how: str = "left",
) -> Tuple[pd.DataFrame, JoinValidationReport]:
    """
    Ingest a PCAP file, extract 19 packet features and 22 flow features, and fuse
    on (src_ip, window_id) to produce the exact 41 canonical model features.

    Args:
        pcap_path: Path to PCAP file on disk.
        output_parquet: Optional path to save fused table.
        max_packets: Optional upper bound on packets parsed.
        how: Merge strategy ('left', 'inner', 'outer').

    Returns:
        Tuple of (fused_41_df, JoinValidationReport).
    """
    pcap_str = str(pcap_path)
    if not os.path.isfile(pcap_str):
        raise FileNotFoundError(f"PCAP file not found: {pcap_str}")

    # 1. Packet extraction (19 packet features)
    packet_records = build_packet_features(pcap_str, max_packets=max_packets)
    packet_df = pd.DataFrame(packet_records)

    # 2. Flow extraction (22 flow features)
    flow_records = extract_flow_features(pcap_str, max_packets=max_packets)
    if flow_records:
        flow_agg_df = aggregate_flows_to_host_windows(flow_records)
    else:
        flow_agg_df = pd.DataFrame()

    if packet_df.empty and flow_agg_df.empty:
        # Return empty DataFrame with canonical 41-feature columns
        cols = ["src_ip", "source_host", "window_id", "timestamp"] + list(CANONICAL_MODEL_FEATURE_NAMES)
        empty_df = pd.DataFrame(columns=cols)
        report = JoinValidationReport(
            packet_rows_before=0,
            flow_rows_before=0,
            packet_host_windows=0,
            flow_host_windows=0,
            rows_after_join=0,
            matched_windows=0,
            unmatched_packet_windows=0,
            unmatched_flow_windows=0,
            packet_coverage_pct=0.0,
            flow_coverage_pct=0.0,
            has_row_multiplication=False,
            has_row_loss=False,
        )
        return empty_df, report

    # 3. Fuse packet and flow features on (src_ip, window_id)
    fused_df, report = fuse_flow_and_packet_dfs(flow_agg_df, packet_df, how=how)

    # 4. Standardize column aliases
    rename_map = {k: v for k, v in FEATURE_ALIASES.items() if k in fused_df.columns and v not in fused_df.columns}
    if rename_map:
        fused_df = fused_df.rename(columns=rename_map)

    # Ensure source_host metadata exists
    if "source_host" not in fused_df.columns and "src_ip" in fused_df.columns:
        fused_df["source_host"] = fused_df["src_ip"]

    # 5. Guarantee presence of all 41 canonical model features (fill missing with 0.0)
    for feat in CANONICAL_MODEL_FEATURE_NAMES:
        if feat not in fused_df.columns:
            fused_df[feat] = 0.0
        else:
            fused_df[feat] = fused_df[feat].fillna(0.0).astype(float)

    # Validate complete 41-feature order
    validate_feature_names(CANONICAL_MODEL_FEATURE_NAMES, expected_order=CANONICAL_MODEL_FEATURE_NAMES)

    # Order columns: metadata first, then 41 canonical features in authoritative order
    meta_cols = [c for c in ["src_ip", "source_host", "window_id", "timestamp", "window_start", "window_end"] if c in fused_df.columns]
    fused_df = fused_df[meta_cols + list(CANONICAL_MODEL_FEATURE_NAMES)].copy()

    if output_parquet is not None:
        os.makedirs(os.path.dirname(os.path.abspath(str(output_parquet))), exist_ok=True)
        fused_df.to_parquet(str(output_parquet), index=False)

    return fused_df, report


def pcap_to_traffic_windows(
    pcap_path: str | Path,
    label_col: Optional[str] = None,
    max_packets: Optional[int] = None,
) -> Tuple[List[TrafficWindow], JoinValidationReport]:
    """
    Ingest a PCAP and construct validated canonical TrafficWindow instances
    with all 41 canonical features in authoritative order.
    """
    fused_df, report = extract_pcap_to_fused_dataframe(pcap_path, max_packets=max_packets)
    windows = fused_df_to_traffic_windows(
        fused_df,
        label_col=label_col,
        feature_names=CANONICAL_MODEL_FEATURE_NAMES,
        allow_prototype_partial_features=False,
    )
    return windows, report


def pcap_to_temporal_sequences(
    pcap_path: str | Path,
    history_length: int = 10,
    forecast_horizons: Tuple[int, ...] = (1, 2, 3),
    window_seconds: int = 10,
    max_packets: Optional[int] = None,
) -> Tuple[TemporalSequenceBatch, JoinValidationReport]:
    """
    Convert a PCAP directly into a canonical 10-window × 41-feature temporal sequence batch.
    """
    traffic_windows, report = pcap_to_traffic_windows(pcap_path, max_packets=max_packets)
    if not traffic_windows:
        empty_X = np.empty((0, history_length, 41), dtype=np.float32)
        empty_y = np.empty((0, len(forecast_horizons)), dtype=np.float32)
        batch = TemporalSequenceBatch(
            X=empty_X,
            y=empty_y,
            hosts=[],
            prediction_times=[],
            target_times=[],
            forecast_steps=forecast_horizons,
            offsets_seconds=tuple(window_seconds * h for h in forecast_horizons),
        )
        return batch, report

    features = np.array([tw.features for tw in traffic_windows], dtype=np.float32)
    targets = np.array([tw.label if tw.label is not None else 0.0 for tw in traffic_windows], dtype=np.float32)
    timestamps = [tw.timestamp for tw in traffic_windows]
    hosts = [tw.source_host for tw in traffic_windows]

    batch = build_sequences(
        features=features,
        targets=targets,
        timestamps=timestamps,
        source_hosts=hosts,
        history_length=history_length,
        forecast_horizons=forecast_horizons,
        window_seconds=window_seconds,
        return_metadata=True,
    )
    return batch, report


def pcap_to_model_forecast(
    pcap_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    max_packets: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full end-to-end execution from raw PCAP through the frozen LSTM WorldModel.

    PCAP
      → Packet features (19)
      → Flow features (22)
      → Fusion on (src_ip, window_id)
      → 41 Canonical features
      → Temporal sequence (10 × 41)
      → Frozen Scaler & LSTM WorldModel
      → +10s, +20s, +30s Risk Forecasts
    """
    batch, report = pcap_to_temporal_sequences(pcap_path, max_packets=max_packets)
    if batch.X.shape[0] == 0:
        return {
            "pcap_path": str(pcap_path),
            "status": "insufficient_windows",
            "message": "PCAP contained fewer than 13 consecutive windows required for 10-window history + 3 forecast steps",
            "coverage_report": report.as_dict(),
            "forecast_risks": [],
        }

    # Load frozen production artifacts
    model, scaler, feature_order, stage_classes = load_artifacts(Path(artifacts_dir))
    if model is None or scaler is None:
        raise RuntimeError(f"Could not load production model or scaler from {artifacts_dir}")

    # Ensure canonical feature order contract
    assert list(feature_order) == list(CANONICAL_MODEL_FEATURE_NAMES), (
        "Artifact feature_order does not match CANONICAL_MODEL_FEATURE_NAMES"
    )

    # Scale the (N, 10, 41) features using the frozen scaler
    N, T, F = batch.X.shape
    X_reshaped = batch.X.reshape(N * T, F)
    X_scaled = scaler.transform(X_reshaped).reshape(N, T, F).astype(np.float32)

    # Run inference through the production model
    predictions = []
    for i in range(N):
        X_sample = np.expand_dims(X_scaled[i], axis=0)
        risk_preds, pred_stage, stage_conf = forecast(model, X_sample, stage_classes)
        predictions.append({
            "host": batch.hosts[i],
            "prediction_time": str(batch.prediction_times[i]),
            "forecast_risks": [float(r) for r in risk_preds],
            "risk_10s": float(risk_preds[0]),
            "risk_20s": float(risk_preds[1]),
            "risk_30s": float(risk_preds[2]),
            "predicted_stage": pred_stage,
            "stage_confidence": float(stage_conf),
        })

    return {
        "pcap_path": str(pcap_path),
        "status": "success",
        "num_sequences": N,
        "input_tensor_shape": list(batch.X.shape),
        "coverage_report": report.as_dict(),
        "predictions": predictions,
    }
