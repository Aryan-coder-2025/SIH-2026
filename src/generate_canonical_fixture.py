"""
Canonical Feature Matrix Generator and Fusion Pipeline for SIH26153.
Project: SIH26153 - AI-Based Network Attack Forecasting from Network Traffic Data

Generates the canonical 41-feature dataset via flow-level and packet-level fusion:
    Flow Features (22 features: NetFlow / Shaurya contract)
            +
    Packet Features (19 features: PCAP statistics / Aman contract)
            ↓
    Traffic Fusion (src.fusion.traffic_fusion.fuse_flow_and_packet_dfs)
            ↓
    Canonical 41 Model Features (CANONICAL_MODEL_FEATURE_NAMES)
            ↓
    data/processed/feature_matrix.parquet  (Authoritative for src.train)
    data/mock/canonical_feature_matrix.csv (Authoritative CSV fixture)

Enforces:
    1. Continuous 10-second temporal windows per host.
    2. Exact 41-feature canonical order from src/schemas/features.py.
    3. Realistic multi-phase episodes: Benign -> Discovery (Recon) -> Impact (DoS/Attack) -> Recovery.
    4. Anti-leakage: targets and host identifiers are isolated from feature tensors.
    5. Clean execution of fusion with 100% window alignment and 0 row multiplication.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.traffic_fusion import fuse_flow_and_packet_dfs
from src.schemas.features import (
    CANONICAL_FLOW_FEATURE_NAMES,
    CANONICAL_MODEL_FEATURE_NAMES,
    CANONICAL_PACKET_FEATURE_NAMES,
    validate_feature_names,
)
from src.temporal.sequences import build_sequences_from_dataframe

SEED = 42
WINDOW_SECONDS = 10
WINDOWS_PER_HOST = 240  # 40 minutes of continuous 10s windows per host
START_TIME = datetime(2026, 9, 18, 8, 0, 0, tzinfo=timezone.utc)

HOSTS = [
    "192.168.10.10",
    "192.168.10.11",
    "192.168.10.12",
    "192.168.10.13",
    "192.168.10.14",
    "192.168.10.15",
]

# Attack episodes structured to ensure all stages (Benign, Discovery, Impact, Recovery)
# appear across train (0-167), val (168-203), and test (204-239) partitions.
ATTACK_SCHEDULE = {
    "192.168.10.10": [
        {"disc_start": 35, "att_start": 50, "att_end": 75, "rec_end": 88},
        {"disc_start": 208, "att_start": 218, "att_end": 230, "rec_end": 238},
    ],
    "192.168.10.11": [
        {"disc_start": 70, "att_start": 85, "att_end": 110, "rec_end": 122},
        {"disc_start": 172, "att_start": 182, "att_end": 195, "rec_end": 202},
    ],
    "192.168.10.12": [
        {"disc_start": 20, "att_start": 35, "att_end": 60, "rec_end": 72},
        {"disc_start": 212, "att_start": 222, "att_end": 232, "rec_end": 239},
    ],
    "192.168.10.13": [
        {"disc_start": 95, "att_start": 110, "att_end": 135, "rec_end": 148},
        {"disc_start": 175, "att_start": 185, "att_end": 198, "rec_end": 203},
    ],
    "192.168.10.14": [
        {"disc_start": 55, "att_start": 70, "att_end": 95, "rec_end": 107},
    ],
    "192.168.10.15": [
        {"disc_start": 125, "att_start": 140, "att_end": 160, "rec_end": 167},
    ],
}

HOST_BASELINES = {
    "192.168.10.10": {"flows": 14.0, "bytes": 18000.0, "packets": 120.0, "ttl": 64.0, "tcp_win": 29200.0},
    "192.168.10.11": {"flows": 20.0, "bytes": 26000.0, "packets": 170.0, "ttl": 64.0, "tcp_win": 32768.0},
    "192.168.10.12": {"flows": 10.0, "bytes": 13000.0, "packets": 85.0, "ttl": 128.0, "tcp_win": 16384.0},
    "192.168.10.13": {"flows": 24.0, "bytes": 32000.0, "packets": 220.0, "ttl": 64.0, "tcp_win": 43800.0},
    "192.168.10.14": {"flows": 16.0, "bytes": 20000.0, "packets": 140.0, "ttl": 128.0, "tcp_win": 28960.0},
    "192.168.10.15": {"flows": 18.0, "bytes": 24000.0, "packets": 160.0, "ttl": 64.0, "tcp_win": 32768.0},
}


def get_episode_state(host: str, window_idx: int) -> tuple[str, int, float, float]:
    """
    Returns (stage, is_malicious, anomaly_intensity [0..1], base_risk).
    Stages align with MITRE mapping expectations: 'Benign', 'Discovery', 'Impact', 'Recovery'.
    """
    episodes = ATTACK_SCHEDULE.get(host, [])
    for ep in episodes:
        disc_start = ep["disc_start"]
        att_start = ep["att_start"]
        att_end = ep["att_end"]
        rec_end = ep["rec_end"]

        if disc_start <= window_idx < att_start:
            # Reconnaissance / Discovery precursor phase
            progress = (window_idx - disc_start) / max(1, att_start - disc_start)
            intensity = 0.20 + 0.35 * progress
            risk = 0.15 + 0.35 * progress
            return "Discovery", 0, intensity, risk

        if att_start <= window_idx < att_end:
            # Active attack / Impact phase
            att_len = max(1, att_end - att_start)
            progress = (window_idx - att_start) / att_len
            intensity = 0.75 + 0.25 * np.sin(np.pi * progress)
            risk = 0.85 + 0.14 * np.sin(np.pi * progress)
            return "Impact", 1, intensity, risk

        if att_end <= window_idx < rec_end:
            # Recovery cooldown phase
            progress = (window_idx - att_end) / max(1, rec_end - att_end)
            intensity = 0.40 * (1.0 - progress)
            risk = 0.35 * (1.0 - progress)
            return "Recovery", 0, intensity, risk

    return "Benign", 0, 0.0, 0.02


def generate_flow_and_packet_dfs(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic Flow (22 features) and Packet (19 features) DataFrames
    with common join keys ('src_ip', 'window_id') and continuous temporal alignment.
    """
    rng = np.random.default_rng(seed)
    flow_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []

    for host in HOSTS:
        baseline = HOST_BASELINES[host]
        base_flows = baseline["flows"]
        base_bytes = baseline["bytes"]
        base_packets = baseline["packets"]
        base_ttl = baseline["ttl"]
        base_tcp_win = baseline["tcp_win"]

        for w_idx in range(WINDOWS_PER_HOST):
            ts = START_TIME + timedelta(seconds=w_idx * WINDOW_SECONDS)
            stage, is_malicious, intensity, base_risk = get_episode_state(host, w_idx)

            smooth_cycle = 1.0 + 0.05 * np.sin(w_idx / 20.0) + 0.03 * np.cos(w_idx / 8.0)
            noise = rng.normal(1.0, 0.04)

            # -------------------------------------------------------------
            # Attack vs Benign Modulations
            # -------------------------------------------------------------
            if stage == "Discovery":
                # Multi-port scanning / probing pattern
                flow_mult = 1.8 + intensity * 1.5
                byte_mult = 1.2 + intensity * 0.8
                packet_mult = 1.5 + intensity * 1.0
                syn_val = 18.0 + intensity * 35.0 + rng.normal(0, 3)
                ports_val = 16.0 + intensity * 30.0 + rng.normal(0, 2)
                seq_port_val = min(0.95, 0.55 + intensity * 0.30 + rng.normal(0, 0.05))
                port_scan_score = min(1.0, 0.65 + intensity * 0.35)
                iat_val = max(0.005, 0.08 * (1.0 - 0.4 * intensity) + rng.normal(0, 0.005))
                ttl_val = base_ttl - 2.0
                rst_val = 6.0 + intensity * 8.0
            elif stage == "Impact":
                # High-volume flood / DoS attack
                flow_mult = 3.5 + intensity * 4.0
                byte_mult = 4.0 + intensity * 8.0
                packet_mult = 5.0 + intensity * 9.0
                syn_val = 60.0 + intensity * 120.0 + rng.normal(0, 10)
                ports_val = 8.0 + intensity * 12.0
                seq_port_val = 0.20 + rng.normal(0, 0.05)
                port_scan_score = 0.40
                iat_val = max(0.001, 0.015 * (1.0 - 0.7 * intensity) + rng.normal(0, 0.002))
                ttl_val = base_ttl - 8.0
                rst_val = 25.0 + intensity * 35.0
            elif stage == "Recovery":
                # Post-attack decay
                flow_mult = 1.0 + 0.8 * intensity
                byte_mult = 1.0 + 1.2 * intensity
                packet_mult = 1.0 + 1.5 * intensity
                syn_val = 3.0 + 8.0 * intensity
                ports_val = 3.0 + 4.0 * intensity
                seq_port_val = 0.10
                port_scan_score = 0.15
                iat_val = max(0.02, 0.10 + rng.normal(0, 0.01))
                ttl_val = base_ttl
                rst_val = 2.0 + 4.0 * intensity
            else:
                # Benign baseline
                flow_mult = 1.0
                byte_mult = 1.0
                packet_mult = 1.0
                syn_val = 2.0 + rng.uniform(0, 2)
                ports_val = 2.0 + rng.uniform(0, 2)
                seq_port_val = 0.05 + rng.uniform(0, 0.05)
                port_scan_score = 0.10
                iat_val = max(0.03, 0.14 + rng.normal(0, 0.02))
                ttl_val = base_ttl + rng.normal(0, 0.5)
                rst_val = 0.5 + rng.uniform(0, 1.0)

            flows = max(1.0, base_flows * smooth_cycle * noise * flow_mult)
            p_total = max(5.0, base_packets * smooth_cycle * noise * packet_mult)
            b_total = max(100.0, base_bytes * smooth_cycle * noise * byte_mult)
            b_mean = b_total / flows
            p_mean = p_total / flows

            # -------------------------------------------------------------
            # 22 Flow Features (NetFlow / Shaurya canonical contract)
            # -------------------------------------------------------------
            flow_row = {
                "src_ip": host,
                "window_id": w_idx,
                # 22 canonical flow features in exact order:
                "flow_count": round(float(flows), 2),
                "unique_dst_ip_count": round(float(max(1.0, min(flows, 2.0 + intensity * 6.0))), 2),
                "unique_dst_port_count": round(float(max(1.0, ports_val)), 2),
                "bytes_total": round(float(b_total), 2),
                "bytes_mean": round(float(b_mean), 2),
                "bytes_std": round(float(max(10.0, b_mean * 0.45)), 2),
                "packets_total": round(float(p_total), 2),
                "packets_mean": round(float(p_mean), 2),
                "duration_mean": round(float(max(0.05, 1.8 + rng.normal(0, 0.2))), 4),
                "duration_std": round(float(max(0.01, 0.6 + rng.normal(0, 0.1))), 4),
                "iat_mean": round(float(iat_val), 6),
                "iat_std": round(float(max(0.001, iat_val * 0.5)), 6),
                "iat_max": round(float(max(iat_val * 1.5, 0.8 + rng.normal(0, 0.1))), 6),
                "tcp_flow_ratio": round(float(min(1.0, max(0.60, 0.85 + rng.normal(0, 0.03)))), 4),
                "udp_flow_ratio": round(float(min(0.40, max(0.0, 0.15 + rng.normal(0, 0.03)))), 4),
                "bidirectional_ratio": round(float(min(1.0, max(0.10, 0.50 + intensity * 0.25))), 4),
                "syn_count": round(float(max(0.0, syn_val)), 2),
                "ack_count": round(float(max(1.0, p_total * 0.25)), 2),
                "fin_count": round(float(max(0.0, flows * 0.8 + rng.normal(0, 1))), 2),
                "rst_count": round(float(max(0.0, rst_val)), 2),
                "psh_count": round(float(max(0.0, p_total * 0.15 + rng.normal(0, 2))), 2),
                "urg_count": 0.0,
            }
            flow_rows.append(flow_row)

            # -------------------------------------------------------------
            # 19 Packet Features (PCAP / Aman canonical contract)
            # -------------------------------------------------------------
            ttl_mean_val = max(32.0, min(128.0, ttl_val))
            p_count_val = max(5.0, p_total)
            packet_row = {
                "src_ip": host,
                "window_id": w_idx,
                "timestamp": ts.isoformat(),
                # Metadata / Targets kept with packet table for post-fusion verification
                "is_malicious": int(is_malicious),
                "stage": stage,
                # 19 canonical packet features in exact order:
                "packet_count": round(float(p_count_val), 2),
                "ttl_mean": round(float(ttl_mean_val), 4),
                "ttl_std": round(float(max(0.1, 0.8 + intensity * 3.5)), 4),
                "ttl_min": round(float(max(30.0, ttl_mean_val - 4.0)), 2),
                "ttl_max": round(float(min(128.0, ttl_mean_val + 4.0)), 2),
                "tcp_window_mean": round(float(max(2048.0, base_tcp_win * (1.0 - 0.3 * intensity))), 2),
                "tcp_window_std": round(float(max(100.0, 800.0 + intensity * 1500.0)), 2),
                "fragment_count": round(float(1.0 if intensity > 0.6 and rng.random() < 0.4 else 0.0), 2),
                "payload_mean": round(float(max(30.0, 350.0 + intensity * 150.0 + rng.normal(0, 20))), 2),
                "payload_std": round(float(max(10.0, 160.0 + intensity * 60.0)), 2),
                "payload_min": 0.0,
                "payload_max": round(float(min(1460.0, 1200.0 + intensity * 260.0)), 2),
                "retransmission_count": round(float(max(0.0, (1.0 + intensity * 15.0) + rng.normal(0, 1))), 2),
                "port_scan_score": round(float(port_scan_score), 4),
                "sequential_port_ratio": round(float(max(0.0, min(1.0, seq_port_val))), 4),
                "unique_dst_ports": round(float(max(1.0, ports_val)), 2),
                "packet_iat_mean": round(float(iat_val), 6),
                "packet_iat_std": round(float(max(0.0005, iat_val * 0.4)), 6),
                "packet_iat_max": round(float(max(iat_val * 1.8, 0.5)), 6),
            }
            packet_rows.append(packet_row)

    flow_df = pd.DataFrame(flow_rows)
    packet_df = pd.DataFrame(packet_rows)
    return flow_df, packet_df


def build_and_save_canonical_dataset() -> pd.DataFrame:
    """
    Executes flow + packet generation, traffic fusion, schema enforcement,
    and saves to both authoritative parquet and CSV fixture.
    """
    print("=" * 70)
    print("CANONICAL 41-FEATURE GENERATOR & TRAFFIC FUSION")
    print("=" * 70)

    flow_df, packet_df = generate_flow_and_packet_dfs(seed=SEED)
    print(f"Generated flow records   : {len(flow_df)} rows")
    print(f"Generated packet records : {len(packet_df)} rows")

    # Step 1: Execute authoritative traffic fusion
    fused_df, report = fuse_flow_and_packet_dfs(flow_df, packet_df, how="inner")
    print("\nFusion Cardinality & Coverage Audit:")
    print(f"  Packet rows before     : {report.packet_rows_before}")
    print(f"  Flow rows before       : {report.flow_rows_before}")
    print(f"  Fused rows after join  : {report.rows_after_join}")
    print(f"  Packet coverage        : {report.packet_coverage_pct}%")
    print(f"  Flow coverage          : {report.flow_coverage_pct}%")
    print(f"  Row multiplication     : {report.has_row_multiplication}")
    print(f"  Row loss               : {report.has_row_loss}")

    assert not report.has_row_multiplication, "Fusion produced row multiplication!"
    assert not report.has_row_loss, "Fusion produced row loss!"
    assert report.packet_coverage_pct == 100.0, "Incomplete packet coverage in fusion!"
    assert report.flow_coverage_pct == 100.0, "Incomplete flow coverage in fusion!"

    # Step 2: Establish canonical source_host identity
    fused_df["source_host"] = fused_df["src_ip"]

    # Step 3: Validate canonical feature columns and ordering
    feature_cols = list(CANONICAL_MODEL_FEATURE_NAMES)
    validate_feature_names(feature_cols, expected_order=CANONICAL_MODEL_FEATURE_NAMES)
    assert len(feature_cols) == 41, f"Expected 41 canonical features, got {len(feature_cols)}"

    for col in feature_cols:
        assert col in fused_df.columns, f"Missing canonical feature: {col}"
        assert pd.api.types.is_numeric_dtype(fused_df[col]), f"Feature {col} is non-numeric!"
        assert not fused_df[col].isna().any(), f"Feature {col} contains NaNs!"
        assert not np.isinf(fused_df[col].to_numpy()).any(), f"Feature {col} contains Infs!"

    # Step 4: Arrange clean DataFrame with authoritative ordering
    metadata_cols = ["source_host", "src_ip", "timestamp", "window_id", "is_malicious", "stage"]
    ordered_cols = ["source_host", "timestamp"] + feature_cols + ["is_malicious", "stage", "src_ip", "window_id"]
    final_df = fused_df[ordered_cols].sort_values(["source_host", "timestamp"]).reset_index(drop=True)

    # Step 5: Save to data/processed/feature_matrix.parquet (authoritative for training)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = processed_dir / "feature_matrix.parquet"
    final_df.to_parquet(parquet_path, index=False)
    print(f"\nSaved authoritative Parquet: {parquet_path} ({len(final_df)} rows, {len(final_df.columns)} columns)")

    # Step 6: Save to data/mock/canonical_feature_matrix.csv (authoritative CSV fixture)
    mock_dir = Path("data/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)
    csv_path = mock_dir / "canonical_feature_matrix.csv"
    final_df.to_csv(csv_path, index=False)
    print(f"Saved canonical CSV fixture : {csv_path}")

    # Step 7: Sequence construction verification test
    print("\nValidating sequence construction from final DataFrame...")
    X, y_risk, y_stage = build_sequences_from_dataframe(
        df=final_df,
        feature_columns=feature_cols,
        sequence_length=10,
        horizon=3,
        entity_column="source_host",
        timestamp_column="timestamp",
        risk_column="is_malicious",
        stage_column="stage",
    )
    print(f"  X shape       : {X.shape} (Expected: N x 10 x 41)")
    print(f"  y_risk shape  : {y_risk.shape} (Expected: N x 3)")
    print(f"  y_stage shape : {y_stage.shape} (Expected: N)")
    assert X.shape[1] == 10 and X.shape[2] == 41, f"Unexpected X shape: {X.shape}"
    assert y_risk.shape[1] == 3, f"Unexpected y_risk shape: {y_risk.shape}"
    assert len(X) == len(y_risk) == len(y_stage)

    print("\nCanonical Dataset Verification PASSED!")
    print(f"  Total hosts: {final_df['source_host'].nunique()}")
    print(f"  Stage distribution:\n{final_df['stage'].value_counts().to_string()}")
    print(f"  Malicious windows: {int(final_df['is_malicious'].sum())} / {len(final_df)}")
    print("=" * 70)

    return final_df


if __name__ == "__main__":
    build_and_save_canonical_dataset()
