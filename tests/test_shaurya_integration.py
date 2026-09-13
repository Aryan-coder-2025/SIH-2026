from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from src.features.flow_features import aggregate_flow_features
from src.features.windowing import add_time_columns, add_window_labels
from src.schemas.traffic import TrafficWindow
from src.temporal.sequences import TemporalSequenceBatch, build_sequences_from_windows
from src.temporal.split import temporal_split_by_time


# ---------------------------------------------------------------------------
# Flow-to-Canonical Window Adapter
# ---------------------------------------------------------------------------
# We adapt Shaurya's flow window aggregation into the canonical TrafficWindow contract
# because downstream temporal splitting, continuity enforcement, and multi-step forecasting
# must operate on a single, validated network state identity (source_host, UTC timestamp,
# integer packet/byte totals, and numerical feature vector) rather than allowing disparate
# tabular column formats to leak into model sequence construction.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "flow_count",
    "unique_dst_ip_count",
    "unique_dst_port_count",
    "bytes_total",
    "bytes_mean",
    "bytes_std",
    "packets_total",
    "packets_mean",
    "duration_mean",
    "duration_std",
    "iat_mean",
    "iat_std",
    "iat_max",
    "tcp_flow_ratio",
    "udp_flow_ratio",
    "bidirectional_ratio",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
]


def flow_windows_to_canonical_traffic(df_merged: pd.DataFrame) -> list[TrafficWindow]:
    """
    Convert merged Shaurya flow feature & label rows into canonical TrafficWindow objects.
    """
    windows = []
    # Sort deterministically by host and window_id
    df_sorted = df_merged.sort_values(["src_ip", "window_id"]).reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        # Physical epoch seconds to UTC datetime
        epoch_sec = int(row["window_id"]) * 10
        ts = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)

        # Numerical feature vector
        feats = tuple(float(row[col]) for col in FEATURE_COLUMNS)

        # Target label mapping
        raw_cat = str(row.get("traffic_category", "normal"))
        label_val = 1.0 if raw_cat == "botnet" else 0.0

        windows.append(
            TrafficWindow(
                timestamp=ts,
                source_host=str(row["src_ip"]),
                packet_count=int(row["packets_total"]),
                byte_count=int(row["bytes_total"]),
                features=feats,
                label=label_val,
                window_id=f"{row['src_ip']}_{row['window_id']}",
            )
        )
    return windows


def test_shaurya_flow_pipeline_to_canonical_temporal_sequences():
    """
    CROSS-MODULE INTEGRATION SMOKE TEST:
    Shaurya Flow Preprocessing → Canonical TrafficWindow → Temporal Split → Sequence Construction.

    Proves:
    1. Shaurya's 10s flow aggregation maps seamlessly into TrafficWindow.
    2. Temporal split happens BEFORE sequence construction.
    3. Host isolation is strictly preserved (no cross-host sequences).
    4. Anti-leakage is maintained (train partition contains zero test observations).
    5. Final tensor dimensions strictly match (N, 10, 22) and target shape (N, 3).
    """
    # 1. Create realistic mock flows across 2 hosts over 20 consecutive 10s windows
    # Host A: 10.0.0.1 (20 windows)
    # Host B: 10.0.0.2 (20 windows)
    base_epoch = 1672563600  # 2023-01-01 09:00:00 UTC
    records = []

    for window_idx in range(20):
        t_sec = base_epoch + (window_idx * 10)
        dt_str = datetime.fromtimestamp(t_sec, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S.000000")

        # Host A: normal traffic, but windows 16..19 are botnet
        label_A = "botnet" if window_idx >= 16 else "normal"
        records.append({
            "StartTime": dt_str,
            "SrcAddr": "10.0.0.1",
            "DstAddr": "192.168.1.50",
            "Dport": 80,
            "TotBytes": 1500 + window_idx * 50,
            "TotPkts": 10,
            "Dur": 0.5,
            "Proto": "tcp",
            "Dir": "<->",
            "State": "S",
            "Label": label_A,
        })

        # Host B: strictly normal traffic
        records.append({
            "StartTime": dt_str,
            "SrcAddr": "10.0.0.2",
            "DstAddr": "192.168.1.100",
            "Dport": 443,
            "TotBytes": 3000,
            "TotPkts": 20,
            "Dur": 1.2,
            "Proto": "tcp",
            "Dir": "->",
            "State": "A",
            "Label": "normal",
        })

    df_raw = pd.DataFrame(records)

    # 2. Run Shaurya's flow processing
    df_timed = add_time_columns(df_raw)
    feats = aggregate_flow_features(df_timed)
    labels = add_window_labels(df_timed)
    merged = pd.merge(feats, labels, on=["src_ip", "window_id"], how="inner")

    assert len(merged) == 40  # 20 windows * 2 hosts
    assert "packets_total" in merged.columns
    assert "bytes_total" in merged.columns

    # 3. Adapt into canonical TrafficWindow representation
    windows = flow_windows_to_canonical_traffic(merged)
    assert len(windows) == 40

    for w in windows:
        assert isinstance(w, TrafficWindow)
        assert w.source_host in ("10.0.0.1", "10.0.0.2")
        assert len(w.features) == 22
        assert isinstance(w.packet_count, int)

    # 4. Enforce Temporal Split BEFORE Sequence Construction
    # Split chronologically at 70% (first 14 windows -> train, remaining 6 -> test)
    split = temporal_split_by_time(windows, train_ratio=0.7)
    assert len(split.train) == 28  # 14 windows * 2 hosts
    assert len(split.test) == 12   # 6 windows * 2 hosts

    # 5. Build sequences strictly per partition
    # Train: 14 contiguous windows per host -> 14 - 10 - 3 + 1 = 2 sequences per host -> 4 total
    train_batch: TemporalSequenceBatch = build_sequences_from_windows(split.train, return_metadata=True)
    assert len(train_batch.X) == 4
    assert train_batch.X.shape == (4, 10, 22)
    assert train_batch.y.shape == (4, 3)

    # Verify Host Isolation: Host A sequences contain strictly Host A data
    for seq_idx, host in enumerate(train_batch.hosts):
        if host == "10.0.0.1":
            # Host A has 10 packets per window
            # Feature index 6 is 'packets_total'
            pkt_feature = train_batch.X[seq_idx, :, 6]
            assert np.all(pkt_feature == 10.0)
        elif host == "10.0.0.2":
            # Host B has 20 packets per window
            pkt_feature = train_batch.X[seq_idx, :, 6]
            assert np.all(pkt_feature == 20.0)

    # Verify Anti-Leakage: botnet onset at window 16 is in the test partition
    # and MUST NOT appear in the training targets or features
    assert np.all(train_batch.y == 0.0), "No future botnet targets should leak into train batch"

    # Test partition: 6 windows per host (< 13 needed for history=10 + horizon=3) -> 0 sequences
    test_batch = build_sequences_from_windows(split.test, return_metadata=True)
    assert len(test_batch.X) == 0, "6 windows is insufficient to build 10-window history + 3 forecast"
