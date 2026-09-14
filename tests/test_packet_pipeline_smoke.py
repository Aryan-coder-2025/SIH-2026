"""
End-to-End Synthetic Integration Smoke Test.

Validates the full structural pipeline:
    Synthetic PCAP Packets
            ↓
    Packet Features (build_packet_features)
            ↓
    Flow Features (extract_flow_features)
            ↓
    Traffic Fusion (fuse_flow_and_packet_dfs)
            ↓
    Canonical TrafficWindow (fused_df_to_traffic_windows)
            ↓
    Temporal Train/Test Split (temporal_train_test_split)
            ↓
    Canonical Sequence Construction (src.temporal.sequences.build_sequences)
            ↓
    Model-Compatible PyTorch Tensor Verification

NOTE: Uses synthetic data strictly for structural and contract verification.
Does not claim any scientific predictive performance.
"""
import os
import tempfile
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest
import torch
from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap  # type: ignore

from src.features.build_packet_features import build_packet_features
from src.flow.flow_extractor import extract_flow_features
from src.fusion.traffic_fusion import fuse_flow_and_packet_dfs, fused_df_to_traffic_windows
from src.schemas.traffic import TrafficWindow
from src.temporal.sequences import build_sequences
from src.temporal.split import temporal_train_test_split


def generate_synthetic_pcap_session(
    pcap_path: str,
    target_host: str = "192.168.1.100",
    bg_host: str = "192.168.1.200",
    num_windows: int = 16,
    window_duration: float = 10.0,
) -> None:
    """
    Generate synthetic multi-window PCAP packets across multiple hosts.
    Ensures packets exist in each 10-second window from 0 to num_windows - 1.
    """
    packets = []

    for w_idx in range(num_windows):
        t_base = w_idx * window_duration

        # Target host packets: multiple TCP packets in each window
        # Packet 1 (SYN-like or data)
        p1 = (
            Ether()
            / IP(src=target_host, dst="10.0.0.1")
            / TCP(sport=10000 + w_idx, dport=80, seq=1000)
            / Raw(b"GET / HTTP/1.1\r\n")
        )
        p1.time = t_base + 1.0

        # Packet 2
        p2 = (
            Ether()
            / IP(src=target_host, dst="10.0.0.1")
            / TCP(sport=10000 + w_idx, dport=80, seq=1020)
            / Raw(b"User-Agent: Synthetic\r\n\r\n")
        )
        p2.time = t_base + 4.5

        # Background host packet in the same window (tests host isolation)
        p_bg = (
            Ether()
            / IP(src=bg_host, dst="8.8.8.8")
            / UDP(sport=5353, dport=53)
            / Raw(b"dnsquery")
        )
        p_bg.time = t_base + 2.0

        packets.extend([p1, p2, p_bg])

    wrpcap(pcap_path, packets)


def test_end_to_end_pipeline_smoke():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        pcap_path = f.name

    try:
        target_host = "192.168.1.100"
        bg_host = "192.168.1.200"
        num_windows = 16

        # Step 1: Generate synthetic multi-window PCAP
        generate_synthetic_pcap_session(
            pcap_path,
            target_host=target_host,
            bg_host=bg_host,
            num_windows=num_windows,
        )

        # Step 2: Extract packet-level features
        packet_features = build_packet_features(pcap_path)
        assert len(packet_features) > 0
        packet_df = pd.DataFrame(packet_features)

        # Verify host isolation in packet features
        hosts_in_packet_df = set(packet_df["src_ip"])
        assert target_host in hosts_in_packet_df
        assert bg_host in hosts_in_packet_df

        # Step 3: Extract flow-level features
        flow_features = extract_flow_features(pcap_path)
        assert len(flow_features) > 0
        flow_df = pd.DataFrame(flow_features)

        # Step 4: Traffic Fusion (Packet + Flow)
        fused_df, report = fuse_flow_and_packet_dfs(flow_df, packet_df, how="left")
        assert len(fused_df) > 0
        assert report.packet_coverage_pct > 0.0
        assert not report.has_row_multiplication
        assert not report.has_row_loss

        # Step 5: Filter for target host and convert to canonical TrafficWindows
        target_fused_df = (
            fused_df[fused_df["src_ip"] == target_host]
            .sort_values("window_id")
            .reset_index(drop=True)
            .copy()
        )
        assert len(target_fused_df) == num_windows

        # Assign synthetic target labels for sequence verification
        target_fused_df["label"] = [0.0 if i < 12 else 1.0 for i in range(num_windows)]

        traffic_windows = fused_df_to_traffic_windows(
            target_fused_df,
            label_col="label",
            allow_prototype_partial_features=True,
        )
        assert len(traffic_windows) == num_windows
        for tw in traffic_windows:
            assert isinstance(tw, TrafficWindow)
            assert tw.source_host == target_host
            assert tw.packet_count > 0
            assert tw.byte_count >= 0
            assert tw.features is not None
            assert all(np.isfinite(f) for f in tw.features)

        # Step 6: Temporal Split
        train_windows, test_windows = temporal_train_test_split(
            traffic_windows,
            train_ratio=0.875,  # 14 train windows, 2 test windows
        )
        assert len(train_windows) == 14
        assert len(test_windows) == 2
        # Temporal boundary: max train timestamp < min test timestamp
        assert max(tw.timestamp for tw in train_windows) < min(tw.timestamp for tw in test_windows)

        # Step 7: Canonical Sequence Construction
        train_features = np.array([tw.features for tw in train_windows], dtype=np.float32)
        train_targets = np.array([tw.label for tw in train_windows], dtype=np.float32)
        train_timestamps = [tw.timestamp for tw in train_windows]
        train_hosts = [tw.source_host for tw in train_windows]

        batch = build_sequences(
            features=train_features,
            targets=train_targets,
            timestamps=train_timestamps,
            source_hosts=train_hosts,
            history_length=10,
            forecast_horizons=(1, 2, 3),
            window_seconds=10,
            return_metadata=True,
        )

        # Step 8: Verify Model-Compatible PyTorch Tensor
        # 14 windows with history=10 and horizons=(1, 2, 3) yields exactly 2 sequence samples
        # (predictions at window indices 9 and 10)
        assert batch.X.shape[0] == 2
        assert batch.X.shape[1] == 10  # history length
        num_feats = batch.X.shape[2]
        assert num_feats > 0

        assert batch.y.shape == (2, 3)  # +10s, +20s, +30s horizons

        # Convert to PyTorch tensors as consumed by LSTM
        X_tensor = torch.tensor(batch.X, dtype=torch.float32)
        y_tensor = torch.tensor(batch.y, dtype=torch.float32)

        assert X_tensor.shape == (2, 10, num_feats)
        assert y_tensor.shape == (2, 3)
        assert torch.all(torch.isfinite(X_tensor))
        assert torch.all(torch.isfinite(y_tensor))

        # Invariant: No cross-host contamination
        for host in batch.hosts:
            assert host == target_host

        # Invariant: Anti-leakage / temporal ordering
        # The prediction_times must be strictly greater than historical observations
        for i, pred_time in enumerate(batch.prediction_times):
            # Target times must be in the physical future
            for t_targ in batch.target_times[i]:
                assert t_targ > pred_time

    finally:
        if os.path.exists(pcap_path):
            os.unlink(pcap_path)
