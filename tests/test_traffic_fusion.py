import os
import tempfile
import pandas as pd
import pytest
from src.fusion.traffic_fusion import fuse_flow_and_packet_data
from src.schemas.model20 import CANONICAL_MODEL_20_FEATURES


def _sample_packet_df():
    return pd.DataFrame([
        {
            "src_ip": "10.0.0.1",
            "window_id": 0,
            "timestamp": 1.0,
            "packet_count": 5,
            "ttl_mean": 64.0,
            "ttl_std": 0.0,
            "ttl_min": 64,
            "ttl_max": 64,
            "tcp_window_mean": 8192.0,
            "tcp_window_std": 0.0,
            "fragment_count": 0,
            "payload_mean": 100.0,
            "payload_std": 10.0,
            "payload_min": 80,
            "payload_max": 120,
            "retransmission_count": 0,
            "port_scan_score": 0.05,
            "sequential_port_ratio": 0.0,
            "unique_dst_ports": 1,
            "packet_iat_mean": 0.2,
            "packet_iat_std": 0.01,
            "packet_iat_max": 0.25,
        },
        {
            "src_ip": "10.0.0.2",
            "window_id": 1,
            "timestamp": 11.0,
            "packet_count": 3,
            "ttl_mean": 128.0,
            "ttl_std": 0.0,
            "ttl_min": 128,
            "ttl_max": 128,
            "tcp_window_mean": 4096.0,
            "tcp_window_std": 0.0,
            "fragment_count": 0,
            "payload_mean": 50.0,
            "payload_std": 0.0,
            "payload_min": 50,
            "payload_max": 50,
            "retransmission_count": 0,
            "port_scan_score": 0.05,
            "sequential_port_ratio": 0.0,
            "unique_dst_ports": 1,
            "packet_iat_mean": 0.5,
            "packet_iat_std": 0.0,
            "packet_iat_max": 0.5,
        }
    ])


def _sample_flow_df():
    return pd.DataFrame([
        {
            "src_ip": "10.0.0.1",
            "window_id": 0,
            "timestamp": 1.0,
            "flow_count": 1,
            "bytes_total": 500.0,
            "bytes_mean": 500.0,
            "bytes_std": 0.0,
            "packets_total": 5.0,
            "packets_mean": 5.0,
            "duration_mean": 1.0,
            "duration_std": 0.0,
            "iat_mean": 0.2,
            "iat_std": 0.01,
            "iat_max": 0.25,
            "syn_count": 1,
            "ack_count": 4,
            "fin_count": 0,
            "rst_count": 0,
            "psh_count": 1,
            "urg_count": 0,
            "tcp_flow_ratio": 1.0,
            "udp_flow_ratio": 0.0,
            "bidirectional_ratio": 1.0,
        },
        {
            "src_ip": "10.0.0.2",
            "window_id": 1,
            "timestamp": 11.0,
            "flow_count": 1,
            "bytes_total": 150.0,
            "bytes_mean": 150.0,
            "bytes_std": 0.0,
            "packets_total": 3.0,
            "packets_mean": 3.0,
            "duration_mean": 0.8,
            "duration_std": 0.0,
            "iat_mean": 0.5,
            "iat_std": 0.0,
            "iat_max": 0.5,
            "syn_count": 0,
            "ack_count": 0,
            "fin_count": 0,
            "rst_count": 0,
            "psh_count": 0,
            "urg_count": 0,
            "tcp_flow_ratio": 0.0,
            "udp_flow_ratio": 1.0,
            "bidirectional_ratio": 0.0,
        }
    ])


def test_fusion_perfect_match():
    packet_df = _sample_packet_df()
    flow_df = _sample_flow_df()

    fused = fuse_flow_and_packet_data(flow_df, packet_df, missing_data_policy="reject")
    assert len(fused) == 2
    assert "src_ip" in fused.columns
    assert "window_id" in fused.columns
    assert "flow_count" in fused.columns
    assert "packet_count" in fused.columns


def test_fusion_unmatched_reject_policy():
    packet_df = _sample_packet_df()
    flow_df = _sample_flow_df().iloc[:1]

    with pytest.raises(ValueError, match="Fusion rejected: packet and flow windows do not match"):
        fuse_flow_and_packet_data(flow_df, packet_df, missing_data_policy="reject")


def test_fusion_fill_zero_policy():
    packet_df = _sample_packet_df()
    flow_df = _sample_flow_df().iloc[:1]

    fused = fuse_flow_and_packet_data(flow_df, packet_df, missing_data_policy="fill_zero")
    assert len(fused) == 2
    numeric_cols = fused.select_dtypes(include="number").columns
    assert not fused[numeric_cols].isna().any().any()

    row2 = fused[fused["src_ip"] == "10.0.0.2"].iloc[0]
    assert row2["flow_count"] == 0.0
    assert row2["packet_count"] == 3


def test_fusion_inner_policy():
    packet_df = _sample_packet_df()
    flow_df = _sample_flow_df().iloc[:1]

    fused = fuse_flow_and_packet_data(flow_df, packet_df, missing_data_policy="inner")
    assert len(fused) == 1
    assert fused["src_ip"].iloc[0] == "10.0.0.1"


def test_fusion_duplicate_rejection():
    packet_df = pd.concat([_sample_packet_df(), _sample_packet_df().iloc[:1]], ignore_index=True)
    flow_df = _sample_flow_df()

    with pytest.raises(ValueError, match="duplicate source-host/window rows"):
        fuse_flow_and_packet_data(flow_df, packet_df)


def test_fusion_anti_leakage_rejection():
    """Verify that forbidden label/target columns in packet data are caught and rejected."""
    packet_df = _sample_packet_df()
    packet_df["is_malicious"] = 1  # Injected label leakage!
    flow_df = _sample_flow_df()

    with pytest.raises(ValueError, match="Feature leakage detected"):
        fuse_flow_and_packet_data(flow_df, packet_df)


def test_fusion_return_metrics():
    packet_df = _sample_packet_df()
    flow_df = _sample_flow_df()

    fused, metrics = fuse_flow_and_packet_data(
        flow_df, packet_df, return_metrics=True
    )
    assert len(fused) == 2
    assert metrics["packet_coverage"] == 100.0
    assert metrics["flow_coverage"] == 100.0
    assert metrics["unmatched_packet_windows"] == 0
    assert metrics["matched_windows"] == 2


def test_fusion_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        fuse_flow_and_packet_data("non_existent_flow.parquet", "non_existent_pkt.parquet")


def test_fusion_type_normalization():
    """Verify float window_id in packet data normalizes to int and matches flow data."""
    packet_df = _sample_packet_df()
    packet_df["window_id"] = packet_df["window_id"].astype(float)
    flow_df = _sample_flow_df()

    fused = fuse_flow_and_packet_data(flow_df, packet_df, missing_data_policy="reject")
    assert len(fused) == 2
    assert fused["window_id"].dtype.kind == "i"
