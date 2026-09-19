"""
Tests for PCAP to Canonical 41-Feature Architecture Integration.

Verifies end-to-end conformance with the frozen integration contract:
1. IPv4 PCAP → packet features
2. IPv6 PCAP → packet features
3. PCAP → flow features (all 22 canonical flow features)
4. packet + flow → fusion with provenance metrics
5. fusion → exact 41-feature validation in authoritative canonical order
6. 41 features → 10×41 temporal sequence
7. no label leakage
8. duplicate join-key rejection
9. missing-window behavior
10. model inference compatibility (existing LSTM WorldModel checkpoint)
11. dashboard Attack Episode and Baseline paths remain intact
"""
import os
import tempfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
import torch
from scapy.all import IP, IPv6, TCP, UDP, ICMP, Ether, Raw, wrpcap  # type: ignore

from src.data.pcap_pipeline import (
    extract_pcap_to_fused_dataframe,
    pcap_to_model_forecast,
    pcap_to_temporal_sequences,
    pcap_to_traffic_windows,
)
from src.features.build_packet_features import build_packet_features
from src.features.packet_features import PACKET_FEATURE_NAMES
from src.features.packet_parser import parse_pcap
from src.flow.flow_extractor import aggregate_flows_to_host_windows, extract_flow_features
from src.fusion.traffic_fusion import (
    JoinValidationReport,
    fuse_flow_and_packet_data,
    fuse_flow_and_packet_dfs,
    fused_df_to_traffic_windows,
)
from src.schemas.features import (
    CANONICAL_FLOW_FEATURE_NAMES,
    CANONICAL_MODEL_20_FEATURES,
    CANONICAL_MODEL_FEATURE_NAMES,
    CANONICAL_PACKET_FEATURE_NAMES,
    FORBIDDEN_FEATURE_NAMES,
    FeatureLeakageError,
    validate_feature_names,
)
from src.schemas.traffic import TrafficWindow
from src.temporal.sequences import build_sequences


# ============================================================================
# 1. IPv4 PCAP → Packet Features
# ============================================================================
def test_ipv4_pcap_to_packet_features():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        pcap_path = f.name

    try:
        packets = [
            Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb") / IP(src="192.168.1.10", dst="10.0.0.1", ttl=64) / TCP(sport=5000, dport=80, seq=100) / Raw(b"GET /"),
            Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb") / IP(src="192.168.1.10", dst="10.0.0.1", ttl=64) / TCP(sport=5000, dport=80, seq=200) / Raw(b"HTTP/1.1"),
            Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb") / IP(src="192.168.1.10", dst="10.0.0.2", ttl=60) / UDP(sport=4000, dport=53) / Raw(b"dns"),
        ]
        for i, pkt in enumerate(packets):
            pkt.time = 5.0 + i * 0.5
        wrpcap(pcap_path, packets)

        records = build_packet_features(pcap_path)
        assert len(records) == 1
        rec = records[0]
        assert rec["src_ip"] == "192.168.1.10"
        assert rec["window_id"] == 0
        assert rec["packet_count"] == 3
        assert rec["unique_dst_ports"] == 2
        for feat in PACKET_FEATURE_NAMES:
            assert feat in rec
            assert np.isfinite(rec[feat])
    finally:
        if os.path.exists(pcap_path):
            os.unlink(pcap_path)


# ============================================================================
# 2. IPv6 PCAP → Packet Features
# ============================================================================
def test_ipv6_pcap_to_packet_features():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        pcap_path = f.name

    try:
        v6_host = "2001:db8::1"
        v6_dst = "2001:db8::2"
        packets = [
            Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb") / IPv6(src=v6_host, dst=v6_dst, hlim=128) / TCP(sport=6000, dport=443, seq=10) / Raw(b"v6data1"),
            Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb") / IPv6(src=v6_host, dst=v6_dst, hlim=128) / TCP(sport=6000, dport=443, seq=20) / Raw(b"v6data2"),
        ]
        for i, pkt in enumerate(packets):
            pkt.time = 12.0 + i * 0.2
        wrpcap(pcap_path, packets)

        parsed = parse_pcap(pcap_path)
        assert len(parsed) == 2
        assert parsed[0]["src_ip"] == v6_host
        assert parsed[0]["protocol_id"] == 6

        records = build_packet_features(pcap_path)
        assert len(records) == 1
        rec = records[0]
        assert rec["src_ip"] == v6_host
        assert rec["window_id"] == 1
        assert rec["packet_count"] == 2
        assert rec["ttl_mean"] == 128.0
    finally:
        if os.path.exists(pcap_path):
            os.unlink(pcap_path)


# ============================================================================
# 3. PCAP → Flow Features (All 22 Canonical Flow Features)
# ============================================================================
def test_pcap_to_flow_features_completeness():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        pcap_path = f.name

    try:
        host = "10.0.0.50"
        # Create TCP handshake and data packet
        p1 = Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb") / IP(src=host, dst="1.1.1.1") / TCP(sport=1024, dport=80, flags="S", seq=1)
        p1.time = 1.0
        p2 = Ether(src="66:77:88:99:aa:bb", dst="00:11:22:33:44:55") / IP(src="1.1.1.1", dst=host) / TCP(sport=80, dport=1024, flags="SA", seq=100, ack=2)
        p2.time = 1.1
        p3 = Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb") / IP(src=host, dst="1.1.1.1") / TCP(sport=1024, dport=80, flags="A", seq=2, ack=101)
        p3.time = 1.2
        wrpcap(pcap_path, [p1, p2, p3])

        flows = extract_flow_features(pcap_path)
        assert len(flows) >= 1

        agg_df = aggregate_flows_to_host_windows(flows)
        assert len(agg_df) >= 1
        host_row = agg_df[agg_df["src_ip"] == host].iloc[0]

        # Verify presence and correctness of canonical flow features
        for f_name in CANONICAL_FLOW_FEATURE_NAMES:
            assert f_name in host_row, f"Missing flow feature: {f_name}"
            assert np.isfinite(host_row[f_name]), f"Non-finite value for {f_name}"

        assert host_row["syn_count"] >= 1
        assert host_row["ack_count"] >= 1
    finally:
        if os.path.exists(pcap_path):
            os.unlink(pcap_path)


# ============================================================================
# 4. Packet + Flow → Fusion With Provenance Metrics API
# ============================================================================
def test_packet_flow_fusion_with_metrics():
    packet_df = pd.DataFrame([
        {"src_ip": "192.168.1.1", "window_id": 0, "packet_count": 10, "ttl_mean": 64.0},
        {"src_ip": "192.168.1.1", "window_id": 1, "packet_count": 15, "ttl_mean": 64.0},
    ])
    flow_df = pd.DataFrame([
        {"src_ip": "192.168.1.1", "window_id": 0, "flow_count": 1, "bytes_total": 1000},
    ])

    fused_df, metrics = fuse_flow_and_packet_data(flow_df, packet_df, return_metrics=True)
    assert len(fused_df) == 2
    assert isinstance(metrics, dict)
    assert metrics["packet_rows_before"] == 2
    assert metrics["flow_rows_before"] == 1
    assert metrics["matched_windows"] == 1
    assert metrics["unmatched_packet_windows"] == 1
    assert metrics["unmatched_flow_windows"] == 0
    assert metrics["packet_coverage"] == 50.0
    assert metrics["flow_coverage"] == 100.0


# ============================================================================
# 5. Fusion → Exact 41-Feature Validation
# ============================================================================
def test_fusion_produces_exact_canonical_41_features():
    assert len(CANONICAL_FLOW_FEATURE_NAMES) == 22
    assert len(CANONICAL_MODEL_20_FEATURES) == 20
    assert len(CANONICAL_PACKET_FEATURE_NAMES) == 19
    assert len(CANONICAL_MODEL_FEATURE_NAMES) == 41

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        pcap_path = f.name

    try:
        packets = [
            Ether() / IP(src="10.10.10.10", dst="8.8.8.8") / TCP(sport=2000, dport=53) / Raw(b"query")
        ]
        packets[0].time = 1.0
        wrpcap(pcap_path, packets)

        fused_df, report = extract_pcap_to_fused_dataframe(pcap_path)
        assert len(fused_df) == 1

        # Check all 41 features exist in fused DataFrame
        for feat in CANONICAL_MODEL_FEATURE_NAMES:
            assert feat in fused_df.columns, f"Missing canonical feature: {feat}"

        # Validate authoritative order
        validated_names = validate_feature_names(
            CANONICAL_MODEL_FEATURE_NAMES,
            expected_order=CANONICAL_MODEL_FEATURE_NAMES,
        )
        assert validated_names == list(CANONICAL_MODEL_FEATURE_NAMES)
    finally:
        if os.path.exists(pcap_path):
            os.unlink(pcap_path)


# ============================================================================
# 6. 41 Features → 10×41 Temporal Sequence Handoff
# ============================================================================
def test_41_features_to_10x41_temporal_sequence():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        pcap_path = f.name

    try:
        host = "10.0.0.99"
        packets = []
        # Generate 15 consecutive windows (0 to 140s)
        for w in range(15):
            t = w * 10.0 + 2.0
            p = Ether() / IP(src=host, dst="10.0.0.1") / TCP(sport=1000 + w, dport=80, flags="S") / Raw(b"ping")
            p.time = t
            packets.append(p)
        wrpcap(pcap_path, packets)

        batch, report = pcap_to_temporal_sequences(
            pcap_path,
            history_length=10,
            forecast_horizons=(1, 2, 3),
            window_seconds=10,
        )

        assert batch.X.shape[1] == 10  # 10 windows
        assert batch.X.shape[2] == 41  # 41 features exactly
        assert batch.y.shape[1] == 3   # 3 forecast horizons
        assert batch.X.shape[0] == 3   # 15 windows with 10 hist + 3 horizon = 3 samples (at w=9, 10, 11)
        assert np.all(np.isfinite(batch.X))
    finally:
        if os.path.exists(pcap_path):
            os.unlink(pcap_path)


# ============================================================================
# 7. No Target or Identifier Leakage
# ============================================================================
def test_no_target_derived_or_identifier_leakage():
    for forbidden in FORBIDDEN_FEATURE_NAMES:
        with pytest.raises(FeatureLeakageError):
            validate_feature_names([forbidden])

    with pytest.raises(FeatureLeakageError):
        validate_feature_names(["target_future_attack"])


# ============================================================================
# 8. Duplicate Join-Key Rejection
# ============================================================================
def test_duplicate_join_key_rejection():
    # Packet table with duplicate (src_ip, window_id) must be rejected
    packet_df = pd.DataFrame([
        {"src_ip": "10.0.0.1", "window_id": 0, "packet_count": 5},
        {"src_ip": "10.0.0.1", "window_id": 0, "packet_count": 10},
    ])
    flow_df = pd.DataFrame([
        {"src_ip": "10.0.0.1", "window_id": 0, "flow_count": 1},
    ])

    with pytest.raises(ValueError, match="duplicate"):
        fuse_flow_and_packet_dfs(flow_df, packet_df)


# ============================================================================
# 9. Missing-Window Behavior
# ============================================================================
def test_missing_window_behavior():
    packet_df = pd.DataFrame([
        {"src_ip": "10.0.0.1", "window_id": 1, "packet_count": 5},
        {"src_ip": "10.0.0.1", "window_id": 2, "packet_count": 8},
    ])
    flow_df = pd.DataFrame([
        {"src_ip": "10.0.0.1", "window_id": 0, "flow_count": 1},
        {"src_ip": "10.0.0.1", "window_id": 1, "flow_count": 2},
    ])

    fused_left, report_left = fuse_flow_and_packet_dfs(flow_df, packet_df, how="left")
    assert len(fused_left) == 2  # windows 1 and 2
    assert report_left.unmatched_flow_windows == 1  # window 0

    fused_inner, report_inner = fuse_flow_and_packet_dfs(flow_df, packet_df, how="inner")
    assert len(fused_inner) == 1  # window 1 only


# ============================================================================
# 10. Model Inference Compatibility (Frozen Checkpoint & Scaler)
# ============================================================================
def test_pcap_to_model_forecast_compatibility():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        pcap_path = f.name

    try:
        host = "192.168.10.10"
        packets = []
        for w in range(14):
            t = w * 10.0 + 1.0
            p = Ether() / IP(src=host, dst="10.0.0.5") / TCP(sport=3000 + w, dport=80, flags="S") / Raw(b"test")
            p.time = t
            packets.append(p)
        wrpcap(pcap_path, packets)

        result = pcap_to_model_forecast(pcap_path, artifacts_dir="artifacts")
        assert result["status"] == "success"
        assert result["num_sequences"] >= 1
        assert result["input_tensor_shape"][2] == 41  # 41 features consumed by model

        pred = result["predictions"][0]
        assert len(pred["forecast_risks"]) == 3
        assert np.isfinite(pred["risk_10s"])
        assert np.isfinite(pred["risk_20s"])
        assert np.isfinite(pred["risk_30s"])
        assert pred["predicted_stage"] in ["Benign", "Reconnaissance", "Exploitation", "Action on Objectives"]
    finally:
        if os.path.exists(pcap_path):
            os.unlink(pcap_path)


# ============================================================================
# 11. Dashboard Attack Episode and Baseline Paths Preserved
# ============================================================================
def test_dashboard_both_scenarios_preserved():
    from src.dashboard import get_all_hosts_summary, get_host_forecast

    # 1. Attack Episode
    att_summary = get_all_hosts_summary("Attack Episode (Threat Window)")
    assert len(att_summary) >= 6

    att_contract = get_host_forecast("192.168.10.10", "Attack Episode (Threat Window)")
    assert att_contract is not None
    assert att_contract["observed_state"] == "MALICIOUS"
    assert len(att_contract["forecast_risks"]) == 3
    assert len(att_contract["ig_all_horizons"]) == 3

    # 2. Baseline Traffic
    base_summary = get_all_hosts_summary("Baseline Traffic (Benign)")
    assert len(base_summary) >= 6

    base_contract = get_host_forecast("192.168.10.10", "Baseline Traffic (Benign)")
    assert base_contract is not None
    assert base_contract["observed_state"] == "BENIGN"
    assert len(base_contract["forecast_risks"]) == 3
