"""
Tests for Aman Packet / Flow / Fusion Integration and Contract Hardening.

Covers:
1. 10-second window assignment
2. Flow crossing a window boundary
3. Source-host isolation
4. Protocol normalization
5. Empty PCAP handling
6. Malformed / non-IP packet handling
7. Duplicate flow-state aggregation (no silent row drop)
8. Join cardinality validation
9. Packet / flow coverage reporting
10. Fused output = source_host × window
11. No target-derived features entering feature matrix
12. Compatibility with canonical TrafficWindow schema
13. Compatibility with canonical temporal sequence construction
"""
from datetime import datetime, timedelta, timezone
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from scapy.all import IP, TCP, UDP, ICMP, Ether, Raw, wrpcap  # type: ignore

from src.features.build_packet_features import build_packet_features
from src.features.packet_features import (
    PACKET_FEATURE_NAMES,
    calculate_iat,
    calculate_packet_features,
)
from src.features.packet_parser import CANONICAL_PACKET_FIELDS, parse_pcap
from src.features.packet_windowing import (
    WINDOW_SIZE,
    assign_packet_window,
    get_window_end,
    get_window_id,
    get_window_start,
    group_packets_by_host_and_window,
    group_packets_by_window,
)
from src.features.protocol import (
    PROTOCOL_ICMP,
    PROTOCOL_OTHER,
    PROTOCOL_TCP,
    PROTOCOL_UDP,
    normalize_protocol,
)
from src.flow.flow_extractor import (
    aggregate_flows_to_host_windows,
    extract_flow_features,
)
from src.fusion.traffic_fusion import (
    JoinValidationReport,
    fuse_flow_and_packet_dfs,
    fused_df_to_traffic_windows,
    validate_join_cardinality,
)
from src.schemas.features import FORBIDDEN_FEATURE_NAMES, validate_feature_names
from src.schemas.traffic import TrafficWindow
from src.temporal.sequences import build_sequences


# ============================================================================
# 1. 10-Second Window Assignment
# ============================================================================
def test_10_second_window_assignment():
    assert get_window_id(0.0) == 0
    assert get_window_id(9.999) == 0
    assert get_window_id(10.0) == 1
    assert get_window_id(25.5) == 2

    assert get_window_start(0.0) == 0.0
    assert get_window_start(14.2) == 10.0
    assert get_window_start(25.0) == 20.0

    assert get_window_end(0.0) == 10.0
    assert get_window_end(14.2) == 20.0
    assert get_window_end(25.0) == 30.0

    pkt = Ether() / IP(src="192.168.1.10", dst="8.8.8.8") / TCP(sport=1000, dport=80)
    pkt.time = 24.5
    win_meta = assign_packet_window(pkt)
    assert win_meta["window_id"] == 2
    assert win_meta["window_start"] == 20.0
    assert win_meta["window_end"] == 30.0


# ============================================================================
# 2. Flow Crossing a Window Boundary
# ============================================================================
def test_flow_crossing_window_boundary():
    """
    Packets for the same 5-tuple flow arriving at t=9.8, t=10.2, and t=10.8.
    Must be attributed to window 0 (1 packet) and window 1 (2 packets) separately.
    No future packets may leak into window 0.
    """
    p1 = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5000, dport=443) / Raw(b"pkt1")
    p1.time = 9.8

    p2 = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5000, dport=443) / Raw(b"pkt2")
    p2.time = 10.2

    p3 = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5000, dport=443) / Raw(b"pkt3_long")
    p3.time = 10.8

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        pcap_path = f.name
    try:
        wrpcap(pcap_path, [p1, p2, p3])
        records = extract_flow_features(pcap_path)

        assert len(records) == 2

        rec_w0 = [r for r in records if r["window_id"] == 0]
        rec_w1 = [r for r in records if r["window_id"] == 1]

        assert len(rec_w0) == 1
        assert len(rec_w1) == 1

        # Window 0: exactly 1 packet, duration 0.0, timestamp 9.8
        assert rec_w0[0]["flow_packet_count"] == 1
        assert rec_w0[0]["timestamp"] == 9.8
        assert rec_w0[0]["flow_duration"] == 0.0
        assert rec_w0[0]["window_start"] == 0.0
        assert rec_w0[0]["window_end"] == 10.0

        # Window 1: exactly 2 packets, duration 0.6s (10.8 - 10.2), timestamp 10.2
        assert rec_w1[0]["flow_packet_count"] == 2
        assert rec_w1[0]["timestamp"] == 10.2
        assert pytest.approx(rec_w1[0]["flow_duration"], rel=1e-3) == 0.6
        assert pytest.approx(rec_w1[0]["packet_iat_mean"], rel=1e-3) == 0.6
        assert rec_w1[0]["window_start"] == 10.0
        assert rec_w1[0]["window_end"] == 20.0
    finally:
        if os.path.exists(pcap_path):
            os.unlink(pcap_path)


# ============================================================================
# 3. Source-Host Isolation
# ============================================================================
def test_source_host_isolation():
    """Packets from distinct hosts in the same window remain strictly partitioned."""
    p_host1 = Ether() / IP(src="192.168.1.10", dst="1.1.1.1") / UDP(sport=100, dport=53)
    p_host1.time = 5.0

    p_host2 = Ether() / IP(src="192.168.1.20", dst="1.1.1.1") / UDP(sport=200, dport=53)
    p_host2.time = 6.0

    host_windows = group_packets_by_host_and_window([p_host1, p_host2])

    assert ("192.168.1.10", 0) in host_windows
    assert ("192.168.1.20", 0) in host_windows
    assert len(host_windows[("192.168.1.10", 0)]) == 1
    assert len(host_windows[("192.168.1.20", 0)]) == 1


# ============================================================================
# 4. Protocol Normalization
# ============================================================================
def test_protocol_normalization():
    # TCP
    assert normalize_protocol("TCP") == PROTOCOL_TCP
    assert normalize_protocol("tcp") == PROTOCOL_TCP
    assert normalize_protocol(6) == PROTOCOL_TCP
    assert normalize_protocol("6") == PROTOCOL_TCP
    assert normalize_protocol(6.0) == PROTOCOL_TCP

    # UDP
    assert normalize_protocol("UDP") == PROTOCOL_UDP
    assert normalize_protocol("udp") == PROTOCOL_UDP
    assert normalize_protocol(17) == PROTOCOL_UDP
    assert normalize_protocol("17") == PROTOCOL_UDP

    # ICMP
    assert normalize_protocol("ICMP") == PROTOCOL_ICMP
    assert normalize_protocol("icmp") == PROTOCOL_ICMP
    assert normalize_protocol(1) == PROTOCOL_ICMP
    assert normalize_protocol("1") == PROTOCOL_ICMP

    # Unknown and non-canonical
    assert normalize_protocol("GRE") == PROTOCOL_OTHER
    assert normalize_protocol(47) == PROTOCOL_OTHER
    assert normalize_protocol("ARP") == PROTOCOL_OTHER
    assert normalize_protocol(2) == PROTOCOL_OTHER
    assert normalize_protocol(None) == PROTOCOL_OTHER
    assert normalize_protocol("") == PROTOCOL_OTHER
    assert normalize_protocol("   ") == PROTOCOL_OTHER
    assert normalize_protocol(float("nan")) == PROTOCOL_OTHER

    # Packet objects
    pkt_tcp = Ether() / IP(proto=6) / TCP()
    pkt_udp = Ether() / IP(proto=17) / UDP()
    pkt_icmp = Ether() / IP(proto=1) / ICMP()
    assert normalize_protocol(pkt_tcp) == PROTOCOL_TCP
    assert normalize_protocol(pkt_udp) == PROTOCOL_UDP
    assert normalize_protocol(pkt_icmp) == PROTOCOL_ICMP


# ============================================================================
# 5. Empty PCAP Handling
# ============================================================================
def test_empty_pcap_handling():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        empty_pcap = f.name
    try:
        wrpcap(empty_pcap, [])

        parsed = parse_pcap(empty_pcap)
        assert parsed == []

        pkt_feats = build_packet_features(empty_pcap)
        assert pkt_feats == []

        flow_feats = extract_flow_features(empty_pcap)
        assert flow_feats == []
    finally:
        if os.path.exists(empty_pcap):
            os.unlink(empty_pcap)


# ============================================================================
# 6. Malformed / Non-IP Packet Handling
# ============================================================================
def test_malformed_and_non_ip_packet_handling():
    # Pure Ethernet frame (ARP-like without IP)
    eth_pkt = Ether(type=0x0806)
    eth_pkt.time = 5.0

    ip_pkt = Ether() / IP(src="10.1.1.1", dst="10.1.1.2") / TCP(sport=80, dport=80)
    ip_pkt.time = 5.5

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        pcap_path = f.name
    try:
        wrpcap(pcap_path, [eth_pkt, ip_pkt])

        parsed = parse_pcap(pcap_path)
        assert len(parsed) == 2
        # eth_pkt should have empty src_ip and protocol_id -1
        assert parsed[0]["src_ip"] == ""
        assert parsed[0]["protocol_id"] == -1
        # ip_pkt should have valid IP
        assert parsed[1]["src_ip"] == "10.1.1.1"
        assert parsed[1]["protocol_id"] == 6

        # build_packet_features should only aggregate valid IP packets
        pkt_feats = build_packet_features(pcap_path)
        assert len(pkt_feats) == 1
        assert pkt_feats[0]["src_ip"] == "10.1.1.1"
    finally:
        if os.path.exists(pcap_path):
            os.unlink(pcap_path)


# ============================================================================
# 7. Duplicate Flow-State Handling (Aggregation Instead of Drop)
# ============================================================================
def test_duplicate_flow_state_aggregation():
    """
    Multiple distinct flows (e.g. ports 80 and 443) for the same host and window
    must be aggregated together so no flow volume is lost.
    """
    flow_rows = [
        {
            "src_ip": "192.168.1.100",
            "dst_ip": "1.1.1.1",
            "src_port": 1234,
            "dst_port": 80,
            "protocol": 6,
            "flow_id": "flow-80",
            "window_id": 5,
            "flow_packet_count": 10,
            "flow_bytes": 1000,
            "flow_duration": 2.0,
            "packet_iat_mean": 0.2,
            "packet_iat_max": 0.5,
        },
        {
            "src_ip": "192.168.1.100",
            "dst_ip": "8.8.8.8",
            "src_port": 1235,
            "dst_port": 443,
            "protocol": 6,
            "flow_id": "flow-443",
            "window_id": 5,
            "flow_packet_count": 20,
            "flow_bytes": 4000,
            "flow_duration": 4.0,
            "packet_iat_mean": 0.1,
            "packet_iat_max": 0.3,
        },
    ]

    agg_df = aggregate_flows_to_host_windows(flow_rows)
    assert len(agg_df) == 1
    row = agg_df.iloc[0]
    assert row["src_ip"] == "192.168.1.100"
    assert row["window_id"] == 5
    assert row["flow_count"] == 2
    assert row["unique_dst_ip_count"] == 2
    assert row["unique_dst_port_count"] == 2
    assert row["flow_packets_total"] == 30
    assert row["flow_bytes_total"] == 5000


# ============================================================================
# 8 & 9. Join Cardinality & Packet/Flow Coverage Reporting
# ============================================================================
def test_join_cardinality_and_coverage_reporting():
    packet_df = pd.DataFrame([
        {"src_ip": "10.0.0.1", "window_id": 0, "packet_count": 5},
        {"src_ip": "10.0.0.1", "window_id": 1, "packet_count": 12},
        {"src_ip": "10.0.0.2", "window_id": 0, "packet_count": 8},
    ])

    flow_df = pd.DataFrame([
        {"src_ip": "10.0.0.1", "window_id": 0, "flow_count": 1, "flow_bytes_total": 500},
        {"src_ip": "10.0.0.1", "window_id": 1, "flow_count": 2, "flow_bytes_total": 1200},
        {"src_ip": "10.0.0.3", "window_id": 0, "flow_count": 1, "flow_bytes_total": 300},
    ])

    fused_df, report = fuse_flow_and_packet_dfs(flow_df, packet_df, how="left")

    assert len(fused_df) == 3
    assert isinstance(report, JoinValidationReport)
    assert report.packet_rows_before == 3
    assert report.flow_rows_before == 3
    assert report.rows_after_join == 3
    assert report.matched_windows == 2  # (10.0.0.1, 0) and (10.0.0.1, 1)
    assert report.unmatched_packet_windows == 1  # (10.0.0.2, 0)
    assert report.unmatched_flow_windows == 1  # (10.0.0.3, 0)
    assert pytest.approx(report.packet_coverage_pct, rel=1e-2) == 66.67
    assert pytest.approx(report.flow_coverage_pct, rel=1e-2) == 66.67
    assert not report.has_row_multiplication
    assert not report.has_row_loss


# ============================================================================
# 10. Fused Output = Source Host × Window
# ============================================================================
def test_fused_output_entity_representation():
    packet_df = pd.DataFrame([
        {"src_ip": "10.0.0.5", "window_id": 10, "packet_count": 15, "timestamp": 100.0},
        {"src_ip": "10.0.0.5", "window_id": 11, "packet_count": 25, "timestamp": 110.0},
    ])
    flow_df = pd.DataFrame([
        {"src_ip": "10.0.0.5", "window_id": 10, "flow_count": 1, "flow_bytes_total": 1500},
        {"src_ip": "10.0.0.5", "window_id": 11, "flow_count": 2, "flow_bytes_total": 3000},
    ])

    fused_df, _ = fuse_flow_and_packet_dfs(flow_df, packet_df)
    assert len(fused_df) == 2
    # Verify exact 1-to-1 host-window pairing
    keys = list(zip(fused_df["src_ip"], fused_df["window_id"]))
    assert keys == [("10.0.0.5", 10), ("10.0.0.5", 11)]


# ============================================================================
# 11. No Target-Derived Features Entering Fused Feature Matrix
# ============================================================================
def test_no_target_derived_features_in_feature_matrix():
    feature_candidates = [
        "packet_count",
        "flow_count",
        "flow_bytes_total",
        "ttl_mean",
        "port_scan_score",
        "sequential_port_ratio",
    ]
    # Should validate cleanly
    valid_features = validate_feature_names(feature_candidates)
    assert valid_features == feature_candidates

    # Target-derived or raw identifier names must be strictly rejected
    for forbidden in FORBIDDEN_FEATURE_NAMES:
        with pytest.raises(ValueError):
            validate_feature_names([forbidden])


# ============================================================================
# 12. Compatibility with Canonical TrafficWindow Schema
# ============================================================================
def test_fused_df_to_canonical_traffic_window():
    fused_df = pd.DataFrame([
        {
            "src_ip": "192.168.1.50",
            "window_id": 4,
            "timestamp": 40.0,
            "packet_count": 10,
            "flow_bytes_total": 2048,
            "ttl_mean": 64.0,
            "port_scan_score": 0.05,
        }
    ])

    traffic_windows = fused_df_to_traffic_windows(
        fused_df,
        allow_prototype_partial_features=True,
    )
    assert len(traffic_windows) == 1

    tw = traffic_windows[0]
    assert isinstance(tw, TrafficWindow)
    assert tw.source_host == "192.168.1.50"
    assert tw.packet_count == 10
    assert tw.byte_count == 2048
    assert tw.window_id == "192.168.1.50_4"
    assert tw.timestamp.tzinfo == timezone.utc
    assert tw.features is not None
    assert all(np.isfinite(f) for f in tw.features)


# ============================================================================
# 13. Compatibility with Canonical Temporal Sequence Construction
# ============================================================================
def test_compatibility_with_canonical_sequences():
    """
    Generate 15 contiguous 10-second windows for host 10.0.0.1.
    Feed into src.temporal.sequences.build_sequences.
    Verify successful creation of history=10, forecast=(10, 20, 30) batch.
    """
    rows = []
    base_ts = datetime(2026, 9, 14, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(15):
        ts = base_ts + timedelta(seconds=i * 10)
        rows.append({
            "src_ip": "10.0.0.1",
            "window_id": i,
            "timestamp": ts.timestamp(),
            "packet_count": 10 + i,
            "flow_bytes_total": 1000 + i * 50,
            "ttl_mean": 64.0,
            "port_scan_score": 0.1,
            "label": 0.0 if i < 12 else 1.0,
        })

    fused_df = pd.DataFrame(rows)
    traffic_windows = fused_df_to_traffic_windows(
        fused_df,
        label_col="label",
        allow_prototype_partial_features=True,
    )

    features = np.array([tw.features for tw in traffic_windows], dtype=np.float32)
    targets = np.array([tw.label for tw in traffic_windows], dtype=np.float32)
    timestamps = [tw.timestamp for tw in traffic_windows]
    hosts = [tw.source_host for tw in traffic_windows]

    batch = build_sequences(
        features=features,
        targets=targets,
        timestamps=timestamps,
        source_hosts=hosts,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        window_seconds=10,
        return_metadata=True,
    )

    assert batch.X.shape[1] == 10
    assert batch.y.shape[1] == 3
    assert len(batch.hosts) == batch.X.shape[0]
    assert np.all(np.isfinite(batch.X))
    assert np.all(np.isfinite(batch.y))
