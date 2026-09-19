import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from scapy.all import wrpcap, IP, IPv6, TCP, UDP, Ether, Raw
from src.flow.flow_extractor import (
    extract_flow_features,
    aggregate_flow_features_to_host,
    _safe_float,
    RAW_FLOW_COLUMNS,
    HOST_FLOW_COLUMNS,
)
from src.schemas.model20 import CANONICAL_MODEL_20_FEATURES

MAC_SRC = "00:11:22:33:44:55"
MAC_DST = "66:77:88:99:aa:bb"


def test_extract_flow_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        tmp_path = f.name
    try:
        assert extract_flow_features(tmp_path) == []
        # Aggregation of empty list yields DataFrame with all HOST_FLOW_COLUMNS
        empty_host_df = aggregate_flow_features_to_host([])
        assert list(empty_host_df.columns) == HOST_FLOW_COLUMNS
        assert len(empty_host_df) == 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_extract_flow_ipv4_and_tcp_flags():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        tmp_path = f.name
    try:
        # 3 packets in the same flow within the same 10-second window (t=1.0, 1.5, 2.0 -> window_id=0)
        p1 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5000, dport=80, flags="S") / Raw(load=b"a"*50)
        p1.time = 1.0
        p2 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5000, dport=80, flags="A") / Raw(load=b"b"*100)
        p2.time = 1.5
        p3 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=5000, dport=80, flags="PA") / Raw(load=b"c"*200)
        p3.time = 2.0

        wrpcap(tmp_path, [p1, p2, p3])

        flows = extract_flow_features(tmp_path)
        assert len(flows) == 1
        flow = flows[0]

        assert flow["src_ip"] == "10.0.0.1"
        assert flow["dst_ip"] == "10.0.0.2"
        assert flow["src_port"] == 5000
        assert flow["dst_port"] == 80
        assert flow["protocol"] == 6
        assert flow["window_id"] == 0
        assert flow["flow_packet_count"] == 3
        assert flow["syn_count"] == 1
        assert flow["ack_count"] == 2  # 'A' and 'PA'
        assert flow["psh_count"] == 1
        assert flow["rst_count"] == 0
        assert flow["flow_duration"] == pytest.approx(1.0)
        assert flow["packet_iat_mean"] == pytest.approx(0.5)

        # Test aggregation to host level
        host_df = aggregate_flow_features_to_host(flows)
        assert len(host_df) == 1
        for col in CANONICAL_MODEL_20_FEATURES:
            assert col in host_df.columns
        assert host_df["flow_count"].iloc[0] == 1
        assert host_df["packets_total"].iloc[0] == 3
        assert host_df["syn_count"].iloc[0] == 1
        assert host_df["ack_count"].iloc[0] == 2
        assert host_df["psh_count"].iloc[0] == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_extract_flow_ipv6_no_crash():
    """Verify IPv6 packets do not crash with KeyError: IP (regression test)."""
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        tmp_path = f.name
    try:
        p1 = Ether(src=MAC_SRC, dst=MAC_DST) / IPv6(src="2001:db8::10", dst="2001:db8::20") / TCP(sport=443, dport=55000, flags="SA") / Raw(load=b"data")
        p1.time = 12.0  # window_id = 1
        wrpcap(tmp_path, [p1])

        flows = extract_flow_features(tmp_path)
        assert len(flows) == 1
        flow = flows[0]
        assert flow["src_ip"] == "2001:db8::10"
        assert flow["dst_ip"] == "2001:db8::20"
        assert flow["protocol"] == 6
        assert flow["window_id"] == 1
        assert flow["syn_count"] == 1
        assert flow["ack_count"] == 1

        host_df = aggregate_flow_features_to_host(flows)
        assert len(host_df) == 1
        assert host_df["src_ip"].iloc[0] == "2001:db8::10"
        assert host_df["window_id"].iloc[0] == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_extract_flow_bidirectional_detection():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        tmp_path = f.name
    try:
        p_fwd = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="192.168.1.1", dst="192.168.1.2") / TCP(sport=1000, dport=80, flags="S")
        p_fwd.time = 5.0

        p_rev = Ether(src=MAC_DST, dst=MAC_SRC) / IP(src="192.168.1.2", dst="192.168.1.1") / TCP(sport=80, dport=1000, flags="SA")
        p_rev.time = 5.1

        wrpcap(tmp_path, [p_fwd, p_rev])

        flows = extract_flow_features(tmp_path)
        assert len(flows) == 2
        assert flows[0]["is_bidirectional"] == 1
        assert flows[1]["is_bidirectional"] == 1

        host_df = aggregate_flow_features_to_host(flows)
        assert len(host_df) == 2
        assert (host_df["bidirectional_ratio"] == 1.0).all()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_anti_leakage_rejection():
    """Verify that forbidden target/label columns are strictly rejected to prevent leakage."""
    flows = [
        {
            "src_ip": "10.0.0.1",
            "window_id": 0,
            "timestamp": 1.0,
            "flow_bytes": 100,
            "flow_packet_count": 2,
            "flow_duration": 0.5,
            "packet_iat_mean": 0.1,
            "protocol": 6,
            "is_malicious": 1,  # Forbidden label leakage!
        }
    ]
    with pytest.raises(ValueError, match="Feature leakage detected"):
        aggregate_flow_features_to_host(flows)


def test_safe_float_sanitizer():
    """Verify NaN, Inf, and negative values are cleanly sanitized."""
    assert _safe_float(float("nan"), default=0.0) == 0.0
    assert _safe_float(float("inf"), default=0.0) == 0.0
    assert _safe_float(-10.0, default=0.0, min_val=0.0) == 0.0
    assert _safe_float(42.5) == 42.5
    assert _safe_float("invalid", default=0.0) == 0.0
