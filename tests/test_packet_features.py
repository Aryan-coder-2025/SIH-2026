import os
import tempfile
import pytest
from scapy.all import wrpcap, IP, IPv6, TCP, UDP, Ether, Raw
from src.features.packet_features import calculate_packet_features
from src.features.build_packet_features import build_packet_features

MAC_SRC = "00:11:22:33:44:55"
MAC_DST = "66:77:88:99:aa:bb"


def test_calculate_packet_features_basic():
    p1 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="10.0.0.1", dst="10.0.0.2", ttl=64) / TCP(sport=1000, dport=80, window=8192) / Raw(load=b"X"*100)
    p1.time = 1.0
    p2 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="10.0.0.1", dst="10.0.0.2", ttl=64) / TCP(sport=1000, dport=80, window=8192) / Raw(load=b"Y"*200)
    p2.time = 1.2

    feats = calculate_packet_features([p1, p2])
    assert feats["packet_count"] == 2
    assert feats["ttl_mean"] == 64.0
    assert feats["ttl_min"] == 64
    assert feats["ttl_max"] == 64
    assert feats["tcp_window_mean"] == 8192.0
    assert feats["payload_mean"] == 150.0
    assert feats["packet_iat_mean"] == pytest.approx(0.2)
    assert feats["unique_dst_ports"] == 1


def test_build_packet_features_dual_stack():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        tmp_path = f.name
    try:
        # IPv4 packet
        p_v4 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="192.168.1.1", dst="10.0.0.1") / TCP(sport=111, dport=22)
        p_v4.time = 2.0  # window 0

        # IPv6 packet
        p_v6 = Ether(src=MAC_SRC, dst=MAC_DST) / IPv6(src="fe80::1", dst="fe80::2") / UDP(sport=222, dport=53)
        p_v6.time = 2.5  # window 0

        wrpcap(tmp_path, [p_v4, p_v6])

        rows = build_packet_features(tmp_path)
        assert len(rows) == 2
        src_ips = {r["src_ip"] for r in rows}
        assert "192.168.1.1" in src_ips
        assert "fe80::1" in src_ips

        for r in rows:
            assert r["window_id"] == 0
            assert r["window_start"] == 0.0
            assert r["window_end"] == 10.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
