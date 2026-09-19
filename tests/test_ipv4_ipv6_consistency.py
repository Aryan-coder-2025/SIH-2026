import os
import tempfile
import pytest
from scapy.all import wrpcap, IP, IPv6, TCP, UDP, Ether, Raw
from src.features.build_packet_features import build_packet_features
from src.flow.flow_extractor import extract_flow_features
from src.fusion.traffic_fusion import fuse_flow_and_packet_data
from src.schemas.model20 import (
    CANONICAL_MODEL_20_FEATURES,
    validate_model_20_features,
    extract_model_20_features,
)

MAC_SRC = "00:11:22:33:44:55"
MAC_DST = "66:77:88:99:aa:bb"


def test_end_to_end_dual_stack_consistency():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        tmp_pcap = f.name
    try:
        packets = []

        # Host 1 (IPv4): 192.168.1.50 -> sends 3 TCP packets in window 0 (t=1.0, 1.5, 2.0)
        p1 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="192.168.1.50", dst="93.184.216.34") / TCP(sport=49152, dport=80, flags="S", seq=100) / Raw(load=b"GET / HTTP/1.1\r\n")
        p1.time = 1.0
        p2 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="192.168.1.50", dst="93.184.216.34") / TCP(sport=49152, dport=80, flags="A", seq=115) / Raw(load=b"Host: example.com\r\n\r\n")
        p2.time = 1.5
        p3 = Ether(src=MAC_DST, dst=MAC_SRC) / IP(src="93.184.216.34", dst="192.168.1.50") / TCP(sport=80, dport=49152, flags="SA", seq=500) / Raw(load=b"HTTP/1.1 200 OK\r\n")
        p3.time = 1.6
        packets.extend([p1, p2, p3])

        # Host 2 (IPv6): 2001:db8:85a3::8a2e:370:7334 -> sends 2 UDP packets in window 0 (t=3.0, 3.2)
        p4 = Ether(src=MAC_SRC, dst=MAC_DST) / IPv6(src="2001:db8::1", dst="2001:db8::2") / UDP(sport=5353, dport=53) / Raw(load=b"query1")
        p4.time = 3.0
        p5 = Ether(src=MAC_SRC, dst=MAC_DST) / IPv6(src="2001:db8::1", dst="2001:db8::2") / UDP(sport=5353, dport=53) / Raw(load=b"query2")
        p5.time = 3.2
        packets.extend([p4, p5])

        wrpcap(tmp_pcap, packets)

        # 1. Packet features
        packet_rows = build_packet_features(tmp_pcap)
        assert len(packet_rows) >= 2
        packet_hosts = {r["src_ip"] for r in packet_rows}
        assert "192.168.1.50" in packet_hosts
        assert "2001:db8::1" in packet_hosts

        # 2. Flow features
        flow_rows = extract_flow_features(tmp_pcap)
        assert len(flow_rows) >= 2
        flow_hosts = {r["src_ip"] for r in flow_rows}
        assert "192.168.1.50" in flow_hosts
        assert "2001:db8::1" in flow_hosts

        import pandas as pd
        packet_df = pd.DataFrame(packet_rows)
        flow_df = pd.DataFrame(flow_rows)

        # 3. Traffic fusion with missing_data_policy="inner" or "fill_zero"
        fused_df = fuse_flow_and_packet_data(
            flow_input=flow_df,
            packet_input=packet_df,
            missing_data_policy="inner",
        )
        assert len(fused_df) >= 2
        fused_hosts = set(fused_df["src_ip"])
        assert "192.168.1.50" in fused_hosts
        assert "2001:db8::1" in fused_hosts

        # 4. Canonical Model-20 extraction & validation
        model20_df = extract_model_20_features(fused_df)
        assert len(model20_df) == len(fused_df)
        validate_model_20_features(model20_df)

        for feat in CANONICAL_MODEL_20_FEATURES:
            assert feat in model20_df.columns
            # Confirm finite
            assert not model20_df[feat].isna().any()

    finally:
        if os.path.exists(tmp_pcap):
            os.remove(tmp_pcap)
