import os
import tempfile
import pytest
from scapy.all import wrpcap, IP, IPv6, TCP, UDP, Ether, Raw
from src.features.packet_parser import parse_pcap

MAC_SRC = "00:11:22:33:44:55"
MAC_DST = "66:77:88:99:aa:bb"


def test_parse_empty_pcap():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        tmp_path = f.name
    try:
        rows = parse_pcap(tmp_path)
        assert rows == []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_parse_ipv4_tcp():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        tmp_path = f.name
    try:
        pkt1 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="192.168.1.10", dst="10.0.0.1") / TCP(sport=12345, dport=80, flags="S") / Raw(load=b"test")
        pkt1.time = 100.0
        wrpcap(tmp_path, [pkt1])

        rows = parse_pcap(tmp_path)
        assert len(rows) == 1
        assert rows[0]["src_ip"] == "192.168.1.10"
        assert rows[0]["dst_ip"] == "10.0.0.1"
        assert rows[0]["protocol_id"] == 6
        assert rows[0]["src_port"] == 12345
        assert rows[0]["dst_port"] == 80
        assert rows[0]["timestamp"] == 100.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_parse_ipv6_udp():
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        tmp_path = f.name
    try:
        pkt = Ether(src=MAC_SRC, dst=MAC_DST) / IPv6(src="2001:db8::1", dst="2001:db8::2") / UDP(sport=53, dport=5353) / Raw(load=b"dns")
        pkt.time = 200.0
        wrpcap(tmp_path, [pkt])

        rows = parse_pcap(tmp_path)
        assert len(rows) == 1
        assert rows[0]["src_ip"] == "2001:db8::1"
        assert rows[0]["dst_ip"] == "2001:db8::2"
        assert rows[0]["protocol_id"] == 17
        assert rows[0]["src_port"] == 53
        assert rows[0]["dst_port"] == 5353
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
