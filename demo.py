"""
End-to-End Pipeline Demo for Network State Fusion and Model-20 Integration.

Demonstrates:
1. Generation of dual-stack (IPv4 & IPv6) network traffic with multiple hosts across 10s windows.
2. Packet feature extraction (19 packet features).
3. Rich flow feature extraction (TCP flags, duration, inter-arrival times, bidirectional tracking).
4. Traffic fusion with coverage metrics and missing-data policy enforcement.
5. Extraction and validation of the canonical 20-feature matrix for downstream LSTM risk forecasting.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
import pandas as pd
from scapy.all import wrpcap, IP, IPv6, TCP, UDP, Ether, Raw

from src.features.build_packet_features import build_packet_features
from src.flow.flow_extractor import extract_flow_features
from src.fusion.traffic_fusion import fuse_flow_and_packet_data
from src.schemas.model20 import (
    CANONICAL_MODEL_20_FEATURES,
    extract_model_20_features,
    validate_model_20_features,
    get_schema_info,
)

OUTPUT_DIR = Path("artifacts")
DEMO_PCAP = OUTPUT_DIR / "demo_traffic.pcap"
PACKET_PARQUET = OUTPUT_DIR / "packet_features.parquet"
FLOW_PARQUET = OUTPUT_DIR / "flow_features.parquet"
FUSED_PARQUET = OUTPUT_DIR / "fused_features.parquet"
MODEL20_PARQUET = OUTPUT_DIR / "model20_features.parquet"

MAC_SRC = "00:11:22:33:44:55"
MAC_DST = "66:77:88:99:aa:bb"


def generate_synthetic_demo_pcap(pcap_path: str | Path) -> None:
    """Generate multi-window, dual-stack synthetic traffic for demonstration."""
    packets = []

    # --- Window 0 (t = 0.0s to 9.9s) ---
    # Host A (IPv4): 192.168.1.100 - Web browsing session with 3-way handshake
    # SYN
    p1 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="192.168.1.100", dst="93.184.216.34") / TCP(sport=51234, dport=80, flags="S", seq=1000)
    p1.time = 1.0
    # SYN-ACK (server response)
    p2 = Ether(src=MAC_DST, dst=MAC_SRC) / IP(src="93.184.216.34", dst="192.168.1.100") / TCP(sport=80, dport=51234, flags="SA", seq=2000, ack=1001)
    p2.time = 1.05
    # ACK + HTTP Request
    p3 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="192.168.1.100", dst="93.184.216.34") / TCP(sport=51234, dport=80, flags="PA", seq=1001, ack=2001) / Raw(load=b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n")
    p3.time = 1.10
    # HTTP Response
    p4 = Ether(src=MAC_DST, dst=MAC_SRC) / IP(src="93.184.216.34", dst="192.168.1.100") / TCP(sport=80, dport=51234, flags="PA", seq=2001, ack=1050) / Raw(load=b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n" + b"A"*100)
    p4.time = 1.20
    # FIN-ACK
    p5 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="192.168.1.100", dst="93.184.216.34") / TCP(sport=51234, dport=80, flags="FA", seq=1050, ack=2140)
    p5.time = 1.25

    # Host B (IPv6): 2001:db8::1 - DNS queries over UDP
    p6 = Ether(src=MAC_SRC, dst=MAC_DST) / IPv6(src="2001:db8::1", dst="2001:db8::53") / UDP(sport=54321, dport=53) / Raw(load=b"\x00\x01query_example")
    p6.time = 2.0
    p7 = Ether(src=MAC_DST, dst=MAC_SRC) / IPv6(src="2001:db8::53", dst="2001:db8::1") / UDP(sport=53, dport=54321) / Raw(load=b"\x00\x01resp_example")
    p7.time = 2.05

    # --- Window 1 (t = 10.0s to 19.9s) ---
    # Host A continues into window 1
    p8 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="192.168.1.100", dst="93.184.216.34") / TCP(sport=51235, dport=443, flags="S", seq=3000)
    p8.time = 12.0
    p9 = Ether(src=MAC_DST, dst=MAC_SRC) / IP(src="93.184.216.34", dst="192.168.1.100") / TCP(sport=443, dport=51235, flags="SA", seq=4000, ack=3001)
    p9.time = 12.08

    # Host C (IPv4 port scan attacker): 10.10.10.99
    p10 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="10.10.10.99", dst="192.168.1.100") / TCP(sport=60000, dport=21, flags="S")
    p10.time = 15.0
    p11 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="10.10.10.99", dst="192.168.1.100") / TCP(sport=60000, dport=22, flags="S")
    p11.time = 15.1
    p12 = Ether(src=MAC_SRC, dst=MAC_DST) / IP(src="10.10.10.99", dst="192.168.1.100") / TCP(sport=60000, dport=23, flags="S")
    p12.time = 15.2

    packets = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
    wrpcap(str(pcap_path), packets)
    print(f"[*] Generated synthetic dual-stack PCAP with {len(packets)} packets at {pcap_path}")


def run_demo():
    print("=" * 70)
    print("  SIH 2026: End-to-End Network Forensics & State Fusion Demo")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate or load PCAP
    print("\n[Step 1] Preparing Traffic Capture...")
    generate_synthetic_demo_pcap(DEMO_PCAP)

    # 2. Extract Packet-Level Features
    print("\n[Step 2] Extracting Packet-Level Features...")
    packet_rows = build_packet_features(str(DEMO_PCAP))
    packet_df = pd.DataFrame(packet_rows)
    packet_df.to_parquet(PACKET_PARQUET, index=False)
    print(f"  -> Generated {len(packet_df)} source-host window states")
    print(f"  -> Saved to {PACKET_PARQUET}")

    # 3. Extract Flow-Level Features
    print("\n[Step 3] Extracting Rich Flow-Level Features...")
    flow_rows = extract_flow_features(str(DEMO_PCAP))
    flow_df = pd.DataFrame(flow_rows)
    flow_df.to_parquet(FLOW_PARQUET, index=False)
    print(f"  -> Extracted {len(flow_df)} discrete flow sessions")
    print(f"  -> Saved to {FLOW_PARQUET}")

    # 4. Traffic Fusion
    print("\n[Step 4] Performing Canonical Traffic Fusion...")
    fused_df = fuse_flow_and_packet_data(
        flow_input=flow_df,
        packet_input=packet_df,
        output_file=str(FUSED_PARQUET),
        missing_data_policy="inner",
        extract_model20=True,
        model20_output_file=str(MODEL20_PARQUET),
    )
    print(f"  -> Fused matrix shape: {fused_df.shape}")
    print(f"  -> Saved to {FUSED_PARQUET}")

    # 5. Model-20 Validation & Inspection
    print("\n[Step 5] Canonical Model-20 Schema Verification...")
    schema_info = get_schema_info()
    print(f"  -> Schema Version: {schema_info['schema_version']}")
    print(f"  -> Exact Feature Count: {schema_info['feature_count']}")
    
    model20_df = pd.read_parquet(MODEL20_PARQUET)
    validate_model_20_features(model20_df)
    print("  -> Schema validation: PASSED! (0 NaNs, 0 Infs, valid non-negative counts)")

    # 6. Display Results
    print("\n[Step 6] Pipeline Output Preview:")
    print("-" * 70)
    preview_cols = ["src_ip", "window_id", "flow_count", "bytes_total", "packets_total", "syn_count", "ack_count", "tcp_flow_ratio", "bidirectional_ratio"]
    print(model20_df[preview_cols].to_string(index=False))
    print("-" * 70)

    print("\n[SUCCESS] End-to-End Network Pipeline completed successfully!")
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}\n")


if __name__ == "__main__":
    run_demo()
