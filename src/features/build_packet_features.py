"""
Build Packet Features Module.

Constructs aggregated packet-level features per (source host × 10-second window)
from a PCAP file using memory-safe streaming extraction.
"""
import os
import sys
from typing import Any

import pandas as pd
from scapy.all import IP, PcapReader  # type: ignore

from src.features.packet_features import calculate_packet_features
from src.features.packet_windowing import (
    WINDOW_SIZE,
    get_window_end,
    get_window_id,
    get_window_start,
    group_packets_by_host_and_window,
)
from src.features.protocol import normalize_protocol


def build_packet_features(
    pcap_path: str,
    output_file: str | None = None,
    max_packets: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build packet-level features from a PCAP file aggregated by
    (src_ip, window_id) in canonical 10-second windows.

    Args:
        pcap_path: Path to PCAP file.
        output_file: Optional path to save resulting DataFrame as Parquet.
        max_packets: Optional limit on packets parsed.

    Returns:
        List of dicts representing (source host × 10-second window) records.

    Raises:
        FileNotFoundError: If pcap_path does not exist.
    """
    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    # Read packets using streaming PcapReader
    packets: list[Any] = []
    try:
        with PcapReader(pcap_path) as reader:
            for i, pkt in enumerate(reader, start=1):
                if max_packets is not None and i > max_packets:
                    break
                packets.append(pkt)
    except Exception as e:
        if os.path.getsize(pcap_path) == 0:
            packets = []
        else:
            raise ValueError(f"Failed reading PCAP file {pcap_path}: {e}") from e

    if not packets:
        if output_file is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            pd.DataFrame().to_parquet(output_file, index=False)
        return []

    # Group directly by canonical entity: (src_ip, window_id)
    host_windows = group_packets_by_host_and_window(packets)

    results: list[dict[str, Any]] = []

    # Deterministic sorting of keys by (src_ip, window_id)
    for (src_ip, window_id), src_packets in sorted(host_windows.items()):
        if not src_packets:
            continue

        row: dict[str, Any] = calculate_packet_features(src_packets)

        timestamps = [
            float(p.time) for p in src_packets if hasattr(p, "time")
        ]
        first_timestamp = min(timestamps) if timestamps else float(window_id * WINDOW_SIZE)

        row["src_ip"] = src_ip
        row["window_id"] = window_id
        row["window_start"] = get_window_start(first_timestamp)
        row["window_end"] = get_window_end(first_timestamp)
        row["timestamp"] = first_timestamp

        # Majority protocol for the host in this window
        protocols = [
            normalize_protocol(p[IP].proto) for p in src_packets if IP in p
        ]
        row["protocol_id"] = (
            max(set(protocols), key=protocols.count) if protocols else -1
        )

        results.append(row)

    if output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        df = pd.DataFrame(results)
        df.to_parquet(output_file, index=False)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_packet_features.py <pcap_file> [output_parquet]")
        sys.exit(1)

    pcap_in = sys.argv[1]
    parquet_out = sys.argv[2] if len(sys.argv) > 2 else "packet_features.parquet"

    records = build_packet_features(pcap_in, output_file=parquet_out)
    print(f"Packet windows generated: {len(records)}")
    print(f"Saved packet features to: {parquet_out}")