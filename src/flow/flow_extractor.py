"""
Flow Feature Extraction Module.

Extracts 5-tuple network flows from PCAP packets and attributes flow activity
strictly to the canonical 10-second temporal windows in which the packets occurred.

TEMPORAL BOUNDARY CONTRACT:
    A continuous transport flow can span multiple 10-second windows.
    Packets arriving at t in [W * 10, (W + 1) * 10) are attributed to window W.
    A flow spanning multiple windows generates a distinct state record for EACH
    window in which it was active, representing only the packets and metrics
    observed during that window. This prevents future packets from leaking into
    earlier window states.
"""
import os
import sys
from collections import defaultdict
from typing import Any

import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader  # type: ignore

from src.features.protocol import normalize_protocol

WINDOW_SIZE: float = 10.0


def extract_flow_features(
    pcap_path: str,
    output_file: str | None = None,
    max_packets: int | None = None,
    window_size: float = WINDOW_SIZE,
) -> list[dict[str, Any]]:
    """
    Extract per-(flow, window_id) features from a PCAP file.

    Enforces temporal correctness: packets belonging to window W are assigned
    strictly to window W. If a flow crosses multiple windows, it produces
    records for each window it is active in, with zero future-packet leakage.

    Args:
        pcap_path: Path to PCAP file.
        output_file: Optional path to save Parquet output.
        max_packets: Optional upper bound on packets parsed.
        window_size: Window duration in seconds (default 10.0).

    Returns:
        List of dictionaries containing per-(flow, window_id) feature records.

    Raises:
        FileNotFoundError: If pcap_path does not exist.
    """
    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    # Read packets via streaming PcapReader
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

    # Partition packets by (flow_key, window_id)
    # flow_key: (src_ip, dst_ip, src_port, dst_port, protocol_id)
    windowed_flows: dict[tuple[tuple[str, str, int, int, int], int], list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for pkt in packets:
        if IP not in pkt:
            continue

        ts = float(getattr(pkt, "time", 0.0))
        src_ip = str(pkt[IP].src)
        dst_ip = str(pkt[IP].dst)

        protocol_id = normalize_protocol(pkt)

        src_port = 0
        dst_port = 0
        if TCP in pkt:
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
        elif UDP in pkt:
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)

        flow_key = (src_ip, dst_ip, src_port, dst_port, protocol_id)
        w_id = int(ts // window_size)

        windowed_flows[(flow_key, w_id)].append({
            "timestamp": ts,
            "packet_length": len(pkt),
        })

    results: list[dict[str, Any]] = []

    for (flow_key, window_id), flow_packets in sorted(windowed_flows.items()):
        src_ip, dst_ip, src_port, dst_port, protocol_id = flow_key

        flow_packets.sort(key=lambda x: x["timestamp"])
        timestamps = [p["timestamp"] for p in flow_packets]
        lengths = [p["packet_length"] for p in flow_packets]

        first_ts = timestamps[0]
        last_ts = timestamps[-1]

        window_start = float(window_id * window_size)
        window_end = float(window_start + window_size)

        iat_values = [
            timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))
        ]

        duration = (last_ts - first_ts) if len(timestamps) > 1 else 0.0

        results.append({
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol_id,
            "protocol_id": protocol_id,
            "flow_id": f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{protocol_id}",
            "window_id": window_id,
            "window_start": window_start,
            "window_end": window_end,
            "timestamp": first_ts,
            "flow_packet_count": len(flow_packets),
            "flow_bytes": sum(lengths),
            "flow_duration": duration,
            "packet_length_mean": sum(lengths) / len(lengths),
            "packet_iat_mean": sum(iat_values) / len(iat_values) if iat_values else 0.0,
            "packet_iat_max": max(iat_values) if iat_values else 0.0,
        })

    if output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        df = pd.DataFrame(results)
        df.to_parquet(output_file, index=False)

    return results


def aggregate_flows_to_host_windows(
    flows_data: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-(flow, window_id) records into canonical
    (source host × 10-second window) summary records.

    Instead of arbitrarily dropping duplicate flows for the same (src_ip, window_id),
    this aggregates all flows belonging to the source host during that window.
    """
    if isinstance(flows_data, list):
        if not flows_data:
            return pd.DataFrame(columns=[
                "src_ip",
                "window_id",
                "flow_count",
                "unique_dst_ip_count",
                "unique_dst_port_count",
                "flow_bytes_total",
                "flow_packets_total",
                "flow_duration_mean",
                "flow_iat_mean",
                "flow_iat_max",
            ])
        df = pd.DataFrame(flows_data)
    else:
        df = flows_data.copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "src_ip",
            "window_id",
            "flow_count",
            "unique_dst_ip_count",
            "unique_dst_port_count",
            "flow_bytes_total",
            "flow_packets_total",
            "flow_duration_mean",
            "flow_iat_mean",
            "flow_iat_max",
        ])

    grouped = df.groupby(["src_ip", "window_id"], as_index=False)

    agg_df = grouped.agg(
        flow_count=("flow_id", "nunique"),
        unique_dst_ip_count=("dst_ip", "nunique"),
        unique_dst_port_count=("dst_port", "nunique"),
        flow_bytes_total=("flow_bytes", "sum"),
        flow_packets_total=("flow_packet_count", "sum"),
        flow_duration_mean=("flow_duration", "mean"),
        flow_iat_mean=("packet_iat_mean", "mean"),
        flow_iat_max=("packet_iat_max", "max"),
    )

    return agg_df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python flow_extractor.py <pcap_file> <output_parquet>")
        sys.exit(1)

    pcap_in = sys.argv[1]
    parquet_out = sys.argv[2]
    res = extract_flow_features(pcap_in, output_file=parquet_out)
    print(f"Flow-window records generated: {len(res)}")
    print(f"Saved flow features to: {parquet_out}")