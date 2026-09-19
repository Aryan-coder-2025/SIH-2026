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
from scapy.all import IP, IPv6, TCP, UDP, PcapReader  # type: ignore

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
        ValueError: If PCAP reading fails due to corruption.
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

    # Also track active 5-tuple conversations per window for bidirectional detection
    endpoints_seen_in_window: dict[int, set[tuple[str, str, int, int, int]]] = defaultdict(set)

    for pkt in packets:
        if IP in pkt:
            src_ip = str(pkt[IP].src)
            dst_ip = str(pkt[IP].dst)
        elif IPv6 in pkt:
            src_ip = str(pkt[IPv6].src)
            dst_ip = str(pkt[IPv6].dst)
        else:
            continue

        try:
            ts = float(getattr(pkt, "time", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue

        if ts < 0.0:
            continue

        protocol_id = normalize_protocol(pkt)

        src_port = 0
        dst_port = 0
        syn_cnt = 0
        ack_cnt = 0
        fin_cnt = 0
        rst_cnt = 0
        psh_cnt = 0
        urg_cnt = 0
        is_tcp = 1 if protocol_id == 6 or TCP in pkt else 0
        is_udp = 1 if protocol_id == 17 or UDP in pkt else 0

        if TCP in pkt:
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
            flags = int(pkt[TCP].flags)
            if flags & 0x02:
                syn_cnt = 1
            if flags & 0x10:
                ack_cnt = 1
            if flags & 0x01:
                fin_cnt = 1
            if flags & 0x04:
                rst_cnt = 1
            if flags & 0x08:
                psh_cnt = 1
            if flags & 0x20:
                urg_cnt = 1
        elif UDP in pkt:
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)

        flow_key = (src_ip, dst_ip, src_port, dst_port, protocol_id)
        w_id = int(ts // window_size)

        endpoints_seen_in_window[w_id].add(flow_key)

        windowed_flows[(flow_key, w_id)].append({
            "timestamp": ts,
            "packet_length": len(pkt),
            "is_tcp": is_tcp,
            "is_udp": is_udp,
            "syn_count": syn_cnt,
            "ack_count": ack_cnt,
            "fin_count": fin_cnt,
            "rst_count": rst_cnt,
            "psh_count": psh_cnt,
            "urg_count": urg_cnt,
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

        # Bidirectional detection: reverse 5-tuple active in the same window
        reverse_key = (dst_ip, src_ip, dst_port, src_port, protocol_id)
        is_bidirectional = 1.0 if reverse_key in endpoints_seen_in_window[window_id] else 0.0

        syn_tot = sum(p["syn_count"] for p in flow_packets)
        ack_tot = sum(p["ack_count"] for p in flow_packets)
        fin_tot = sum(p["fin_count"] for p in flow_packets)
        rst_tot = sum(p["rst_count"] for p in flow_packets)
        psh_tot = sum(p["psh_count"] for p in flow_packets)
        urg_tot = sum(p["urg_count"] for p in flow_packets)
        is_tcp_val = 1.0 if (protocol_id == 6 or any(p["is_tcp"] for p in flow_packets)) else 0.0
        is_udp_val = 1.0 if (protocol_id == 17 or any(p["is_udp"] for p in flow_packets)) else 0.0

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
            "is_tcp": is_tcp_val,
            "is_udp": is_udp_val,
            "is_bidirectional": is_bidirectional,
            "syn_count": syn_tot,
            "ack_count": ack_tot,
            "fin_count": fin_tot,
            "rst_count": rst_tot,
            "psh_count": psh_tot,
            "urg_count": urg_tot,
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
    (source host × 10-second window) summary records with all 22 canonical flow features.

    Instead of arbitrarily dropping duplicate flows for the same (src_ip, window_id),
    this aggregates all flows belonging to the source host during that window.
    """
    canonical_flow_cols = [
        "src_ip",
        "window_id",
        "flow_count",
        "unique_dst_ip_count",
        "unique_dst_port_count",
        "bytes_total",
        "bytes_mean",
        "bytes_std",
        "packets_total",
        "packets_mean",
        "duration_mean",
        "duration_std",
        "iat_mean",
        "iat_std",
        "iat_max",
        "tcp_flow_ratio",
        "udp_flow_ratio",
        "bidirectional_ratio",
        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "psh_count",
        "urg_count",
        # Backward-compatible aliases
        "flow_bytes_total",
        "flow_packets_total",
        "flow_duration_mean",
        "flow_iat_mean",
        "flow_iat_max",
    ]

    if isinstance(flows_data, list):
        if not flows_data:
            return pd.DataFrame(columns=canonical_flow_cols)
        df = pd.DataFrame(flows_data)
    else:
        df = flows_data.copy()

    if df.empty:
        return pd.DataFrame(columns=canonical_flow_cols)

    # Standardize column names if needed
    if "flow_bytes" in df.columns and "bytes" not in df.columns:
        df["bytes"] = df["flow_bytes"]
    if "flow_packet_count" in df.columns and "packets" not in df.columns:
        df["packets"] = df["flow_packet_count"]
    if "flow_duration" in df.columns and "duration" not in df.columns:
        df["duration"] = df["flow_duration"]
    if "packet_iat_mean" in df.columns and "iat" not in df.columns:
        df["iat"] = df["packet_iat_mean"]

    # Protocol indicators
    if "is_tcp" not in df.columns:
        if "protocol" in df.columns:
            df["is_tcp"] = (df["protocol"] == 6).astype(float)
        else:
            df["is_tcp"] = 0.0
    if "is_udp" not in df.columns:
        if "protocol" in df.columns:
            df["is_udp"] = (df["protocol"] == 17).astype(float)
        else:
            df["is_udp"] = 0.0
    if "is_bidirectional" not in df.columns:
        df["is_bidirectional"] = 0.0

    # Flag counts default to 0 if not extracted
    for flg in ["syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count"]:
        if flg not in df.columns:
            df[flg] = 0.0

    grouped = df.groupby(["src_ip", "window_id"], as_index=False)

    def _std(series: pd.Series) -> float:
        return float(series.std()) if len(series) > 1 and not pd.isna(series.std()) else 0.0

    agg_df = grouped.agg(
        flow_count=("flow_id", "nunique") if "flow_id" in df.columns else ("dst_ip", "count"),
        unique_dst_ip_count=("dst_ip", "nunique"),
        unique_dst_port_count=("dst_port", "nunique"),
        bytes_total=("bytes", "sum") if "bytes" in df.columns else ("flow_bytes", "sum"),
        bytes_mean=("bytes", "mean") if "bytes" in df.columns else ("flow_bytes", "mean"),
        bytes_std=("bytes", _std) if "bytes" in df.columns else ("flow_bytes", _std),
        packets_total=("packets", "sum") if "packets" in df.columns else ("flow_packet_count", "sum"),
        packets_mean=("packets", "mean") if "packets" in df.columns else ("flow_packet_count", "mean"),
        duration_mean=("duration", "mean") if "duration" in df.columns else ("flow_duration", "mean"),
        duration_std=("duration", _std) if "duration" in df.columns else ("flow_duration", _std),
        iat_mean=("iat", "mean") if "iat" in df.columns else ("packet_iat_mean", "mean"),
        iat_std=("iat", _std) if "iat" in df.columns else ("packet_iat_mean", _std),
        iat_max=("packet_iat_max", "max") if "packet_iat_max" in df.columns else ("iat", "max"),
        tcp_flow_ratio=("is_tcp", "mean"),
        udp_flow_ratio=("is_udp", "mean"),
        bidirectional_ratio=("is_bidirectional", "mean"),
        syn_count=("syn_count", "sum"),
        ack_count=("ack_count", "sum"),
        fin_count=("fin_count", "sum"),
        rst_count=("rst_count", "sum"),
        psh_count=("psh_count", "sum"),
        urg_count=("urg_count", "sum"),
    )

    # Add backward-compatible alias columns
    agg_df["flow_bytes_total"] = agg_df["bytes_total"]
    agg_df["flow_packets_total"] = agg_df["packets_total"]
    agg_df["flow_duration_mean"] = agg_df["duration_mean"]
    agg_df["flow_iat_mean"] = agg_df["iat_mean"]
    agg_df["flow_iat_max"] = agg_df["iat_max"]

    # Fill NaNs that can occur in std calculations
    agg_df = agg_df.fillna(0.0)

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