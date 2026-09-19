"""
Flow Extractor Module.

Extracts discrete flow sessions and canonical source-host aggregated network-state
vectors from PCAP captures. Enforces strict temporal boundaries, dual-stack
IPv4/IPv6 support, and anti-leakage invariants.
"""
from __future__ import annotations

import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, IPv6, TCP, UDP
from scapy.error import Scapy_Exception

from src.features.packet_windowing import (
    WINDOW_SIZE,
    get_window_id,
    get_window_start,
    get_window_end,
)
from src.features.protocol import normalize_protocol
from src.schemas.model20 import (
    CANONICAL_MODEL_20_FEATURES,
    FORBIDDEN_COLUMNS,
)

RAW_FLOW_COLUMNS: list[str] = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "protocol_id",
    "flow_id",
    "window_id",
    "window_start",
    "window_end",
    "timestamp",
    "flow_packet_count",
    "flow_bytes",
    "flow_duration",
    "packet_length_mean",
    "packet_length_std",
    "packet_iat_mean",
    "packet_iat_std",
    "packet_iat_max",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "is_bidirectional",
]

HOST_FLOW_COLUMNS: list[str] = [
    "src_ip",
    "window_id",
    "timestamp",
    "window_start",
    "window_end",
] + CANONICAL_MODEL_20_FEATURES + [
    "unique_dst_ip_count",
    "unique_dst_port_count",
]


def _safe_float(val: Any, default: float = 0.0, min_val: float | None = None) -> float:
    """Sanitize float values, guarding against NaN, Inf, and negative values."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        if min_val is not None and f < min_val:
            return min_val
        return f
    except (TypeError, ValueError):
        return default


def _extract_packet_ip_proto(packet) -> tuple[str | None, str | None, int]:
    """Extract src_ip, dst_ip, and canonical protocol_id from IPv4 or IPv6 packet."""
    if IP in packet:
        src_ip = str(packet[IP].src)
        dst_ip = str(packet[IP].dst)
        if TCP in packet:
            proto = 6
        elif UDP in packet:
            proto = 17
        else:
            proto = normalize_protocol(packet[IP].proto)
        return src_ip, dst_ip, proto
    elif IPv6 in packet:
        src_ip = str(packet[IPv6].src)
        dst_ip = str(packet[IPv6].dst)
        if TCP in packet:
            proto = 6
        elif UDP in packet:
            proto = 17
        else:
            proto = normalize_protocol(packet[IPv6].nh)
        return src_ip, dst_ip, proto
    return None, None, -1


def extract_flow_features(pcap_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract per-flow features within canonical time windows from a PCAP file.
    Safely streams packets with dual-stack IPv4/IPv6 support and zero temporal leakage.
    """
    pcap_path_str = str(pcap_path)
    if not os.path.isfile(pcap_path_str) or os.path.getsize(pcap_path_str) == 0:
        return []

    try:
        packets = PcapReader(pcap_path_str)
    except Exception:
        return []

    # Store packets strictly partitioned per (flow_5tuple, window_id)
    flow_windows: dict[tuple[tuple[str, str, int, int, int], int], list[dict[str, Any]]] = (
        defaultdict(list)
    )

    # Track within-window traffic directions strictly isolated per window_id
    window_flow_directions: dict[int, dict[tuple[frozenset, int], set]] = (
        defaultdict(lambda: defaultdict(set))
    )

    try:
        for packet in packets:
            src_ip, dst_ip, protocol = _extract_packet_ip_proto(packet)
            if src_ip is None or dst_ip is None:
                continue

            try:
                timestamp = float(packet.time)
                # Leakage and validity check: reject NaN, Inf, and negative timestamps
                if math.isnan(timestamp) or math.isinf(timestamp) or timestamp < 0:
                    continue
            except (AttributeError, TypeError, ValueError):
                continue

            # Extract ports and TCP control flags
            src_port = 0
            dst_port = 0
            tcp_flags = {"S": 0, "A": 0, "F": 0, "R": 0, "P": 0, "U": 0}

            if TCP in packet:
                try:
                    src_port = int(packet[TCP].sport)
                    dst_port = int(packet[TCP].dport)
                except (AttributeError, ValueError, TypeError):
                    src_port = 0
                    dst_port = 0

                try:
                    raw_flags = packet[TCP].flags
                    # Support integer bitmask flags or string representations
                    if isinstance(raw_flags, int):
                        tcp_flags["S"] = 1 if (raw_flags & 0x02) else 0
                        tcp_flags["A"] = 1 if (raw_flags & 0x10) else 0
                        tcp_flags["F"] = 1 if (raw_flags & 0x01) else 0
                        tcp_flags["R"] = 1 if (raw_flags & 0x04) else 0
                        tcp_flags["P"] = 1 if (raw_flags & 0x08) else 0
                        tcp_flags["U"] = 1 if (raw_flags & 0x20) else 0
                    else:
                        flag_str = str(raw_flags)
                        tcp_flags["S"] = 1 if "S" in flag_str else 0
                        tcp_flags["A"] = 1 if "A" in flag_str else 0
                        tcp_flags["F"] = 1 if "F" in flag_str else 0
                        tcp_flags["R"] = 1 if "R" in flag_str else 0
                        tcp_flags["P"] = 1 if "P" in flag_str else 0
                        tcp_flags["U"] = 1 if "U" in flag_str else 0
                except Exception:
                    pass

            elif UDP in packet:
                try:
                    src_port = int(packet[UDP].sport)
                    dst_port = int(packet[UDP].dport)
                except (AttributeError, ValueError, TypeError):
                    src_port = 0
                    dst_port = 0

            # Strict temporal window assignment
            window_id = get_window_id(timestamp)
            flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)

            # Symmetrical pair key isolated strictly to this window
            pair_key = (
                frozenset({(src_ip, src_port), (dst_ip, dst_port)}),
                protocol,
            )
            window_flow_directions[window_id][pair_key].add((src_ip, dst_ip, src_port, dst_port))

            flow_windows[(flow_key, window_id)].append({
                "timestamp": timestamp,
                "packet_length": len(packet),
                "tcp_flags": tcp_flags,
                "pair_key": pair_key,
            })
    except Exception:
        pass
    finally:
        try:
            packets.close()
        except Exception:
            pass

    results: list[dict[str, Any]] = []

    for (flow_key, window_id), flow_packets in flow_windows.items():
        src_ip, dst_ip, src_port, dst_port, protocol = flow_key
        flow_packets.sort(key=lambda x: x["timestamp"])

        timestamps = [p["timestamp"] for p in flow_packets]
        lengths = [p["packet_length"] for p in flow_packets]

        # Enforce canonical window boundaries
        window_start = float(window_id * WINDOW_SIZE)
        window_end = float((window_id + 1) * WINDOW_SIZE)

        iat_values = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
        ]

        syn_count = sum(p["tcp_flags"]["S"] for p in flow_packets)
        ack_count = sum(p["tcp_flags"]["A"] for p in flow_packets)
        fin_count = sum(p["tcp_flags"]["F"] for p in flow_packets)
        rst_count = sum(p["tcp_flags"]["R"] for p in flow_packets)
        psh_count = sum(p["tcp_flags"]["P"] for p in flow_packets)
        urg_count = sum(p["tcp_flags"]["U"] for p in flow_packets)

        # Detect bidirectional conversation within this window
        pair_key = flow_packets[0]["pair_key"]
        is_bidirectional = 1 if len(window_flow_directions[window_id][pair_key]) > 1 else 0

        flow_duration = _safe_float(timestamps[-1] - timestamps[0], default=0.0, min_val=0.0) if len(timestamps) > 1 else 0.0
        iat_mean = _safe_float(statistics.mean(iat_values), default=0.0, min_val=0.0) if iat_values else 0.0
        iat_std = _safe_float(statistics.stdev(iat_values), default=0.0, min_val=0.0) if len(iat_values) > 1 else 0.0
        iat_max = _safe_float(max(iat_values), default=0.0, min_val=0.0) if iat_values else 0.0

        length_mean = _safe_float(statistics.mean(lengths), default=0.0, min_val=0.0) if lengths else 0.0
        length_std = _safe_float(statistics.stdev(lengths), default=0.0, min_val=0.0) if len(lengths) > 1 else 0.0

        results.append({
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "protocol_id": protocol,
            "flow_id": f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{protocol}",
            "window_id": window_id,
            "window_start": window_start,
            "window_end": window_end,
            "timestamp": timestamps[0],
            "flow_packet_count": len(flow_packets),
            "flow_bytes": sum(lengths),
            "flow_duration": flow_duration,
            "packet_length_mean": length_mean,
            "packet_length_std": length_std,
            "packet_iat_mean": iat_mean,
            "packet_iat_std": iat_std,
            "packet_iat_max": iat_max,
            "syn_count": syn_count,
            "ack_count": ack_count,
            "fin_count": fin_count,
            "rst_count": rst_count,
            "psh_count": psh_count,
            "urg_count": urg_count,
            "is_bidirectional": is_bidirectional,
        })

    return results


def aggregate_flow_features_to_host(flows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-flow records or DataFrame into host-level window features
    matching the canonical model-20 feature schema.

    Strictly protects against feature and label leakage.
    """
    if isinstance(flows, list):
        if not flows:
            return pd.DataFrame(columns=HOST_FLOW_COLUMNS)
        df = pd.DataFrame(flows)
    elif isinstance(flows, pd.DataFrame):
        if flows.empty:
            return pd.DataFrame(columns=HOST_FLOW_COLUMNS)
        df = flows.copy()
    else:
        raise TypeError("Expected list of flow dicts or pandas DataFrame")

    # Anti-Leakage Invariant Check: Reject any forbidden target or label columns
    leaked_cols = [col for col in df.columns if col in FORBIDDEN_COLUMNS]
    if leaked_cols:
        raise ValueError(
            f"Feature leakage detected: forbidden target/label columns {leaked_cols} "
            f"cannot be processed in flow aggregation."
        )

    records = []
    grouped = df.groupby(["src_ip", "window_id"], sort=True)

    for (src_ip, window_id), group in grouped:
        flow_count = len(group)
        bytes_series = group["flow_bytes"]
        packets_series = group["flow_packet_count"]
        duration_series = group["flow_duration"]
        iat_series = group["packet_iat_mean"]

        bytes_total = _safe_float(bytes_series.sum(), default=0.0, min_val=0.0)
        bytes_mean = _safe_float(bytes_series.mean(), default=0.0, min_val=0.0)
        bytes_std = _safe_float(bytes_series.std(ddof=1), default=0.0, min_val=0.0) if flow_count > 1 else 0.0

        packets_total = _safe_float(packets_series.sum(), default=0.0, min_val=0.0)
        packets_mean = _safe_float(packets_series.mean(), default=0.0, min_val=0.0)

        duration_mean = _safe_float(duration_series.mean(), default=0.0, min_val=0.0)
        duration_std = _safe_float(duration_series.std(ddof=1), default=0.0, min_val=0.0) if flow_count > 1 else 0.0

        iat_mean = _safe_float(iat_series.mean(), default=0.0, min_val=0.0)
        iat_std = _safe_float(iat_series.std(ddof=1), default=0.0, min_val=0.0) if flow_count > 1 else 0.0
        iat_max = _safe_float(group["packet_iat_max"].max(), default=0.0, min_val=0.0) if "packet_iat_max" in group else 0.0

        syn_count = int(group["syn_count"].sum()) if "syn_count" in group else 0
        ack_count = int(group["ack_count"].sum()) if "ack_count" in group else 0
        fin_count = int(group["fin_count"].sum()) if "fin_count" in group else 0
        rst_count = int(group["rst_count"].sum()) if "rst_count" in group else 0
        psh_count = int(group["psh_count"].sum()) if "psh_count" in group else 0
        urg_count = int(group["urg_count"].sum()) if "urg_count" in group else 0

        proto_col = group["protocol"] if "protocol" in group else group.get("protocol_id", pd.Series([0]))
        tcp_flow_ratio = _safe_float((proto_col == 6).sum() / flow_count, default=0.0, min_val=0.0)
        udp_flow_ratio = _safe_float((proto_col == 17).sum() / flow_count, default=0.0, min_val=0.0)

        if "is_bidirectional" in group:
            bidirectional_ratio = _safe_float((group["is_bidirectional"] == 1).sum() / flow_count, default=0.0, min_val=0.0)
        else:
            bidirectional_ratio = 0.0

        min_timestamp = float(group["timestamp"].min()) if "timestamp" in group else float(window_id * WINDOW_SIZE)
        unique_dst_ips = int(group["dst_ip"].nunique()) if "dst_ip" in group else 0
        unique_dst_ports = int(group["dst_port"].nunique()) if "dst_port" in group else 0

        records.append({
            "src_ip": str(src_ip),
            "window_id": int(window_id),
            "timestamp": min_timestamp,
            "window_start": float(window_id * WINDOW_SIZE),
            "window_end": float((window_id + 1) * WINDOW_SIZE),
            # Canonical 20 Flow Features
            "flow_count": flow_count,
            "bytes_total": bytes_total,
            "bytes_mean": bytes_mean,
            "bytes_std": bytes_std,
            "packets_total": packets_total,
            "packets_mean": packets_mean,
            "duration_mean": duration_mean,
            "duration_std": duration_std,
            "iat_mean": iat_mean,
            "iat_std": iat_std,
            "iat_max": iat_max,
            "syn_count": syn_count,
            "ack_count": ack_count,
            "fin_count": fin_count,
            "rst_count": rst_count,
            "psh_count": psh_count,
            "urg_count": urg_count,
            "tcp_flow_ratio": tcp_flow_ratio,
            "udp_flow_ratio": udp_flow_ratio,
            "bidirectional_ratio": bidirectional_ratio,
            # Rich flow metadata
            "unique_dst_ip_count": unique_dst_ips,
            "unique_dst_port_count": unique_dst_ports,
        })

    out_df = pd.DataFrame(records)
    # Ensure correct column ordering matching HOST_FLOW_COLUMNS
    for c in HOST_FLOW_COLUMNS:
        if c not in out_df.columns:
            out_df[c] = 0.0
    return out_df[HOST_FLOW_COLUMNS]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.flow.flow_extractor <pcap_file> [output_parquet] [--aggregate]")
        sys.exit(1)

    pcap_file = sys.argv[1]
    aggregate = "--aggregate" in sys.argv
    clean_args = [a for a in sys.argv[2:] if not a.startswith("--")]
    output_file = clean_args[0] if clean_args else "flow_features.parquet"

    print(f"Reading PCAP: {pcap_file}")
    results = extract_flow_features(pcap_file)

    if aggregate:
        df = aggregate_flow_features_to_host(results)
        print(f"Aggregated source-host windows generated: {len(df)}")
    else:
        if results:
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame(columns=RAW_FLOW_COLUMNS)
        print(f"Flows generated: {len(df)}")

    df.to_parquet(output_file, index=False)
    print(f"Saved flow features to: {output_file}")