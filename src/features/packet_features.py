"""
Packet-Level Feature Extraction Module.

Extracts statistical and heuristic features from a collection of Scapy packets
belonging to a discrete (source host × 10-second window) aggregation.

ENGINEERING / HEURISTIC NOTICE:
    - 'port_scan_score' is an engineered heuristic feature normalized over an
      arbitrary threshold of 20 unique ports; it is NOT a verified ground-truth attack label.
    - 'sequential_port_ratio' measures the proportion of numerically adjacent port
      pairs among sorted UNIQUE destination ports. It reflects port density/contiguity
      in the port space rather than temporal sequential order of port visitation.
"""
import statistics
from typing import Any

from scapy.all import IP, IPv6, TCP, UDP, Raw  # type: ignore

PACKET_FEATURE_NAMES: tuple[str, ...] = (
    "packet_count",
    "ttl_mean",
    "ttl_std",
    "ttl_min",
    "ttl_max",
    "tcp_window_mean",
    "tcp_window_std",
    "fragment_count",
    "payload_mean",
    "payload_std",
    "payload_min",
    "payload_max",
    "retransmission_count",
    "port_scan_score",
    "sequential_port_ratio",
    "unique_dst_ports",
    "packet_iat_mean",
    "packet_iat_std",
    "packet_iat_max",
)


def calculate_iat(timestamps: list[float]) -> list[float]:
    """
    Calculate packet inter-arrival times (in seconds).
    
    Returns an empty list if fewer than 2 timestamps are provided.
    """
    if len(timestamps) < 2:
        return []

    sorted_ts = sorted(timestamps)
    return [sorted_ts[i] - sorted_ts[i - 1] for i in range(1, len(sorted_ts))]


def calculate_packet_features(packets: list[Any]) -> dict[str, float | int]:
    """
    Extract packet-level features from a list of Scapy packets.

    Args:
        packets: List of Scapy packet objects within the same host-window.

    Returns:
        Dictionary mapping feature names to numerical values. Returns zeroed
        feature dictionary if packets is empty.
    """
    if not packets:
        return {k: (0 if "count" in k or k == "unique_dst_ports" else 0.0) for k in PACKET_FEATURE_NAMES}

    ttl_values: list[float] = []
    tcp_window_values: list[float] = []
    payload_sizes: list[float] = []
    timestamps: list[float] = []
    destination_ports: list[int] = []

    fragment_count: int = 0
    retransmission_count: int = 0

    seen_tcp_packets: set[tuple[str, str, int, int, int]] = set()

    for packet in packets:
        # Timestamp
        if hasattr(packet, "time"):
            timestamps.append(float(packet.time))

        # IPv4 / IPv6 features
        if IP in packet:
            ip_layer = packet[IP]
            ttl_values.append(float(ip_layer.ttl))

            # Fragmentation detection (More Fragments flag or non-zero fragment offset)
            if bool(ip_layer.flags.MF) or ip_layer.frag > 0:
                fragment_count += 1
        elif IPv6 in packet:
            ip_layer = packet[IPv6]
            ttl_values.append(float(getattr(ip_layer, "hlim", 64.0)))

        # TCP features
        if TCP in packet:
            tcp_layer = packet[TCP]
            tcp_window_values.append(float(tcp_layer.window))
            destination_ports.append(int(tcp_layer.dport))

            # Retransmission heuristic: identical (src_ip, dst_ip, sport, dport, seq)
            src_ip = str(packet[IP].src) if IP in packet else (str(packet[IPv6].src) if IPv6 in packet else "")
            dst_ip = str(packet[IP].dst) if IP in packet else (str(packet[IPv6].dst) if IPv6 in packet else "")
            tcp_key = (
                src_ip,
                dst_ip,
                int(tcp_layer.sport),
                int(tcp_layer.dport),
                int(tcp_layer.seq),
            )

            if tcp_key in seen_tcp_packets:
                retransmission_count += 1
            else:
                seen_tcp_packets.add(tcp_key)
        elif UDP in packet:
            destination_ports.append(int(packet[UDP].dport))

        # Payload size
        if Raw in packet:
            payload_sizes.append(float(len(packet[Raw].load)))
        else:
            payload_sizes.append(0.0)

    # Packet Inter-Arrival Times
    iat_values = calculate_iat(timestamps)

    # Unique destination ports
    unique_ports = sorted(set(destination_ports))

    # Sequential port ratio heuristic:
    # Measures the fraction of numerically adjacent port pairs among sorted unique ports.
    sequential_port_ratio = 0.0
    if len(unique_ports) >= 2:
        sequential_count = sum(
            1 for i in range(1, len(unique_ports)) if unique_ports[i] == unique_ports[i - 1] + 1
        )
        sequential_port_ratio = float(sequential_count) / float(len(unique_ports) - 1)

    # Port scan heuristic score: normalized over arbitrary threshold of 20 unique ports
    port_scan_score = min(float(len(unique_ports)) / 20.0, 1.0)

    features: dict[str, float | int] = {
        "packet_count": len(packets),
        "ttl_mean": statistics.mean(ttl_values) if ttl_values else 0.0,
        "ttl_std": statistics.stdev(ttl_values) if len(ttl_values) > 1 else 0.0,
        "ttl_min": min(ttl_values) if ttl_values else 0.0,
        "ttl_max": max(ttl_values) if ttl_values else 0.0,
        "tcp_window_mean": (
            statistics.mean(tcp_window_values) if tcp_window_values else 0.0
        ),
        "tcp_window_std": (
            statistics.stdev(tcp_window_values) if len(tcp_window_values) > 1 else 0.0
        ),
        "fragment_count": fragment_count,
        "payload_mean": statistics.mean(payload_sizes) if payload_sizes else 0.0,
        "payload_std": statistics.stdev(payload_sizes) if len(payload_sizes) > 1 else 0.0,
        "payload_min": min(payload_sizes) if payload_sizes else 0.0,
        "payload_max": max(payload_sizes) if payload_sizes else 0.0,
        "retransmission_count": retransmission_count,
        "port_scan_score": port_scan_score,
        "sequential_port_ratio": sequential_port_ratio,
        "unique_dst_ports": len(unique_ports),
        "packet_iat_mean": statistics.mean(iat_values) if iat_values else 0.0,
        "packet_iat_std": statistics.stdev(iat_values) if len(iat_values) > 1 else 0.0,
        "packet_iat_max": max(iat_values) if iat_values else 0.0,
    }

    return features