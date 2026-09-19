"""
Packet Parser Module.

Performs streaming PCAP packet inspection and field extraction.
Enforces resource safety (streaming iterator), empty/malformed PCAP resilience,
and separates library parsing from output persistence.
"""
import csv
import os
import sys
from typing import Any

from scapy.all import IP, IPv6, TCP, UDP, ICMP, PcapReader  # type: ignore

from src.features.protocol import normalize_protocol

CANONICAL_PACKET_FIELDS = [
    "packet_number",
    "timestamp",
    "src_mac",
    "dst_mac",
    "src_ip",
    "dst_ip",
    "protocol",
    "protocol_id",
    "src_port",
    "dst_port",
    "packet_length",
]


def parse_pcap(
    pcap_path: str,
    output_file: str | None = None,
    max_packets: int | None = None,
) -> list[dict[str, Any]]:
    """
    Parse a PCAP file and extract packet-level metadata.

    Args:
        pcap_path: Path to the PCAP file on disk.
        output_file: Optional path to save parsed packets as CSV.
                     If None, no file side effects are produced.
        max_packets: Optional upper bound on packets to process (guards against
                     memory exhaustion on oversized or untrusted PCAPs).

    Returns:
        List of dictionaries containing parsed packet fields.

    Raises:
        FileNotFoundError: If pcap_path does not exist on disk.
    """
    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    rows: list[dict[str, Any]] = []

    # Use streaming PcapReader to avoid unbounded memory growth
    try:
        with PcapReader(pcap_path) as reader:
            for i, pkt in enumerate(reader, start=1):
                if max_packets is not None and i > max_packets:
                    break

                row: dict[str, Any] = {
                    "packet_number": i,
                    "timestamp": float(getattr(pkt, "time", 0.0)),
                    "src_mac": getattr(pkt, "src", ""),
                    "dst_mac": getattr(pkt, "dst", ""),
                    "src_ip": "",
                    "dst_ip": "",
                    "protocol": "",
                    "protocol_id": -1,
                    "src_port": 0,
                    "dst_port": 0,
                    "packet_length": len(pkt),
                }

                try:
                    # IPv4 extraction
                    if IP in pkt:
                        ip_layer = pkt[IP]
                        row["src_ip"] = str(ip_layer.src)
                        row["dst_ip"] = str(ip_layer.dst)
                        row["protocol_id"] = normalize_protocol(ip_layer.proto)
                    elif IPv6 in pkt:
                        ip_layer = pkt[IPv6]
                        row["src_ip"] = str(ip_layer.src)
                        row["dst_ip"] = str(ip_layer.dst)
                        row["protocol_id"] = normalize_protocol(ip_layer.nh)

                    # Layer 4 protocol parsing
                    if TCP in pkt:
                        tcp_layer = pkt[TCP]
                        row["protocol"] = "TCP"
                        row["protocol_id"] = 6
                        row["src_port"] = int(tcp_layer.sport)
                        row["dst_port"] = int(tcp_layer.dport)
                    elif UDP in pkt:
                        udp_layer = pkt[UDP]
                        row["protocol"] = "UDP"
                        row["protocol_id"] = 17
                        row["src_port"] = int(udp_layer.sport)
                        row["dst_port"] = int(udp_layer.dport)
                    elif ICMP in pkt:
                        row["protocol"] = "ICMP"
                        row["protocol_id"] = 1
                    elif row["protocol_id"] != -1:
                        row["protocol"] = str(pkt[IP].proto) if IP in pkt else "OTHER"
                    else:
                        row["protocol"] = "OTHER"

                except Exception:
                    # Degraded / malformed layers are recorded safely with protocol_id = -1
                    row["protocol"] = "MALFORMED"
                    row["protocol_id"] = -1

                rows.append(row)
    except Exception as e:
        # If PCAP is completely corrupt or truncated, re-raise with informative context
        if not rows:
            # Check if file was simply empty (0 bytes)
            if os.path.getsize(pcap_path) == 0:
                rows = []
            else:
                raise ValueError(f"Failed to parse PCAP {pcap_path}: {e}") from e

    # Persist output only if output_file is explicitly provided
    if output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        fieldnames = CANONICAL_PACKET_FIELDS
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if rows:
                writer.writerows(rows)

    return rows


if __name__ == "__main__":
    if len(sys.argv) > 2:
        parse_pcap(sys.argv[1], output_file=sys.argv[2])
    elif len(sys.argv) == 2:
        parse_pcap(sys.argv[1], output_file="packet_data.csv")
    else:
        print("Usage: python packet_parser.py <pcap_file> [output_csv]")