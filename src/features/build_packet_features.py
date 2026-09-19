import os
import pandas as pd
from scapy.all import PcapReader, IP, IPv6, TCP, UDP
from scapy.error import Scapy_Exception

from src.features.packet_features import calculate_packet_features
from src.features.packet_windowing import (
    WINDOW_SIZE,
    get_window_id,
    get_window_start,
    get_window_end,
)
from src.features.protocol import normalize_protocol


def _get_packet_src_ip(packet):
    """Extract source IP from IPv4 or IPv6 packet."""
    if IP in packet:
        return packet[IP].src
    if IPv6 in packet:
        return packet[IPv6].src
    return None


def _get_packet_protocol(packet):
    """Extract normalized protocol from packet."""
    if TCP in packet:
        return 6
    if UDP in packet:
        return 17
    if IP in packet:
        return normalize_protocol(packet[IP].proto)
    if IPv6 in packet:
        return normalize_protocol(packet[IPv6].nh)
    return -1


def build_packet_features(pcap_path):
    """
    Build packet-level features from a PCAP file using canonical 10-second windows.
    Supports both IPv4 and IPv6 traffic.
    """
    if not os.path.exists(pcap_path) or os.path.getsize(pcap_path) == 0:
        return []

    windows = {}
    try:
        reader = PcapReader(pcap_path)
    except Scapy_Exception:
        return []

    try:
        for packet in reader:
            if IP not in packet and IPv6 not in packet:
                continue

            try:
                timestamp = float(packet.time)
                if timestamp < 0:
                    continue
            except (AttributeError, TypeError, ValueError):
                continue

            window_id = get_window_id(timestamp)
            if window_id not in windows:
                windows[window_id] = []
            windows[window_id].append(packet)
    except Scapy_Exception:
        pass
    finally:
        reader.close()

    results = []

    for window_id in sorted(windows.keys()):
        window_packets = windows[window_id]

        src_ips = set()
        for packet in window_packets:
            sip = _get_packet_src_ip(packet)
            if sip:
                src_ips.add(sip)

        for src_ip in sorted(src_ips):
            src_packets = [
                packet
                for packet in window_packets
                if _get_packet_src_ip(packet) == src_ip
            ]

            if not src_packets:
                continue

            row = calculate_packet_features(src_packets)

            timestamps = [
                float(p.time)
                for p in src_packets
                if hasattr(p, "time") and float(p.time) >= 0
            ]
            first_timestamp = min(timestamps) if timestamps else float(window_id * WINDOW_SIZE)

            row["src_ip"] = src_ip
            row["window_id"] = window_id
            row["window_start"] = get_window_start(first_timestamp)
            row["window_end"] = get_window_end(first_timestamp)
            row["timestamp"] = first_timestamp

            protocols = [_get_packet_protocol(p) for p in src_packets]
            valid_protocols = [p for p in protocols if p != -1]

            row["protocol_id"] = (
                max(set(valid_protocols), key=valid_protocols.count)
                if valid_protocols
                else -1
            )

            results.append(row)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.features.build_packet_features <pcap_file> [output_parquet]")
        sys.exit(1)

    pcap_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "packet_features.parquet"

    results = build_packet_features(pcap_file)
    df = pd.DataFrame(results)
    df.to_parquet(output_file, index=False)

    print(f"Packet windows generated: {len(results)}")
    print(f"Saved packet features to: {output_file}")