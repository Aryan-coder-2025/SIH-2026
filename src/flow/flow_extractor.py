from scapy.all import PcapReader, IP, TCP, UDP
import pandas as pd
import sys
from collections import defaultdict


# Read the canonical window size from project.yaml.
from src.features.packet_windowing import WINDOW_SIZE


def normalize_protocol(packet):
    if TCP in packet:
        return 6
    if UDP in packet:
        return 17
    if IP in packet:
        return int(packet[IP].proto)
    return -1


def extract_flow_features(pcap_path):
    # Read the PCAP file.
    packets = PcapReader(pcap_path)

    # Store packets separately for each flow and actual time window.
    flow_windows = defaultdict(list)

    for packet in packets:
        # Only process IPv4 packets for the current flow extractor.
        if IP not in packet:
            continue

                # Validate the packet timestamp before using it.
        try:
            timestamp = float(packet.time)
        except (AttributeError, TypeError, ValueError):
            continue

        # Ignore invalid or negative timestamps.
        if timestamp < 0:
            continue

        # Extract source and destination IP addresses.
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst

        # Normalize the protocol to the canonical numeric ID.
        protocol = normalize_protocol(packet)

        # Default ports for non-TCP/UDP traffic.
        src_port = 0
        dst_port = 0

        # Extract TCP ports.
        if TCP in packet:
            src_port = int(packet[TCP].sport)
            dst_port = int(packet[TCP].dport)

        # Extract UDP ports.
        elif UDP in packet:
            src_port = int(packet[UDP].sport)
            dst_port = int(packet[UDP].dport)

        # Build the canonical 5-tuple flow identity.
        flow_key = (
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol
        )

        # Assign this packet to its actual 10-second window.
        window_id = int(timestamp // WINDOW_SIZE)

        # Store the packet inside its flow and actual window.
        flow_windows[(flow_key, window_id)].append({
            "timestamp": timestamp,
            "packet_length": len(packet)
        })

    results = []

    # Calculate features independently for every flow-window pair.
    for (flow_key, window_id), flow_packets in flow_windows.items():

        src_ip, dst_ip, src_port, dst_port, protocol = flow_key

        # Sort packets chronologically within the window.
        flow_packets.sort(key=lambda x: x["timestamp"])

        timestamps = [p["timestamp"] for p in flow_packets]
        lengths = [p["packet_length"] for p in flow_packets]

        # Calculate the canonical window boundaries.
        window_start = window_id * WINDOW_SIZE
        window_end = window_start + WINDOW_SIZE

        # Calculate packet inter-arrival times.
        iat_values = []

        for i in range(1, len(timestamps)):
            iat_values.append(
                timestamps[i] - timestamps[i - 1]
            )

        # Create a feature row for this flow-window pair.
        results.append({
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "protocol_id": protocol,

            "flow_id": (
                f"{src_ip}-{dst_ip}-"
                f"{src_port}-{dst_port}-{protocol}"
            ),

            "window_id": window_id,
            "window_start": window_start,
            "window_end": window_end,

            "timestamp": timestamps[0],

            "flow_packet_count": len(flow_packets),

            "flow_bytes": sum(lengths),

            "flow_duration": (
                timestamps[-1] - timestamps[0]
                if len(timestamps) > 1
                else 0.0
            ),

            "packet_length_mean": (
                sum(lengths) / len(lengths)
            ),

            "packet_iat_mean": (
                sum(iat_values) / len(iat_values)
                if iat_values
                else 0.0
            ),

            "packet_iat_max": (
                max(iat_values)
                if iat_values
                else 0.0
            )
        })

    return results

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python flow_extractor.py "
            "<pcap_file> <output_parquet>"
        )
        sys.exit(1)

    pcap_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Reading PCAP: {pcap_file}")

    results = extract_flow_features(pcap_file)

    df = pd.DataFrame(results)

    df.to_parquet(
        output_file,
        index=False
    )

    print(f"Flows generated: {len(df)}")
    print(f"Saved flow features to: {output_file}")