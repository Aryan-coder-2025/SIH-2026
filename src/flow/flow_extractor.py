from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import sys
from collections import defaultdict


WINDOW_SIZE = 10


def normalize_protocol(packet):
    if TCP in packet:
        return 6
    if UDP in packet:
        return 17
    if IP in packet:
        return int(packet[IP].proto)
    return -1


def extract_flow_features(pcap_path):
    packets = rdpcap(pcap_path)

    flows = defaultdict(list)

    for packet in packets:
        if IP not in packet:
            continue

        timestamp = float(packet.time)

        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = normalize_protocol(packet)

        src_port = 0
        dst_port = 0

        if TCP in packet:
            src_port = int(packet[TCP].sport)
            dst_port = int(packet[TCP].dport)

        elif UDP in packet:
            src_port = int(packet[UDP].sport)
            dst_port = int(packet[UDP].dport)

        flow_key = (
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol
        )

        flows[flow_key].append({
            "timestamp": timestamp,
            "packet_length": len(packet)
        })

    results = []

    for flow_key, flow_packets in flows.items():

        src_ip, dst_ip, src_port, dst_port, protocol = flow_key

        flow_packets.sort(key=lambda x: x["timestamp"])

        timestamps = [p["timestamp"] for p in flow_packets]
        lengths = [p["packet_length"] for p in flow_packets]

        first_timestamp = timestamps[0]

        window_id = int(first_timestamp // WINDOW_SIZE)
        window_start = window_id * WINDOW_SIZE
        window_end = window_start + WINDOW_SIZE

        iat_values = []

        for i in range(1, len(timestamps)):
            iat_values.append(
                timestamps[i] - timestamps[i - 1]
            )

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

            "timestamp": first_timestamp,

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