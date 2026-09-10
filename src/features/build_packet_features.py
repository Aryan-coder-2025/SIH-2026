from scapy.all import rdpcap, IP
import pandas as pd
from src.features.packet_features import calculate_packet_features
from src.features.packet_windowing import group_packets_by_window
from src.features.protocol import normalize_protocol


def build_packet_features(pcap_path):
    """
    Build packet-level features from a PCAP file
    using 10-second windows.
    """

    packets = rdpcap(pcap_path)
    windows = group_packets_by_window(packets)

    results = []

    for window_id, window_packets in windows.items():

        features = calculate_packet_features(window_packets)

        src_ips = []

        for packet in window_packets:
            if IP in packet:
                src_ips.append(packet[IP].src)

        unique_src_ips = sorted(set(src_ips))

        for src_ip in unique_src_ips:

            src_packets = [
                packet
                for packet in window_packets
                if IP in packet and packet[IP].src == src_ip
            ]

            if not src_packets:
                continue

            row = calculate_packet_features(src_packets)

            first_timestamp = min(float(p.time) for p in src_packets)

            row["src_ip"] = src_ip
            row["window_id"] = window_id
            row["window_start"] = window_id * 10.0
            row["window_end"] = (window_id + 1) * 10.0
            row["timestamp"] = first_timestamp

            protocols = []

            for packet in src_packets:
                if IP in packet:
                    protocols.append(
                        normalize_protocol(packet[IP].proto)
                    )

            row["protocol_id"] = (
                max(set(protocols), key=protocols.count)
                if protocols else -1
            )

            results.append(row)

    return results


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python build_packet_features.py <pcap_file>")
        sys.exit(1)

    pcap_file = sys.argv[1]

    results = build_packet_features(pcap_file)

    import pandas as pd

output_file = "packet_features.parquet"

df = pd.DataFrame(results)

df.to_parquet(output_file, index=False)

print(f"Packet windows generated: {len(results)}")
print(f"Saved packet features to: {output_file}")