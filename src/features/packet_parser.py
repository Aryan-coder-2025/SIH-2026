from scapy.all import rdpcap, IP, TCP, UDP
import csv
import os
import sys


def parse_pcap(pcap_path):
    packets = rdpcap(pcap_path)
    rows = []

    for i, pkt in enumerate(packets, start=1):
        row = {
            "packet_number": i,
            "timestamp": float(pkt.time),
            "src_mac": getattr(pkt, "src", ""),
            "dst_mac": getattr(pkt, "dst", ""),
            "src_ip": "",
            "dst_ip": "",
            "protocol": "",
            "src_port": "",
            "dst_port": "",
            "packet_length": len(pkt),
        }

        if IP in pkt:
            row["src_ip"] = pkt[IP].src
            row["dst_ip"] = pkt[IP].dst
            row["protocol"] = pkt[IP].proto

        if TCP in pkt:
            row["protocol"] = "TCP"
            row["src_port"] = pkt[TCP].sport
            row["dst_port"] = pkt[TCP].dport

        elif UDP in pkt:
            row["protocol"] = "UDP"
            row["src_port"] = pkt[UDP].sport
            row["dst_port"] = pkt[UDP].dport

        rows.append(row)

    output_file = "packet_data.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Packet data saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_pcap(sys.argv[1])
    else:
        print("Usage: python packet_parser.py <pcap_file>")