from scapy.all import PcapReader, IP, IPv6, TCP, UDP
from scapy.error import Scapy_Exception
import os
import sys

from src.features.protocol import normalize_protocol


def parse_pcap(pcap_path):
    # Handle an empty PCAP safely.
    if os.path.getsize(pcap_path) == 0:
        return []

    # Open the PCAP using streaming instead of loading the whole file into RAM.
    try:
        packets = PcapReader(pcap_path)
    except Scapy_Exception:
        return []

    rows = []

    try:
        # Process one packet at a time.
        for packet_number, pkt in enumerate(packets, start=1):
            # Skip packets that do not contain a valid timestamp.
            try:
                timestamp = float(pkt.time)
            except (AttributeError, TypeError, ValueError):
                continue

            # Create the basic packet record.
            row = {
                "packet_number": packet_number,
                "timestamp": timestamp,
                "src_mac": getattr(pkt, "src", ""),
                "dst_mac": getattr(pkt, "dst", ""),
                "src_ip": "",
                "dst_ip": "",
                "protocol_id": -1,
                "src_port": "",
                "dst_port": "",
                "packet_length": len(pkt),
            }

            # Extract IPv4 information.
            if IP in pkt:
                row["src_ip"] = pkt[IP].src
                row["dst_ip"] = pkt[IP].dst
                row["protocol_id"] = normalize_protocol(pkt[IP].proto)

            # Extract IPv6 information.
            elif IPv6 in pkt:
                row["src_ip"] = pkt[IPv6].src
                row["dst_ip"] = pkt[IPv6].dst
                row["protocol_id"] = normalize_protocol(pkt[IPv6].nh)

            # Extract TCP ports and use the canonical TCP ID.
            if TCP in pkt:
                row["protocol_id"] = normalize_protocol("TCP")
                row["src_port"] = pkt[TCP].sport
                row["dst_port"] = pkt[TCP].dport

            # Extract UDP ports and use the canonical UDP ID.
            elif UDP in pkt:
                row["protocol_id"] = normalize_protocol("UDP")
                row["src_port"] = pkt[UDP].sport
                row["dst_port"] = pkt[UDP].dport

            rows.append(row)

    except Scapy_Exception:
        # Keep packets already processed if a PCAP read error occurs.
        pass

    finally:
        # Always close the PCAP reader and release the file resource.
        packets.close()

    # Return parsed data without creating files as a side effect.
    return rows


if __name__ == "__main__":
    # Keep command-line usage available for manual testing.
    if len(sys.argv) > 1:
        parsed_rows = parse_pcap(sys.argv[1])
        print(f"Parsed {len(parsed_rows)} packets.")
    else:
        print("Usage: python -m src.features.packet_parser <pcap_file>")