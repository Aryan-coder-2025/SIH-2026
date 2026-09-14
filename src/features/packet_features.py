from scapy.all import IP, TCP, Raw
import statistics


def calculate_iat(timestamps):
    """Calculate packet inter-arrival times."""
    if len(timestamps) < 2:
        return []

    timestamps = sorted(timestamps)

    return [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]


def calculate_packet_features(packets):
    """Extract packet-level features from a list of Scapy packets."""

    ttl_values = []
    tcp_window_values = []
    payload_sizes = []
    timestamps = []
    destination_ports = []

    fragment_count = 0
    retransmission_count = 0

    seen_tcp_packets = set()

    for packet in packets:

        # Timestamp
        if hasattr(packet, "time"):
            timestamps.append(float(packet.time))

        # IP features
        if IP in packet:

            # TTL
            ttl_values.append(packet[IP].ttl)

            # Fragmentation
            if packet[IP].flags.MF or packet[IP].frag > 0:
                fragment_count += 1

        # TCP features
        if TCP in packet:

            tcp_window_values.append(packet[TCP].window)

            destination_ports.append(packet[TCP].dport)

            # Simple retransmission signature
            tcp_key = (
                packet[IP].src if IP in packet else "",
                packet[IP].dst if IP in packet else "",
                packet[TCP].sport,
                packet[TCP].dport,
                packet[TCP].seq
            )

            if tcp_key in seen_tcp_packets:
                retransmission_count += 1
            else:
                seen_tcp_packets.add(tcp_key)

        # Payload size
        if Raw in packet:
            payload_sizes.append(len(packet[Raw].load))
        else:
            payload_sizes.append(0)

    # Packet IAT
    iat_values = calculate_iat(timestamps)

    # Unique destination ports
    unique_ports = set(destination_ports)

    # Sequential port ratio
    sequential_port_ratio = 0.0

    if len(destination_ports) >= 2:
        sequential_count = 0

        sorted_ports = sorted(unique_ports)

        for i in range(1, len(sorted_ports)):
            if sorted_ports[i] == sorted_ports[i - 1] + 1:
                sequential_count += 1

        sequential_port_ratio = (
            sequential_count / (len(sorted_ports) - 1)
            if len(sorted_ports) > 1
            else 0.0
        )

    # Port scan score
    port_scan_score = min(len(unique_ports) / 20.0, 1.0)

    features = {
        "packet_count": len(packets),

        "ttl_mean": statistics.mean(ttl_values) if ttl_values else 0.0,
        "ttl_std": statistics.stdev(ttl_values) if len(ttl_values) > 1 else 0.0,
        "ttl_min": min(ttl_values) if ttl_values else 0.0,
        "ttl_max": max(ttl_values) if ttl_values else 0.0,

        "tcp_window_mean": (
            statistics.mean(tcp_window_values)
            if tcp_window_values else 0.0
        ),

        "tcp_window_std": (
            statistics.stdev(tcp_window_values)
            if len(tcp_window_values) > 1 else 0.0
        ),

        "fragment_count": fragment_count,

        "payload_mean": (
            statistics.mean(payload_sizes)
            if payload_sizes else 0.0
        ),

        "payload_std": (
            statistics.stdev(payload_sizes)
            if len(payload_sizes) > 1 else 0.0
        ),

        "payload_min": min(payload_sizes) if payload_sizes else 0.0,
        "payload_max": max(payload_sizes) if payload_sizes else 0.0,

        "retransmission_count": retransmission_count,

        "port_scan_score": port_scan_score,

        "sequential_port_ratio": sequential_port_ratio,

        "unique_dst_ports": len(unique_ports),

        "packet_iat_mean": (
            statistics.mean(iat_values)
            if iat_values else 0.0
        ),

        "packet_iat_std": (
            statistics.stdev(iat_values)
            if len(iat_values) > 1 else 0.0
        ),

        "packet_iat_max": max(iat_values) if iat_values else 0.0,
    }

    return features