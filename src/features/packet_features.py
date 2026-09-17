from scapy.all import IP, IPv6, TCP, UDP, Raw
import statistics


def calculate_iat(timestamps):
    # Calculate packet inter-arrival times in chronological order.
    if len(timestamps) < 2:
        return []

    timestamps = sorted(timestamps)

    return [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]


def calculate_packet_features(packets):
    # Extract packet-level features from a list of Scapy packets.

    ttl_values = []
    tcp_window_values = []
    payload_sizes = []
    timestamps = []
    destination_ports = []
    protocol_ids = []

    fragment_count = 0
    retransmission_count = 0

    # Track TCP packets using flow identity, sequence number,
    # and payload length to reduce false retransmission matches.
    seen_tcp_packets = set()

    for packet in packets:

        # Validate and collect packet timestamps.
        if hasattr(packet, "time"):
            try:
                timestamp = float(packet.time)

                # Ignore invalid or negative timestamps.
                if timestamp >= 0:
                    timestamps.append(timestamp)
            except (TypeError, ValueError):
                pass

        # IPv4 features.
        if IP in packet:

            # Record the IPv4 protocol ID.
            protocol_ids.append(int(packet[IP].proto))

            # Collect TTL values.
            ttl_values.append(int(packet[IP].ttl))

            # Count fragmented IPv4 packets.
            if packet[IP].flags.MF or packet[IP].frag > 0:
                fragment_count += 1

        # IPv6 features.
        elif IPv6 in packet:

            # Record the IPv6 next-header protocol ID.
            protocol_ids.append(int(packet[IPv6].nh))

        # TCP features.
        if TCP in packet:

            # Collect TCP window sizes.
            tcp_window_values.append(int(packet[TCP].window))

            # Collect destination ports in packet arrival order.
            destination_ports.append(int(packet[TCP].dport))

            # Use flow identity, sequence number, and payload length
            # as a retransmission signature.
            src_ip = ""
            dst_ip = ""

            if IP in packet:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
            elif IPv6 in packet:
                src_ip = packet[IPv6].src
                dst_ip = packet[IPv6].dst

            payload_length = (
                len(packet[Raw].load)
                if Raw in packet
                else 0
            )

            tcp_key = (
                src_ip,
                dst_ip,
                int(packet[TCP].sport),
                int(packet[TCP].dport),
                int(packet[TCP].seq),
                payload_length
            )

            if tcp_key in seen_tcp_packets:
                retransmission_count += 1
            else:
                seen_tcp_packets.add(tcp_key)

        # UDP destination ports.
        elif UDP in packet:

            # Include UDP traffic in port-based features.
            destination_ports.append(int(packet[UDP].dport))

        # Payload size.
        if Raw in packet:
            payload_sizes.append(len(packet[Raw].load))
        else:
            payload_sizes.append(0)

    # Calculate packet inter-arrival times.
    iat_values = calculate_iat(timestamps)

    # Count unique destination ports.
    unique_ports = set(destination_ports)

    # Calculate sequential port ratio using actual packet order.
    sequential_port_ratio = 0.0

    if len(destination_ports) >= 2:
        sequential_count = 0

        # Compare each port with the immediately previous packet's port.
        for i in range(1, len(destination_ports)):
            if destination_ports[i] == destination_ports[i - 1] + 1:
                sequential_count += 1

        sequential_port_ratio = (
            sequential_count / (len(destination_ports) - 1)
        )

    # Normalize the port scan score between 0 and 1.
    # A score of 1.0 is reached at 20 unique destination ports.
    port_scan_score = min(len(unique_ports) / 20.0, 1.0)

    # Count protocol occurrences to preserve multi-protocol information.
    tcp_count = protocol_ids.count(6)
    udp_count = protocol_ids.count(17)
    icmp_count = protocol_ids.count(1)

    other_protocol_count = (
        len(protocol_ids)
        - tcp_count
        - udp_count
        - icmp_count
    )

    features = {
        "packet_count": len(packets),

        "ttl_mean": (
            statistics.mean(ttl_values)
            if ttl_values
            else 0.0
        ),

        "ttl_std": (
            statistics.stdev(ttl_values)
            if len(ttl_values) > 1
            else 0.0
        ),

        "ttl_min": (
            min(ttl_values)
            if ttl_values
            else 0.0
        ),

        "ttl_max": (
            max(ttl_values)
            if ttl_values
            else 0.0
        ),

        "tcp_window_mean": (
            statistics.mean(tcp_window_values)
            if tcp_window_values
            else 0.0
        ),

        "tcp_window_std": (
            statistics.stdev(tcp_window_values)
            if len(tcp_window_values) > 1
            else 0.0
        ),

        "fragment_count": fragment_count,

        "payload_mean": (
            statistics.mean(payload_sizes)
            if payload_sizes
            else 0.0
        ),

        "payload_std": (
            statistics.stdev(payload_sizes)
            if len(payload_sizes) > 1
            else 0.0
        ),

        "payload_min": (
            min(payload_sizes)
            if payload_sizes
            else 0.0
        ),

        "payload_max": (
            max(payload_sizes)
            if payload_sizes
            else 0.0
        ),

        "retransmission_count": retransmission_count,

        "port_scan_score": port_scan_score,

        "sequential_port_ratio": sequential_port_ratio,

        "unique_dst_ports": len(unique_ports),

        "packet_iat_mean": (
            statistics.mean(iat_values)
            if iat_values
            else 0.0
        ),

        "packet_iat_std": (
            statistics.stdev(iat_values)
            if len(iat_values) > 1
            else 0.0
        ),

        "packet_iat_max": (
            max(iat_values)
            if iat_values
            else 0.0
        ),

        # Preserve multi-protocol information.
        "tcp_count": tcp_count,
        "udp_count": udp_count,
        "icmp_count": icmp_count,
        "other_protocol_count": other_protocol_count
    }

    return features