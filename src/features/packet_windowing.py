WINDOW_SIZE = 10.0


def get_window_id(timestamp):
    """Return the 10-second window ID for a packet timestamp."""
    return int(float(timestamp) // WINDOW_SIZE)


def get_window_start(timestamp):
    """Return the start timestamp of the packet's 10-second window."""
    return get_window_id(timestamp) * WINDOW_SIZE


def get_window_end(timestamp):
    """Return the end timestamp of the packet's 10-second window."""
    return get_window_start(timestamp) + WINDOW_SIZE


def assign_packet_window(packet):
    """
    Add window information to a packet.

    Returns:
        dict containing timestamp, window_id,
        window_start and window_end.
    """

    timestamp = float(packet.time)

    return {
        "timestamp": timestamp,
        "window_id": get_window_id(timestamp),
        "window_start": get_window_start(timestamp),
        "window_end": get_window_end(timestamp),
    }


def group_packets_by_window(packets):
    """
    Group Scapy packets into 10-second windows.

    Returns:
        Dictionary:
        {
            window_id: [packet1, packet2, ...]
        }
    """

    windows = {}

    for packet in packets:
        window_id = get_window_id(float(packet.time))

        if window_id not in windows:
            windows[window_id] = []

        windows[window_id].append(packet)

    return windows