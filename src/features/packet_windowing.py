"""
Packet Windowing Module.

Defines temporal windowing functions aligned with canonical project configuration:
    WINDOW_SIZE = 10.0 seconds
    Primary entity = source host (src_ip) × 10-second window (window_id)
"""
from typing import Any

from scapy.all import IP  # type: ignore

WINDOW_SIZE: float = 10.0


def get_window_id(timestamp: float | int) -> int:
    """Return the canonical 10-second integer window ID for an epoch timestamp."""
    return int(float(timestamp) // WINDOW_SIZE)


def get_window_start(timestamp: float | int) -> float:
    """Return the start timestamp (seconds) of the 10-second window."""
    return float(get_window_id(timestamp) * WINDOW_SIZE)


def get_window_end(timestamp: float | int) -> float:
    """Return the end timestamp (seconds) of the 10-second window."""
    return float(get_window_start(timestamp) + WINDOW_SIZE)


def assign_packet_window(packet: Any) -> dict[str, Any]:
    """
    Extract timestamp and calculate window metadata for a single packet.

    Returns:
        dict containing timestamp, window_id, window_start, window_end.
    """
    timestamp = float(getattr(packet, "time", 0.0))
    return {
        "timestamp": timestamp,
        "window_id": get_window_id(timestamp),
        "window_start": get_window_start(timestamp),
        "window_end": get_window_end(timestamp),
    }


def group_packets_by_window(packets: list[Any]) -> dict[int, list[Any]]:
    """
    Group Scapy packets into 10-second windows.

    Returns:
        Dictionary: {window_id: [packet1, packet2, ...]}
    """
    windows: dict[int, list[Any]] = {}
    for packet in packets:
        ts = float(getattr(packet, "time", 0.0))
        w_id = get_window_id(ts)
        if w_id not in windows:
            windows[w_id] = []
        windows[w_id].append(packet)
    return windows


def group_packets_by_host_and_window(
    packets: list[Any],
) -> dict[tuple[str, int], list[Any]]:
    """
    Group Scapy packets by canonical identity: (src_ip, window_id).

    Only packets containing a valid IP layer with a non-empty source IP
    are indexed by host.

    Returns:
        Dictionary: {(src_ip, window_id): [packet1, packet2, ...]}
    """
    host_windows: dict[tuple[str, int], list[Any]] = {}
    for packet in packets:
        if IP not in packet:
            continue
        src_ip = str(packet[IP].src).strip()
        if not src_ip:
            continue
        ts = float(getattr(packet, "time", 0.0))
        w_id = get_window_id(ts)
        key = (src_ip, w_id)
        if key not in host_windows:
            host_windows[key] = []
        host_windows[key].append(packet)
    return host_windows