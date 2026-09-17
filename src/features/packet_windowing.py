from pathlib import Path
import yaml

# Find the project root and canonical configuration file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "project.yaml"

# Load the canonical project configuration.
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

# Read the canonical window size from project.yaml.
WINDOW_SIZE = float(CONFIG["window"]["size_seconds"])


def get_window_id(timestamp):
    # Convert a packet timestamp into its deterministic window ID.
    return int(float(timestamp) // WINDOW_SIZE)


def get_window_start(timestamp):
    # Return the start of the packet's window.
    return get_window_id(timestamp) * WINDOW_SIZE


def get_window_end(timestamp):
    # Return the exclusive end of the packet's window.
    return get_window_start(timestamp) + WINDOW_SIZE


def assign_packet_window(packet):
    # Add canonical window information to a packet.
    timestamp = float(packet.time)

    return {
        "timestamp": timestamp,
        "window_id": get_window_id(timestamp),
        "window_start": get_window_start(timestamp),
        "window_end": get_window_end(timestamp),
    }


def group_packets_by_window(packets):
    # Group packets according to the canonical window size.
    windows = {}

    for packet in packets:
        timestamp = float(packet.time)
        window_id = get_window_id(timestamp)

        if window_id not in windows:
            windows[window_id] = []

        windows[window_id].append(packet)

    return windows