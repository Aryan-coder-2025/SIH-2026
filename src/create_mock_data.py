"""
Mock Dataset Generator for SIH26153
AI-Based Network Attack Forecasting from Network Traffic Data

Generates:
- 10-second network traffic windows
- Multiple source hosts
- Realistic-looking network features
- Benign traffic
- Scanning attacks
- Exploitation attacks
- Infiltration attacks
- Temporal attack progression
- Malicious labels
- Stage labels

Output:
    data/mock/mock_feature_matrix.parquet
    data/mock/mock_feature_matrix.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

OUTPUT_DIR = Path("data/mock")

PARQUET_PATH = OUTPUT_DIR / "mock_feature_matrix.parquet"
CSV_PATH = OUTPUT_DIR / "mock_feature_matrix.csv"

# 10-second windows
WINDOW_SECONDS = 10

# 6 hours of data
HOURS = 6

# Number of source hosts
NUM_HOSTS = 6

# Start timestamp
START_TIME = "2026-09-08 00:00:00"

# Host names
HOSTS = [
    "host_A",
    "host_B",
    "host_C",
    "host_D",
    "host_E",
    "host_F",
]


# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(SEED)


# ============================================================
# ATTACK EPISODES
# ============================================================
#
# Each attack is represented as:
#
#   SCANNING
#       ↓
#   EXPLOITATION
#       ↓
#   INFILTRATION
#
# The values are window indices.
#
# Since every window = 10 seconds:
#
# 180 -> 30 minutes
# 360 -> 60 minutes
#
# ============================================================

ATTACK_EPISODES = {
    "host_A": [
        {
            "start": 180,
            "scan_end": 200,
            "exploit_end": 215,
            "end": 230,
        },
        {
            "start": 900,
            "scan_end": 920,
            "exploit_end": 935,
            "end": 950,
        },
    ],

    "host_B": [
        {
            "start": 300,
            "scan_end": 320,
            "exploit_end": 335,
            "end": 350,
        }
    ],

    "host_C": [
        {
            "start": 600,
            "scan_end": 620,
            "exploit_end": 640,
            "end": 660,
        }
    ],

    "host_D": [
        {
            "start": 1050,
            "scan_end": 1070,
            "exploit_end": 1085,
            "end": 1100,
        }
    ],

    "host_E": [
        {
            "start": 750,
            "scan_end": 770,
            "exploit_end": 785,
            "end": 805,
        }
    ],

    "host_F": [
        {
            "start": 1200,
            "scan_end": 1220,
            "exploit_end": 1235,
            "end": 1250,
        }
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clip(value, low, high):
    """Keep a value within a specified range."""
    return max(low, min(value, high))


def poisson(mean):
    """Generate a Poisson-distributed value."""
    return int(rng.poisson(max(mean, 0.1)))


def positive_normal(mean, std, minimum=0):
    """Generate a non-negative normally distributed value."""
    value = rng.normal(mean, std)
    return max(minimum, value)


def get_attack_stage(host, window_index):
    """
    Determine the attack stage for a host at a particular window.

    Returns:
        BENIGN
        SCANNING
        EXPLOITATION
        INFILTRATION
    """

    episodes = ATTACK_EPISODES.get(host, [])

    for episode in episodes:

        if episode["start"] <= window_index < episode["scan_end"]:
            return "SCANNING"

        if episode["scan_end"] <= window_index < episode["exploit_end"]:
            return "EXPLOITATION"

        if episode["exploit_end"] <= window_index < episode["end"]:
            return "INFILTRATION"

    return "BENIGN"


def stage_multiplier(stage):
    """
    Controls how strongly network behaviour changes
    during different attack stages.
    """

    if stage == "BENIGN":
        return {
            "traffic": 1.0,
            "packets": 1.0,
            "syn": 1.0,
            "connections": 1.0,
            "duration": 1.0,
            "entropy": 1.0,
        }

    if stage == "SCANNING":
        return {
            "traffic": 1.8,
            "packets": 2.2,
            "syn": 6.0,
            "connections": 5.0,
            "duration": 0.7,
            "entropy": 1.3,
        }

    if stage == "EXPLOITATION":
        return {
            "traffic": 3.0,
            "packets": 3.5,
            "syn": 3.5,
            "connections": 3.0,
            "duration": 1.8,
            "entropy": 1.7,
        }

    if stage == "INFILTRATION":
        return {
            "traffic": 5.0,
            "packets": 4.5,
            "syn": 2.5,
            "connections": 2.0,
            "duration": 3.0,
            "entropy": 2.0,
        }

    return {
        "traffic": 1.0,
        "packets": 1.0,
        "syn": 1.0,
        "connections": 1.0,
        "duration": 1.0,
        "entropy": 1.0,
    }


# ============================================================
# GENERATE DATA
# ============================================================

def generate_dataset():

    print("=" * 70)
    print("SIH26153 MOCK DATASET GENERATOR")
    print("=" * 70)

    # 6 hours / 10 seconds
    total_windows = int((HOURS * 60 * 60) / WINDOW_SECONDS)

    print(f"Hosts              : {NUM_HOSTS}")
    print(f"Hours              : {HOURS}")
    print(f"Window size        : {WINDOW_SECONDS} seconds")
    print(f"Windows per host   : {total_windows}")
    print(f"Total rows         : {total_windows * NUM_HOSTS}")
    print()

    timestamps = pd.date_range(
        start=START_TIME,
        periods=total_windows,
        freq=f"{WINDOW_SECONDS}s"
    )

    rows = []

    # --------------------------------------------------------
    # Host-specific baseline behaviour
    # --------------------------------------------------------

    host_profiles = {
        "host_A": {
            "bytes": 18000,
            "packets": 120,
            "connections": 8,
        },
        "host_B": {
            "bytes": 24000,
            "packets": 160,
            "connections": 12,
        },
        "host_C": {
            "bytes": 14000,
            "packets": 95,
            "connections": 6,
        },
        "host_D": {
            "bytes": 30000,
            "packets": 190,
            "connections": 15,
        },
        "host_E": {
            "bytes": 21000,
            "packets": 135,
            "connections": 10,
        },
        "host_F": {
            "bytes": 26000,
            "packets": 175,
            "connections": 13,
        },
    }

    # --------------------------------------------------------
    # Generate each host
    # --------------------------------------------------------

    for host in HOSTS:

        profile = host_profiles[host]

        for window_index, timestamp in enumerate(timestamps):

            stage = get_attack_stage(host, window_index)

            multipliers = stage_multiplier(stage)

            # ------------------------------------------------
            # Daily/time-of-day effect
            # ------------------------------------------------

            hour = timestamp.hour

            if 8 <= hour < 18:
                activity_factor = 1.20
            elif 18 <= hour < 23:
                activity_factor = 0.90
            else:
                activity_factor = 0.55

            # Small smooth temporal variation
            temporal_factor = (
                1.0
                + 0.08 * np.sin(window_index / 20)
                + 0.04 * np.sin(window_index / 7)
            )

            # ------------------------------------------------
            # Core traffic features
            # ------------------------------------------------

            bytes_value = positive_normal(
                profile["bytes"]
                * activity_factor
                * temporal_factor
                * multipliers["traffic"],
                profile["bytes"] * 0.20,
                minimum=100
            )

            packets_value = positive_normal(
                profile["packets"]
                * activity_factor
                * temporal_factor
                * multipliers["packets"],
                profile["packets"] * 0.20,
                minimum=5
            )

            # Average packet size
            avg_packet_size = bytes_value / max(packets_value, 1)

            # ------------------------------------------------
            # Connection features
            # ------------------------------------------------

            connection_count = positive_normal(
                profile["connections"]
                * activity_factor
                * multipliers["connections"],
                profile["connections"] * 0.30,
                minimum=1
            )

            unique_dst_ips = positive_normal(
                profile["connections"]
                * 0.7
                * multipliers["connections"],
                2.0,
                minimum=1
            )

            unique_dst_ports = positive_normal(
                3
                * multipliers["connections"],
                1.5,
                minimum=1
            )

            # ------------------------------------------------
            # TCP/SYN behaviour
            # ------------------------------------------------

            if stage == "BENIGN":
                syn_mean = 2
                rst_mean = 1
            elif stage == "SCANNING":
                syn_mean = 18
                rst_mean = 10
            elif stage == "EXPLOITATION":
                syn_mean = 12
                rst_mean = 7
            else:
                syn_mean = 8
                rst_mean = 4

            syn_count = poisson(syn_mean * multipliers["syn"])
            rst_count = poisson(rst_mean * multipliers["syn"])

            ack_count = poisson(
                max(
                    packets_value * 0.18,
                    1
                )
            )

            fin_count = poisson(
                max(
                    packets_value * 0.05,
                    1
                )
            )

            # ------------------------------------------------
            # Timing features
            # ------------------------------------------------

            flow_duration = positive_normal(
                2.0 * multipliers["duration"],
                0.8,
                minimum=0.05
            )

            inter_arrival_mean = positive_normal(
                0.12 / multipliers["packets"],
                0.03,
                minimum=0.001
            )

            inter_arrival_std = positive_normal(
                0.05 * multipliers["duration"],
                0.02,
                minimum=0.001
            )

            # ------------------------------------------------
            # TTL
            # ------------------------------------------------

            ttl_mean = clip(
                positive_normal(
                    64,
                    8
                ),
                32,
                128
            )

            ttl_std = clip(
                positive_normal(
                    7,
                    3
                ),
                0.5,
                30
            )

            # ------------------------------------------------
            # Packet statistics
            # ------------------------------------------------

            packet_size_std = positive_normal(
                180 * multipliers["entropy"],
                40,
                minimum=5
            )

            packet_size_min = clip(
                positive_normal(40, 10),
                20,
                100
            )

            packet_size_max = clip(
                positive_normal(
                    1400 * multipliers["traffic"],
                    150
                ),
                200,
                9000
            )

            # ------------------------------------------------
            # Protocol / service diversity
            # ------------------------------------------------

            protocol_diversity = clip(
                positive_normal(
                    3.0 * multipliers["entropy"],
                    0.8
                ),
                1,
                20
            )

            service_diversity = clip(
                positive_normal(
                    4.0 * multipliers["connections"],
                    1.0
                ),
                1,
                30
            )

            # ------------------------------------------------
            # Byte/packet rates
            # ------------------------------------------------

            bytes_per_second = bytes_value / WINDOW_SECONDS

            packets_per_second = packets_value / WINDOW_SECONDS

            # ------------------------------------------------
            # Attack-specific features
            # ------------------------------------------------

            failed_connections = poisson(
                max(
                    2 * multipliers["connections"],
                    1
                )
            )

            successful_connections = poisson(
                max(
                    5 * multipliers["connections"],
                    1
                )
            )

            outbound_ratio = clip(
                rng.normal(
                    0.45 if stage == "BENIGN" else 0.65,
                    0.08
                ),
                0.05,
                0.95
            )

            inbound_ratio = 1.0 - outbound_ratio

            # ------------------------------------------------
            # Risk label
            # ------------------------------------------------

            if stage == "BENIGN":
                is_malicious = 0
            else:
                is_malicious = 1

            # ------------------------------------------------
            # Risk score
            # ------------------------------------------------

            if stage == "BENIGN":
                risk_score = clip(
                    rng.normal(0.10, 0.04),
                    0.0,
                    0.30
                )

            elif stage == "SCANNING":
                risk_score = clip(
                    rng.normal(0.60, 0.08),
                    0.35,
                    0.85
                )

            elif stage == "EXPLOITATION":
                risk_score = clip(
                    rng.normal(0.82, 0.07),
                    0.60,
                    0.97
                )

            elif stage == "INFILTRATION":
                risk_score = clip(
                    rng.normal(0.94, 0.04),
                    0.75,
                    1.00
                )

            # ------------------------------------------------
            # Build row
            # ------------------------------------------------

            row = {
                "timestamp": timestamp,
                "source_host": host,

                # Traffic volume
                "bytes": round(bytes_value, 2),
                "packets": round(packets_value, 2),
                "bytes_per_second": round(bytes_per_second, 4),
                "packets_per_second": round(packets_per_second, 4),
                "avg_packet_size": round(avg_packet_size, 4),

                # Connections
                "connection_count": round(connection_count, 2),
                "unique_dst_ips": round(unique_dst_ips, 2),
                "unique_dst_ports": round(unique_dst_ports, 2),

                # TCP behaviour
                "syn_count": syn_count,
                "ack_count": ack_count,
                "fin_count": fin_count,
                "rst_count": rst_count,

                # Timing
                "flow_duration_mean": round(flow_duration, 4),
                "inter_arrival_mean": round(inter_arrival_mean, 6),
                "inter_arrival_std": round(inter_arrival_std, 6),

                # TTL
                "ttl_mean": round(ttl_mean, 4),
                "ttl_std": round(ttl_std, 4),

                # Packet sizes
                "packet_size_std": round(packet_size_std, 4),
                "packet_size_min": round(packet_size_min, 4),
                "packet_size_max": round(packet_size_max, 4),

                # Diversity
                "protocol_diversity": round(protocol_diversity, 4),
                "service_diversity": round(service_diversity, 4),

                # Connection outcomes
                "failed_connections": failed_connections,
                "successful_connections": successful_connections,

                # Direction
                "outbound_ratio": round(outbound_ratio, 4),
                "inbound_ratio": round(inbound_ratio, 4),

                # Targets
                "risk_score": round(risk_score, 4),
                "is_malicious": is_malicious,
                "stage": stage,
            }

            rows.append(row)

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(rows)

    # Sort by host and time
    df = df.sort_values(
        ["source_host", "timestamp"]
    ).reset_index(drop=True)

    # ========================================================
    # SAVE DATA
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        PARQUET_PATH,
        index=False
    )

    df.to_csv(
        CSV_PATH,
        index=False
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"Parquet file : {PARQUET_PATH}")
    print(f"CSV file     : {CSV_PATH}")
    print()

    print("Shape:")
    print(df.shape)

    print()
    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")

    print()
    print("Hosts:")
    print(df["source_host"].value_counts())

    print()
    print("Stage distribution:")
    print(df["stage"].value_counts())

    print()
    print("Malicious distribution:")
    print(df["is_malicious"].value_counts())

    print()
    print("Sample rows:")
    print(df.head(10).to_string())

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_dataset()