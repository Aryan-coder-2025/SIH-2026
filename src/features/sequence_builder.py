#!/usr/bin/env python3
"""
sequence_builder.py
-------------------
Converts the per-(src_ip, window_id) feature DataFrame into LSTM-ready sequences.

A sequence is a sliding window of HISTORY_WINDOWS time steps, and the label
is whether ANY of the next 1/2/3 windows will be classified as botnet.
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HISTORY_WINDOWS = 10
FORECAST_STEPS = [1, 2, 3]   # produce y_10, y_20, y_30

FEATURE_COLUMNS = [
    "flow_count",
    "unique_dst_ip_count",
    "unique_dst_port_count",
    "bytes_total",
    "bytes_mean",
    "bytes_std",
    "packets_total",
    "packets_mean",
    "duration_mean",
    "duration_std",
    "iat_mean",
    "iat_std",
    "iat_max",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "tcp_flow_ratio",
    "udp_flow_ratio",
    "bidirectional_ratio",
]


def _check_columns(df: pd.DataFrame) -> None:
    needed = set(FEATURE_COLUMNS) | {"src_ip", "window_id", "traffic_category"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns for sequence building: {missing}")


def build_sequences(df: pd.DataFrame) -> list:
    """Return a list of sequence dicts from the processed DataFrame.

    Each dict contains:
        src_ip       - str
        window_id    - int (the prediction moment)
        X            - float32 array of shape (HISTORY_WINDOWS, n_features)
        y_10, y_20, y_30 - int (1 = botnet in next 10 / 20 / 30 seconds)
    """
    _check_columns(df)
    sequences = []

    min_windows = HISTORY_WINDOWS + max(FORECAST_STEPS)
    host_counts = df["src_ip"].value_counts()
    valid_hosts = set(host_counts[host_counts >= min_windows].index)
    if not valid_hosts:
        log.info("Built 0 sequences from %d hosts.", df["src_ip"].nunique())
        return []

    # Sort once globally by src_ip and window_id for blazing-fast sequential traversal
    df_filtered = df[df["src_ip"].isin(valid_hosts)].sort_values(["src_ip", "window_id"])

    for src_ip, host_df in df_filtered.groupby("src_ip", sort=False):
        window_ids = host_df["window_id"].to_numpy(dtype=np.int64)
        features = host_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        is_botnet = (host_df["traffic_category"] == "botnet").to_numpy(dtype=np.int32)
        n_rows = len(window_ids)

        if n_rows < min_windows:
            continue

        diffs = np.diff(window_ids)
        for i in range(HISTORY_WINDOWS, n_rows - 2):
            # Verify all consecutive windows in [i - HISTORY_WINDOWS : i + 3] are contiguous (1-step diff)
            if not np.all(diffs[i - HISTORY_WINDOWS : i + 2] == 1):
                continue

            sequences.append({
                "src_ip": src_ip,
                "window_id": int(window_ids[i]),
                "X": features[i - HISTORY_WINDOWS : i],
                "y_10": int(is_botnet[i]),
                "y_20": int(is_botnet[i + 1]),
                "y_30": int(is_botnet[i + 2]),
            })

    log.info("Built %d sequences from %d hosts.", len(sequences), df["src_ip"].nunique())
    return sequences


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = "data/processed/ctu13_scenario1_windows.parquet"
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"[ERROR] Processed data not found: {path}")
        sys.exit(1)

    seqs = build_sequences(df)
    print(f"Sequences: {len(seqs)}")
    if seqs:
        print("First sequence - X shape:", seqs[0]["X"].shape)
        print("Targets y10/y20/y30:", seqs[0]["y_10"], seqs[0]["y_20"], seqs[0]["y_30"])