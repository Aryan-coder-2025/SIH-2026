#!/usr/bin/env python3
"""
sequence_builder.py
-------------------
Converts the per-(host, window) feature DataFrame into LSTM-ready sequences.

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
    needed = set(FEATURE_COLUMNS) | {"host_id", "window_start", "traffic_category"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns for sequence building: {missing}")


def build_sequences(df: pd.DataFrame) -> list:
    """Return a list of sequence dicts from the processed DataFrame.

    Each dict contains:
        host_id      – str
        window_start – Timestamp (the prediction moment)
        X            – float32 array of shape (HISTORY_WINDOWS, n_features)
        y_10, y_20, y_30 – int (1 = botnet in next 10 / 20 / 30 seconds)
    """
    _check_columns(df)
    sequences = []

    for host_id, host_df in df.groupby("host_id"):
        host_df = host_df.sort_values("window_start").reset_index(drop=True)

        for i in range(HISTORY_WINDOWS, len(host_df)):
            future = host_df.iloc[i: i + 3]
            if len(future) < 3:
                continue  # not enough future windows

            # Verify all consecutive windows are exactly 10 s apart
            combined = host_df.iloc[i - HISTORY_WINDOWS: i + 3]
            time_diffs = combined["window_start"].diff().dt.total_seconds().dropna()
            if not (time_diffs == 10).all():
                continue

            history = host_df.iloc[i - HISTORY_WINDOWS: i]
            X = history[FEATURE_COLUMNS].values.astype(np.float32)

            sequences.append({
                "host_id": host_id,
                "window_start": host_df.iloc[i]["window_start"],
                "X": X,
                "y_10": int(future.iloc[0]["traffic_category"] == "botnet"),
                "y_20": int(future.iloc[1]["traffic_category"] == "botnet"),
                "y_30": int(future.iloc[2]["traffic_category"] == "botnet"),
            })

    log.info("Built %d sequences from %d hosts.", len(sequences), df["host_id"].nunique())
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
        print("First sequence – X shape:", seqs[0]["X"].shape)
        print("Targets y10/y20/y30:", seqs[0]["y_10"], seqs[0]["y_20"], seqs[0]["y_30"])
