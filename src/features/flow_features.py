#!/usr/bin/env python3
"""
flow_features.py
----------------
Aggregate per-flow columns into per-(host_id, window_start) feature vectors.

All 22 features match exactly the FEATURE_COLUMNS list in train_lstm.py and
sequence_builder.py so the pipeline stays consistent.
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLS = ["host_id", "window_start", "StartTime", "SrcAddr", "DstAddr",
                 "Dport", "TotBytes", "TotPkts", "Dur", "Proto", "Dir", "State"]


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")


def aggregate_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 22 aggregate features per (host_id, window_start).

    Expects `df` to already have `host_id` and `window_start` columns
    (added by `add_time_columns`).
    """
    _validate_columns(df)
    df = df.sort_values(["host_id", "StartTime"]).copy()

    # Inter-arrival time (IAT) per host
    df["iat"] = (
        df.groupby("host_id")["StartTime"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )

    # Protocol / direction flags
    df["tcp_flag"] = (df["Proto"].str.lower() == "tcp").astype(int)
    df["udp_flag"] = (df["Proto"].str.lower() == "udp").astype(int)
    df["bidirectional_flag"] = df["Dir"].str.contains("<->", na=False).astype(int)

    grouped = df.groupby(["host_id", "window_start"], dropna=False)

    features = grouped.agg(
        flow_count=("SrcAddr", "count"),
        unique_dst_ip_count=("DstAddr", "nunique"),
        unique_dst_port_count=("Dport", "nunique"),
        bytes_total=("TotBytes", "sum"),
        bytes_mean=("TotBytes", "mean"),
        bytes_std=("TotBytes", "std"),
        packets_total=("TotPkts", "sum"),
        packets_mean=("TotPkts", "mean"),
        duration_mean=("Dur", "mean"),
        duration_std=("Dur", "std"),
        iat_mean=("iat", "mean"),
        iat_std=("iat", "std"),
        iat_max=("iat", "max"),
        tcp_flow_ratio=("tcp_flag", "mean"),
        udp_flow_ratio=("udp_flag", "mean"),
        bidirectional_ratio=("bidirectional_flag", "mean"),
        syn_count=("State", lambda x: x.str.contains("S", na=False).sum()),
        ack_count=("State", lambda x: x.str.contains("A", na=False).sum()),
        fin_count=("State", lambda x: x.str.contains("F", na=False).sum()),
        rst_count=("State", lambda x: x.str.contains("R", na=False).sum()),
        psh_count=("State", lambda x: x.str.contains("P", na=False).sum()),
        urg_count=("State", lambda x: x.str.contains("U", na=False).sum()),
    ).reset_index()

    # Fill NaN that can appear in std cols for single-flow windows
    features = features.fillna(0)
    log.debug("Feature matrix shape: %s", features.shape)
    return features


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    path = "data/raw/CTU-13/sample_100k.binetflow"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    from src.features.windowing import add_time_columns
    df = add_time_columns(df)
    feats = aggregate_flow_features(df)
    print("Feature shape:", feats.shape)
    print(feats.head())
