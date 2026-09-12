#!/usr/bin/env python3
"""
windowing.py
------------
Time-window utilities for the CTU-13 pipeline.
Adds `window_start` and `host_id` columns to the raw flow DataFrame,
and aggregates per-(host, window) traffic labels.
"""
import logging

import pandas as pd

from src.features.labels import classify_label

log = logging.getLogger(__name__)

WINDOW_SIZE = "10s"


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse StartTime, floor to 10-second windows, derive host_id.

    Accepts both the real CTU-13 format ('YYYY/MM/DD HH:MM:SS.ffffff')
    and plain ISO-8601 strings so the synthetic dataset works too.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["StartTime"]):
        # Try CTU-13 format first, fall back to format="mixed"
        try:
            df["StartTime"] = pd.to_datetime(df["StartTime"], format="%Y/%m/%d %H:%M:%S.%f")
        except Exception:
            df["StartTime"] = pd.to_datetime(df["StartTime"], format="mixed")

    df["window_start"] = df["StartTime"].dt.floor(WINDOW_SIZE)
    df["host_id"] = df["SrcAddr"].astype(str)
    return df


def add_window_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per (host_id, window_start) and a majority label.

    A window is labelled 'botnet' if ANY flow in it is botnet, otherwise 'normal'.
    Windows that are entirely 'background' are still kept but labelled 'normal' so
    the classifier can learn from them (they carry no positive signal).
    """
    df = df.copy()
    df["traffic_category"] = df["Label"].apply(classify_label)

    labels = (
        df.groupby(["host_id", "window_start"])["traffic_category"]
        .apply(lambda x: "botnet" if "botnet" in x.values else "normal")
        .reset_index(name="traffic_category")
    )
    log.debug("Window label distribution:\n%s", labels["traffic_category"].value_counts())
    return labels


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = "data/raw/CTU-13/sample_100k.binetflow"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    df = add_time_columns(df)
    labels = add_window_labels(df)
    print("Window labels (head):\n", labels.head(20))
    print("\nLabel distribution:\n", labels["traffic_category"].value_counts())
