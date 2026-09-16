from __future__ import annotations

import numpy as np
import pandas as pd

# Bump this whenever feature names/order or the model input contract changes.
SCHEMA_VERSION = "1.1"

CANONICAL_FEATURES = [
    "flow_count",
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

METADATA_COLUMNS = {"source_host", "timestamp", "is_malicious"}
TARGET_COLUMNS = {
    "risk_score", "is_malicious", "stage", "future_label",
    "future_risk", "target", "label",
}


def validate_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Strictly validate the exact model-input schema and reject leakage."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Feature list must be non-empty and contain no duplicates.")

    missing = [f for f in feature_columns if f not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    allowed = set(feature_columns) | METADATA_COLUMNS | TARGET_COLUMNS
    unexpected = [c for c in df.columns if c not in allowed]
    if unexpected:
        raise ValueError(f"Unexpected columns found in model dataframe: {unexpected}")

    forbidden = [f for f in feature_columns if f in TARGET_COLUMNS]
    if forbidden:
        raise ValueError(f"Target columns cannot be model features: {forbidden}")

    for feature in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise TypeError(f"Feature '{feature}' must be numeric.")
        values = df[feature].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature '{feature}' contains NaN or Inf.")


def validate_feature_order(feature_columns: list[str]) -> None:
    if list(feature_columns) != CANONICAL_FEATURES:
        raise ValueError(
            "Feature order/schema mismatch. "
            f"Expected: {CANONICAL_FEATURES}; received: {feature_columns}"
        )


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_feature_columns(df, CANONICAL_FEATURES)
    validate_feature_order(CANONICAL_FEATURES)
    return df[CANONICAL_FEATURES].copy()


def validate_window_dataframe(df: pd.DataFrame) -> None:
    """Validate the complete window-level dataframe before sequence creation."""
    validate_feature_columns(df, CANONICAL_FEATURES)

    for column in ("timestamp", "is_malicious"):
        if column not in df.columns:
            raise ValueError(f"Required window-level column missing: {column}")

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp contains invalid values.")

    if timestamps.duplicated().any() and "source_host" not in df.columns:
        raise ValueError("Duplicate network-level timestamps are not allowed.")

    if df["is_malicious"].isna().any():
        raise ValueError("is_malicious contains missing values.")

    values = set(df["is_malicious"].unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"is_malicious must contain only 0 or 1, found {values}.")

    if "source_host" in df.columns:
        if df["source_host"].isna().any():
            raise ValueError("source_host contains missing values.")
        if (df["source_host"].astype(str).str.len() == 0).any():
            raise ValueError("source_host contains empty values.")
        if df.duplicated(["source_host", "timestamp"]).any():
            raise ValueError("Duplicate source_host/timestamp rows detected.")


def get_schema_info() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_count": len(CANONICAL_FEATURES),
        "feature_order": list(CANONICAL_FEATURES),
        "metadata_columns": sorted(METADATA_COLUMNS),
        "excluded_target_columns": sorted(TARGET_COLUMNS),
    }
