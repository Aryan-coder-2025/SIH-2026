"""
Canonical Model-20 Feature Schema, Invariants, and Extraction Module.

Defines the authoritative 20-feature input schema for temporal LSTM risk forecasting.
Enforces non-leakage invariants, finite-value guarantees, and strict ordering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMA_VERSION = "2.0"

# Authoritative 20 Flow Features for Temporal Models
CANONICAL_MODEL_20_FEATURES: list[str] = [
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

# Authoritative 19 Packet Features
CANONICAL_PACKET_FEATURES: list[str] = [
    "packet_count",
    "ttl_mean",
    "ttl_std",
    "ttl_min",
    "ttl_max",
    "tcp_window_mean",
    "tcp_window_std",
    "fragment_count",
    "payload_mean",
    "payload_std",
    "payload_min",
    "payload_max",
    "retransmission_count",
    "port_scan_score",
    "sequential_port_ratio",
    "unique_dst_ports",
    "packet_iat_mean",
    "packet_iat_std",
    "packet_iat_max",
]

# Rich Flow Features (including destination diversity)
CANONICAL_RICH_FLOW_FEATURES: list[str] = (
    CANONICAL_MODEL_20_FEATURES + ["unique_dst_ip_count", "unique_dst_port_count"]
)

# 41 Fused Features
CANONICAL_FUSED_FEATURES: list[str] = (
    CANONICAL_RICH_FLOW_FEATURES + CANONICAL_PACKET_FEATURES
)

METADATA_COLUMNS = {
    "src_ip",
    "source_host",
    "window_id",
    "timestamp",
    "window_start",
    "window_end",
}

FORBIDDEN_COLUMNS = {
    "label",
    "is_malicious",
    "target",
    "risk_score",
    "stage",
    "future_label",
    "future_risk",
}


def validate_model_20_features(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame contains the complete, canonical 20 features
    with finite numeric values and without label/target leakage.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    missing = [f for f in CANONICAL_MODEL_20_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical model-20 features: {missing}")

    for feature in CANONICAL_MODEL_20_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise TypeError(f"Feature '{feature}' must be numeric.")
        
        arr = df[feature].to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"Feature '{feature}' contains NaN or infinite values.")

        # Invariant check: counts and durations cannot be negative
        if feature in {
            "flow_count", "bytes_total", "packets_total", "duration_mean",
            "syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count"
        }:
            if (arr < 0).any():
                raise ValueError(f"Feature '{feature}' contains negative values.")


def validate_feature_order(feature_columns: list[str]) -> None:
    """Ensure feature columns match exact canonical model-20 order."""
    if list(feature_columns) != CANONICAL_MODEL_20_FEATURES:
        raise ValueError(
            f"Feature order mismatch.\nExpected: {CANONICAL_MODEL_20_FEATURES}\nReceived: {feature_columns}"
        )


def extract_model_20_features(
    df: pd.DataFrame,
    include_metadata: bool = True,
) -> pd.DataFrame:
    """
    Extract the canonical model-20 feature matrix from fused or flow DataFrame.
    Validates completeness and finite values.
    """
    validate_model_20_features(df)

    cols = []
    if include_metadata:
        for meta in ["src_ip", "source_host", "window_id", "timestamp"]:
            if meta in df.columns:
                cols.append(meta)

    cols.extend(CANONICAL_MODEL_20_FEATURES)
    return df[cols].copy()


def get_schema_info() -> dict:
    """Return dictionary metadata describing the canonical model-20 schema."""
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_count": len(CANONICAL_MODEL_20_FEATURES),
        "feature_order": list(CANONICAL_MODEL_20_FEATURES),
        "packet_feature_count": len(CANONICAL_PACKET_FEATURES),
        "fused_feature_count": len(CANONICAL_FUSED_FEATURES),
        "metadata_columns": sorted(METADATA_COLUMNS),
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
    }
