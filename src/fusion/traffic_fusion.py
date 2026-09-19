"""
Traffic Fusion Module.

Combines packet-level features and flow-level features into the canonical
(source host × 10-second window) network state representation.

ENFORCEMENT:
    - Replaces unsafe 'drop_duplicates' with host-window aggregation.
    - Computes and validates join cardinality and coverage metrics per DATA_JOIN_CONTRACT.md.
    - Prevents target or raw identifier leakage into feature vectors.
    - Connects directly to Aryan's canonical TrafficWindow dataclass.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.flow.flow_extractor import aggregate_flows_to_host_windows
from src.schemas.features import (
    CANONICAL_FLOW_FEATURE_NAMES,
    CANONICAL_MODEL_FEATURE_NAMES,
    CANONICAL_PACKET_FEATURE_NAMES,
    FEATURE_ALIASES,
    FORBIDDEN_FEATURE_NAMES,
    validate_feature_names,
)
from src.schemas.traffic import TrafficWindow

REQUIRED_PACKET_KEYS = ["src_ip", "window_id"]
REQUIRED_FLOW_KEYS = ["src_ip", "window_id"]


@dataclass(frozen=True)
class JoinValidationReport:
    """Detailed join cardinality and coverage metrics."""
    packet_rows_before: int
    flow_rows_before: int
    packet_host_windows: int
    flow_host_windows: int
    rows_after_join: int
    matched_windows: int
    unmatched_packet_windows: int
    unmatched_flow_windows: int
    packet_coverage_pct: float
    flow_coverage_pct: float
    has_row_multiplication: bool
    has_row_loss: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "packet_rows_before": self.packet_rows_before,
            "flow_rows_before": self.flow_rows_before,
            "packet_host_windows": self.packet_host_windows,
            "flow_host_windows": self.flow_host_windows,
            "rows_after_join": self.rows_after_join,
            "matched_windows": self.matched_windows,
            "unmatched_packet_windows": self.unmatched_packet_windows,
            "unmatched_flow_windows": self.unmatched_flow_windows,
            "packet_coverage_pct": self.packet_coverage_pct,
            "flow_coverage_pct": self.flow_coverage_pct,
            "has_row_multiplication": self.has_row_multiplication,
            "has_row_loss": self.has_row_loss,
        }


def validate_join_cardinality(
    packet_df: pd.DataFrame,
    flow_agg_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    how: str = "left",
) -> JoinValidationReport:
    """
    Validate join metrics, detecting duplicates, unmapped windows,
    row multiplication, and row loss.
    """
    p_keys = set(zip(packet_df["src_ip"], packet_df["window_id"])) if not packet_df.empty else set()
    f_keys = set(zip(flow_agg_df["src_ip"], flow_agg_df["window_id"])) if not flow_agg_df.empty else set()

    matched_keys = p_keys & f_keys
    unmatched_p = p_keys - f_keys
    unmatched_f = f_keys - p_keys

    p_cov = (len(matched_keys) / len(p_keys) * 100.0) if p_keys else 0.0
    f_cov = (len(matched_keys) / len(f_keys) * 100.0) if f_keys else 0.0

    expected_len = len(p_keys) if how == "left" else (len(p_keys | f_keys) if how == "outer" else len(matched_keys))
    has_mult = len(fused_df) > expected_len
    has_loss = len(fused_df) < expected_len

    return JoinValidationReport(
        packet_rows_before=len(packet_df),
        flow_rows_before=len(flow_agg_df),
        packet_host_windows=len(p_keys),
        flow_host_windows=len(f_keys),
        rows_after_join=len(fused_df),
        matched_windows=len(matched_keys),
        unmatched_packet_windows=len(unmatched_p),
        unmatched_flow_windows=len(unmatched_f),
        packet_coverage_pct=round(p_cov, 2),
        flow_coverage_pct=round(f_cov, 2),
        has_row_multiplication=has_mult,
        has_row_loss=has_loss,
    )


def fuse_flow_and_packet_dfs(
    flow_df: pd.DataFrame,
    packet_df: pd.DataFrame,
    how: str = "left",
) -> tuple[pd.DataFrame, JoinValidationReport]:
    """
    Merge flow-level and packet-level dataframes into the canonical
    source_host × 10-second window representation.

    Correctness guarantees:
        1. If flow_df contains multiple flows per (src_ip, window_id), they are
           aggregated instead of dropped.
        2. Cardinality and coverage are explicitly audited.
        3. No target or identifier leakage.
    """
    for col in REQUIRED_PACKET_KEYS:
        if col not in packet_df.columns:
            raise ValueError(f"Packet data missing required key column: {col}")

    for col in REQUIRED_FLOW_KEYS:
        if col not in flow_df.columns:
            raise ValueError(f"Flow data missing required key column: {col}")

    packet_df = packet_df.copy()
    flow_df = flow_df.copy()
    packet_df["window_id"] = packet_df["window_id"].astype(int)
    flow_df["window_id"] = flow_df["window_id"].astype(int)
    packet_df["src_ip"] = packet_df["src_ip"].astype(str).str.strip()
    flow_df["src_ip"] = flow_df["src_ip"].astype(str).str.strip()

    # Step 1: Ensure flow table is aggregated per (src_ip, window_id)
    # Check if flow_df has multiple rows per key
    flow_dup_mask = flow_df.duplicated(subset=["src_ip", "window_id"])
    if flow_dup_mask.any():
        flow_agg = aggregate_flows_to_host_windows(flow_df)
    else:
        flow_agg = flow_df.copy()

    # Step 2: Ensure packet table has no duplicate (src_ip, window_id) keys
    if packet_df.duplicated(subset=["src_ip", "window_id"]).any():
        raise ValueError("Packet data contains duplicate source-host/window rows.")

    # Step 3: Prefix non-key columns of flow table if overlapping with packet table
    overlapping = (set(packet_df.columns) & set(flow_agg.columns)) - {"src_ip", "window_id"}
    if overlapping:
        rename_map = {col: f"flow_{col}" if not col.startswith("flow_") else col for col in overlapping}
        flow_agg = flow_agg.rename(columns=rename_map)

    # Step 4: Perform merge
    fused_df = pd.merge(
        packet_df,
        flow_agg,
        on=["src_ip", "window_id"],
        how=how,
    )

    # Step 5: Chronological sort by (src_ip, window_id)
    if not fused_df.empty:
        sort_cols = ["src_ip", "window_id"]
        if "timestamp" in fused_df.columns:
            sort_cols = ["src_ip", "timestamp", "window_id"]
        fused_df = fused_df.sort_values(sort_cols).reset_index(drop=True)

    # Step 6: Validate join cardinality and generate audit report
    report = validate_join_cardinality(packet_df, flow_agg, fused_df, how=how)

    return fused_df, report


def fuse_flow_and_packet_data(
    flow_file: str | pd.DataFrame,
    packet_file: str | pd.DataFrame,
    output_file: str | None = None,
    how: str = "left",
    return_metrics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load data (Parquet or DataFrame), fuse packet and flow features, and optionally return coverage metrics.

    Args:
        flow_file: Path to flow Parquet file or flow DataFrame.
        packet_file: Path to packet Parquet file or packet DataFrame.
        output_file: Optional path to save result as Parquet.
        how: Merge strategy ('left', 'inner', 'outer').
        return_metrics: If True, returns (fused_df, metrics_dict).

    Returns:
        fused_df if return_metrics is False, else (fused_df, metrics_dict).
    """
    if isinstance(flow_file, pd.DataFrame):
        flow_df = flow_file
    else:
        if not os.path.isfile(flow_file):
            raise FileNotFoundError(f"Flow file not found: {flow_file}")
        flow_df = pd.read_parquet(flow_file)

    if isinstance(packet_file, pd.DataFrame):
        packet_df = packet_file
    else:
        if not os.path.isfile(packet_file):
            raise FileNotFoundError(f"Packet file not found: {packet_file}")
        packet_df = pd.read_parquet(packet_file)

    fused_df, report = fuse_flow_and_packet_dfs(flow_df, packet_df, how=how)

    if output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        fused_df.to_parquet(output_file, index=False)

    if return_metrics:
        metrics: dict[str, Any] = {
            "packet_rows_before": report.packet_rows_before,
            "flow_rows_before": report.flow_rows_before,
            "packet_host_windows": report.packet_host_windows,
            "flow_host_windows": report.flow_host_windows,
            "rows_after_join": report.rows_after_join,
            "matched_windows": report.matched_windows,
            "unmatched_packet_windows": report.unmatched_packet_windows,
            "unmatched_flow_windows": report.unmatched_flow_windows,
            "packet_coverage": report.packet_coverage_pct,
            "flow_coverage": report.flow_coverage_pct,
            "packet_coverage_pct": report.packet_coverage_pct,
            "flow_coverage_pct": report.flow_coverage_pct,
            "has_row_multiplication": report.has_row_multiplication,
            "has_row_loss": report.has_row_loss,
        }
        return fused_df, metrics

    return fused_df


def fused_df_to_traffic_windows(
    fused_df: pd.DataFrame,
    label_col: str | None = None,
    feature_names: Sequence[str] | None = None,
    allow_prototype_partial_features: bool = False,
) -> list[TrafficWindow]:
    """
    Convert a fused feature DataFrame into canonical TrafficWindow objects.

    Enforces Aryan's canonical TrafficWindow schema:
        - timestamp: datetime (UTC)
        - source_host: str (non-empty)
        - packet_count: int >= 0
        - byte_count: int >= 0
        - window_id: str
        - features: tuple[float, ...] (finite floats, no target/identifier leakage)
        - label: float | int | None
    """
    if fused_df.empty:
        return []

    # Normalize known feature aliases if present
    df = fused_df.rename(
        columns={
            k: v
            for k, v in FEATURE_ALIASES.items()
            if k in fused_df.columns and v not in fused_df.columns
        }
    )

    # Resolve authoritative active features
    if feature_names is not None:
        active_features = validate_feature_names(
            feature_names,
            expected_order=CANONICAL_MODEL_FEATURE_NAMES
            if not allow_prototype_partial_features
            else None,
        )
    else:
        all_cols = set(df.columns)
        missing_canonical = [c for c in CANONICAL_MODEL_FEATURE_NAMES if c not in all_cols]

        if not missing_canonical:
            active_features = list(CANONICAL_MODEL_FEATURE_NAMES)
        elif allow_prototype_partial_features:
            has_packet = any(c in all_cols for c in CANONICAL_PACKET_FEATURE_NAMES)
            has_flow = any(c in all_cols for c in CANONICAL_FLOW_FEATURE_NAMES)

            if has_flow and not has_packet:
                active_features = list(CANONICAL_FLOW_FEATURE_NAMES)
            elif has_packet and not has_flow:
                active_features = list(CANONICAL_PACKET_FEATURE_NAMES)
            else:
                metadata_cols = {
                    "src_ip",
                    "dst_ip",
                    "src_port",
                    "dst_port",
                    "window_id",
                    "window_start",
                    "window_end",
                    "timestamp",
                    "flow_id",
                    "label",
                    "target",
                    "attack_type",
                    "y_true",
                }
                if label_col:
                    metadata_cols.add(label_col)

                numeric_cols = [
                    c for c in df.columns
                    if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])
                ]
                active_features = validate_feature_names(numeric_cols)
        else:
            missing_flow = [c for c in CANONICAL_FLOW_FEATURE_NAMES if c not in all_cols]
            missing_packet = [c for c in CANONICAL_PACKET_FEATURE_NAMES if c not in all_cols]
            raise ValueError(
                "Canonical forecasting requires all 41 flow+packet features. "
                f"Missing flow features: {missing_flow}; missing packet features: {missing_packet}"
            )

    if not allow_prototype_partial_features:
        validate_feature_names(active_features, expected_order=CANONICAL_MODEL_FEATURE_NAMES)
        if len(active_features) != 41:
            raise ValueError(f"Canonical forecasting requires 41 features, got {len(active_features)}")

    windows: list[TrafficWindow] = []
    col_set = set(df.columns)

    for _, row in df.iterrows():
        src_host = str(row["src_ip"]).strip()

        # Timestamp normalization
        if "timestamp" in col_set and pd.notna(row["timestamp"]):
            raw_ts = row["timestamp"]
            if isinstance(raw_ts, (int, float)):
                ts = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
            elif isinstance(raw_ts, pd.Timestamp):
                ts = raw_ts.to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            elif isinstance(raw_ts, datetime):
                ts = raw_ts if raw_ts.tzinfo is not None else raw_ts.replace(tzinfo=timezone.utc)
            else:
                ts = datetime.fromtimestamp(float(row["window_id"]) * 10.0, tz=timezone.utc)
        elif "window_start" in col_set and pd.notna(row["window_start"]):
            ts = datetime.fromtimestamp(float(row["window_start"]), tz=timezone.utc)
        else:
            ts = datetime.fromtimestamp(float(row["window_id"]) * 10.0, tz=timezone.utc)

        # Discrete packet and byte quantities
        p_count = int(row.get("packet_count", row.get("flow_packets_total", row.get("packets_total", 0))))
        b_count = int(row.get("flow_bytes_total", row.get("bytes_total", row.get("payload_max", 0))))

        # Feature vector (deterministic canonical feature ordering)
        feat_vals = tuple(float(row[c]) if c in col_set and pd.notna(row[c]) else 0.0 for c in active_features)

        # Label extraction if available
        lbl = None
        if label_col and label_col in row and pd.notna(row[label_col]):
            lbl = float(row[label_col])

        w_id_str = f"{src_host}_{int(row['window_id'])}"

        tw = TrafficWindow(
            timestamp=ts,
            source_host=src_host,
            packet_count=p_count,
            byte_count=b_count,
            window_id=w_id_str,
            features=feat_vals,
            label=lbl,
        )
        windows.append(tw)

    return windows


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python traffic_fusion.py <flow_parquet> <packet_parquet> <output_parquet>")
        sys.exit(1)

    f_in = sys.argv[1]
    p_in = sys.argv[2]
    out_file = sys.argv[3]

    fused = fuse_flow_and_packet_data(f_in, p_in, output_file=out_file)
    print(f"Traffic fusion completed. Fused rows: {len(fused)}")
