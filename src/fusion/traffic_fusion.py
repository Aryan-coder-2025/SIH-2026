"""
Traffic Fusion Module.
Joins flow-level and packet-level features per (src_ip, window_id) into a unified
canonical network-state matrix. Enforces missing-data policies and reports coverage metrics.
"""
from _pytest import assertion
from __future__ import annotations

import pandas as pd
from src.flow.flow_extractor import aggregate_flow_features_to_host
from src.schemas.model20 import (
    CANONICAL_MODEL_20_FEATURES,
    extract_model_20_features,
)


def fuse_flow_and_packet_data(
    flow_input: str | pd.DataFrame,
    packet_input: str | pd.DataFrame,
    output_file: str | None = None,
    missing_data_policy: str = "reject",
    extract_model20: bool = False,
    model20_output_file: str | None = None,
) -> pd.DataFrame:
    """
    Merge packet-level and flow-level features on canonical primary key (src_ip, window_id).

    Args:
        flow_input: Path to flow parquet file or flow DataFrame.
        packet_input: Path to packet parquet file or packet DataFrame.
        output_file: Path to save fused parquet file (optional).
        missing_data_policy: 'reject' (default), 'fill_zero', or 'inner'.
        extract_model20: If True, also extract and validate canonical model-20 features.
        model20_output_file: Path to save model-20 parquet file (optional).

    Returns:
        pd.DataFrame: The canonical fused DataFrame.
    """
    # 1. Load packet and flow data
    flow_df = (
        pd.read_parquet(flow_input)
        if isinstance(flow_input, str)
        else flow_input.copy()
    )
    packet_df = (
        pd.read_parquet(packet_input)
        if isinstance(packet_input, str)
        else packet_input.copy()
    )

    # 2. Validate required key columns
    required_keys = ["src_ip", "window_id"]
    for col in required_keys:
        if col not in packet_df.columns:
            raise ValueError(f"Packe
            t data missing required key column: {col}")
        if col not in flow_df.columns:
            raise ValueError(f"Flow data missing required key column: {col}")

    packet_rows_before = len(packet_df)
    flow_rows_before = len(flow_df)

    # 3. Aggregate flow features to host level (src_ip, window_id) if needed
    is_already_aggregated = (
        "flow_count" in flow_df.columns
        and not flow_df.duplicated(subset=["src_ip", "window_id"]).any()
    )

    if not is_already_aggregated and not flow_df.empty:
        flow_df = aggregate_flow_features_to_host(flow_df)

    # 4. Extract join keys
    packet_keys = set(zip(packet_df["src_ip"], packet_df["window_id"]))
    flow_keys = set(zip(flow_df["src_ip"], flow_df["window_id"]))

    matched_keys = packet_keys & flow_keys
    unmatched_packet_windows = packet_keys - flow_keys
    unmatched_flow_windows = flow_keys - packet_keys

    packet_coverage = (
        len(matched_keys) / len(packet_keys) * 100
        if packet_keys
        else 0.0
    )
    flow_coverage = (
        len(matched_keys) / len(flow_keys) * 100
        if flow_keys
        else 0.0
    )

    # 5. Handle missing-data policy
    policy = missing_data_policy.lower().strip()
    if policy == "reject":
        if unmatched_packet_windows or unmatched_flow_windows:
            raise ValueError(
                f"Fusion rejected: packet and flow windows do not match. "
                f"Unmatched packet windows: {len(unmatched_packet_windows)}, "
                f"Unmatched flow windows: {len(unmatched_flow_windows)}."
            )
        how = "inner"
    elif policy == "fill_zero":
        how = "outer"
    elif policy in ("inner", "drop"):
        how = "inner"
    else:
        raise ValueError(
            f"Unknown missing_data_policy '{missing_data_policy}'. "
            f"Must be 'reject', 'fill_zero', or 'inner'."
        )

    # Ensure no duplicates in input tables on join key
    if packet_df.duplicated(subset=["src_ip", "window_id"]).any():
        raise ValueError("Packet data contains duplicate source-host/window rows.")
    if flow_df.duplicated(subset=["src_ip", "window_id"]).any():
        raise ValueError("Flow data contains duplicate source-host/window rows.")

    # 6. Prepare columns for join and rename conflicting non-key columns
    shared_metadata = {"src_ip", "window_id", "timestamp", "window_start", "window_end"}
    overlapping = (set(packet_df.columns) & set(flow_df.columns)) - shared_metadata

    # Any overlapping feature names in flow get a flow_ prefix unless it's already distinct
    flow_renamed = flow_df.rename(
        columns={col: f"flow_{col}" for col in overlapping}
    )

    # 7. Merge datasets
    fused_df = pd.merge(
        packet_df,
        flow_renamed,
        on=["src_ip", "window_id"],
        how=how,
        suffixes=("", "_flow"),
    )

    # 8. Reconcile metadata (timestamp, window_start, window_end) if filled via outer join
    for meta in ["timestamp", "window_start", "window_end"]:
        meta_flow = f"{meta}_flow"
        if meta in fused_df.columns and meta_flow in fused_df.columns:
            fused_df[meta] = fused_df[meta].combine_first(fused_df[meta_flow])
            fused_df = fused_df.drop(columns=[meta_flow])

    # 9. Handle fill_zero policy: replace NaNs with 0.0 for numeric features
    if policy == "fill_zero":
        numeric_cols = fused_df.select_dtypes(include="number").columns
        fused_df[numeric_cols] = fused_df[numeric_cols].fillna(0.0)

    # 10. Sort chronologically
    sort_cols = [c for c in ["src_ip", "window_id", "timestamp"] if c in fused_df.columns]
    if sort_cols:
        fused_df = fused_df.sort_values(sort_cols)

    fused_df = fused_df.reset_index(drop=True)

    # 11. Reporting
    print("Traffic fusion completed.")
    print(f"Packet rows: {len(packet_df)}")
    print(f"Aggregated flow rows: {len(flow_df)}")
    print(f"Fused rows: {len(fused_df)}")
    print(f"Packet rows before fusion: {packet_rows_before}")
    print(f"Flow rows before fusion: {flow_rows_before}")
    print(f"Packet coverage: {packet_coverage:.2f}%")
    print(f"Flow coverage: {flow_coverage:.2f}%")
    print(f"Unmatched packet windows: {len(unmatched_packet_windows)}")
    print(f"Unmatched flow windows: {len(unmatched_flow_windows)}")

    # 12. Save outputs
    if output_file:
        fused_df.to_parquet(output_file, index=False)
        print(f"Saved to: {output_file}")

    if extract_model20:
        model20_df = extract_model_20_features(fused_df)
        if model20_output_file:
            model20_df.to_parquet(model20_output_file, index=False)
            print(f"Saved canonical model-20 features to: {model20_output_file}")

    return fused_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python -m src.fusion.traffic_fusion "
            "<flow_parquet> <packet_parquet> <output_parquet> [missing_data_policy]"
        )
        sys.exit(1)

    flow_file = sys.argv[1]
    packet_file = sys.argv[2]
    out_file = sys.argv[3]
    policy = sys.argv[4] if len(sys.argv) > 4 else "reject"

    fuse_flow_and_packet_data(
        flow_input=flow_file,
        packet_input=packet_file,
        output_file=out_file,
        missing_data_policy=policy,
    )