import pandas as pd


def fuse_flow_and_packet_data(flow_file, packet_file, output_file):
    """
    Merge packet-level and flow-level features.

    Primary identity:
        src_ip + window_id

    Multiple flows belonging to the same source host and window
    are aggregated before the final join.
    """

    # Load packet and flow feature tables.
    flow_df = pd.read_parquet(flow_file)
    packet_df = pd.read_parquet(packet_file)

    # Validate required columns.
    required_packet = ["src_ip", "window_id"]
    required_flow = ["src_ip", "window_id"]

    for column in required_packet:
        if column not in packet_df.columns:
            raise ValueError(f"Packet data missing column: {column}")

    for column in required_flow:
        if column not in flow_df.columns:
            raise ValueError(f"Flow data missing column: {column}")

    # Aggregate all flows belonging to the same source host and window.
    if not flow_df.empty:
        numeric_columns = flow_df.select_dtypes(
            include="number"
        ).columns.tolist()

        numeric_columns = [
            column
            for column in numeric_columns
            if column not in {"window_id"}
        ]

        aggregation = {}

        for column in numeric_columns:
            if column == "flow_packet_count":
                aggregation[column] = "sum"
            elif column == "flow_bytes":
                aggregation[column] = "sum"
            elif column == "flow_duration":
                aggregation[column] = "sum"
            else:
                aggregation[column] = "mean"

        flow_df = flow_df.groupby(
            ["src_ip", "window_id"],
            as_index=False
        ).agg(aggregation)

    # Make sure packet data has one row per source host and window.
    if packet_df.duplicated(
        subset=["src_ip", "window_id"]
    ).any():
        raise ValueError(
            "Packet data contains duplicate source-host/window rows."
        )

    # Rename overlapping flow columns.
    overlapping = set(packet_df.columns) & set(flow_df.columns)
    overlapping -= {"src_ip", "window_id"}

    flow_df = flow_df.rename(
        columns={
            column: f"flow_{column}"
            for column in overlapping
        }
    )

    # Perform a one-to-one join after aggregation.
    fused_df = pd.merge(
        packet_df,
        flow_df,
        on=["src_ip", "window_id"],
        how="left",
        validate="one_to_one"
    )

    # Sort chronologically when timestamp is available.
    if "timestamp" in fused_df.columns:
        fused_df = fused_df.sort_values(
            ["src_ip", "timestamp"]
        )

    # Reset the DataFrame index.
    fused_df = fused_df.reset_index(drop=True)

    # Save the fused feature table.
    fused_df.to_parquet(
        output_file,
        index=False
    )

    print("Traffic fusion completed.")
    print(f"Packet rows: {len(packet_df)}")
    print(f"Aggregated flow rows: {len(flow_df)}")
    print(f"Fused rows: {len(fused_df)}")
    print(f"Saved to: {output_file}")

    return fused_df

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print(
            "Usage: python traffic_fusion.py "
            "<flow_parquet> <packet_parquet> <output_parquet>"
        )
        sys.exit(1)

    flow_file = sys.argv[1]
    packet_file = sys.argv[2]
    output_file = sys.argv[3]

    fuse_flow_and_packet_data(
        flow_file,
        packet_file,
        output_file
    )