import pandas as pd


def fuse_flow_and_packet_data(flow_file, packet_file, output_file):
    """
    Merge flow-level and packet-level features.

    Primary identity:
        src_ip + window_id

    Output:
        Fused traffic feature table.
    """

    flow_df = pd.read_parquet(flow_file)
    packet_df = pd.read_parquet(packet_file)

    # Validate required columns
    required_packet = ["src_ip", "window_id"]
    required_flow = ["src_ip", "window_id"]

    for column in required_packet:
        if column not in packet_df.columns:
            raise ValueError(f"Packet data missing column: {column}")

    for column in required_flow:
        if column not in flow_df.columns:
            raise ValueError(f"Flow data missing column: {column}")

    # Remove duplicate flow states before joining
    flow_df = flow_df.drop_duplicates(
        subset=["src_ip", "window_id"]
    )

    # Rename overlapping columns
    overlapping = set(packet_df.columns) & set(flow_df.columns)
    overlapping -= {"src_ip", "window_id"}

    flow_df = flow_df.rename(
        columns={column: f"flow_{column}" for column in overlapping}
    )

    # Merge packet + flow features
    fused_df = pd.merge(
        packet_df,
        flow_df,
        on=["src_ip", "window_id"],
        how="left"
    )

    # Sort chronologically
    if "timestamp" in fused_df.columns:
        fused_df = fused_df.sort_values(
            ["src_ip", "timestamp"]
        )

    # Reset index
    fused_df = fused_df.reset_index(drop=True)

    # Save result
    fused_df.to_parquet(output_file, index=False)

    print("Traffic fusion completed.")
    print(f"Packet rows: {len(packet_df)}")
    print(f"Flow rows: {len(flow_df)}")
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