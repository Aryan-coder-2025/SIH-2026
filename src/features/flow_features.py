import pandas as pd


def aggregate_flow_features(df):
    df = df.sort_values(["host_id", "StartTime"]).copy()

    df["iat"] = (
    df.groupby("host_id")["StartTime"]
      .diff()
      .dt.total_seconds()
)

    df["iat"] = df["iat"].fillna(0)

    grouped = df.groupby(
        ["host_id", "window_start"],
        dropna=False
    )

    features = grouped.agg(
        iat_mean=("iat", "mean"),
        iat_std=("iat", "std"),
        iat_max=("iat", "max"),
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
        
        tcp_flow_ratio=("Proto", lambda x: (x == "tcp").mean()),
        udp_flow_ratio=("Proto", lambda x: (x == "udp").mean()),
        bidirectional_ratio=("Dir", lambda x: x.str.contains("<->").mean()),

    ).reset_index()
        

    return features


if __name__ == "__main__":

    path = "data/raw/CTU-13/capture20110810.binetflow"

    df = pd.read_csv(path, nrows=10000)

    df["StartTime"] = pd.to_datetime(
        df["StartTime"],
        format="%Y/%m/%d %H:%M:%S.%f"
    )

    df["window_start"] = df["StartTime"].dt.floor("10s")
    df["host_id"] = df["SrcAddr"]
    
    print("\nAddress types:")
    print(df["SrcAddr"].map(
    lambda x: "ipv4" if isinstance(x, str) and x.count(".") == 3
    else "other"
).value_counts())

    features = aggregate_flow_features(df)
    features = features.fillna(0)
    print("Feature shape:", features.shape)

    print("\nFeatures:")
    print(features.head(10))