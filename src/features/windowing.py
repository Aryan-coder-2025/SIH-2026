import pandas as pd
from labels import classify_label


WINDOW_SIZE = "10s"


def add_time_columns(df):
    df = df.copy()

    df["StartTime"] = pd.to_datetime(
        df["StartTime"],
        format="%Y/%m/%d %H:%M:%S.%f"
    )

    df["window_start"] = df["StartTime"].dt.floor(WINDOW_SIZE)
    df["host_id"] = df["SrcAddr"]

    return df


def add_window_labels(df):
    df = df.copy()

    df["traffic_category"] = df["Label"].apply(
        lambda x: classify_label(x)
    )

    labels = (
        df.groupby(["host_id", "window_start"])["traffic_category"]
        .apply(
            lambda x: "botnet" if "botnet" in x.values
            else "normal"
        )
        .reset_index(name="traffic_category")
    )

    return labels


if __name__ == "__main__":
    path = "data/raw/CTU-13/capture20110810.binetflow"

    df = pd.read_csv(path, nrows=10000)

    df = add_time_columns(df)
    labels = add_window_labels(df)

    print("\nWindow labels:")
    print(labels.head(20))

    print("\nLabel distribution:")
    print(labels["traffic_category"].value_counts())