from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int = 10,
    horizon: int = 3,
    entity_column: str = "source_host",
    timestamp_column: str = "timestamp",
    risk_column: str = "is_malicious",
    stage_column: str = "stage",
):
    """
    Convert window-level network states into temporal sequences.

    Input:
        One row = one 10-second network state for one source host.

    Output:
        X:
            shape = (samples, sequence_length, num_features)

        y_risk:
            shape = (samples, horizon)

        y_stage:
            shape = (samples,)
    """

    required_columns = (
        [entity_column, timestamp_column, risk_column]
        + feature_columns
    )

    if stage_column in df.columns:
        required_columns.append(stage_column)

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    # Convert timestamp
    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column]
    )

    # Sort chronologically within each source host
    df = df.sort_values(
        [entity_column, timestamp_column]
    ).reset_index(drop=True)

    X_sequences = []
    y_risk_sequences = []
    y_stage = []

    # Process each source host independently
    for entity, group in df.groupby(entity_column, sort=False):

        group = group.sort_values(
            timestamp_column
        ).reset_index(drop=True)

        feature_values = group[feature_columns].to_numpy(
            dtype=np.float32
        )

        risk_values = group[risk_column].to_numpy(
            dtype=np.float32
        )

        if stage_column in group.columns:
            stage_values = group[stage_column].astype(str).to_numpy()
        else:
            stage_values = np.array(
                ["UNKNOWN"] * len(group)
            )

        # Need:
        # sequence_length historical windows
        # + horizon future windows
        max_start = (
            len(group)
            - sequence_length
            - horizon
            + 1
        )

        if max_start <= 0:
            continue

        for i in range(max_start):

            # Past 10 windows
            X = feature_values[
                i:i + sequence_length
            ]

            # Future +10, +20, +30 seconds
            future_start = i + sequence_length

            future_end = (
                future_start + horizon
            )

            future_risk = risk_values[
                future_start:future_end
            ]

            # Stage at first future point
            future_stage = stage_values[
                future_start
            ]

            X_sequences.append(X)
            y_risk_sequences.append(future_risk)
            y_stage.append(future_stage)

    if not X_sequences:
        raise ValueError(
            "No sequences could be created. "
            "Check sequence length and data size."
        )

    X = np.asarray(
        X_sequences,
        dtype=np.float32
    )

    y_risk = np.asarray(
        y_risk_sequences,
        dtype=np.float32
    )

    y_stage = np.asarray(
        y_stage
    )

    return X, y_risk, y_stage


if __name__ == "__main__":

    # Small test
    data = {
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=20,
            freq="10s"
        ),
        "source_host": ["host_A"] * 20,
        "bytes": np.random.randint(
            1000, 10000, 20
        ),
        "packets": np.random.randint(
            1, 100, 20
        ),
        "syn_count": np.random.randint(
            0, 20, 20
        ),
        "is_malicious": [
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 1, 1, 1,
            1, 1, 1, 0, 0
        ],
        "stage": [
            "BENIGN"
        ] * 12
        + ["SCANNING"] * 6
        + ["BENIGN"] * 2
    }

    df = pd.DataFrame(data)

    features = [
        "bytes",
        "packets",
        "syn_count"
    ]

    X, y_risk, y_stage = build_sequences(
        df,
        feature_columns=features
    )

    print("X shape:", X.shape)
    print("Risk shape:", y_risk.shape)
    print("Stage shape:", y_stage.shape)