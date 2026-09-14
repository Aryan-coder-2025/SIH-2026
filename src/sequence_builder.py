"""
[PROTOTYPE / ADAPTER MODULE - NON-AUTHORITATIVE]
Authoritative sequence construction engine is strictly src.temporal.sequences.

This module delegates directly to Aryan's canonical sequence engine
to prevent duplicate or competing implementations.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from src.temporal.sequences import build_sequences_from_dataframe as canonical_build_sequences_from_df


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
    Adapter delegating to canonical src.temporal.sequences.build_sequences_from_dataframe.
    """
    return canonical_build_sequences_from_df(
        df=df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        horizon=horizon,
        entity_column=entity_column,
        timestamp_column=timestamp_column,
        risk_column=risk_column,
        stage_column=stage_column,
    )


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