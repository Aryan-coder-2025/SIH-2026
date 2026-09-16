from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .schema import CANONICAL_FEATURES, validate_feature_order, validate_window_dataframe
except ImportError:
    from schema import CANONICAL_FEATURES, validate_feature_order, validate_window_dataframe


def _validate_group_continuity(
    group: pd.DataFrame, timestamp_column: str, window_seconds: int
) -> np.ndarray:
    """Return True where the row starts a continuous 10-second grid."""
    timestamps = pd.to_datetime(group[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Invalid timestamp found in sequence data.")
    if timestamps.duplicated().any():
        raise ValueError("Duplicate timestamp/window detected.")

    differences = timestamps.diff().dt.total_seconds().to_numpy()
    valid_step = np.zeros(len(group), dtype=bool)
    if len(group):
        valid_step[0] = True
    if len(group) > 1:
        valid_step[1:] = differences[1:] == float(window_seconds)
    return valid_step


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    horizon: int,
    entity_column: str = "source_host",
    timestamp_column: str = "timestamp",
    risk_column: str = "is_malicious",
    window_seconds: int = 10,
    forecast_offsets_seconds: list[int] | None = None,
):
    """
    Build leakage-safe temporal samples.

    Target definition:
      y[i] = malicious ground truth at the exact configured future offsets
      from prediction_time (the last window in the history).

    If source_host exists, continuity and duplicate checks are performed
    independently for every host. Otherwise the data is treated as one
    network-level time series.
    """
    if sequence_length <= 0 or horizon <= 0 or window_seconds <= 0:
        raise ValueError("sequence_length, horizon and window_seconds must be positive.")

    if forecast_offsets_seconds is None:
        forecast_offsets_seconds = [window_seconds * (i + 1) for i in range(horizon)]

    if len(forecast_offsets_seconds) != horizon:
        raise ValueError("Number of forecast offsets must equal horizon.")
    if forecast_offsets_seconds != sorted(set(forecast_offsets_seconds)):
        raise ValueError("Forecast offsets must be unique and strictly increasing.")
    if any(x <= 0 or x % window_seconds != 0 for x in forecast_offsets_seconds):
        raise ValueError("Forecast offsets must be positive multiples of window_seconds.")

    validate_feature_order(feature_columns)
    validate_window_dataframe(df)

    required = [timestamp_column, risk_column, *feature_columns]
    if entity_column in df.columns:
        required.append(entity_column)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Cannot build sequences from an empty dataframe.")

    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
    df = df.sort_values(
        [entity_column, timestamp_column] if entity_column in df.columns else [timestamp_column]
    ).reset_index(drop=True)

    if entity_column in df.columns:
        groups = df.groupby(entity_column, sort=False, dropna=False)
    else:
        groups = [("network", df)]

    X_sequences, y_sequences = [], []

    for entity, group in groups:
        group = group.sort_values(timestamp_column).reset_index(drop=True)
        minimum_rows = sequence_length + max(x // window_seconds for x in forecast_offsets_seconds)
        if len(group) < minimum_rows:
            continue

        continuity = _validate_group_continuity(group, timestamp_column, window_seconds)
        features = group[feature_columns].to_numpy(dtype=np.float32)
        risk = group[risk_column].to_numpy(dtype=np.float32)
        timestamps = group[timestamp_column].to_numpy()

        max_future_steps = max(x // window_seconds for x in forecast_offsets_seconds)

        for start in range(0, len(group) - sequence_length - max_future_steps + 1):
            history_end = start + sequence_length
            prediction_time = pd.Timestamp(timestamps[history_end - 1])

            # Entire history must be continuous.
            interval_start = start
            interval_end = history_end + max_future_steps
            if not continuity[interval_start:interval_end].all():
                continue

            target_indices = [
                history_end + (offset // window_seconds) - 1
                for offset in forecast_offsets_seconds
            ]
            if max(target_indices) >= len(group):
                continue

            expected_times = [
                prediction_time + pd.Timedelta(seconds=offset)
                for offset in forecast_offsets_seconds
            ]
            actual_times = [pd.Timestamp(timestamps[i]) for i in target_indices]
            if actual_times != expected_times:
                continue

            X = features[start:history_end]
            y = risk[target_indices]

            if X.shape != (sequence_length, len(feature_columns)):
                raise ValueError(f"Invalid sequence shape: {X.shape}")
            if y.shape != (horizon,):
                raise ValueError(f"Invalid target shape: {y.shape}")

            X_sequences.append(X)
            y_sequences.append(y)

    if not X_sequences:
        raise ValueError(
            "No valid temporal sequences were created. Check the time grid, "
            "history length, forecast horizon and missing/duplicate windows."
        )

    X = np.asarray(X_sequences, dtype=np.float32)
    y = np.asarray(y_sequences, dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (sequence_length, len(feature_columns)):
        raise ValueError(f"Unexpected X shape: {X.shape}")
    if y.ndim != 2 or y.shape[1] != horizon:
        raise ValueError(f"Unexpected y shape: {y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Sequences contain NaN or Inf.")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("Sequence targets must contain only 0 or 1.")

    return X, y
