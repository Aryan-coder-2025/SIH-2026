from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Sequence, TypeVar

from src.schemas.traffic import TrafficWindow, normalize_to_utc

T = TypeVar("T")


@dataclass(frozen=True)
class TemporalSplit:
    """
    Result container for temporal partition with explicit boundary timestamps.
    """

    train: list[Any]
    test: list[Any]
    val: list[Any] | None = None
    train_end_time: datetime | None = None
    test_start_time: datetime | None = None
    val_start_time: datetime | None = None
    val_end_time: datetime | None = None


def _extract_timestamp(
    item: Any,
    timestamp_extractor: Callable[[Any], datetime] | None = None,
) -> datetime | None:
    """
    Extract and normalize a datetime to UTC from an item, TrafficWindow, dict, or custom extractor.

    TIMEZONE CONSISTENCY INVARIANT:
    Generic inputs containing mixed offset-naive and offset-aware datetimes will fail in Python
    with TypeError during comparison or sorting. We strictly normalize all extracted datetimes to UTC.
    Naive datetimes are assumed to represent UTC.
    """
    raw_ts: datetime | None = None
    if timestamp_extractor is not None:
        raw_ts = timestamp_extractor(item)
    elif isinstance(item, TrafficWindow):
        raw_ts = item.timestamp
    elif isinstance(item, dict) and "timestamp" in item:
        ts = item["timestamp"]
        if isinstance(ts, datetime):
            raw_ts = ts
    elif hasattr(item, "timestamp") and isinstance(item.timestamp, datetime):
        raw_ts = item.timestamp

    if raw_ts is not None:
        return normalize_to_utc(raw_ts)
    return None


def temporal_train_test_split(
    data: Sequence[T],
    train_ratio: float = 0.8,
) -> tuple[list[T], list[T]]:
    """
    Split ordered data chronologically without shuffling.

    Enforces:
    - 0 < train_ratio < 1
    - Non-empty input
    - Non-empty partitions
    - Chronological ordering if elements carry timestamps
    - Zero overlap between train and test
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    if not data:
        raise ValueError("data cannot be empty")

    # We validate pairwise against the immediately preceding window (data[i-1])
    # rather than only the first timestamp. A later observation can still be
    # earlier than its immediate predecessor while remaining later than data[0]
    # (e.g., 10:00 -> 10:20 -> 10:10). Without this pairwise check, non-monotonic
    # timelines can enter sequence construction and corrupt temporal causality.
    for i in range(1, len(data)):
        prev_ts = _extract_timestamp(data[i - 1])
        curr_ts = _extract_timestamp(data[i])
        if prev_ts is not None and curr_ts is not None and curr_ts < prev_ts:
            raise ValueError(
                f"Input data is not chronologically sorted: item {i} ({curr_ts}) "
                f"is earlier than preceding item {i - 1} ({prev_ts})."
            )

    split_index = int(len(data) * train_ratio)

    if split_index == 0 or split_index == len(data):
        raise ValueError("train_ratio produces an empty partition")

    train_data = list(data[:split_index])
    test_data = list(data[split_index:])

    return train_data, test_data


def temporal_split_by_time(
    data: Sequence[T],
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
    *,
    timestamp_extractor: Callable[[T], datetime] | None = None,
) -> TemporalSplit:
    """
    Perform multi-host temporal split based on physical timestamps.

    All hosts are partitioned at the exact same physical cutoff timestamp,
    ensuring that evaluation strictly models forecasting into the future.

    Parameters:
    - data: Sequence of records (TrafficWindow, objects with timestamps, or dicts).
    - train_ratio: Fraction of temporal duration allocated to training (e.g. 0.8).
    - val_ratio: Optional fraction for validation. If 0.0, splits into train/test only.
    - timestamp_extractor: Optional callable to extract datetime from an item.

    Boundary & Anti-Leakage Invariants:
    1. Temporal Split Before Sequence Construction: Data is partitioned at global
       timestamps before sequences are built.
    2. Zero Future Contamination: Training partition contains strictly records prior
       to train_end_time.
    3. Multi-Host Synchrony: All hosts share the identical temporal boundary.
    """
    if not data:
        raise ValueError("data cannot be empty")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be >= 0 and < 1")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be strictly less than 1.0")

    # Extract and validate all timestamps
    items_with_ts: list[tuple[datetime, T]] = []
    for idx, item in enumerate(data):
        ts = _extract_timestamp(item, timestamp_extractor)
        if ts is None:
            raise ValueError(
                f"Item at index {idx} does not have a valid datetime timestamp."
            )
        items_with_ts.append((ts, item))

    # Sort deterministically by timestamp
    items_with_ts.sort(key=lambda x: x[0])

    min_time = items_with_ts[0][0]
    max_time = items_with_ts[-1][0]

    if min_time == max_time:
        raise ValueError("All records have the exact same timestamp; cannot split by time.")

    total_duration = max_time - min_time
    train_cutoff = min_time + total_duration * train_ratio

    train_items: list[T] = []
    val_items: list[T] = []
    test_items: list[T] = []

    if val_ratio > 0.0:
        val_cutoff = min_time + total_duration * (train_ratio + val_ratio)
        for ts, item in items_with_ts:
            if ts < train_cutoff:
                train_items.append(item)
            elif ts < val_cutoff:
                val_items.append(item)
            else:
                test_items.append(item)

        if not train_items or not val_items or not test_items:
            raise ValueError("Temporal split resulted in an empty train, val, or test partition")

        return TemporalSplit(
            train=train_items,
            val=val_items,
            test=test_items,
            train_end_time=train_cutoff,
            val_start_time=train_cutoff,
            val_end_time=val_cutoff,
            test_start_time=val_cutoff,
        )
    else:
        for ts, item in items_with_ts:
            if ts < train_cutoff:
                train_items.append(item)
            else:
                test_items.append(item)

        if not train_items or not test_items:
            raise ValueError("Temporal split resulted in an empty train or test partition")

        return TemporalSplit(
            train=train_items,
            test=test_items,
            train_end_time=train_cutoff,
            test_start_time=train_cutoff,
        )