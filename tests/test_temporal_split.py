from datetime import datetime, timedelta, timezone

import pytest

from src.schemas.traffic import TrafficWindow
from src.temporal.split import temporal_split_by_time, temporal_train_test_split


def test_temporal_split_preserves_order():
    data = list(range(10))

    train, test = temporal_train_test_split(
        data,
        train_ratio=0.8,
    )

    assert train == list(range(8))
    assert test == [8, 9]


def test_temporal_split_does_not_overlap():
    data = list(range(10))

    train, test = temporal_train_test_split(
        data,
        train_ratio=0.8,
    )

    assert set(train).isdisjoint(test)


def test_temporal_split_rejects_invalid_ratio():
    data = list(range(10))

    with pytest.raises(ValueError):
        temporal_train_test_split(data, train_ratio=0)

    with pytest.raises(ValueError):
        temporal_train_test_split(data, train_ratio=1)

    with pytest.raises(ValueError):
        temporal_train_test_split(data, train_ratio=1.5)


def test_temporal_split_rejects_empty_data():
    with pytest.raises(ValueError):
        temporal_train_test_split([])


def test_temporal_split_handles_different_ratio():
    data = list(range(10))

    train, test = temporal_train_test_split(
        data,
        train_ratio=0.6,
    )

    assert train == list(range(6))
    assert test == list(range(6, 10))


def test_temporal_split_rejects_unsorted_timestamped_data():
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    w1 = TrafficWindow(timestamp=base + timedelta(seconds=10), source_host="h1", packet_count=1, byte_count=1)
    w2 = TrafficWindow(timestamp=base, source_host="h1", packet_count=1, byte_count=1)

    with pytest.raises(ValueError, match="Input data is not chronologically sorted"):
        temporal_train_test_split([w1, w2], train_ratio=0.5)


def test_temporal_split_by_time_train_test():
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    # 10 windows of 10s = 0s to 90s (span = 90s)
    windows = [
        TrafficWindow(
            timestamp=base + timedelta(seconds=10 * i),
            source_host=f"host_{i % 2}",
            packet_count=10,
            byte_count=1000,
        )
        for i in range(10)
    ]

    split = temporal_split_by_time(windows, train_ratio=0.7)

    assert len(split.train) > 0
    assert len(split.test) > 0
    assert split.train_end_time is not None
    assert split.test_start_time == split.train_end_time

    # Verify that all train windows have timestamp < train_end_time
    for w in split.train:
        assert w.timestamp < split.train_end_time

    # Verify that all test windows have timestamp >= train_end_time
    for w in split.test:
        assert w.timestamp >= split.train_end_time


def test_temporal_split_by_time_train_val_test():
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    windows = [
        TrafficWindow(
            timestamp=base + timedelta(seconds=10 * i),
            source_host="host_A",
            packet_count=10,
            byte_count=1000,
        )
        for i in range(10)
    ]

    split = temporal_split_by_time(windows, train_ratio=0.6, val_ratio=0.2)

    assert split.val is not None
    assert len(split.train) > 0
    assert len(split.val) > 0
    assert len(split.test) > 0

    assert split.train_end_time == split.val_start_time
    assert split.val_end_time == split.test_start_time

    for w in split.train:
        assert w.timestamp < split.train_end_time
    for w in split.val:
        assert split.val_start_time <= w.timestamp < split.val_end_time
    for w in split.test:
        assert w.timestamp >= split.val_end_time


def test_pairwise_non_monotonic_timestamps_rejected():
    """
    CRITICAL REGRESSION TEST (Section 5.B & 6):
    Timeline: 10:00, 10:20, 10:10.
    10:10 is greater than data[0] (10:00), but strictly earlier than its immediate
    predecessor data[1] (10:20). A naive check against data[0] alone would allow this!
    The pairwise invariant check must catch and reject this non-monotonic inversion.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    w0 = TrafficWindow(timestamp=base, source_host="h1", packet_count=1, byte_count=1)
    w1 = TrafficWindow(timestamp=base + timedelta(minutes=20), source_host="h1", packet_count=1, byte_count=1)
    w2 = TrafficWindow(timestamp=base + timedelta(minutes=10), source_host="h1", packet_count=1, byte_count=1)

    with pytest.raises(ValueError, match="is earlier than preceding item 1"):
        temporal_train_test_split([w0, w1, w2], train_ratio=0.5)


def test_temporal_split_by_time_handles_mixed_naive_and_aware_generic_dicts():
    """
    CRITICAL AUDIT INVARIANT (Issue 3):
    Generic inputs (such as dicts or custom objects) containing a mix of naive
    and UTC-aware datetimes must be normalized internally by temporal_split_by_time()
    without raising Python TypeError: can't compare offset-naive and offset-aware datetimes.
    """
    data = [
        {"timestamp": datetime(2026, 9, 12, 10, 0, 0), "id": 0},  # naive
        {"timestamp": datetime(2026, 9, 12, 10, 0, 10, tzinfo=timezone.utc), "id": 1},  # aware
        {"timestamp": datetime(2026, 9, 12, 10, 0, 20), "id": 2},  # naive
        {"timestamp": datetime(2026, 9, 12, 10, 0, 30, tzinfo=timezone.utc), "id": 3},  # aware
    ]

    split = temporal_split_by_time(data, train_ratio=0.5)
    assert len(split.train) == 2
    assert len(split.test) == 2
    assert [d["id"] for d in split.train] == [0, 1]
    assert [d["id"] for d in split.test] == [2, 3]
    assert split.train_end_time.tzinfo == timezone.utc


def test_temporal_split_by_time_custom_extractor_mixed_timezones():
    """
    Verify that temporal_split_by_time normalizes custom timestamp_extractor returns.
    """
    records = [
        ("rec0", datetime(2026, 9, 12, 12, 0, 0)),  # naive
        ("rec1", datetime(2026, 9, 12, 12, 1, 0, tzinfo=timezone.utc)),  # aware
        ("rec2", datetime(2026, 9, 12, 12, 2, 0)),  # naive
    ]

    split = temporal_split_by_time(
        records,
        train_ratio=0.5,
        timestamp_extractor=lambda r: r[1],
    )
    assert len(split.train) >= 1
    assert len(split.test) >= 1


def test_temporal_train_test_split_handles_mixed_timezones_in_dicts():
    """
    Verify that temporal_train_test_split compares mixed naive/aware datetimes seamlessly.
    """
    data = [
        {"timestamp": datetime(2026, 9, 12, 10, 0, 0), "val": "A"},
        {"timestamp": datetime(2026, 9, 12, 10, 0, 10, tzinfo=timezone.utc), "val": "B"},
    ]
    train, test = temporal_train_test_split(data, train_ratio=0.5)
    assert len(train) == 1
    assert len(test) == 1