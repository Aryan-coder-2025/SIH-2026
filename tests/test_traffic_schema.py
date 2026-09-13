from datetime import datetime, timezone
import math

import pytest

from src.schemas.traffic import TrafficWindow


def test_valid_traffic_window():
    now = datetime.now(timezone.utc)
    window = TrafficWindow(
        timestamp=now,
        source_host="192.168.1.10",
        packet_count=100,
        byte_count=50000,
    )

    assert window.source_host == "192.168.1.10"
    assert window.packet_count == 100
    assert window.byte_count == 50000
    assert window.canonical_window_id == f"192.168.1.10_{int(now.timestamp())}"


def test_negative_packet_count_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        TrafficWindow(
            timestamp=datetime.now(timezone.utc),
            source_host="192.168.1.10",
            packet_count=-1,
            byte_count=50000,
        )


def test_negative_byte_count_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        TrafficWindow(
            timestamp=datetime.now(timezone.utc),
            source_host="192.168.1.10",
            packet_count=100,
            byte_count=-1,
        )


def test_empty_source_host_rejected():
    with pytest.raises(ValueError, match="source_host cannot be empty"):
        TrafficWindow(
            timestamp=datetime.now(timezone.utc),
            source_host="",
            packet_count=100,
            byte_count=50000,
        )


def test_whitespace_source_host_rejected():
    with pytest.raises(ValueError, match="source_host cannot be empty"):
        TrafficWindow(
            timestamp=datetime.now(timezone.utc),
            source_host="   ",
            packet_count=100,
            byte_count=50000,
        )


def test_invalid_timestamp_type_rejected():
    with pytest.raises(ValueError, match="timestamp must be a valid datetime instance"):
        TrafficWindow(
            timestamp="2026-09-12 10:00:00",  # type: ignore
            source_host="192.168.1.10",
            packet_count=100,
            byte_count=50000,
        )


def test_boolean_counts_rejected():
    with pytest.raises(ValueError, match="packet_count must be an integer"):
        TrafficWindow(
            timestamp=datetime.now(timezone.utc),
            source_host="192.168.1.10",
            packet_count=True,  # type: ignore
            byte_count=50000,
        )


def test_nan_or_inf_counts_rejected():
    with pytest.raises(ValueError, match="must be a finite number"):
        TrafficWindow(
            timestamp=datetime.now(timezone.utc),
            source_host="192.168.1.10",
            packet_count=float("nan"),  # type: ignore
            byte_count=50000,
        )

    with pytest.raises(ValueError, match="must be a finite number"):
        TrafficWindow(
            timestamp=datetime.now(timezone.utc),
            source_host="192.168.1.10",
            packet_count=100,
            byte_count=float("inf"),  # type: ignore
        )


def test_nan_or_inf_features_rejected():
    with pytest.raises(ValueError, match="must be a finite number"):
        TrafficWindow(
            timestamp=datetime.now(timezone.utc),
            source_host="192.168.1.10",
            packet_count=100,
            byte_count=50000,
            features=(1.0, float("nan"), 3.0),
        )


def test_explicit_window_id_preserved():
    window = TrafficWindow(
        timestamp=datetime.now(timezone.utc),
        source_host="192.168.1.10",
        packet_count=10,
        byte_count=500,
        window_id="custom_win_123",
    )
    assert window.canonical_window_id == "custom_win_123"


def test_features_converted_to_immutable_tuple():
    """
    CRITICAL AUDIT INVARIANT (Section 7):
    A frozen dataclass must not leave nested mutable structures like list modifiable.
    Passing a list must normalize to an immutable tuple.
    """
    mutable_list = [10.0, 20.0, 30.0]
    window = TrafficWindow(
        timestamp=datetime.now(timezone.utc),
        source_host="192.168.1.10",
        packet_count=10,
        byte_count=500,
        features=mutable_list,  # type: ignore
    )

    assert isinstance(window.features, tuple)
    assert window.features == (10.0, 20.0, 30.0)

    # Mutating original list must NOT mutate window.features
    mutable_list[0] = 999.0
    assert window.features[0] == 10.0