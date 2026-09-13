from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.config import ConfigError, TemporalConfig, load_config
from src.schemas.traffic import TrafficWindow
from src.temporal.sequences import (
    TemporalSequenceBatch,
    build_sequences,
    build_sequences_from_windows,
)


# --------------------------------------------------------------------------
# Baseline existing tests (backward compatibility)
# --------------------------------------------------------------------------
def test_build_sequences():
    features = np.arange(13).reshape(13, 1)
    targets = np.arange(13)

    X, y = build_sequences(
        features,
        targets,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        allow_synthetic_fallbacks=True,
    )

    assert X.shape == (1, 10, 1)
    assert y.shape == (1, 3)

    np.testing.assert_array_equal(
        X[0, :, 0],
        np.arange(10),
    )

    np.testing.assert_array_equal(
        y[0],
        np.array([10, 11, 12]),
    )


def test_multiple_sequences_slide_forward():
    features = np.arange(15).reshape(15, 1)
    targets = np.arange(15)

    X, y = build_sequences(
        features,
        targets,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        allow_synthetic_fallbacks=True,
    )

    assert X.shape == (3, 10, 1)

    np.testing.assert_array_equal(
        X[1, :, 0],
        np.arange(1, 11),
    )

    np.testing.assert_array_equal(
        y[1],
        np.array([11, 12, 13]),
    )


def test_mismatched_lengths_rejected():
    features = np.zeros((10, 2))
    targets = np.zeros(9)

    with pytest.raises(ValueError, match="same length"):
        build_sequences(features, targets)


def test_invalid_feature_dimensions_rejected():
    features = np.zeros(10)
    targets = np.zeros(10)

    with pytest.raises(ValueError, match="2D array"):
        build_sequences(features, targets)


# --------------------------------------------------------------------------
# Section 9: Realistic Scenario Tests (Tests 1 - 12)
# --------------------------------------------------------------------------

def test_scenario_1_multi_host_isolation():
    """
    TEST 1: Interleave two source hosts (Host A and Host B) in time.
    Verify that every generated sequence contains observations strictly
    from exactly one source host, and never mixes hosts.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    # 15 windows for Host A (features = 100 + i)
    # 15 windows for Host B (features = 200 + i)
    records = []
    for i in range(15):
        t = base + timedelta(seconds=10 * i)
        # Interleaved: Host A then Host B at same timestamps
        records.append(("10.0.0.1", t, float(100 + i), float(i)))
        records.append(("10.0.0.2", t, float(200 + i), float(i)))

    hosts = [r[0] for r in records]
    times = [r[1] for r in records]
    feats = np.array([[r[2]] for r in records])
    targs = np.array([r[3] for r in records])

    batch: TemporalSequenceBatch = build_sequences(
        features=feats,
        targets=targs,
        timestamps=times,
        source_hosts=hosts,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        return_metadata=True,
    )

    # 15 windows each -> 15 - 10 - 3 + 1 = 3 sequences per host -> 6 total
    assert len(batch.X) == 6
    assert len(batch.hosts) == 6

    for idx, (seq_X, host) in enumerate(zip(batch.X, batch.hosts)):
        if host == "10.0.0.1":
            # All historical feature values must be in [100, 199]
            assert np.all((seq_X >= 100) & (seq_X < 200)), (
                f"Host A sequence {idx} contaminated with non-Host-A data: {seq_X}"
            )
        elif host == "10.0.0.2":
            # All historical feature values must be in [200, 299]
            assert np.all((seq_X >= 200) & (seq_X < 300)), (
                f"Host B sequence {idx} contaminated with non-Host-B data: {seq_X}"
            )
        else:
            pytest.fail(f"Unexpected host {host}")


def test_scenario_2_unordered_input():
    """
    TEST 2: Provide rows in non-chronological order.
    Verify deterministic chronological processing.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    times_ordered = [base + timedelta(seconds=10 * i) for i in range(13)]
    feats_ordered = np.arange(13, dtype=float).reshape(13, 1)
    targs_ordered = np.arange(13, dtype=float)

    # Shuffle indices
    perm = [5, 2, 12, 0, 8, 1, 9, 3, 11, 4, 7, 6, 10]
    times_shuffled = [times_ordered[p] for p in perm]
    feats_shuffled = feats_ordered[perm]
    targs_shuffled = targs_ordered[perm]

    X, y = build_sequences(
        features=feats_shuffled,
        targets=targs_shuffled,
        timestamps=times_shuffled,
        source_hosts=["host_A"] * 13,
        history_length=10,
        forecast_horizons=(1, 2, 3),
    )

    # Output must match chronological ordering: X[0] = [0..9], y[0] = [10, 11, 12]
    assert X.shape == (1, 10, 1)
    assert y.shape == (1, 3)
    np.testing.assert_array_equal(X[0, :, 0], np.arange(10))
    np.testing.assert_array_equal(y[0], np.array([10, 11, 12]))


def test_scenario_3_timestamp_gap():
    """
    TEST 3: Create 10:00:00, 10:00:10, 10:00:20, 10:00:40 (missing 10:00:30).
    Verify that the missing window is NOT silently treated as continuous.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    times = [
        base,
        base + timedelta(seconds=10),
        base + timedelta(seconds=20),
        base + timedelta(seconds=40),  # Gap of 20s instead of 10s
    ]
    feats = np.arange(4, dtype=float).reshape(4, 1)
    targs = np.arange(4, dtype=float)

    # In strict mode, must explicitly raise ValueError
    with pytest.raises(ValueError, match="Temporal discontinuity detected"):
        build_sequences(
            features=feats,
            targets=targs,
            timestamps=times,
            source_hosts=["host_A"] * 4,
            window_seconds=10,
            history_length=2,
            forecast_horizons=(1,),
            strict_continuity=True,
        )

    # In standard segmentation mode, the gap breaks continuity:
    # Segment 1: [0, 1, 2] (len 3)
    # Segment 2: [3] (len 1)
    # With history=3 and horizon=1 (need 4 contiguous), 0 sequences should be generated
    X, y = build_sequences(
        features=feats,
        targets=targs,
        timestamps=times,
        source_hosts=["host_A"] * 4,
        window_seconds=10,
        history_length=3,
        forecast_horizons=(1,),
        strict_continuity=False,
    )
    assert len(X) == 0, "Sequences must not bridge across temporal gap"


def test_scenario_4_duplicate_window():
    """
    TEST 4: Provide duplicate host/time windows.
    Verify explicit rejection with ValueError.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    times = [
        base,
        base + timedelta(seconds=10),
        base + timedelta(seconds=10),  # Duplicate timestamp for same host
    ]
    feats = np.ones((3, 1))
    targs = np.zeros(3)

    with pytest.raises(ValueError, match="Duplicate window detected"):
        build_sequences(
            features=feats,
            targets=targs,
            timestamps=times,
            source_hosts=["host_A", "host_A", "host_A"],
        )


def test_scenario_5_insufficient_history():
    """
    TEST 5: Provide fewer than 10 historical windows.
    Verify no invalid sequence is generated.
    """
    feats = np.ones((9, 1))
    targs = np.zeros(9)

    X, y = build_sequences(
        features=feats,
        targets=targs,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        allow_synthetic_fallbacks=True,
    )
    assert len(X) == 0
    assert len(y) == 0
    assert X.shape == (0, 10, 1)


def test_scenario_6_insufficient_future():
    """
    TEST 6: Provide 10 historical windows but fewer than max_horizon future windows.
    Verify no target is fabricated.
    """
    # 12 windows total: history=10, but horizon requires 3 future windows (needs 13)
    feats = np.ones((12, 1))
    targs = np.zeros(12)

    X, y = build_sequences(
        features=feats,
        targets=targs,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        allow_synthetic_fallbacks=True,
    )
    assert len(X) == 0
    assert len(y) == 0


def test_scenario_7_horizon_alignment():
    """
    TEST 7: Construct a deterministic timeline where expected +10/+20/+30 targets
    are obvious. Verify exact target alignment.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    times = [base + timedelta(seconds=10 * i) for i in range(15)]
    feats = np.arange(15, dtype=float).reshape(15, 1)
    # Target value at window i is 1000 + (i * 10)
    targs = np.array([1000 + (i * 10) for i in range(15)], dtype=float)

    batch: TemporalSequenceBatch = build_sequences(
        features=feats,
        targets=targs,
        timestamps=times,
        source_hosts=["host_A"] * 15,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        return_metadata=True,
    )

    # First sequence: history covers indices 0..9 (time 0s..90s)
    # Prediction time is at index 9 (time 90s)
    assert batch.prediction_times[0] == base + timedelta(seconds=90)

    # Expected target times: 90s + 10s = 100s, 90s + 20s = 110s, 90s + 30s = 120s
    assert batch.target_times[0] == [
        base + timedelta(seconds=100),
        base + timedelta(seconds=110),
        base + timedelta(seconds=120),
    ]

    # Target values at indices 10, 11, 12: 1100, 1110, 1120
    np.testing.assert_array_equal(
        batch.y[0],
        np.array([1100.0, 1110.0, 1120.0]),
    )


def test_scenario_8_future_leakage():
    """
    TEST 8: Create a distinctive attack signature occurring strictly in the future.
    Verify that this feature never appears in the historical context for a prediction
    made prior to the attack.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    times = [base + timedelta(seconds=10 * i) for i in range(14)]

    # Normal traffic has feature value 5.0
    feats = np.full((14, 1), 5.0)
    # Distinctive attack signature appears at window 12 (future)
    feats[12, 0] = 999999.0

    targs = np.zeros(14)
    targs[12] = 1.0  # Attack label

    batch: TemporalSequenceBatch = build_sequences(
        features=feats,
        targets=targs,
        timestamps=times,
        source_hosts=["host_A"] * 14,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        return_metadata=True,
    )

    # Sequence 0: history indices 0..9. Prediction made at window 9.
    # Future target horizon 3 is window 9 + 3 = 12 (the attack).
    assert batch.y[0, 2] == 1.0, "Future target should register the upcoming attack"

    # CRITICAL LEAKAGE TEST:
    # Historical feature vector for sequence 0 MUST NOT contain the 999999.0 attack feature
    assert 999999.0 not in batch.X[0, :, 0], (
        "LEAKAGE DETECTED: Future attack feature appeared in historical observations!"
    )


def test_scenario_9_host_specific_future_target():
    """
    TEST 9: Host A becomes malicious in the future (+30s target = 1.0).
    Host B remains strictly benign (targets = 0.0).
    Verify that Host A's future target never contaminates Host B.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    records = []

    # 13 windows for Host A and Host B
    for i in range(13):
        t = base + timedelta(seconds=10 * i)
        # Host A targets: 1.0 at index 12 (+30s from index 9), 0 otherwise
        target_A = 1.0 if i == 12 else 0.0
        records.append(("Host_A", t, [float(i)], target_A))

        # Host B targets: always 0.0
        records.append(("Host_B", t, [float(i * 10)], 0.0))

    hosts = [r[0] for r in records]
    times = [r[1] for r in records]
    feats = np.array([r[2] for r in records])
    targs = np.array([r[3] for r in records])

    batch: TemporalSequenceBatch = build_sequences(
        features=feats,
        targets=targs,
        timestamps=times,
        source_hosts=hosts,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        return_metadata=True,
    )

    for seq_y, host in zip(batch.y, batch.hosts):
        if host == "Host_A":
            # Target at +30s (index 2) must be 1.0
            assert seq_y[2] == 1.0
        elif host == "Host_B":
            # All targets for Host B must be 0.0
            assert np.all(seq_y == 0.0), (
                f"Contamination: Host B received non-benign targets: {seq_y}"
            )


def test_scenario_10_nan_and_inf_handling():
    """
    TEST 10: Verify explicit rejection of NaN and Inf in features or targets.
    """
    # Features with NaN
    feats_nan = np.array([[1.0], [float("nan")], [3.0]])
    targs = np.zeros(3)
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_sequences(feats_nan, targs, history_length=2, forecast_horizons=(1,))

    # Features with Inf
    feats_inf = np.array([[1.0], [float("inf")], [3.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_sequences(feats_inf, targs, history_length=2, forecast_horizons=(1,))

    # Targets with NaN
    feats_valid = np.ones((3, 1))
    targs_nan = np.array([0.0, float("nan"), 1.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_sequences(feats_valid, targs_nan, history_length=2, forecast_horizons=(1,))


def test_scenario_11_empty_input():
    """
    TEST 11: Verify controlled behavior for empty input.
    """
    empty_feats = np.empty((0, 4))
    empty_targs = np.empty((0,))

    X, y = build_sequences(
        empty_feats,
        empty_targs,
        history_length=10,
        forecast_horizons=(1, 2, 3),
    )
    assert X.shape == (0, 10, 4)
    assert y.shape == (0, 3)

    # With metadata
    batch: TemporalSequenceBatch = build_sequences(
        empty_feats,
        empty_targs,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        return_metadata=True,
    )
    assert len(batch.hosts) == 0
    assert len(batch.prediction_times) == 0


def test_scenario_12_config_integration_and_mismatch():
    """
    TEST 12: Verify that configuration parameters are respected and mismatched
    temporal configs raise errors.
    """
    # Valid config
    cfg = load_config()
    feats = np.arange(15).reshape(15, 1)
    targs = np.arange(15)

    X, y = build_sequences(feats, targs, config=cfg, allow_synthetic_fallbacks=True)
    # Using project config: history=10, horizons=3
    assert X.shape[1] == 10
    assert y.shape[1] == 3

    # Invalid config passed (e.g. window_seconds <= 0)
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        build_sequences(
            feats,
            targs,
            window_seconds=-5,
            allow_synthetic_fallbacks=True,
        )


def test_determinism_invariant():
    """
    Verify that sequence extraction is completely deterministic across repeated runs.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    records = [
        ("Host_Z", base + timedelta(seconds=10 * i), float(i), float(i))
        for i in range(15)
    ] + [
        ("Host_A", base + timedelta(seconds=10 * i), float(100 + i), float(i))
        for i in range(15)
    ]

    hosts = [r[0] for r in records]
    times = [r[1] for r in records]
    feats = np.array([[r[2]] for r in records])
    targs = np.array([r[3] for r in records])

    batch1 = build_sequences(feats, targs, timestamps=times, source_hosts=hosts, return_metadata=True)
    batch2 = build_sequences(feats, targs, timestamps=times, source_hosts=hosts, return_metadata=True)

    np.testing.assert_array_equal(batch1.X, batch2.X)
    np.testing.assert_array_equal(batch1.y, batch2.y)
    assert batch1.hosts == batch2.hosts
    assert batch1.prediction_times == batch2.prediction_times
    # Host_A should be processed before Host_Z due to deterministic alphabetical host sorting
    assert batch1.hosts[0] == "Host_A"
    assert batch1.hosts[-1] == "Host_Z"


def test_build_sequences_from_windows_integration():
    """
    Verify building sequences directly from TrafficWindow objects.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    windows = [
        TrafficWindow(
            timestamp=base + timedelta(seconds=10 * i),
            source_host="192.168.1.50",
            packet_count=100 + i,
            byte_count=5000 + i * 100,
            label=1.0 if i >= 12 else 0.0,
        )
        for i in range(15)
    ]

    batch = build_sequences_from_windows(windows, return_metadata=True)
    assert isinstance(batch, TemporalSequenceBatch)
    assert len(batch.X) == 3
    assert batch.X.shape == (3, 10, 2)
    assert batch.y.shape == (3, 3)


def test_missing_label_none_rejected_never_assumed_benign():
    """
    CRITICAL AUDIT INVARIANT (Section 8):
    Missing ground truth must NEVER be silently converted to benign (0.0).
    Passing windows with label=None must raise an explicit ValueError.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    windows = [
        TrafficWindow(
            timestamp=base + timedelta(seconds=10 * i),
            source_host="192.168.1.50",
            packet_count=100 + i,
            byte_count=5000,
            label=None,  # Missing ground truth!
        )
        for i in range(15)
    ]

    with pytest.raises(ValueError, match="has missing label"):
        build_sequences_from_windows(windows)


def test_require_temporal_metadata_enforces_timestamps_and_hosts():
    """
    CRITICAL AUDIT INVARIANT (Section 11):
    In production mode (default allow_synthetic_fallbacks=False), missing timestamps
    or source_hosts must NOT be silently replaced with synthetic fallbacks.
    """
    feats = np.ones((15, 2))
    targs = np.zeros(15)

    with pytest.raises(ValueError, match="source_hosts must be explicitly provided"):
        build_sequences(feats, targs)

    with pytest.raises(ValueError, match="timestamps must be explicitly provided"):
        build_sequences(feats, targs, source_hosts=["host_A"] * 15)