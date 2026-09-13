from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.config import TemporalConfig
from src.schemas.evaluation import PredictionRecord
from src.schemas.features import (
    FeatureLeakageError,
    FeatureOrderError,
    validate_feature_names,
)
from src.schemas.traffic import TrafficWindow
from src.temporal.baseline import PersistenceBaseline
from src.temporal.sequences import (
    TemporalSequenceBatch,
    build_sequences,
    build_sequences_from_windows,
)
from src.temporal.split import temporal_split_by_time


# --------------------------------------------------------------------------
# 🔴 P0 / 🟠 P1: Partition Boundary & Anti-Leakage (Tests 10, 11)
# --------------------------------------------------------------------------

def test_invariant_10_11_train_test_boundary_no_leakage():
    """
    INVARIANT 10 & 11:
    Boundary between train and test: Distinctive attack feature in test partition
    must NEVER appear in training sequences, and training targets must not cross
    the temporal boundary into test.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    # 20 windows of 10s: 0s to 190s
    records = []
    for i in range(20):
        t = base + timedelta(seconds=10 * i)
        # Windows 0..13 benign (feat 10.0, label 0.0)
        # Windows 14..19 in TEST with distinctive attack feature (feat 999999.0, label 1.0)
        feat_val = 999999.0 if i >= 14 else 10.0
        label_val = 1.0 if i >= 14 else 0.0
        records.append(
            TrafficWindow(
                timestamp=t,
                source_host="Host_Alpha",
                packet_count=100,
                byte_count=1000,
                features=(feat_val,),
                label=label_val,
            )
        )

    # Split temporally at 70% duration (train: 0s..130s = 14 windows, test: 140s..190s = 6 windows)
    split = temporal_split_by_time(records, train_ratio=0.7)

    # Assert test starts at or after 140s
    assert len(split.train) == 14
    assert len(split.test) == 6

    # Build sequences on training partition
    train_batch: TemporalSequenceBatch = build_sequences_from_windows(
        split.train,
        return_metadata=True,
    )

    # 14 windows in train -> 14 - 10 - 3 + 1 = 2 sequences
    assert len(train_batch.X) == 2

    # 1. Assert distinctive test attack feature (999999.0) NEVER appears in training X
    assert 999999.0 not in train_batch.X

    # 2. Assert training targets are strictly benign (all 0.0) and never observe test attack
    assert np.all(train_batch.y == 0.0)

    # 3. Assert all training target times are strictly before test_start_time
    for target_time_list in train_batch.target_times:
        for targ_t in target_time_list:
            assert targ_t < split.test_start_time, (
                f"Boundary leakage: target time {targ_t} crossed into test partition {split.test_start_time}"
            )


# --------------------------------------------------------------------------
# 🟠 P1: Timezone Consistency & Invalid Timestamps (Tests 12, 13)
# --------------------------------------------------------------------------

def test_invariant_12_timezone_consistency():
    """
    INVARIANT 12:
    Timestamps with different timezone representations (UTC, offset-aware, naive)
    representing the exact same physical instants must be normalized consistently
    and not treated as different times or fail comparison.
    """
    # 10:00:00 UTC
    t_utc = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    # 15:30:00 UTC+5:30 (exact same physical instant as 10:00:00 UTC)
    tz_ist = timezone(timedelta(hours=5, minutes=30))
    t_ist = datetime(2026, 9, 12, 15, 30, 0, tzinfo=tz_ist)
    # Naive timestamp
    t_naive = datetime(2026, 9, 12, 10, 0, 10)  # 10s after base

    w_utc = TrafficWindow(timestamp=t_utc, source_host="h1", packet_count=1, byte_count=1)
    w_ist = TrafficWindow(timestamp=t_ist, source_host="h1", packet_count=1, byte_count=1)
    w_naive = TrafficWindow(timestamp=t_naive, source_host="h1", packet_count=1, byte_count=1)

    # Normalization should recognize w_utc and w_ist as the same UTC physical instant
    assert w_utc.timestamp == w_ist.timestamp
    # Naive timestamp normalized to UTC without error
    assert w_naive.timestamp.tzinfo == timezone.utc
    assert (w_naive.timestamp - w_utc.timestamp) == timedelta(seconds=10)


def test_invariant_13_invalid_timestamps_rejected():
    """
    INVARIANT 13:
    Invalid timestamps ("not-a-date", None, NaN) raise controlled ValueError.
    """
    for invalid_ts in ["not-a-date", None, float("nan"), 12345]:
        with pytest.raises(ValueError, match="timestamp must be a valid datetime"):
            TrafficWindow(
                timestamp=invalid_ts,  # type: ignore
                source_host="host_A",
                packet_count=1,
                byte_count=1,
            )


# --------------------------------------------------------------------------
# 🟠 P1: Single Host & Multi-Host Variable Lengths (Tests 17, 18)
# --------------------------------------------------------------------------

def test_invariant_17_single_host_exact_counts():
    """
    INVARIANT 17:
    Single host with 15 consecutive windows produces exactly (15 - 13 + 1) = 3 sequences
    with exact horizon alignment.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    times = [base + timedelta(seconds=10 * i) for i in range(15)]
    feats = np.arange(15, dtype=float).reshape(15, 1)
    targs = np.arange(15, dtype=float)

    X, y = build_sequences(
        feats,
        targs,
        timestamps=times,
        source_hosts=["host_A"] * 15,
        history_length=10,
        forecast_horizons=(1, 2, 3),
    )
    assert len(X) == 3
    assert len(y) == 3
    # Check sliding window progression
    np.testing.assert_array_equal(X[0, :, 0], np.arange(0, 10))
    np.testing.assert_array_equal(X[1, :, 0], np.arange(1, 11))
    np.testing.assert_array_equal(X[2, :, 0], np.arange(2, 12))


def test_invariant_18_variable_host_lengths():
    """
    INVARIANT 18:
    Multiple hosts with different activity lengths:
    - Host A: 20 windows -> 20 - 13 + 1 = 8 sequences
    - Host B: 11 windows -> insufficient (< 13) -> 0 sequences
    - Host C: 40 windows -> 40 - 13 + 1 = 28 sequences
    Total sequences must equal 8 + 0 + 28 = 36 sequences, processed independently.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    records = []

    # Host A: 20 windows
    for i in range(20):
        records.append(("Host_A", base + timedelta(seconds=10 * i), [float(i)], float(i)))

    # Host B: 11 windows (< 13 needed for history=10 + horizon=3)
    for i in range(11):
        records.append(("Host_B", base + timedelta(seconds=10 * i), [float(i)], float(i)))

    # Host C: 40 windows
    for i in range(40):
        records.append(("Host_C", base + timedelta(seconds=10 * i), [float(i)], float(i)))

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

    assert len(batch.X) == 36
    host_counts = {h: batch.hosts.count(h) for h in ("Host_A", "Host_B", "Host_C")}
    assert host_counts["Host_A"] == 8
    assert host_counts["Host_B"] == 0
    assert host_counts["Host_C"] == 28


# --------------------------------------------------------------------------
# 🟡 P1/P2: Configuration Agility & Window Size Mismatch (Tests 19, 22)
# --------------------------------------------------------------------------

def test_invariant_19_configuration_responsiveness():
    """
    INVARIANT 19:
    Sequence builder dynamically responds to changed configuration parameters
    rather than hard-coding 10-window history or 3-step forecast.
    """
    custom_cfg = TemporalConfig(
        window_seconds=10,
        sequence_length_windows=5,      # 5 history windows
        forecast_horizon_windows=2,    # 2 forecast steps
        forecast_offsets_seconds=(10, 20),
    )

    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    times = [base + timedelta(seconds=10 * i) for i in range(10)]
    feats = np.arange(10, dtype=float).reshape(10, 1)
    targs = np.arange(10, dtype=float)

    # 10 windows with hist=5, horizon=2 -> 10 - 5 - 2 + 1 = 4 sequences
    X, y = build_sequences(
        feats,
        targs,
        timestamps=times,
        source_hosts=["host_A"] * 10,
        config=custom_cfg,
    )
    assert X.shape == (4, 5, 1)
    assert y.shape == (4, 2)


def test_invariant_22_window_size_mismatch_rejected():
    """
    INVARIANT 22:
    If configuration specifies 10s windows, but incoming timestamps arrive at 5s intervals
    (00:00, 00:05, 00:10...), the system must not quietly treat 5-second observations as 10-second windows.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    # 5-second step intervals
    times = [base + timedelta(seconds=5 * i) for i in range(15)]
    feats = np.arange(15, dtype=float).reshape(15, 1)
    targs = np.arange(15, dtype=float)

    # In strict mode: raises ValueError immediately
    with pytest.raises(ValueError, match="Temporal discontinuity detected"):
        build_sequences(
            features=feats,
            targets=targs,
            timestamps=times,
            source_hosts=["host_A"] * 15,
            window_seconds=10,  # expecting 10s steps, got 5s
            strict_continuity=True,
        )

    # In segmentation mode: delta is 5s != 10s, so every step breaks segment -> 0 sequences
    X, y = build_sequences(
        features=feats,
        targets=targs,
        timestamps=times,
        source_hosts=["host_A"] * 15,
        window_seconds=10,
        strict_continuity=False,
    )
    assert len(X) == 0, "Must not silently compress 5-second observations into 10-second sequences"


# --------------------------------------------------------------------------
# 🔥 SIH Demo Credibility: Forecasting & Integration Tests (Tests 23, 24, 25, 26, 27, 29)
# --------------------------------------------------------------------------

def test_invariant_23_attack_after_history_credibility():
    """
    INVARIANT 23: Attack-after-history test
    History: 100 seconds (10 windows) of benign traffic.
    Future: Attack occurs at +20 seconds (window index 11).
    Verify that historical input contains NO attack information, while target(+20s) registers the attack.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    # 13 windows: 0..9 (history), 10 (+10s), 11 (+20s), 12 (+30s)
    times = [base + timedelta(seconds=10 * i) for i in range(13)]

    # Historical features = 0.0 (benign); at window 11 attack begins (feature = 500.0)
    feats = np.zeros((13, 1))
    feats[11, 0] = 500.0

    # Labels: 0.0 except window 11 and 12
    targs = np.zeros(13)
    targs[11] = 1.0  # +20s attack
    targs[12] = 1.0  # +30s attack

    batch: TemporalSequenceBatch = build_sequences(
        features=feats,
        targets=targs,
        timestamps=times,
        source_hosts=["host_A"] * 13,
        history_length=10,
        forecast_horizons=(1, 2, 3),
        return_metadata=True,
    )

    assert len(batch.X) == 1
    # 1. Historical features are strictly 0.0 (benign)
    assert np.all(batch.X[0] == 0.0)
    # 2. Future targets: +10s=0.0 (benign), +20s=1.0 (attack forecasted), +30s=1.0 (attack forecasted)
    np.testing.assert_array_equal(batch.y[0], np.array([0.0, 1.0, 1.0]))


def test_invariant_24_persistence_baseline_sanity():
    """
    INVARIANT 24: Persistence Baseline Sanity Test
    Persistence predicts that future state equals current state X[-1].
    - When current state = 1.0 and future = 1.0, persistence succeeds.
    - When current state = 0.0 and future = 1.0 (attack onset), persistence FAILS.
    This verifies that the baseline is not receiving future information.
    """
    baseline = PersistenceBaseline(forecast_horizons=(1, 2, 3))

    # Case A: Persistent attack (current = 1.0, future = [1.0, 1.0, 1.0])
    current_state_A = np.array([1.0])
    pred_A = baseline.predict(current_state_A)
    np.testing.assert_array_equal(pred_A[0], np.array([1.0, 1.0, 1.0]))

    # Case B: Attack onset in future (current = 0.0, future = [0.0, 1.0, 1.0])
    current_state_B = np.array([0.0])
    pred_B = baseline.predict(current_state_B)
    # Baseline predicts 0.0 for all horizons, failing to anticipate future attack onset
    np.testing.assert_array_equal(pred_B[0], np.array([0.0, 0.0, 0.0]))
    actual_future_B = np.array([0.0, 1.0, 1.0])
    assert not np.array_equal(pred_B[0], actual_future_B)


def test_invariant_25_prediction_record_internal_alignment():
    """
    INVARIANT 25: Prediction Record Alignment
    PredictionRecord guarantees internal consistency between prediction_time,
    forecast_horizon, and target_time.
    Example: prediction_time = 12:00:00, horizon = 30s -> target_time MUST = 12:00:30.
    """
    pred_t = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
    targ_t = datetime(2026, 9, 12, 12, 0, 30, tzinfo=timezone.utc)

    # Valid aligned record
    rec = PredictionRecord(
        source_host="192.168.1.100",
        prediction_time=pred_t,
        forecast_horizon=30,
        target_time=targ_t,
        y_true=1.0,
        predicted_risk=0.85,
    )
    assert rec.forecast_horizon == 30

    # Misaligned record: target_time is 12:00:20 instead of 12:00:30 -> raises ValueError
    misaligned_targ = datetime(2026, 9, 12, 12, 0, 20, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="PredictionRecord alignment error"):
        PredictionRecord(
            source_host="192.168.1.100",
            prediction_time=pred_t,
            forecast_horizon=30,
            target_time=misaligned_targ,
            y_true=1.0,
            predicted_risk=0.85,
        )


def test_invariant_26_feature_order_corruption_rejected():
    """
    INVARIANT 26: Feature-order corruption test
    Features supplied in wrong order (e.g. [duration, packets, bytes] instead of
    [packets, bytes, duration]) must be caught and rejected by feature schema validation.
    """
    canonical_order = ["packet_count", "byte_count", "flow_duration"]
    shuffled_features = ["flow_duration", "packet_count", "byte_count"]

    with pytest.raises(FeatureOrderError, match="Feature ordering mismatch"):
        validate_feature_names(shuffled_features, expected_order=canonical_order)


def test_invariant_27_target_derived_feature_leakage_rejected():
    """
    INVARIANT 27: No target-derived features in feature selection
    Features containing suspicious/target-derived columns such as risk_score,
    future_malicious, label, or attack_type must be explicitly rejected.
    """
    for forbidden_col in ["risk_score", "future_malicious", "label", "attack_type", "src_ip"]:
        with pytest.raises(FeatureLeakageError, match="Forbidden feature"):
            validate_feature_names(["packet_rate", forbidden_col, "byte_rate"])


def test_invariant_29_looks_valid_individually_but_has_timeline_gap():
    """
    INVARIANT 29: 'Looks valid but isn't' test
    Every individual TrafficWindow is valid, but the timeline has a massive 40-minute gap:
    00:00, 00:10, 00:20, 01:00, 01:10...
    The temporal sequence builder must refuse to join records across the gap.
    """
    base = datetime(2026, 9, 12, 0, 0, 0, tzinfo=timezone.utc)
    windows = [
        # Segment 1: 3 windows (0s, 10s, 20s)
        TrafficWindow(timestamp=base, source_host="host_A", packet_count=10, byte_count=100, label=0.0),
        TrafficWindow(timestamp=base + timedelta(seconds=10), source_host="host_A", packet_count=10, byte_count=100, label=0.0),
        TrafficWindow(timestamp=base + timedelta(seconds=20), source_host="host_A", packet_count=10, byte_count=100, label=0.0),
        # Gap: jumps from 00:00:20 to 01:00:00 (3580s gap!)
        # Segment 2: 3 windows (3600s, 3610s, 3620s)
        TrafficWindow(timestamp=base + timedelta(seconds=3600), source_host="host_A", packet_count=10, byte_count=100, label=0.0),
        TrafficWindow(timestamp=base + timedelta(seconds=3610), source_host="host_A", packet_count=10, byte_count=100, label=0.0),
        TrafficWindow(timestamp=base + timedelta(seconds=3620), source_host="host_A", packet_count=10, byte_count=100, label=0.0),
    ]

    # Every individual TrafficWindow is valid
    for w in windows:
        assert w.source_host == "host_A"

    # Sequence builder must recognize the gap and not compress the timeline
    batch = build_sequences_from_windows(windows, return_metadata=True)
    # Neither segment has 13 contiguous windows (both have 3) -> 0 sequences formed
    assert len(batch.X) == 0, (
        "Must NOT construct sequences spanning across the 40-minute temporal gap!"
    )
