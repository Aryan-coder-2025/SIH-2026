from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.evaluation.records import PredictionRecord
from src.temporal.sequences import TemporalSequenceBatch


def test_valid_prediction_record():
    pred_t = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    targ_t = datetime(2026, 9, 12, 10, 0, 30, tzinfo=timezone.utc)

    rec = PredictionRecord(
        source_host="192.168.1.10",
        prediction_time=pred_t,
        forecast_horizon=30,
        target_time=targ_t,
        y_true=1.0,
        predicted_risk=0.85,
    )

    assert rec.source_host == "192.168.1.10"
    assert rec.prediction_time == pred_t
    assert rec.target_time == targ_t
    assert rec.forecast_horizon == 30
    assert rec.predicted_risk == 0.85


def test_prediction_record_rejects_misaligned_timestamps():
    pred_t = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    # Target time is 10:00:20 instead of expected 10:00:30
    wrong_targ = datetime(2026, 9, 12, 10, 0, 20, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="PredictionRecord alignment error"):
        PredictionRecord(
            source_host="192.168.1.10",
            prediction_time=pred_t,
            forecast_horizon=30,
            target_time=wrong_targ,
            y_true=1.0,
            predicted_risk=0.5,
        )


def test_prediction_record_rejects_risk_outside_zero_one():
    pred_t = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    targ_t = datetime(2026, 9, 12, 10, 0, 10, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="predicted_risk must be a probability score in"):
        PredictionRecord(
            source_host="192.168.1.10",
            prediction_time=pred_t,
            forecast_horizon=10,
            target_time=targ_t,
            y_true=1.0,
            predicted_risk=1.5,  # > 1.0
        )

    with pytest.raises(ValueError, match="predicted_risk must be a probability score in"):
        PredictionRecord(
            source_host="192.168.1.10",
            prediction_time=pred_t,
            forecast_horizon=10,
            target_time=targ_t,
            y_true=1.0,
            predicted_risk=-0.1,  # < 0.0
        )


def test_prediction_record_rejects_nan_inf():
    pred_t = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    targ_t = datetime(2026, 9, 12, 10, 0, 10, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="must be a finite number"):
        PredictionRecord(
            source_host="192.168.1.10",
            prediction_time=pred_t,
            forecast_horizon=10,
            target_time=targ_t,
            y_true=1.0,
            predicted_risk=float("nan"),
        )


def test_to_prediction_records_rejects_none_predictions():
    """
    CRITICAL AUDIT INVARIANT:
    to_prediction_records must never silently manufacture predictions or default to zeros.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    batch = TemporalSequenceBatch(
        X=np.zeros((1, 10, 2)),
        y=np.zeros((1, 3)),
        hosts=["192.168.1.1"],
        prediction_times=[base],
        target_times=[[base + timedelta(seconds=10), base + timedelta(seconds=20), base + timedelta(seconds=30)]],
    )

    with pytest.raises(ValueError, match="predicted_risks must be explicitly supplied"):
        batch.to_prediction_records(predicted_risks=None)  # type: ignore


def test_to_prediction_records_rejects_shape_mismatch():
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    batch = TemporalSequenceBatch(
        X=np.zeros((1, 10, 2)),
        y=np.zeros((1, 3)),
        hosts=["192.168.1.1"],
        prediction_times=[base],
        target_times=[[base + timedelta(seconds=10), base + timedelta(seconds=20), base + timedelta(seconds=30)]],
    )

    # Mismatched horizon count (2 instead of 3)
    wrong_preds = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="does not match targets shape"):
        batch.to_prediction_records(predicted_risks=wrong_preds)


def test_to_prediction_records_defaults_to_batch_offsets_seconds():
    """
    CRITICAL AUDIT INVARIANT (Issue 2):
    When offsets_seconds is omitted, to_prediction_records automatically uses
    the batch's validated offsets_seconds.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    batch = TemporalSequenceBatch(
        X=np.zeros((1, 10, 2)),
        y=np.array([[0.0, 0.5, 1.0]]),
        hosts=["192.168.1.1"],
        prediction_times=[base],
        target_times=[[base + timedelta(seconds=10), base + timedelta(seconds=20), base + timedelta(seconds=30)]],
        offsets_seconds=(10, 20, 30),
    )

    records = batch.to_prediction_records(predicted_risks=np.array([[0.1, 0.4, 0.9]]))
    assert len(records) == 3
    assert [r.forecast_horizon for r in records] == [10, 20, 30]
    assert [r.target_time for r in records] == [
        base + timedelta(seconds=10),
        base + timedelta(seconds=20),
        base + timedelta(seconds=30),
    ]


def test_to_prediction_records_rejects_mismatched_offsets_seconds():
    """
    CRITICAL AUDIT INVARIANT (Issue 2):
    Supplying offsets_seconds that disagree with the batch's actual sequence horizons
    must raise an explicit ValueError.
    """
    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    batch = TemporalSequenceBatch(
        X=np.zeros((1, 10, 2)),
        y=np.array([[0.0, 0.5, 1.0]]),
        hosts=["192.168.1.1"],
        prediction_times=[base],
        target_times=[[base + timedelta(seconds=10), base + timedelta(seconds=20), base + timedelta(seconds=30)]],
        offsets_seconds=(10, 20, 30),
    )

    preds = np.array([[0.1, 0.4, 0.9]])

    # Disagreeing length
    with pytest.raises(ValueError, match="does not match target horizon count"):
        batch.to_prediction_records(predicted_risks=preds, offsets_seconds=(10, 20))

    # Disagreeing values (e.g. 40s instead of 30s)
    with pytest.raises(ValueError, match="does not match sequence batch offsets_seconds"):
        batch.to_prediction_records(predicted_risks=preds, offsets_seconds=(10, 20, 40))


def test_custom_sequence_horizons_propagate_to_prediction_records():
    """
    Verify that building sequences with custom horizons (e.g. steps 1 and 3 -> 10s and 30s)
    sets matching offsets_seconds on TemporalSequenceBatch and propagates to PredictionRecord.
    """
    from src.temporal.sequences import build_sequences

    base = datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)
    times = [base + timedelta(seconds=10 * i) for i in range(15)]
    feats = np.arange(15, dtype=float).reshape(15, 1)
    targs = np.arange(15, dtype=float)

    batch: TemporalSequenceBatch = build_sequences(
        features=feats,
        targets=targs,
        timestamps=times,
        source_hosts=["host_A"] * 15,
        history_length=10,
        forecast_horizons=(1, 3),  # steps 1 (+10s) and 3 (+30s)
        window_seconds=10,
        return_metadata=True,
    )

    assert batch.forecast_steps == (1, 3)
    assert batch.offsets_seconds == (10, 30)

    preds = np.zeros_like(batch.y)
    records = batch.to_prediction_records(predicted_risks=preds)
    # First sequence's records: horizon 10s and 30s
    assert records[0].forecast_horizon == 10
    assert records[1].forecast_horizon == 30

