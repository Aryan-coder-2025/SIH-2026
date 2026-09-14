import pytest

from src.eval.forecast import (
    AlignmentError,
    GroundTruthLabel,
    RawPrediction,
    align_predictions_with_ground_truth,
    evaluate_forecast,
)
from src.eval.records import (
    PredictionRecord,
    RecordValidationError,
    SUPPORTED_HORIZONS,
    find_duplicate_keys,
    find_missing_prediction_keys,
    validate_prediction_records,
)


def _record(host="hostA", t=100.0, horizon=10, y_true=1, risk=0.9):
    return PredictionRecord(
        source_host=host,
        prediction_time=t,
        forecast_horizon=horizon,
        target_time=t + horizon,
        y_true=y_true,
        risk=risk,
    )


# ----------------------------------------------------------------------
# Supported horizons
# ----------------------------------------------------------------------
def test_supported_horizons_are_10_20_30():
    assert SUPPORTED_HORIZONS == (10, 20, 30)


# ----------------------------------------------------------------------
# Record validation
# ----------------------------------------------------------------------
def test_valid_record_passes():
    validate_prediction_records([_record()])


def test_target_time_mismatch_is_rejected():
    bad = PredictionRecord(
        source_host="hostA", prediction_time=100.0, forecast_horizon=10,
        target_time=999.0,  # wrong: should be 110.0
        y_true=1, risk=0.9,
    )
    with pytest.raises(RecordValidationError, match="target_time"):
        validate_prediction_records([bad])


def test_unsupported_horizon_is_rejected():
    bad = PredictionRecord(
        source_host="hostA", prediction_time=100.0, forecast_horizon=15,
        target_time=115.0, y_true=1, risk=0.9,
    )
    with pytest.raises(RecordValidationError, match="forecast_horizon"):
        validate_prediction_records([bad])


def test_empty_source_host_is_rejected():
    bad = _record(host="")
    with pytest.raises(RecordValidationError, match="source_host"):
        validate_prediction_records([bad])


def test_invalid_y_true_is_rejected():
    bad = _record(y_true=2)
    with pytest.raises(RecordValidationError, match="y_true"):
        validate_prediction_records([bad])


def test_out_of_range_risk_is_rejected():
    bad = _record(risk=1.5)
    with pytest.raises(RecordValidationError, match="risk"):
        validate_prediction_records([bad])


def test_empty_records_rejected():
    with pytest.raises(RecordValidationError):
        validate_prediction_records([])


def test_multiple_problems_are_all_reported():
    bad = PredictionRecord(
        source_host="", prediction_time=100.0, forecast_horizon=99,
        target_time=100.0, y_true=5, risk=2.0,
    )
    with pytest.raises(RecordValidationError) as exc_info:
        validate_prediction_records([bad])
    message = str(exc_info.value)
    assert "source_host" in message
    assert "forecast_horizon" in message
    assert "y_true" in message
    assert "risk" in message


# ----------------------------------------------------------------------
# Duplicate detection (Part 2)
# ----------------------------------------------------------------------
def test_duplicate_keys_detected_not_silently_dropped():
    records = [_record(t=100.0, horizon=10), _record(t=100.0, horizon=10)]
    duplicates = find_duplicate_keys(records)
    assert duplicates == [("hostA", 100.0, 10)]

    with pytest.raises(RecordValidationError, match="Duplicate"):
        validate_prediction_records(records)


def test_no_duplicates_for_distinct_horizons_same_host_time():
    records = [_record(t=100.0, horizon=10), _record(t=100.0, horizon=20)]
    assert find_duplicate_keys(records) == []
    validate_prediction_records(records)  # must not raise


# ----------------------------------------------------------------------
# Missing prediction detection (Part 2)
# ----------------------------------------------------------------------
def test_missing_predictions_detected():
    records = [_record(host="hostA", t=100.0, horizon=10)]
    expected_keys = [
        ("hostA", 100.0, 10),
        ("hostA", 100.0, 20),  # missing
        ("hostB", 100.0, 10),  # missing
    ]
    missing = find_missing_prediction_keys(records, expected_keys)
    assert set(missing) == {("hostA", 100.0, 20), ("hostB", 100.0, 10)}

    with pytest.raises(RecordValidationError, match="Missing"):
        validate_prediction_records(records, expected_keys=expected_keys)


def test_complete_records_pass_expected_keys_check():
    records = [_record(host="hostA", t=100.0, horizon=10)]
    validate_prediction_records(records, expected_keys=[("hostA", 100.0, 10)])


# ----------------------------------------------------------------------
# Key-based (never positional) alignment + host isolation (Part 1.5)
# ----------------------------------------------------------------------
def test_alignment_is_key_based_not_positional():
    # Deliberately out-of-order / interleaved across hosts.
    predictions = [
        RawPrediction("hostB", 100.0, 10, 0.2),
        RawPrediction("hostA", 100.0, 10, 0.9),
    ]
    ground_truth = [
        GroundTruthLabel("hostA", 110.0, 1),
        GroundTruthLabel("hostB", 110.0, 0),
    ]

    records = align_predictions_with_ground_truth(predictions, ground_truth)
    by_host = {r.source_host: r for r in records}

    # hostA's prediction (risk=0.9) must be matched to hostA's label (1),
    # NOT to hostB's -- proving the join is key-based, not row-order-based.
    assert by_host["hostA"].y_true == 1
    assert by_host["hostA"].risk == 0.9
    assert by_host["hostB"].y_true == 0
    assert by_host["hostB"].risk == 0.2


def test_host_isolation_prevents_cross_host_matching():
    # hostA and hostB share the same target_time -- must not cross-match.
    predictions = [RawPrediction("hostA", 100.0, 10, 0.99)]
    ground_truth = [GroundTruthLabel("hostB", 110.0, 1)]  # wrong host

    with pytest.raises(AlignmentError, match="no matching ground truth"):
        align_predictions_with_ground_truth(predictions, ground_truth)


def test_ambiguous_ground_truth_is_rejected():
    ground_truth = [
        GroundTruthLabel("hostA", 110.0, 1),
        GroundTruthLabel("hostA", 110.0, 0),  # conflicting duplicate
    ]
    with pytest.raises(AlignmentError, match="Ambiguous"):
        align_predictions_with_ground_truth([], ground_truth)


def test_unmatched_prediction_raises_not_silently_dropped():
    predictions = [RawPrediction("hostA", 100.0, 10, 0.5)]
    with pytest.raises(AlignmentError, match="no matching ground truth"):
        align_predictions_with_ground_truth(predictions, ground_truth=[])


# ----------------------------------------------------------------------
# Per-horizon forecast evaluation (Part 1)
# ----------------------------------------------------------------------
def test_evaluate_forecast_separates_by_horizon():
    records = [
        _record(host="hostA", t=0.0, horizon=10, y_true=1, risk=0.9),  # correct @10
        _record(host="hostA", t=0.0, horizon=20, y_true=1, risk=0.1),  # wrong @20
        _record(host="hostB", t=0.0, horizon=10, y_true=0, risk=0.1),  # correct @10
    ]

    result = evaluate_forecast(records, threshold=0.5)

    assert set(result.per_horizon.keys()) == {10, 20}
    assert result.per_horizon[10].recall == 1.0  # hostA@10 caught
    assert result.per_horizon[10].fpr == 0.0      # hostB@10 correctly benign
    assert result.per_horizon[20].recall == 0.0   # hostA@20 missed


def test_evaluate_forecast_supports_per_horizon_threshold_dict():
    records = [
        _record(host="hostA", t=0.0, horizon=10, y_true=1, risk=0.65),
        _record(host="hostA", t=0.0, horizon=20, y_true=1, risk=0.65),
    ]
    # 0.60 catches horizon 10, 0.90 does not catch horizon 20
    result = evaluate_forecast(records, threshold={10: 0.60, 20: 0.90})

    assert result.per_horizon[10].recall == 1.0
    assert result.per_horizon[20].recall == 0.0


def test_evaluate_forecast_rejects_duplicate_records():
    records = [_record(t=0.0, horizon=10), _record(t=0.0, horizon=10)]
    with pytest.raises(RecordValidationError):
        evaluate_forecast(records)


def test_evaluate_forecast_to_dict_shape():
    records = [_record(t=0.0, horizon=10, y_true=1, risk=0.9)]
    result = evaluate_forecast(records, threshold=0.7)
    result_dict = result.to_dict()

    assert "per_horizon" in result_dict
    assert "10" in result_dict["per_horizon"]
    assert "aggregate" in result_dict
