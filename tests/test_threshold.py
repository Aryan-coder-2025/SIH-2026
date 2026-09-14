import pytest

from src.eval.metrics import evaluate_risk
from src.eval.threshold import (
    DEFAULT_EXPERIMENTAL_THRESHOLD,
    ThresholdSelectionError,
    select_threshold,
)


def test_default_experimental_threshold_is_070_but_only_a_default():
    assert DEFAULT_EXPERIMENTAL_THRESHOLD == 0.70


def test_select_threshold_picks_best_f1_on_given_data():
    # Risk scores cleanly separable at 0.5; a range of thresholds should
    # all achieve perfect F1, but 0.9+ should start missing positives.
    y_val = [0, 0, 1, 1]
    risk_val = [0.1, 0.2, 0.8, 0.9]

    result = select_threshold(y_val, risk_val, metric="f1")

    # Confirm the selected threshold actually achieves the best score
    # among the standard candidate grid.
    achieved = evaluate_risk(y_val, risk_val, threshold=result.threshold)
    assert achieved.f1 == pytest.approx(result.validation_score)
    assert achieved.f1 == 1.0


def test_select_threshold_respects_custom_candidate_grid():
    y_val = [0, 1, 0, 1]
    risk_val = [0.2, 0.6, 0.3, 0.9]

    result = select_threshold(y_val, risk_val, candidate_thresholds=[0.5, 0.7], metric="f1")

    assert result.threshold in (0.5, 0.7)
    assert result.candidates_evaluated == 2


def test_select_threshold_ties_broken_by_lowest_threshold_deterministically():
    # Both thresholds give perfect separation -> identical F1 -> lowest wins.
    y_val = [0, 1]
    risk_val = [0.1, 0.95]

    result_a = select_threshold(y_val, risk_val, candidate_thresholds=[0.5, 0.6, 0.9], metric="f1")
    result_b = select_threshold(y_val, risk_val, candidate_thresholds=[0.5, 0.6, 0.9], metric="f1")

    assert result_a.threshold == result_b.threshold  # deterministic
    assert result_a.threshold == 0.5  # lowest tied candidate


def test_select_threshold_supports_precision_and_recall_metrics():
    y_val = [0, 0, 1, 1]
    risk_val = [0.1, 0.4, 0.6, 0.9]

    result_precision = select_threshold(y_val, risk_val, metric="precision")
    result_recall = select_threshold(y_val, risk_val, metric="recall")

    assert 0.0 <= result_precision.validation_score <= 1.0
    assert 0.0 <= result_recall.validation_score <= 1.0


def test_unsupported_metric_rejected():
    with pytest.raises(ThresholdSelectionError):
        select_threshold([0, 1], [0.1, 0.9], metric="not_a_real_metric")


def test_empty_candidate_grid_rejected():
    with pytest.raises(ThresholdSelectionError):
        select_threshold([0, 1], [0.1, 0.9], candidate_thresholds=[])


def test_invalid_validation_data_surfaces_as_threshold_error():
    with pytest.raises(ThresholdSelectionError):
        select_threshold([0, 1, 2], [0.1, 0.5, 0.9])  # y_val has invalid label 2


def test_selection_only_uses_the_data_passed_in_not_a_hidden_test_set():
    """
    Selecting on one (small) validation set and then a DIFFERENT
    validation set produces independently-chosen thresholds -- proving
    selection is driven solely by whatever data is passed in, with no
    dependency on any other/hidden dataset (e.g. a final test set).
    """
    y_val_a = [0, 0, 1, 1]
    risk_val_a = [0.1, 0.2, 0.8, 0.9]  # best threshold clusters around 0.3-0.8

    y_val_b = [0, 0, 1, 1]
    risk_val_b = [0.05, 0.06, 0.07, 0.99]  # best threshold clusters higher

    result_a = select_threshold(y_val_a, risk_val_a, metric="f1")
    result_b = select_threshold(y_val_b, risk_val_b, metric="f1")

    # Different validation data -> independently optimal thresholds.
    assert result_a.threshold != result_b.threshold
