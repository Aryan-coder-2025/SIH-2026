import numpy as np
import pytest

from src.eval.metrics import (
    ConfusionMatrix,
    EvaluationError,
    confusion_matrix,
    evaluate_predictions,
    evaluate_risk,
    f1_score,
    false_positive_rate,
    precision,
    recall,
    risk_to_label,
)


# ----------------------------------------------------------------------
# TEST 1: Perfect predictions
# ----------------------------------------------------------------------
def test_perfect_predictions():
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 1, 1, 0]

    result = evaluate_predictions(y_true, y_pred)

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.fpr == 0.0


# ----------------------------------------------------------------------
# TEST 2: Completely wrong predictions
# ----------------------------------------------------------------------
def test_completely_wrong_predictions():
    y_true = [0, 1, 0, 1]
    y_pred = [1, 0, 1, 0]

    result = evaluate_predictions(y_true, y_pred)

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.fpr == 1.0
    assert result.confusion_matrix == ConfusionMatrix(tn=0, fp=2, fn=2, tp=0)


# ----------------------------------------------------------------------
# TEST 3: Mixed predictions (manually verified)
# ----------------------------------------------------------------------
def test_mixed_predictions_manual_verification():
    # index:      0  1  2  3  4  5  6  7
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0, 1, 1]

    # TP: idx1, idx3, idx6 -> 3
    # TN: idx0, idx5 -> 2
    # FP: idx2, idx7 -> 2
    # FN: idx4 -> 1
    cm = confusion_matrix(y_true, y_pred)
    assert cm == ConfusionMatrix(tn=2, fp=2, fn=1, tp=3)

    expected_precision = 3 / (3 + 2)
    expected_recall = 3 / (3 + 1)
    expected_f1 = (
        2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    )
    expected_fpr = 2 / (2 + 2)

    assert precision(cm) == pytest.approx(expected_precision)
    assert recall(cm) == pytest.approx(expected_recall)
    assert f1_score(cm) == pytest.approx(expected_f1)
    assert false_positive_rate(cm) == pytest.approx(expected_fpr)


# ----------------------------------------------------------------------
# TEST 4: No positive predictions
# ----------------------------------------------------------------------
def test_no_positive_predictions_does_not_crash():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 0]

    result = evaluate_predictions(y_true, y_pred)

    assert result.precision == 0.0  # documented zero-division policy
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.fpr == 0.0


# ----------------------------------------------------------------------
# TEST 5: No positive ground-truth samples
# ----------------------------------------------------------------------
def test_no_positive_ground_truth_does_not_crash():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 1, 0, 0]

    result = evaluate_predictions(y_true, y_pred)

    # recall undefined (TP+FN=0) -> documented as 0.0
    assert result.recall == 0.0
    # fpr well-defined here: FP=1, TN=3
    assert result.fpr == pytest.approx(1 / 4)
    # precision well-defined: TP=0, FP=1
    assert result.precision == 0.0


def test_no_positive_predictions_and_no_positive_ground_truth():
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    result = evaluate_predictions(y_true, y_pred)

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.fpr == 0.0


# ----------------------------------------------------------------------
# TEST 6: Threshold conversion
# ----------------------------------------------------------------------
def test_threshold_conversion():
    risk = [0.2, 0.7, 0.69, 0.9]
    threshold = 0.7

    labels = risk_to_label(risk, threshold=threshold)

    assert list(labels) == [0, 1, 0, 1]


def test_evaluate_risk_matches_manual_threshold_conversion():
    y_true = [0, 1, 0, 1]
    risk = [0.2, 0.7, 0.69, 0.9]

    result = evaluate_risk(y_true, risk, threshold=0.7)
    expected = evaluate_predictions(y_true, [0, 1, 0, 1])

    assert result == expected


# ----------------------------------------------------------------------
# TEST 7: Invalid threshold
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1])
def test_invalid_threshold_is_rejected(bad_threshold):
    risk = [0.2, 0.7, 0.69, 0.9]

    with pytest.raises(EvaluationError):
        risk_to_label(risk, threshold=bad_threshold)


# ----------------------------------------------------------------------
# TEST 8: Mismatched input lengths
# ----------------------------------------------------------------------
def test_mismatched_lengths_raise_clear_error():
    y_true = [0, 1, 0]
    y_pred = [0, 1]

    with pytest.raises(EvaluationError):
        confusion_matrix(y_true, y_pred)


# ----------------------------------------------------------------------
# TEST 9: Empty input
# ----------------------------------------------------------------------
def test_empty_input_raises_clear_error():
    with pytest.raises(EvaluationError):
        confusion_matrix([], [])


def test_empty_risk_raises_clear_error():
    with pytest.raises(EvaluationError):
        risk_to_label([], threshold=0.5)


# ----------------------------------------------------------------------
# TEST 10: Invalid labels
# ----------------------------------------------------------------------
def test_invalid_labels_are_rejected():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]

    with pytest.raises(EvaluationError):
        confusion_matrix(y_true, y_pred)


def test_non_numeric_labels_are_rejected():
    y_true = [0, 1, "malicious"]
    y_pred = [0, 1, 1]

    with pytest.raises(EvaluationError):
        confusion_matrix(y_true, y_pred)


# ----------------------------------------------------------------------
# TEST 11: NaN/inf risk values
# ----------------------------------------------------------------------
def test_nan_risk_values_are_rejected():
    risk = [0.1, np.nan, 0.9]

    with pytest.raises(EvaluationError):
        risk_to_label(risk, threshold=0.5)


def test_inf_risk_values_are_rejected():
    risk = [0.1, np.inf, 0.9]

    with pytest.raises(EvaluationError):
        risk_to_label(risk, threshold=0.5)


def test_out_of_range_risk_values_are_rejected():
    risk = [0.1, 1.5, 0.9]

    with pytest.raises(EvaluationError):
        risk_to_label(risk, threshold=0.5)


# ----------------------------------------------------------------------
# TEST 12: Confusion matrix ordering
# ----------------------------------------------------------------------
def test_confusion_matrix_ordering():
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 0]

    cm = confusion_matrix(y_true, y_pred)

    assert cm.tn == 2
    assert cm.fp == 1
    assert cm.fn == 1
    assert cm.tp == 2

    expected_matrix = np.array([[2, 1], [1, 2]])
    assert np.array_equal(cm.matrix(), expected_matrix)

    expected_dict = {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert cm.to_dict() == expected_dict


# ----------------------------------------------------------------------
# Extra: EvaluationResult structured output
# ----------------------------------------------------------------------
def test_evaluation_result_to_dict_shape():
    result = evaluate_predictions([0, 1], [0, 1])
    result_dict = result.to_dict()

    assert set(result_dict.keys()) == {
        "precision",
        "recall",
        "f1",
        "fpr",
        "confusion_matrix",
    }
    assert set(result_dict["confusion_matrix"].keys()) == {"tn", "fp", "fn", "tp"}
