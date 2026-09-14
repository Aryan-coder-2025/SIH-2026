"""
Unit and Contract Tests for Scientific Evaluation, Threshold Freezing,
Lead-Time Analysis, and Unified Feature Schemas.

Project: SIH26153 - AI-Based Network Attack Forecasting from Network Traffic Data
"""
from datetime import datetime, timedelta, timezone
import pytest
import numpy as np

from src.eval.metrics import (
    ConfusionMatrix,
    EvaluationResult,
    LeadTimeResult,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_lead_time,
    evaluate_prediction_records,
    evaluate_risk,
    evaluate_unseen_attacks,
    optimize_threshold,
)
from src.evaluation.records import PredictionRecord
from src.mitre.evidence import CANONICAL_FEATURES
from src.schemas.features import (
    CANONICAL_FLOW_FEATURE_NAMES,
    CANONICAL_FUSED_FEATURE_NAMES,
    CANONICAL_MODEL_FEATURE_NAMES,
    CANONICAL_PACKET_FEATURE_NAMES,
    FORBIDDEN_FEATURE_NAMES,
    validate_feature_names,
)


def test_canonical_feature_schema_unification():
    """Verify that flow and packet feature schemas combine into exactly 41 unique features."""
    assert len(CANONICAL_FLOW_FEATURE_NAMES) == 22
    assert len(CANONICAL_PACKET_FEATURE_NAMES) == 19
    assert len(CANONICAL_FUSED_FEATURE_NAMES) == 41
    assert CANONICAL_MODEL_FEATURE_NAMES == CANONICAL_FUSED_FEATURE_NAMES

    # Zero overlap in names between flow and packet features
    flow_set = set(CANONICAL_FLOW_FEATURE_NAMES)
    packet_set = set(CANONICAL_PACKET_FEATURE_NAMES)
    assert len(flow_set & packet_set) == 0

    # MITRE evidence CANONICAL_FEATURES must match CANONICAL_MODEL_FEATURE_NAMES exactly
    assert CANONICAL_FEATURES == list(CANONICAL_MODEL_FEATURE_NAMES)

    # Validate against forbidden names
    for f in CANONICAL_MODEL_FEATURE_NAMES:
        assert f not in FORBIDDEN_FEATURE_NAMES


def test_threshold_optimization_and_freezing():
    """Test validation threshold optimization and freezing before test evaluation."""
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Predicted risks separated around 0.65
    risk_val = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.85, 0.9])

    frozen_threshold = optimize_threshold(y_val, risk_val, metric="f1")
    # Expected threshold between 0.4 and 0.7 gives perfect F1=1.0
    assert 0.40 <= frozen_threshold <= 0.70

    # Apply frozen threshold to test set
    y_test = np.array([0, 0, 1, 1])
    risk_test = np.array([0.2, 0.3, 0.75, 0.95])
    test_res = evaluate_risk(y_test, risk_test, threshold=frozen_threshold)
    assert test_res.precision == 1.0
    assert test_res.recall == 1.0
    assert test_res.f1 == 1.0


def test_roc_and_pr_auc():
    """Verify continuous threshold-independent metrics (ROC-AUC & PR-AUC)."""
    y_true = np.array([0, 0, 1, 1])
    risk = np.array([0.1, 0.4, 0.6, 0.9])

    roc = compute_roc_auc(y_true, risk)
    pr = compute_pr_auc(y_true, risk)
    assert roc == 1.0
    assert pr == 1.0

    # Single-class batch returns None gracefully
    y_single = np.array([0, 0, 0, 0])
    assert compute_roc_auc(y_single, risk) is None
    assert compute_pr_auc(y_single, risk) is None


def test_lead_time_evaluation():
    """Test lead-time analysis measuring detection advance (+30s, +20s, +10s)."""
    t_base = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    t_attack = t_base + timedelta(seconds=30)

    # An attack occurs at t_attack.
    # Prediction at t_base with horizon=30s predicts high risk (0.85).
    # Prediction at t_base + 10s with horizon=20s predicts high risk (0.90).
    # Prediction at t_base + 20s with horizon=10s predicts high risk (0.95).
    rec30 = PredictionRecord(
        source_host="192.168.1.50",
        prediction_time=t_base,
        forecast_horizon=30,
        target_time=t_attack,
        y_true=1.0,
        predicted_risk=0.85,
    )
    rec20 = PredictionRecord(
        source_host="192.168.1.50",
        prediction_time=t_base + timedelta(seconds=10),
        forecast_horizon=20,
        target_time=t_attack,
        y_true=1.0,
        predicted_risk=0.90,
    )
    rec10 = PredictionRecord(
        source_host="192.168.1.50",
        prediction_time=t_base + timedelta(seconds=20),
        forecast_horizon=10,
        target_time=t_attack,
        y_true=1.0,
        predicted_risk=0.95,
    )

    lead_result = evaluate_lead_time([rec30, rec20, rec10], threshold=0.70)
    assert lead_result.total_events == 1
    assert lead_result.detected_events == 1
    assert lead_result.detection_rate == 1.0
    # Earliest detection was at horizon 30s (+30 seconds advance warning)
    assert lead_result.earliest_detections_by_horizon[30] == 1
    assert lead_result.mean_lead_time_seconds == 30.0


def test_unseen_attack_evaluation():
    """Test partitioning between known attack classes and unseen/zero-day attacks."""
    t_base = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)

    records = [
        PredictionRecord(
            source_host="10.0.0.1",
            prediction_time=t_base,
            forecast_horizon=10,
            target_time=t_base + timedelta(seconds=10),
            y_true=1.0,
            predicted_risk=0.85,
        ),
        PredictionRecord(
            source_host="10.0.0.2",
            prediction_time=t_base,
            forecast_horizon=10,
            target_time=t_base + timedelta(seconds=10),
            y_true=1.0,
            predicted_risk=0.45,
        ),
    ]
    attack_labels = ["PortScan", "ZeroDayExfil"]
    seen_types = {"PortScan"}

    partitioned = evaluate_unseen_attacks(records, attack_labels, seen_types, threshold=0.50)
    assert "seen" in partitioned
    assert "unseen" in partitioned

    # Seen (PortScan): risk 0.85 >= 0.50 -> TP=1, Recall=1.0
    assert partitioned["seen"].recall == 1.0
    # Unseen (ZeroDayExfil): risk 0.45 < 0.50 -> FN=1, Recall=0.0
    assert partitioned["unseen"].recall == 0.0
