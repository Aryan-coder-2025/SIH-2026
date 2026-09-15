import datetime
import numpy as np
import pytest

from src.eval.forecast import (
    GroundTruthLabel,
    align_predictions_with_ground_truth,
    evaluate_forecast
)
from src.integration_adapter import convert_lstm_forecast_to_raw_predictions

def test_lstm_to_evaluation_pipeline_integration():
    """
    Contract-level integration test proving the end-to-end flow from
    the integrated LSTM model output to the evaluation pipeline.
    
    This acts as a proof of correctness for:
    1. Real LSTM output shape entering the adapter.
    2. Producing valid RawPrediction records.
    3. Passing these records directly to the existing forecast evaluation pipeline.
    
    NOTE: The real PyTorch model is NOT executed here because the required
    dependencies and checkpoint artifacts might not be present in the CI 
    or test environment. Instead, this uses the exact API contract 
    discovered from `origin/duplicate:src/inference.py`.
    """
    
    # 1. Simulate the exact output from the integrated LSTM inference pipeline
    # The `forecast(model, X, stage_classes)` function returns `risk, stage, stage_conf`
    # where `risk` is a numpy array of shape (3,) for horizons 10, 20, 30.
    simulated_risk_array = np.array([0.85, 0.60, 0.20], dtype=np.float32)
    prediction_time = datetime.datetime(2026, 9, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    source_host = "hostA"
    
    # Simulate a second host to have multiple records for the evaluation
    simulated_risk_array_2 = np.array([0.10, 0.20, 0.95], dtype=np.float32)
    prediction_time_2 = datetime.datetime(2026, 9, 15, 12, 0, 10, tzinfo=datetime.timezone.utc)
    source_host_2 = "hostB"
    
    # 2. Pass the simulated LSTM outputs through our integration adapter
    raw_preds_1 = convert_lstm_forecast_to_raw_predictions(
        source_host=source_host,
        prediction_timestamp=prediction_time,
        risk_array=simulated_risk_array
    )
    raw_preds_2 = convert_lstm_forecast_to_raw_predictions(
        source_host=source_host_2,
        prediction_timestamp=prediction_time_2,
        risk_array=simulated_risk_array_2
    )
    
    all_raw_preds = raw_preds_1 + raw_preds_2
    
    # Verify adapter produced the correct number of records
    assert len(all_raw_preds) == 6
    
    # 3. Simulate existing Ground Truth data in the evaluation framework
    ground_truths = [
        GroundTruthLabel(source_host="hostA", target_time=prediction_time.timestamp() + 10, y_true=1),
        GroundTruthLabel(source_host="hostA", target_time=prediction_time.timestamp() + 20, y_true=0),
        GroundTruthLabel(source_host="hostA", target_time=prediction_time.timestamp() + 30, y_true=0),
        
        GroundTruthLabel(source_host="hostB", target_time=prediction_time_2.timestamp() + 10, y_true=0),
        GroundTruthLabel(source_host="hostB", target_time=prediction_time_2.timestamp() + 20, y_true=0),
        GroundTruthLabel(source_host="hostB", target_time=prediction_time_2.timestamp() + 30, y_true=1),
    ]
    
    # 4. Pass into existing alignment logic
    aligned_records = align_predictions_with_ground_truth(all_raw_preds, ground_truths)
    
    # 5. Evaluate
    eval_result = evaluate_forecast(aligned_records, threshold=0.50)
    
    # 6. Verify evaluation computed successfully without altering the eval layer
    assert 10 in eval_result.per_horizon
    assert 20 in eval_result.per_horizon
    assert 30 in eval_result.per_horizon
    
    # For horizon 10: hostA pred 0.85 (y_true 1), hostB pred 0.10 (y_true 0)
    # TP=1, TN=1, FP=0, FN=0 -> Precision=1.0, Recall=1.0
    h10_metrics = eval_result.per_horizon[10]
    assert h10_metrics.confusion_matrix.tp == 1
    assert h10_metrics.confusion_matrix.tn == 1
    assert h10_metrics.precision == 1.0
    assert h10_metrics.recall == 1.0

    # For horizon 20: hostA pred 0.60 (y_true 0), hostB pred 0.20 (y_true 0)
    # TP=0, TN=1, FP=1, FN=0 -> FPR=0.5, Specificity=0.5
    h20_metrics = eval_result.per_horizon[20]
    assert h20_metrics.confusion_matrix.fp == 1
    assert h20_metrics.confusion_matrix.tn == 1
    assert h20_metrics.fpr == 0.5
    
    # For horizon 30: hostA pred 0.20 (y_true 0), hostB pred 0.95 (y_true 1)
    # TP=1, TN=1, FP=0, FN=0
    h30_metrics = eval_result.per_horizon[30]
    assert h30_metrics.confusion_matrix.tp == 1
    assert h30_metrics.confusion_matrix.tn == 1
    assert h30_metrics.precision == 1.0
