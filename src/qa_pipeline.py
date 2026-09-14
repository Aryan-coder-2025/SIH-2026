import json
from datetime import datetime
from typing import Dict, List, Any

from src.qa_mock_data import generate_mock_data
from src.eval.temporal_split import temporal_train_validation_test_split
from src.baseline.temporal_features import flatten_temporal_history
from src.baseline.random_forest import RandomForestBaseline
from src.eval.forecast import (
    RawPrediction,
    GroundTruthLabel,
    align_predictions_with_ground_truth,
    evaluate_forecast,
    PredictionRecord
)
from src.eval.experiment import ExperimentArtifact

def run_qa_pipeline() -> List[Dict[str, Any]]:
    # 1. Generate Data
    n_samples = 200
    n_features = 5
    n_windows = 10
    features, timestamps, hosts, targets = generate_mock_data(
        n_samples=n_samples, n_features=n_features, n_windows=n_windows, random_seed=42
    )

    # 2. Flatten History
    # features shape: (N, 10, 5) -> (N, 50)
    flat_features = flatten_temporal_history(features)

    # 3. Temporal Split
    # Since timestamps are base + i*10, base is 1700000000
    # Let's split 60% train, 20% val, 20% test
    # 200 samples -> 120 train, 40 val, 40 test
    train_end = timestamps[120]
    val_end = timestamps[160]
    split = temporal_train_validation_test_split(timestamps, train_end, val_end)

    train_idx = split.train_idx
    test_idx = split.test_idx

    X_train = flat_features[train_idx]
    X_test = flat_features[test_idx]

    all_prediction_records: List[PredictionRecord] = []
    
    # 4. Train, predict and align per horizon
    horizons = [10, 20, 30]
    for horizon in horizons:
        y_all = targets[horizon]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        # Fit Baseline
        baseline = RandomForestBaseline(random_state=42)
        baseline.fit(X_train, y_train)

        # Predict Risk
        test_risk = baseline.predict_risk(X_test)

        # Create Alignment records
        raw_preds = []
        ground_truths = []
        for i, idx in enumerate(test_idx):
            host = hosts[idx]
            pred_time = timestamps[idx]
            target_time = pred_time + horizon
            
            raw_preds.append(
                RawPrediction(
                    source_host=host,
                    prediction_time=pred_time,
                    forecast_horizon=horizon,
                    risk=float(test_risk[i])
                )
            )
            ground_truths.append(
                GroundTruthLabel(
                    source_host=host,
                    target_time=target_time,
                    y_true=int(y_test[i])
                )
            )
        
        aligned_records = align_predictions_with_ground_truth(raw_preds, ground_truths)
        all_prediction_records.extend(aligned_records)

    # 5. Evaluate all records together (which evaluate_forecast supports)
    # The default threshold is 0.70
    eval_result = evaluate_forecast(all_prediction_records, threshold=0.70)

    # 6. Build ExperimentArtifact for each horizon
    artifacts = []
    for horizon in horizons:
        horizon_metrics = eval_result.per_horizon[horizon].to_dict()
        
        # Flatten confusion matrix so metrics is purely Dict[str, float]
        # (ExperimentArtifact type hint is Dict[str, float] and it was designed for flat metrics)
        # Actually, let's keep the precision, recall, f1, fpr.
        flat_metrics = {
            "precision": float(horizon_metrics["precision"]),
            "recall": float(horizon_metrics["recall"]),
            "f1": float(horizon_metrics["f1"]),
            "fpr": float(horizon_metrics["fpr"]),
            "accuracy": float(horizon_metrics["accuracy"]),
            "specificity": float(horizon_metrics["specificity"])
        }
        
        # We might also want to add confusion matrix elements as flat values
        cm = horizon_metrics["confusion_matrix"]
        flat_metrics["cm_tp"] = float(cm["tp"])
        flat_metrics["cm_fp"] = float(cm["fp"])
        flat_metrics["cm_tn"] = float(cm["tn"])
        flat_metrics["cm_fn"] = float(cm["fn"])

        artifact = ExperimentArtifact(
            experiment_name=f"qa_mock_rf_h{horizon}",
            model_name="random_forest",
            dataset_version="mock_qa_v1",
            train_period=(float(timestamps[train_idx][0]), float(timestamps[train_idx][-1] + 10.0)),
            validation_period=(float(timestamps[split.validation_idx][0]), float(timestamps[split.validation_idx][-1] + 10.0)),
            test_period=(float(timestamps[test_idx][0]), float(timestamps[test_idx][-1] + 10.0)),
            forecast_horizon=horizon,
            feature_set=[f"feat_{i}" for i in range(n_windows * n_features)],
            feature_schema_version="v1",
            history_length=n_windows,
            random_seed=42,
            hyperparameters={"n_estimators": 100},
            threshold=0.70,
            metrics=flat_metrics,
            created_at=datetime.utcnow().isoformat() + "Z"
        )
        artifacts.append(artifact.to_dict())

    return artifacts

if __name__ == "__main__":
    import os
    results = run_qa_pipeline()
    
    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/qa_mock_experiment.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Successfully wrote {len(results)} artifacts to {out_path}")
