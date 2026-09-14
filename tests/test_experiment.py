import json

import pytest

from src.eval.experiment import ExperimentArtifact, ExperimentArtifactError


def _valid_kwargs(**overrides):
    kwargs = dict(
        experiment_name="rf_flow_packet_h10_v1",
        model_name="random_forest",
        dataset_version="CSE-CIC-IDS2018",
        train_period=(0.0, 1000.0),
        validation_period=(1000.0, 1500.0),
        test_period=(1500.0, 2000.0),
        forecast_horizon=10,
        feature_set=["syn_count", "bytes_total", "ttl_mean"],
        feature_schema_version="v1",
        history_length=10,
        random_seed=42,
        hyperparameters={"n_estimators": 100, "max_depth": None},
        threshold=0.70,
        metrics={"precision": 0.9, "recall": 0.85, "f1": 0.87, "fpr": 0.05},
        created_at="2026-09-10T00:00:00Z",
    )
    kwargs.update(overrides)
    return kwargs


# ----------------------------------------------------------------------
# Valid construction
# ----------------------------------------------------------------------
def test_valid_artifact_constructs_cleanly():
    artifact = ExperimentArtifact(**_valid_kwargs())
    assert artifact.model_name == "random_forest"
    assert artifact.random_seed == 42


def test_to_dict_is_json_serializable():
    artifact = ExperimentArtifact(**_valid_kwargs())
    d = artifact.to_dict()
    serialized = json.dumps(d)  # must not raise
    assert json.loads(serialized)["experiment_name"] == "rf_flow_packet_h10_v1"


def test_round_trip_to_dict_from_dict():
    artifact = ExperimentArtifact(**_valid_kwargs())
    restored = ExperimentArtifact.from_dict(artifact.to_dict())
    assert restored == artifact


# ----------------------------------------------------------------------
# Temporal ordering validation
# ----------------------------------------------------------------------
def test_train_period_must_end_before_validation_starts():
    with pytest.raises(ExperimentArtifactError, match="train_period"):
        ExperimentArtifact(
            **_valid_kwargs(
                train_period=(0.0, 1200.0),  # overlaps validation_period start (1000)
                validation_period=(1000.0, 1500.0),
            )
        )


def test_validation_period_must_end_before_test_starts():
    with pytest.raises(ExperimentArtifactError, match="validation_period"):
        ExperimentArtifact(
            **_valid_kwargs(
                validation_period=(1000.0, 1600.0),  # overlaps test_period start (1500)
                test_period=(1500.0, 2000.0),
            )
        )


def test_period_end_must_be_after_start():
    with pytest.raises(ExperimentArtifactError):
        ExperimentArtifact(**_valid_kwargs(train_period=(500.0, 100.0)))


def test_adjacent_periods_with_equal_boundary_are_valid():
    # train ends exactly where validation begins -- allowed (no gap required).
    artifact = ExperimentArtifact(
        **_valid_kwargs(
            train_period=(0.0, 1000.0),
            validation_period=(1000.0, 1500.0),
            test_period=(1500.0, 2000.0),
        )
    )
    assert artifact.test_period == (1500.0, 2000.0)


# ----------------------------------------------------------------------
# Field validation
# ----------------------------------------------------------------------
def test_invalid_forecast_horizon_rejected():
    with pytest.raises(ExperimentArtifactError, match="forecast_horizon"):
        ExperimentArtifact(**_valid_kwargs(forecast_horizon=15))


def test_empty_feature_set_rejected():
    with pytest.raises(ExperimentArtifactError, match="feature_set"):
        ExperimentArtifact(**_valid_kwargs(feature_set=[]))


def test_invalid_threshold_rejected():
    with pytest.raises(ExperimentArtifactError, match="threshold"):
        ExperimentArtifact(**_valid_kwargs(threshold=1.5))


def test_empty_metrics_rejected():
    with pytest.raises(ExperimentArtifactError, match="metrics"):
        ExperimentArtifact(**_valid_kwargs(metrics={}))


def test_non_positive_history_length_rejected():
    with pytest.raises(ExperimentArtifactError, match="history_length"):
        ExperimentArtifact(**_valid_kwargs(history_length=0))


def test_empty_experiment_name_rejected():
    with pytest.raises(ExperimentArtifactError, match="experiment_name"):
        ExperimentArtifact(**_valid_kwargs(experiment_name=""))


def test_empty_dataset_version_rejected():
    with pytest.raises(ExperimentArtifactError, match="dataset_version"):
        ExperimentArtifact(**_valid_kwargs(dataset_version=""))


# ----------------------------------------------------------------------
# from_dict validation
# ----------------------------------------------------------------------
def test_from_dict_rejects_missing_keys():
    incomplete = {"experiment_name": "x", "model_name": "random_forest"}
    with pytest.raises(ExperimentArtifactError, match="Missing"):
        ExperimentArtifact.from_dict(incomplete)


def test_from_dict_converts_period_lists_back_to_tuples():
    artifact = ExperimentArtifact(**_valid_kwargs())
    serialized = artifact.to_dict()
    assert isinstance(serialized["train_period"], list)  # JSON-friendly on the way out

    restored = ExperimentArtifact.from_dict(serialized)
    assert isinstance(restored.train_period, tuple)  # back to tuple on the way in
