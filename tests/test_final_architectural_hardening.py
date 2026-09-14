from datetime import datetime, timedelta, timezone
import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from src.eval.metrics import evaluate_lead_time, evaluate_unseen_attacks
from src.evaluation.records import PredictionRecord
from src.explain.shap_explain import AttributionExplainer
from src.fusion.traffic_fusion import fused_df_to_traffic_windows
from src.inference import load_artifacts
from src.schemas.features import (
    CANONICAL_FLOW_FEATURE_NAMES,
    CANONICAL_MODEL_FEATURE_NAMES,
    CANONICAL_PACKET_FEATURE_NAMES,
    FeatureOrderError,
)
from src.temporal.sequences import build_sequences, build_sequences_from_dataframe
from src.temporal.split import temporal_split_by_time
from src.train import select_canonical_model_features, split_dataframe_temporally


class IdentityScaler:
    def transform(self, x):
        return x


def _canonical_row(i: int, host: str = "10.0.0.1") -> dict:
    row = {
        "source_host": host,
        "src_ip": host,
        "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=10 * i),
        "window_id": i,
        "is_malicious": 1.0 if i >= 18 else 0.0,
        "stage": "malicious" if i >= 18 else "benign",
    }
    row.update({name: float(i + idx + 1) for idx, name in enumerate(CANONICAL_MODEL_FEATURE_NAMES)})
    return row


def test_training_feature_selection_ignores_arbitrary_numeric_columns():
    df = pd.DataFrame([_canonical_row(i) for i in range(20)])
    df["stray_numeric_column"] = np.arange(len(df), dtype=float) + 9999.0

    assert select_canonical_model_features(df) == list(CANONICAL_MODEL_FEATURE_NAMES)
    assert "stray_numeric_column" not in select_canonical_model_features(df)


def test_training_feature_selection_fails_on_missing_canonical_feature():
    df = pd.DataFrame([_canonical_row(i) for i in range(20)]).drop(columns=["packet_count"])

    with pytest.raises(ValueError, match="Missing canonical model feature"):
        select_canonical_model_features(df)


def test_training_temporal_split_occurs_before_sequence_construction():
    df = pd.DataFrame([_canonical_row(i) for i in range(24)])
    df.loc[df["window_id"] >= 18, "flow_count"] = 999999.0
    train_df, _, test_df, split = split_dataframe_temporally(
        df,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    X_train, y_train, _ = build_sequences_from_dataframe(
        train_df,
        list(CANONICAL_MODEL_FEATURE_NAMES),
        sequence_length=10,
        horizon=3,
    )

    assert not test_df.empty
    assert train_df["timestamp"].max() < split.val_start_time
    assert 999999.0 not in X_train
    assert np.all(y_train == 0.0)


def test_canonical_fusion_rejects_missing_packet_and_flow_features():
    flow_only = pd.DataFrame([{**_canonical_row(0), **{name: 1.0 for name in CANONICAL_FLOW_FEATURE_NAMES}}])
    flow_only = flow_only.drop(columns=list(CANONICAL_PACKET_FEATURE_NAMES))
    with pytest.raises(ValueError, match="Canonical forecasting requires all 41"):
        fused_df_to_traffic_windows(flow_only)

    packet_only = pd.DataFrame([{**_canonical_row(0), **{name: 1.0 for name in CANONICAL_PACKET_FEATURE_NAMES}}])
    packet_only = packet_only.drop(columns=list(CANONICAL_FLOW_FEATURE_NAMES))
    with pytest.raises(ValueError, match="Canonical forecasting requires all 41"):
        fused_df_to_traffic_windows(packet_only)


def test_checkpoint_with_22_features_is_rejected_by_inference(tmp_path):
    artifact_dir = tmp_path
    (artifact_dir / "feature_order.json").write_text(
        json.dumps(list(CANONICAL_FLOW_FEATURE_NAMES)),
        encoding="utf-8",
    )
    with (artifact_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(IdentityScaler(), f)
    (artifact_dir / "stage_classes.json").write_text(json.dumps(["benign", "malicious"]), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": {},
            "project_id": "SIH26153",
            "feature_order": list(CANONICAL_FLOW_FEATURE_NAMES),
            "feature_count": 22,
            "input_size": 22,
            "sequence_length": 10,
            "horizon": 3,
            "forecast_offsets_seconds": [10, 20, 30],
        },
        artifact_dir / "world_model.pt",
    )

    with pytest.raises((ValueError, FeatureOrderError), match="feature|Checkpoint|ordering"):
        load_artifacts(artifact_dir)


def test_xai_feature_order_mismatch_is_rejected():
    wrong_order = list(CANONICAL_MODEL_FEATURE_NAMES)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]

    with pytest.raises(FeatureOrderError):
        AttributionExplainer(feature_names=wrong_order)


def test_xai_requires_three_by_ten_by_forty_one_attribution_tensor():
    explainer = AttributionExplainer(feature_names=list(CANONICAL_MODEL_FEATURE_NAMES))

    with pytest.raises(ValueError, match="Attribution tensor must have shape"):
        explainer.explain_multi_horizon(
            risk_timeline=[0.1, 0.2, 0.3],
            attributions_by_horizon=np.zeros((3, 10, 40)),
            temporal_attribution_tensor=np.zeros((3, 10, 40)),
        )


def test_source_hosts_never_mix_and_future_windows_stay_out_of_history():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = []
    hosts = []
    features = []
    targets = []
    for host, marker in [("host_a", 1.0), ("host_b", 2.0)]:
        for i in range(13):
            times.append(base + timedelta(seconds=10 * i))
            hosts.append(host)
            features.append([marker, 999.0 if i == 12 else 0.0])
            targets.append(1.0 if i == 12 else 0.0)

    batch = build_sequences(
        np.asarray(features),
        np.asarray(targets),
        timestamps=times,
        source_hosts=hosts,
        return_metadata=True,
    )

    assert batch.X.shape == (2, 10, 2)
    for idx, host in enumerate(batch.hosts):
        expected_marker = 1.0 if host == "host_a" else 2.0
        assert np.all(batch.X[idx, :, 0] == expected_marker)
        assert 999.0 not in batch.X[idx]
    assert np.all(batch.y[:, 2] == 1.0)


def test_split_before_sequence_would_fail_if_sequences_were_split_afterward():
    rows = [_canonical_row(i) for i in range(16)]
    split = temporal_split_by_time(rows, train_ratio=0.875, timestamp_extractor=lambda r: r["timestamp"])
    train_batch = build_sequences(
        np.array([[r["flow_count"]] for r in split.train]),
        np.array([r["is_malicious"] for r in split.train]),
        timestamps=[r["timestamp"] for r in split.train],
        source_hosts=[r["source_host"] for r in split.train],
        return_metadata=True,
    )

    assert len(train_batch.X) == 2
    assert all(target_time < split.test_start_time for targets in train_batch.target_times for target_time in targets)


def test_lead_time_uses_target_attack_time_not_horizon_labels():
    attack_time = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    records = [
        PredictionRecord("host_a", attack_time - timedelta(seconds=30), 30, attack_time, 1.0, 0.40),
        PredictionRecord("host_a", attack_time - timedelta(seconds=20), 20, attack_time, 1.0, 0.90),
        PredictionRecord("host_a", attack_time - timedelta(seconds=10), 10, attack_time, 1.0, 0.95),
    ]

    result = evaluate_lead_time(records, threshold=0.70)

    assert result.total_events == 1
    assert result.lead_times_seconds == [20]
    assert result.earliest_detections_by_horizon[20] == 1


def test_unseen_attack_evaluation_uses_predefined_seen_set():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = [
        PredictionRecord("host_a", base, 10, base + timedelta(seconds=10), 1.0, 0.9),
        PredictionRecord("host_b", base, 10, base + timedelta(seconds=10), 1.0, 0.1),
    ]

    result = evaluate_unseen_attacks(
        records,
        attack_labels=["PortScan", "NovelFamilyHeldOutBeforeEval"],
        seen_attack_types={"PortScan"},
        threshold=0.5,
    )

    assert set(result) == {"seen", "unseen"}
    assert result["seen"].recall == 1.0
    assert result["unseen"].recall == 0.0
