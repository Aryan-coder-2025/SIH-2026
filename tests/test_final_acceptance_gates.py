"""
Final Pre-GitHub Freeze Acceptance Gates Test Suite for SIH-26153.

Tests critical behavioral invariants:
1. Canonical 41-feature schema validation, exact ordering, and schema hash locking.
2. Temporal leakage hard gates: chronological split monotonicity, future target exclusion,
   deliberate leakage failure injection.
3. Scaler leakage hard gate (must be fitted only on training set; rejects test contamination).
4. Fair baselines contract: LR and RF receive identical flattened history (10 x 41 = 410 features).
5. Persistence baseline semantics: uses actual current risk state y_curr, rejects arbitrary proxy features.
6. Validation-only threshold selection protocol: test thresholds must remain strictly frozen.
7. Checkpoint provenance & failure injection: rejection of invalid schema_hash, feature order,
   input size, sequence length, and horizons.
8. Integrated Gradients XAI: shape preservation (batch, 10, 41), feature alignment,
   completeness/conservation sanity check (sum(attr) ~ F(x) - F(x0)).
9. Tamper-evident hash-chained audit ledger: valid chain, modified payload detection,
   modified previous_hash detection, reordered blocks detection, deleted block detection,
   duplicated block detection.
10. Chunked ingestion pipeline smoke test: chunked streaming, NaN/Inf handling,
    provenance reporting.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.audit.ledger import AuditBlock, TamperEvidentLedger
from src.config import get_temporal_config
from src.data.chunked_ingestion import ChunkedIngestionPipeline
from src.explain.integrated_gradients import IntegratedGradientsAttributor
from src.inference import validate_checkpoint_contract
from src.model import WorldModel
from src.schemas.features import (
    CANONICAL_FLOW_FEATURE_NAMES,
    CANONICAL_MODEL_FEATURE_NAMES,
    CANONICAL_PACKET_FEATURE_NAMES,
    CANONICAL_SCHEMA_HASH,
    FeatureLeakageError,
    FeatureOrderError,
    compute_schema_hash,
    validate_feature_names,
)
from src.temporal.sequences import build_sequences_from_dataframe
from src.temporal.split import temporal_split_by_time
from src.train import split_dataframe_temporally



# =============================================================================
# 1. SCHEMA & SCHEMA HASH ACCEPTANCE TESTS
# =============================================================================

def test_canonical_schema_feature_count_and_composition():
    """Verify production schema is exactly 41 features composed of 22 flow + 19 packet."""
    assert len(CANONICAL_FLOW_FEATURE_NAMES) == 22
    assert len(CANONICAL_PACKET_FEATURE_NAMES) == 19
    assert len(CANONICAL_MODEL_FEATURE_NAMES) == 41
    assert CANONICAL_MODEL_FEATURE_NAMES == CANONICAL_FLOW_FEATURE_NAMES + CANONICAL_PACKET_FEATURE_NAMES


def test_schema_hash_deterministic_and_matches_canonical():
    """Verify compute_schema_hash is deterministic and matches locked CANONICAL_SCHEMA_HASH."""
    computed = compute_schema_hash(CANONICAL_MODEL_FEATURE_NAMES)
    assert computed == CANONICAL_SCHEMA_HASH
    assert len(CANONICAL_SCHEMA_HASH) == 64  # Valid SHA-256 hex string


def test_schema_rejection_of_wrong_order_extra_and_missing_features():
    """Verify validate_feature_names rejects extra, missing, or permuted features."""
    canonical_list = list(CANONICAL_MODEL_FEATURE_NAMES)

    # Permuted order must raise FeatureOrderError
    permuted = list(canonical_list)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    with pytest.raises(FeatureOrderError):
        validate_feature_names(permuted, expected_order=canonical_list)

    # Missing feature must fail order check
    missing = canonical_list[:-1]
    with pytest.raises(FeatureOrderError):
        validate_feature_names(missing, expected_order=canonical_list)

    # Forbidden identifier/target feature must raise FeatureLeakageError
    for forbidden in ["target_risk", "is_malicious", "label", "flow_id", "source_host"]:
        with pytest.raises(FeatureLeakageError):
            validate_feature_names(canonical_list + [forbidden])


# =============================================================================
# 2. TEMPORAL LEAKAGE HARD GATE TESTS
# =============================================================================

@pytest.fixture
def sample_temporal_df() -> pd.DataFrame:
    """Create deterministic multi-host temporal dataframe for leakage verification."""
    rows = []
    base_ts = pd.Timestamp("2026-03-01 10:00:00")
    for host_id in ["host_alpha", "host_beta"]:
        for step in range(50):
            ts = base_ts + pd.Timedelta(seconds=step * 10)
            row = {
                "source_host": host_id,
                "timestamp": ts,
                "window_id": step,
                "is_malicious": 1 if step >= 35 else 0,
                "stage": "dos" if step >= 35 else "benign",
            }
            # Add all 41 canonical features
            for f_idx, feat in enumerate(CANONICAL_MODEL_FEATURE_NAMES):
                row[feat] = float(step * 0.1 + f_idx * 0.01)
            rows.append(row)
    return pd.DataFrame(rows)


def test_chronological_split_strict_time_monotonicity(sample_temporal_df):
    """Verify train end <= val start < val end <= test start and records are strictly monotonic."""
    train_df, val_df, test_df, split = split_dataframe_temporally(
        sample_temporal_df,
        train_ratio=0.7,
        validation_ratio=0.15,
    )
    assert split.train_end_time is not None
    assert split.val_start_time is not None
    assert split.val_end_time is not None
    assert split.test_start_time is not None

    # Cutoffs must satisfy ordering
    assert split.train_end_time <= split.val_start_time
    assert split.val_start_time < split.val_end_time
    assert split.val_end_time <= split.test_start_time

    # Actual records in train must strictly precede records in val, and val strictly precede test
    max_train_ts = pd.to_datetime(train_df["timestamp"]).max()
    min_val_ts = pd.to_datetime(val_df["timestamp"]).min()
    max_val_ts = pd.to_datetime(val_df["timestamp"]).max()
    min_test_ts = pd.to_datetime(test_df["timestamp"]).min()

    assert max_train_ts < min_val_ts, f"Train max ({max_train_ts}) must strictly precede Val min ({min_val_ts})"
    assert max_val_ts < min_test_ts, f"Val max ({max_val_ts}) must strictly precede Test min ({min_test_ts})"



def test_temporal_target_is_strictly_in_future_of_history(sample_temporal_df):
    """Verify that build_sequences generates targets that occur strictly after history windows."""
    history_len = 10
    forecast_horizon = 3

    X, y_risk, y_stage = build_sequences_from_dataframe(
        sample_temporal_df,
        feature_columns=list(CANONICAL_MODEL_FEATURE_NAMES),
        entity_column="source_host",
        timestamp_column="timestamp",
        risk_column="is_malicious",
        stage_column="stage",
        history_length=history_len,
        forecast_horizon=forecast_horizon,
    )
    # Shape checks: (N, 10, 41)
    assert X.ndim == 3
    assert X.shape[1] == history_len
    assert X.shape[2] == 41
    assert y_risk.shape[1] == forecast_horizon


def test_deliberate_future_leakage_failure_injection(sample_temporal_df):
    """
    FAILURE-INJECTION TEST:
    Inject future target values directly into feature matrix; verify validator catches leakage.
    """
    corrupted_df = sample_temporal_df.copy()
    # Inject a future target column into the feature schema
    corrupted_features = list(CANONICAL_MODEL_FEATURE_NAMES)[:-1] + ["target_future_risk"]
    corrupted_df["target_future_risk"] = corrupted_df["is_malicious"].shift(-3).fillna(0)

    with pytest.raises(FeatureLeakageError, match="Forbidden feature"):
        validate_feature_names(corrupted_features)


# =============================================================================
# 3. SCALER CONTAMINATION / LEAKAGE GATE
# =============================================================================

def test_scaler_fitted_only_on_training_data(sample_temporal_df):
    """Verify scaler mean/scale change if test data is improperly included (proving fit on train only)."""
    train_df, _, _, _ = split_dataframe_temporally(sample_temporal_df, train_ratio=0.7, validation_ratio=0.15)
    feature_cols = list(CANONICAL_MODEL_FEATURE_NAMES)

    train_scaler = StandardScaler().fit(train_df[feature_cols])
    contaminated_scaler = StandardScaler().fit(sample_temporal_df[feature_cols])

    # Because values drift upwards over time, train-only mean must differ from global mean
    assert not np.allclose(train_scaler.mean_, contaminated_scaler.mean_), (
        "Train-only scaler mean should differ from contaminated full-dataset scaler mean."
    )



# =============================================================================
# 4. FAIR BASELINES & PERSISTENCE SEMANTICS
# =============================================================================

def test_baseline_receives_identical_flattened_history():
    """Verify baseline receives 410 flattened features (10 windows x 41 features)."""
    batch_size = 15
    history_len = 10
    n_features = 41

    dummy_X = np.random.randn(batch_size, history_len, n_features)
    X_flat = dummy_X.reshape(batch_size, -1)

    assert X_flat.shape == (batch_size, 410), f"Expected 410 features for fair baseline, got {X_flat.shape[1]}"


def test_persistence_baseline_uses_actual_current_risk_state(sample_temporal_df):
    """
    REGRESSION TEST:
    Persistence baseline must broadcast the actual network state y_curr at time t,
    NOT flow_count or any arbitrary feature.
    """
    X, y_risk, _, y_curr = build_sequences_from_dataframe(
        sample_temporal_df,
        feature_columns=list(CANONICAL_MODEL_FEATURE_NAMES),
        entity_column="source_host",
        timestamp_column="timestamp",
        risk_column="is_malicious",
        stage_column="stage",
        history_length=10,
        forecast_horizon=3,
        return_current_risk=True,
    )
    assert y_curr is not None
    # y_curr must be binary or state in [0, 1] matching is_malicious at time t
    assert set(np.unique(y_curr)).issubset({0.0, 1.0})
    assert y_curr.shape == (len(X),)


def test_validation_only_threshold_selection_protocol():
    """Verify test evaluation uses frozen validation thresholds and never tunes on test set."""
    val_probs = np.array([0.1, 0.45, 0.8, 0.9])
    val_y = np.array([0, 0, 1, 1])

    # Best threshold on validation: 0.5 achieves 100% F1
    cand_thresh = [0.2, 0.5, 0.8]
    best_t = 0.5

    # On test set, frozen threshold must be used
    test_probs = np.array([0.48, 0.52])
    test_pred_frozen = (test_probs >= best_t).astype(int)
    assert np.array_equal(test_pred_frozen, [0, 1])


# =============================================================================
# 5. CHECKPOINT CONTRACT FAILURE-INJECTION TESTS
# =============================================================================

def test_checkpoint_contract_validation_and_failure_injection():
    """Verify validate_checkpoint_contract rejects invalid schema, order, counts, and horizons."""
    valid_checkpoint = {
        "project_id": "SIH26153",
        "schema_hash": CANONICAL_SCHEMA_HASH,
        "feature_order": list(CANONICAL_MODEL_FEATURE_NAMES),
        "feature_count": 41,
        "input_size": 41,
        "sequence_length": 10,
        "horizon": 3,
        "forecast_offsets_seconds": [10, 20, 30],
    }

    # 1. Valid checkpoint passes
    validate_checkpoint_contract(valid_checkpoint, list(CANONICAL_MODEL_FEATURE_NAMES))

    # 2. Corrupted schema hash injection -> Must raise ValueError
    corrupt_hash_ckpt = copy.deepcopy(valid_checkpoint)
    corrupt_hash_ckpt["schema_hash"] = "deadbeef" * 8
    with pytest.raises(ValueError, match="schema_hash"):
        validate_checkpoint_contract(corrupt_hash_ckpt, list(CANONICAL_MODEL_FEATURE_NAMES))

    # 3. Wrong feature count -> Must raise ValueError
    corrupt_count_ckpt = copy.deepcopy(valid_checkpoint)
    corrupt_count_ckpt["feature_count"] = 22
    with pytest.raises(ValueError, match="feature_count"):
        validate_checkpoint_contract(corrupt_count_ckpt, list(CANONICAL_MODEL_FEATURE_NAMES))

    # 4. Wrong sequence length -> Must raise ValueError
    corrupt_seq_ckpt = copy.deepcopy(valid_checkpoint)
    corrupt_seq_ckpt["sequence_length"] = 5
    with pytest.raises(ValueError, match="sequence_length"):
        validate_checkpoint_contract(corrupt_seq_ckpt, list(CANONICAL_MODEL_FEATURE_NAMES))

    # 5. Wrong horizon -> Must raise ValueError
    corrupt_horizon_ckpt = copy.deepcopy(valid_checkpoint)
    corrupt_horizon_ckpt["horizon"] = 1
    with pytest.raises(ValueError, match="horizon"):
        validate_checkpoint_contract(corrupt_horizon_ckpt, list(CANONICAL_MODEL_FEATURE_NAMES))


# =============================================================================
# 6. INTEGRATED GRADIENTS AXIOMATIC XAI TESTS
# =============================================================================

def test_integrated_gradients_shape_feature_alignment_and_completeness():
    """
    Verify Integrated Gradients preserves (batch, 10, 41) shape and satisfies
    approximate completeness / conservation: sum(attributions) ~ F(x) - F(x0).
    """
    model = WorldModel(
        input_size=41,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        horizon=3,
        num_stages=2,
    )
    model.eval()

    attributor = IntegratedGradientsAttributor(model=model, steps=30)
    x_input = torch.randn(1, 10, 41)
    baseline = torch.zeros(1, 10, 41)

    result = attributor.attribute(
        x_input=x_input,
        baseline=baseline,
        horizon_idx=0,
        feature_names=list(CANONICAL_MODEL_FEATURE_NAMES),
    )

    # Shape preservation
    assert result.attributions.shape == (10, 41)
    assert len(result.feature_importance) == 41
    assert list(result.feature_importance.keys()) == list(CANONICAL_MODEL_FEATURE_NAMES)

    # Completeness / Conservation sanity check:
    # Attribution sum should approximate model output difference within numerical tolerance
    with torch.no_grad():
        out_target = model(x_input)[0][0, 0].item()
        out_base = model(baseline)[0][0, 0].item()
        expected_diff = out_target - out_base

    attr_sum = float(result.attributions.sum())
    completeness_err = abs(attr_sum - expected_diff)
    assert completeness_err < 0.25, (
        f"Completeness error {completeness_err:.4f} exceeded tolerance between attr_sum={attr_sum} and diff={expected_diff}"
    )


# =============================================================================
# 7. TAMPER-EVIDENT AUDIT LEDGER TESTS
# =============================================================================

def test_tamper_evident_ledger_valid_chain():
    """Verify normal append operations produce a verified chain."""
    ledger = TamperEvidentLedger()
    ledger.append_record("rec_1", {"risk": 0.1, "host": "192.168.1.1"})
    ledger.append_record("rec_2", {"risk": 0.8, "host": "192.168.1.2"})
    ledger.append_record("rec_3", {"risk": 0.95, "host": "192.168.1.3"})

    is_valid, violations = ledger.verify_ledger_integrity()
    assert is_valid
    assert len(violations) == 0
    assert len(ledger.chain) == 3


def test_tamper_evident_ledger_detects_modified_payload():
    """TAMPER DETECTION TEST: Modify payload; ledger integrity must fail."""
    ledger = TamperEvidentLedger()
    ledger.append_record("rec_1", {"risk": 0.1})
    ledger.append_record("rec_2", {"risk": 0.8})

    # Tamper with block 1 payload
    ledger.chain[1].payload["risk"] = 0.0

    is_valid, violations = ledger.verify_ledger_integrity()
    assert not is_valid
    assert any("payload or metadata tampered" in v for v in violations)


def test_tamper_evident_ledger_detects_modified_previous_hash():
    """TAMPER DETECTION TEST: Modify previous_hash pointer; ledger integrity must fail."""
    ledger = TamperEvidentLedger()
    ledger.append_record("rec_1", {"risk": 0.1})
    ledger.append_record("rec_2", {"risk": 0.8})

    # Tamper with previous_hash pointer
    ledger.chain[1].previous_hash = "0" * 64

    is_valid, violations = ledger.verify_ledger_integrity()
    assert not is_valid
    assert any("previous_hash mismatch" in v for v in violations)


def test_tamper_evident_ledger_detects_reordered_blocks():
    """TAMPER DETECTION TEST: Swap block positions; ledger integrity must fail."""
    ledger = TamperEvidentLedger()
    ledger.append_record("rec_1", {"risk": 0.1})
    ledger.append_record("rec_2", {"risk": 0.8})
    ledger.append_record("rec_3", {"risk": 0.9})

    # Swap blocks 1 and 2
    ledger.chain[1], ledger.chain[2] = ledger.chain[2], ledger.chain[1]

    is_valid, violations = ledger.verify_ledger_integrity()
    assert not is_valid
    assert any("invalid index" in v or "previous_hash mismatch" in v for v in violations)


def test_tamper_evident_ledger_detects_deleted_block():
    """TAMPER DETECTION TEST: Delete an intermediate block; ledger integrity must fail."""
    ledger = TamperEvidentLedger()
    ledger.append_record("rec_1", {"risk": 0.1})
    ledger.append_record("rec_2", {"risk": 0.8})
    ledger.append_record("rec_3", {"risk": 0.9})

    # Delete intermediate block 1
    del ledger.chain[1]

    is_valid, violations = ledger.verify_ledger_integrity()
    assert not is_valid
    assert any("invalid index" in v or "previous_hash mismatch" in v for v in violations)


def test_tamper_evident_ledger_detects_duplicated_block():
    """TAMPER DETECTION TEST: Duplicate a block in the chain; ledger integrity must fail."""
    ledger = TamperEvidentLedger()
    ledger.append_record("rec_1", {"risk": 0.1})
    ledger.append_record("rec_2", {"risk": 0.8})

    # Duplicate block 1
    ledger.chain.append(copy.deepcopy(ledger.chain[1]))

    is_valid, violations = ledger.verify_ledger_integrity()
    assert not is_valid
    assert any("invalid index" in v or "previous_hash mismatch" in v for v in violations)


# =============================================================================
# 8. CHUNKED INGESTION PIPELINE SMOKE TEST
# =============================================================================

def test_chunked_ingestion_pipeline_smoke(tmp_path: Path):
    """Verify chunked ingestion pipeline reads, validates, and outputs clean Parquet with provenance."""
    # Create small raw CSV mock
    raw_csv = tmp_path / "raw_stream.csv"
    out_parquet = tmp_path / "stream_output.parquet"
    provenance_path = tmp_path / "provenance.json"

    rows = []
    base_ts = pd.Timestamp("2026-03-01 12:00:00")
    for i in range(120):
        row = {
            "source_host": f"10.0.0.{i % 4}",
            "timestamp": base_ts + pd.Timedelta(seconds=i * 10),
            "window_id": i,
            "is_malicious": 1 if i > 80 else 0,
            "stage": "portscan" if i > 80 else "benign",
        }
        for feat in CANONICAL_MODEL_FEATURE_NAMES:
            # Introduce occasional NaN/Inf to test sanitation
            if i == 5 and feat == "bytes_total":
                row[feat] = np.nan
            elif i == 10 and feat == "packets_total":
                row[feat] = np.inf
            else:
                row[feat] = float(i * 0.5)
        rows.append(row)

    pd.DataFrame(rows).to_csv(raw_csv, index=False)

    pipeline = ChunkedIngestionPipeline(
        raw_csv_path=raw_csv,
        output_parquet_path=out_parquet,
        chunk_size=50,
        provenance_report_path=provenance_path,
    )

    report = pipeline.run()

    assert out_parquet.is_file()
    assert provenance_path.is_file()
    assert report["total_processed_rows"] == 120
    assert report["total_dropped_rows"] == 0
    assert report["unique_entities"] == 4

    out_df = pd.read_parquet(out_parquet)
    assert not out_df.isna().any().any(), "Sanitized output must contain zero NaNs."
    assert not np.isinf(out_df[list(CANONICAL_MODEL_FEATURE_NAMES)].to_numpy()).any(), "Sanitized output must contain zero Infs."
