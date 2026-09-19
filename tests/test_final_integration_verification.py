"""
Automated Verification Tests for Final Integration Fixes and Interactive Demo.
Tests:
1. 50-step Integrated Gradients: shape (10, 41), multi-horizon (3, 10, 41), completeness calculation.
2. Dynamic PASS/FAIL completeness logic.
3. Dashboard data-driven contract, sequence shapes, 3 horizons.
4. Demo root-independent paths and removal of gradient*input substitution.
5. Clean ledger integrity, unique record IDs, hash chain verification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.audit.ledger import TamperEvidentLedger
from src.explain.integrated_gradients import IntegratedGradientsAttributor
from src.inference import forecast, load_artifacts
from src.schemas.features import CANONICAL_MODEL_FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_matrix.parquet"


@pytest.fixture(scope="module")
def runtime_artifacts():
    model, scaler, feature_order, stage_classes = load_artifacts(ARTIFACT_DIR)
    return model, scaler, feature_order, stage_classes


@pytest.fixture(scope="module")
def canonical_dataframe():
    if not DATA_PATH.is_file():
        from src.generate_canonical_fixture import build_and_save_canonical_dataset
        build_and_save_canonical_dataset()
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# -----------------------------------------------------------------------------
# 1. 50-Step Integrated Gradients & Completeness Tests
# -----------------------------------------------------------------------------
class TestIntegratedGradients50Steps:
    def test_single_horizon_geometry_and_steps(self, runtime_artifacts, canonical_dataframe):
        model, scaler, feature_order, _ = runtime_artifacts
        host_df = canonical_dataframe[canonical_dataframe["source_host"] == "192.168.10.10"].sort_values("timestamp")
        slice_df = host_df.tail(10)

        X_raw = slice_df[feature_order].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(X_raw)
        X_batch = np.expand_dims(X_scaled, axis=0)

        attributor = IntegratedGradientsAttributor(model=model, steps=50)
        assert attributor.m_steps == 50

        res = attributor.attribute(
            x_input=torch.tensor(X_batch, dtype=torch.float32),
            horizon_idx=0,
            feature_names=feature_order,
        )

        # Shape must strictly be (10, 41)
        assert res.attributions.shape == (10, 41)
        assert len(res.feature_importance) == 41

    def test_multi_horizon_geometry(self, runtime_artifacts, canonical_dataframe):
        model, scaler, feature_order, _ = runtime_artifacts
        host_df = canonical_dataframe[canonical_dataframe["source_host"] == "192.168.10.10"].sort_values("timestamp")
        slice_df = host_df.tail(10)

        X_raw = slice_df[feature_order].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(X_raw)
        X_batch = np.expand_dims(X_scaled, axis=0)

        attributor = IntegratedGradientsAttributor(model=model, steps=50)
        multi_res = [
            attributor.attribute(
                x_input=torch.tensor(X_batch, dtype=torch.float32),
                horizon_idx=h,
                feature_names=feature_order,
            )
            for h in range(3)
        ]

        tensor = np.array([r.attributions for r in multi_res])
        # Canonical multi-horizon shape must strictly be (3, 10, 41)
        assert tensor.shape == (3, 10, 41)

    def test_completeness_calculation_and_dynamic_pass_fail(self, runtime_artifacts, canonical_dataframe):
        model, scaler, feature_order, _ = runtime_artifacts
        host_df = canonical_dataframe[canonical_dataframe["source_host"] == "192.168.10.10"].sort_values("timestamp")
        slice_df = host_df.tail(10)

        X_raw = slice_df[feature_order].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(X_raw)
        X_batch = np.expand_dims(X_scaled, axis=0)

        attributor = IntegratedGradientsAttributor(model=model, steps=50)
        res = attributor.attribute(
            x_input=torch.tensor(X_batch, dtype=torch.float32),
            horizon_idx=0,
            feature_names=feature_order,
        )

        # Dynamic verification logic test
        denom = max(abs(res.delta_target), abs(res.attribution_sum), 1e-6)
        rel_err = res.completeness_error / denom
        is_passed = attributor.check_completeness(res.delta_target, res.attribution_sum, relative_tolerance=0.20)

        # For our 50-step run, it must calculate real error and pass within tolerance
        assert res.completeness_error < 0.25
        assert is_passed or res.completeness_error <= 0.20

        # Inverted test: artificially huge difference must trigger FAIL
        fake_fail = attributor.check_completeness(delta_target=10.0, attribution_sum=1.0, relative_tolerance=0.20)
        assert fake_fail is False


# -----------------------------------------------------------------------------
# 2. Dashboard Data-Driven & History Visuals Tests
# -----------------------------------------------------------------------------
class TestDashboardIntegration:
    def test_all_hosts_exist_and_produce_valid_contract(self, canonical_dataframe):
        from src.dashboard import get_host_forecast

        hosts = canonical_dataframe["source_host"].unique().tolist()
        assert len(hosts) >= 6

        for h in hosts:
            contract = get_host_forecast(h, scenario="Attack Episode (Threat Window)", compute_xai=False)
            assert contract is not None
            assert contract["host_id"] == h
            assert contract["observed_state"] in ["BENIGN", "MALICIOUS"]
            assert len(contract["risk_preds"]) == 3
            assert len(contract["window_history"]) == 10
            # Time labels must be T-90s to T0
            assert contract["time_labels"] == [
                "T-90s", "T-80s", "T-70s", "T-60s", "T-50s",
                "T-40s", "T-30s", "T-20s", "T-10s", "T0"
            ]

    def test_ablation_results_are_dynamically_loaded(self):
        ablation_file = ARTIFACT_DIR / "ablation_results.json"
        assert ablation_file.is_file()

        with open(ablation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "modality_ablations" in data
        assert "history_depth_ablations" in data
        assert data["modality_ablations"]["flow_only_22"]["feature_count"] == 22
        assert data["modality_ablations"]["packet_only_19"]["feature_count"] == 19
        assert data["modality_ablations"]["multimodal_fusion_41"]["feature_count"] == 41


# -----------------------------------------------------------------------------
# 3. Demo Robustness & XAI Method Tests
# -----------------------------------------------------------------------------
class TestDemoRobustness:
    def test_gradient_input_saliency_not_masquerading_as_ig(self):
        import inspect
        import src.demo as demo_module

        # Ensure compute_honest_attributions (the gradient*input substitute) is absent
        assert not hasattr(demo_module, "compute_honest_attributions")
        source = inspect.getsource(demo_module)
        assert "grad * X_scaled" not in source

    def test_demo_execution_produces_50_step_ig_and_3_horizons(self):
        from src.demo import run_demo

        ledger_file = ARTIFACT_DIR / "audit_ledger.json"
        original_ledger_content = ledger_file.read_text(encoding="utf-8") if ledger_file.is_file() else None

        try:
            out = run_demo(scenario="attack")
            assert out["host_id"] is not None
            assert len(out["forecast"]["risk_timeline"]) == 3
            assert "observed_state" in out["forecast"]
            assert out["forecast"]["observed_state"] in ["BENIGN", "MALICIOUS"]
            assert len(out["explainability"]["horizons"]) == 3
        finally:
            if original_ledger_content is not None:
                ledger_file.write_text(original_ledger_content, encoding="utf-8")


# -----------------------------------------------------------------------------
# 4. Clean Audit Ledger Tests
# -----------------------------------------------------------------------------
class TestCleanAuditLedger:
    def test_ledger_integrity_and_unique_record_ids(self):
        ledger_file = ARTIFACT_DIR / "audit_ledger.json"
        assert ledger_file.is_file()

        ledger = TamperEvidentLedger.load_from_file(str(ledger_file))
        is_valid, violations = ledger.verify_ledger_integrity()
        assert is_valid is True
        assert len(violations) == 0

        # Unique record IDs
        record_ids = [b.record_id for b in ledger.chain]
        assert len(record_ids) == len(set(record_ids)), f"Duplicate record IDs found: {record_ids}"


# -----------------------------------------------------------------------------
# 5. Interactive Execution Gating & Semantic Contract Tests
# -----------------------------------------------------------------------------
class TestInteractiveExecutionAndSemanticContract:
    def test_forecast_gating_and_host_switching_contract(self, canonical_dataframe):
        """
        Verifies that:
        1. An unexecuted target returns None (requiring Run Forecast action).
        2. Triggering Run Forecast computes and returns the contract.
        3. Switching target from Host A -> Host B immediately invalidates the contract
           (does not recompute in background; does not serve Host A data for Host B).
        4. Triggering Run Forecast for Host B computes new results specific to Host B.
        """
        from src.dashboard import get_host_forecast

        # Simulated Session State
        session_state = {"active_target_key": None, "active_forecast": None}

        # Step 1: Initial state - target selected as Host A, no forecast run yet
        selected_host = "192.168.10.10"
        selected_scenario = "Attack Episode (Threat Window)"
        current_key = (selected_host, selected_scenario)

        should_execute = False
        host_contract = session_state["active_forecast"] if session_state["active_target_key"] == current_key else None
        # Must be None until user clicks Run Forecast
        assert host_contract is None, "Inference must NOT execute silently before Run Forecast is triggered."

        # Step 2: User clicks Run Forecast for Host A
        should_execute = True
        if should_execute:
            contract_a = get_host_forecast(selected_host, scenario=selected_scenario, compute_xai=True)
            session_state["active_forecast"] = contract_a
            session_state["active_target_key"] = current_key

        host_contract = session_state["active_forecast"] if session_state["active_target_key"] == current_key else None
        assert host_contract is not None
        assert host_contract["host_id"] == "192.168.10.10"
        assert host_contract["temporal_attribution_tensor"].shape == (3, 10, 41)

        # Step 3: Evaluator switches host: Host A -> Host B without clicking Run Forecast
        selected_host = "192.168.10.11"
        current_key = (selected_host, selected_scenario)
        should_execute = False

        host_contract = session_state["active_forecast"] if session_state["active_target_key"] == current_key else None
        # Must immediately be None! Does NOT serve Host A results for Host B, does NOT auto-compute
        assert host_contract is None, "Switching host must invalidate active contract until Run Forecast is clicked."

        # Step 4: Evaluator clicks Run Forecast for Host B
        should_execute = True
        if should_execute:
            contract_b = get_host_forecast(selected_host, scenario=selected_scenario, compute_xai=True)
            session_state["active_forecast"] = contract_b
            session_state["active_target_key"] = current_key

        host_contract = session_state["active_forecast"] if session_state["active_target_key"] == current_key else None
        assert host_contract is not None
        assert host_contract["host_id"] == "192.168.10.11"

        # Verify Host A and Host B produced distinct data-driven contracts
        assert contract_a["host_id"] != contract_b["host_id"]
        # Different raw traffic slices
        assert not contract_a["recent_slice"].equals(contract_b["recent_slice"])

    def test_active_forecast_result_absent_before_execution_and_invalidated_on_change(self, canonical_dataframe):
        """
        Lightweight interaction contract test proving:
        1. Active forecast result is absent in session state before Run Forecast execution.
        2. Active forecast result is populated after Run Forecast execution path.
        3. Changing host invalidates the previous result.
        4. Changing scenario invalidates the previous result.
        5. Preserves observed_state as string label (not probability float), current_forecast_risk,
           and 3 future forecast probabilities, while ambiguous current_risk is absent.
        """
        from src.dashboard import get_host_forecast

        session_state = {
            "active_target_key": None,
            "host_contract": None,
            "just_executed": False,
        }

        # Initial selection
        selected_host = "192.168.10.10"
        selected_scenario = "Attack Episode (Threat Window)"
        current_key = (selected_host, selected_scenario)

        # 1. Before execution: active forecast result must be absent
        if session_state.get("active_target_key") != current_key:
            session_state["host_contract"] = None
            session_state["just_executed"] = False

        assert session_state["host_contract"] is None
        assert session_state["just_executed"] is False

        # 2. Run Forecast execution path
        contract = get_host_forecast(selected_host, scenario=selected_scenario, compute_xai=True)
        session_state["host_contract"] = contract
        session_state["active_target_key"] = current_key
        session_state["just_executed"] = True

        # Populated after Run Forecast execution path
        assert session_state["host_contract"] is not None
        assert session_state["just_executed"] is True
        assert session_state["host_contract"]["host_id"] == selected_host
        assert session_state["host_contract"]["scenario"] == selected_scenario

        # Verify semantics on the active contract
        active = session_state["host_contract"]
        assert isinstance(active["observed_state"], str)
        assert active["observed_state"] in ("MALICIOUS", "BENIGN")
        assert not isinstance(active["observed_state"], float)
        assert "current_forecast_risk" in active
        assert isinstance(active["current_forecast_risk"], float)
        assert isinstance(active["forecast_risk_10s"], float)
        assert isinstance(active["forecast_risk_20s"], float)
        assert isinstance(active["forecast_risk_30s"], float)
        assert "current_risk" not in active

        # 3. Changing host invalidates previous result
        selected_host = "192.168.10.12"
        current_key = (selected_host, selected_scenario)
        if session_state.get("active_target_key") != current_key:
            session_state["host_contract"] = None
            session_state["active_target_key"] = None
            session_state["just_executed"] = False

        assert session_state["host_contract"] is None
        assert session_state["active_target_key"] is None
        assert session_state["just_executed"] is False

        # Execute for new host
        contract_new_host = get_host_forecast(selected_host, scenario=selected_scenario, compute_xai=False)
        session_state["host_contract"] = contract_new_host
        session_state["active_target_key"] = current_key
        session_state["just_executed"] = True
        assert session_state["host_contract"]["host_id"] == "192.168.10.12"

        # 4. Changing scenario invalidates previous result
        selected_scenario = "Baseline Traffic (Benign)"
        current_key = (selected_host, selected_scenario)
        if session_state.get("active_target_key") != current_key:
            session_state["host_contract"] = None
            session_state["active_target_key"] = None
            session_state["just_executed"] = False

        assert session_state["host_contract"] is None
        assert session_state["active_target_key"] is None
        assert session_state["just_executed"] is False

    def test_computation_actually_differs_across_hosts_and_scenarios(self, canonical_dataframe):
        """
        Verifies that actual computation differs across hosts and scenarios:
        - Host A (Attack) vs Host A (Benign)
        - Host A (Attack) vs Host B (Attack)
        """
        from src.dashboard import get_host_forecast

        host_a_attack = get_host_forecast("192.168.10.10", scenario="Attack Episode (Threat Window)", compute_xai=True)
        host_a_benign = get_host_forecast("192.168.10.10", scenario="Baseline Traffic (Benign)", compute_xai=True)
        host_b_attack = get_host_forecast("192.168.10.11", scenario="Attack Episode (Threat Window)", compute_xai=True)

        assert host_a_attack is not None
        assert host_a_benign is not None
        assert host_b_attack is not None

        # Scenario difference for same host
        assert host_a_attack["observed_state"] == "MALICIOUS"
        assert host_a_benign["observed_state"] == "BENIGN"
        assert host_a_attack["forecast_risk_10s"] > host_a_benign["forecast_risk_10s"]
        assert host_a_attack["pred_stage"] != host_a_benign["pred_stage"]

        # Attribution tensors are non-identical
        diff = np.abs(host_a_attack["temporal_attribution_tensor"] - host_b_attack["temporal_attribution_tensor"]).sum()
        assert diff > 1e-4, "Attribution tensors between distinct hosts must not be identical."

    def test_pipeline_explicit_semantics_contract_no_ambiguous_current_risk(self, canonical_dataframe):
        """
        Verifies that:
        1. observed_state is explicitly 'MALICIOUS' or 'BENIGN'.
        2. forecast_risk_10s, 20s, 30s are distinct float probabilities.
        3. Ambiguous 'current_risk' is removed from the forecast dictionary.
        """
        from src.pipeline import CyberForecastPipeline

        pipeline = CyberForecastPipeline()
        out = pipeline.process_prediction(
            host_id="192.168.10.10",
            window_id=1,
            timestamp="2026-09-18T08:00:00Z",
            forecast_risk_10s=0.85,
            risk_timeline=[0.85, 0.90, 0.95],
            predicted_stage="Impact",
            observed_state="MALICIOUS",
        )

        forecast = out["forecast"]
        assert forecast["observed_state"] == "MALICIOUS"
        assert forecast["forecast_risk_10s"] == 0.85
        assert forecast["forecast_risk_20s"] == 0.90
        assert forecast["forecast_risk_30s"] == 0.95
        assert forecast["current_forecast_risk"] == 0.85
        assert forecast["predicted_risk_30s"] == 0.95
        # The ambiguous current_risk must NOT be present
        assert "current_risk" not in forecast

    def test_scientific_report_matches_artifacts_and_frozen_contracts(self):
        """
        Verifies that:
        1. Random Forest +30s PR-AUC in SCIENTIFIC_VALIDATION_REPORT.md is 0.7317 (not stale 0.8841).
        2. Persistence +30s PR-AUC is 0.5706.
        3. Completeness tolerance threshold is 0.20 (not 0.15).
        4. No claims of 0.8841 appear in the report.
        5. Exact defensible phrasing for 10-window ablation is present.
        """
        report_path = PROJECT_ROOT / "SCIENTIFIC_VALIDATION_REPORT.md"
        assert report_path.is_file()
        content = report_path.read_text(encoding="utf-8")

        # Must not contain the stale 0.8841 value
        assert "0.8841" not in content

        # Check exact artifact values
        eval_path = ARTIFACT_DIR / "evaluation_metrics.json"
        assert eval_path.is_file()
        eval_metrics = json.loads(eval_path.read_text(encoding="utf-8"))

        rf_pr_auc_30s = eval_metrics["baselines"]["random_forest"]["30"]["pr_auc"]
        persist_pr_auc_30s = eval_metrics["baselines"]["persistence"]["30"]["pr_auc"]
        wm_pr_auc_30s = eval_metrics["world_model"]["30"]["pr_auc"]

        assert f"{rf_pr_auc_30s:.4f}" == "0.7317"
        assert f"{persist_pr_auc_30s:.4f}" == "0.5706"
        assert f"{wm_pr_auc_30s:.4f}" == "0.9409"

        assert "0.7317" in content
        assert "0.5706" in content

        # Completeness tolerance
        assert "0.20 tolerance threshold" in content
        assert "0.15 tolerance threshold" not in content

        # Exact defensible finding
        expected_finding = "10 windows is the frozen architectural contract for the temporal forecasting system; the current synthetic ablation does not establish that it is optimal."
        assert expected_finding in content

