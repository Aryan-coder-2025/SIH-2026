"""
Master Cross-Module System Integration and Contract Test.
Project: SIH26153 - AI-Based Network Attack Forecasting from Network Traffic Data

Validates the full unified pipeline across all team components:
    Raw Traffic / Synthetic PCAP (Aman)
            ↓
    Packet Features (Aman)
            ↓
    Flow Features (Shaurya / Aman)
            ↓
    Traffic Fusion (Aman / Aryan)
            ↓
    Canonical TrafficWindow Schema (Aryan)
            ↓
    Temporal Split before Sequences (Aryan)
            ↓
    Canonical Sequence Construction (Aryan)
            ↓
    LSTM Model Ingestion & Forecasting (Sohini)
            ↓
    Multi-Horizon XAI Explanations (Srijani)
            ↓
    MITRE ATT&CK Candidate Interpretation & Evidence (Srijani)
            ↓
    Advisory SOC Recommendations (Srijani)
            ↓
    Multi-Horizon Forecast Evaluation & Baselines (Ankit / Aryan)
"""
from datetime import datetime, timedelta, timezone
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap  # type: ignore

from src.baseline.persistence import PersistenceBaseline as AnkitPersistenceBaseline
from src.eval.metrics import evaluate_prediction_records, evaluate_risk
from src.evaluation.records import PredictionRecord
from src.explain.shap_explain import ShapExplainer
from src.features.build_packet_features import build_packet_features
from src.flow.flow_extractor import extract_flow_features
from src.fusion.traffic_fusion import (
    fuse_flow_and_packet_dfs,
    fused_df_to_traffic_windows,
)
from src.mitre.evidence import EvidenceRuleEvaluator
from src.mitre.mapping import MitreMapper
from src.mitre.recommendations import RecommendationEngine
from src.model import WorldModel
from src.pipeline import CyberForecastPipeline
from src.schemas.features import (
    CANONICAL_FLOW_FEATURE_NAMES,
    CANONICAL_MODEL_FEATURE_NAMES,
    FORBIDDEN_FEATURE_NAMES,
    validate_feature_names,
)
from src.schemas.traffic import TrafficWindow
from src.temporal.baseline import PersistenceBaseline as AryanPersistenceBaseline
from src.temporal.sequences import build_sequences
from src.temporal.split import temporal_train_test_split


def generate_multi_host_pcap(pcap_path: str, num_windows: int = 16) -> None:
    """Generate deterministic synthetic packets across 2 hosts over 16 discrete 10s windows."""
    packets = []
    target_host = "192.168.10.15"
    benign_host = "192.168.10.20"

    for w in range(num_windows):
        t_base = w * 10.0

        # Target host: reconnaissance surge starting at window 8
        is_recon = w >= 8
        p_count = 15 if is_recon else 2

        for p_idx in range(p_count):
            dport = 20 + p_idx if is_recon else 80
            pkt = (
                Ether()
                / IP(src=target_host, dst="10.0.0.1")
                / TCP(sport=20000 + p_idx, dport=dport, flags="S" if is_recon else "A")
                / Raw(b"probe_payload" if is_recon else b"data")
            )
            pkt.time = t_base + 0.2 * p_idx
            packets.append(pkt)

        # Benign host: steady normal traffic
        pkt_b = (
            Ether()
            / IP(src=benign_host, dst="8.8.8.8")
            / UDP(sport=5353, dport=53)
            / Raw(b"normal_query")
        )
        pkt_b.time = t_base + 0.5
        packets.append(pkt_b)

    wrpcap(pcap_path, packets)


class TestMasterSystemIntegration:
    """Comprehensive end-to-end integration and architectural invariant tests."""

    def test_full_chain_from_pcap_to_evaluation_and_mitre(self):
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
            pcap_path = f.name

        try:
            target_host = "192.168.10.15"
            benign_host = "192.168.10.20"
            num_windows = 16

            # -------------------------------------------------------------
            # Stage 1: PCAP Ingestion (Aman)
            # -------------------------------------------------------------
            generate_multi_host_pcap(pcap_path, num_windows=num_windows)

            # -------------------------------------------------------------
            # Stage 2: Packet Feature Extraction (Aman)
            # -------------------------------------------------------------
            packet_records = build_packet_features(pcap_path)
            assert len(packet_records) > 0
            packet_df = pd.DataFrame(packet_records)

            # Verify host separation
            assert target_host in packet_df["src_ip"].values
            assert benign_host in packet_df["src_ip"].values

            # -------------------------------------------------------------
            # Stage 3: Flow Feature Extraction (Shaurya / Aman)
            # -------------------------------------------------------------
            flow_records = extract_flow_features(pcap_path)
            assert len(flow_records) > 0
            flow_df = pd.DataFrame(flow_records)

            # -------------------------------------------------------------
            # Stage 4: Traffic Fusion (Aman / Aryan)
            # -------------------------------------------------------------
            fused_df, report = fuse_flow_and_packet_dfs(flow_df, packet_df, how="left")
            assert len(fused_df) > 0
            assert not report.has_row_multiplication
            assert not report.has_row_loss

            # -------------------------------------------------------------
            # Stage 5: Canonical TrafficWindow Conversion (Aryan)
            # -------------------------------------------------------------
            target_df = (
                fused_df[fused_df["src_ip"] == target_host]
                .sort_values("window_id")
                .reset_index(drop=True)
            )
            # Assign future malicious risk target (0 for early windows, 1 for attack onset)
            target_df["label"] = [0.0 if w < 10 else 1.0 for w in range(len(target_df))]

            traffic_windows = fused_df_to_traffic_windows(target_df, label_col="label")
            assert len(traffic_windows) == num_windows

            for tw in traffic_windows:
                assert isinstance(tw, TrafficWindow)
                assert tw.source_host == target_host
                assert tw.timestamp.tzinfo == timezone.utc
                assert tw.packet_count >= 0
                assert tw.byte_count >= 0
                assert tw.features is not None
                assert all(np.isfinite(val) for val in tw.features)

            # -------------------------------------------------------------
            # Stage 6: Temporal Split BEFORE Sequence Construction (Aryan)
            # -------------------------------------------------------------
            # Split into train (14 windows) and test (2 windows)
            train_windows, test_windows = temporal_train_test_split(
                traffic_windows,
                train_ratio=0.875,
            )
            assert len(train_windows) == 14
            assert len(test_windows) == 2

            # Anti-leakage: strict time boundary
            max_train_ts = max(tw.timestamp for tw in train_windows)
            min_test_ts = min(tw.timestamp for tw in test_windows)
            assert max_train_ts < min_test_ts

            # -------------------------------------------------------------
            # Stage 7: Canonical Sequence Construction (Aryan)
            # -------------------------------------------------------------
            train_feats = np.array([tw.features for tw in train_windows], dtype=np.float32)
            train_targets = np.array([tw.label for tw in train_windows], dtype=np.float32)
            train_ts = [tw.timestamp for tw in train_windows]
            train_hosts = [tw.source_host for tw in train_windows]

            batch = build_sequences(
                features=train_feats,
                targets=train_targets,
                timestamps=train_ts,
                source_hosts=train_hosts,
                history_length=10,
                forecast_horizons=(1, 2, 3),
                window_seconds=10,
                return_metadata=True,
            )

            # Assert sequence geometry
            assert batch.X.shape[1] == 10  # 10 historical windows
            assert batch.y.shape[1] == 3   # 3 forecast horizons (+10s, +20s, +30s)
            num_features = batch.X.shape[2]

            # -------------------------------------------------------------
            # Stage 8: Model Ingestion & Forecasting (Sohini)
            # -------------------------------------------------------------
            model = WorldModel(
                input_size=num_features,
                hidden_size=64,
                num_layers=2,
                dropout=0.1,
                horizon=3,
                num_stages=2,
            )
            model.eval()

            X_tensor = torch.tensor(batch.X, dtype=torch.float32)
            with torch.no_grad():
                risk_logits, stage_logits = model(X_tensor)
                risk_probs = torch.sigmoid(risk_logits).numpy()
                stage_probs = torch.softmax(stage_logits, dim=1).numpy()

            assert risk_probs.shape == (batch.X.shape[0], 3)
            assert np.all(risk_probs >= 0.0) and np.all(risk_probs <= 1.0)
            assert stage_probs.shape == (batch.X.shape[0], 2)

            # -------------------------------------------------------------
            # Stage 9: Multi-Horizon XAI Explanations (Srijani)
            # -------------------------------------------------------------
            # Attribution tensor of shape (3 horizons, 10 windows, num_features)
            sample_attributions = np.zeros((3, 10, num_features))
            # Put synthetic attribution on feature 0 (e.g. packet_count or syn_count)
            sample_attributions[:, -1, 0] = 0.35

            explainer = ShapExplainer(feature_names=[f"feature_{i}" for i in range(num_features)])
            sample_risk_timeline = [float(risk_probs[0, 0]), float(risk_probs[0, 1]), float(risk_probs[0, 2])]

            xai_result = explainer.explain_multi_horizon(
                risk_timeline=sample_risk_timeline,
                attributions_by_horizon=sample_attributions,
                top_k=3,
            )

            assert len(xai_result["horizons"]) == 3
            assert xai_result["horizons"][0]["horizon_seconds"] == 10
            assert xai_result["horizons"][1]["horizon_seconds"] == 20
            assert xai_result["horizons"][2]["horizon_seconds"] == 30

            # -------------------------------------------------------------
            # Stage 10: MITRE Candidate Mapping & Evidence (Srijani)
            # -------------------------------------------------------------
            evaluator = EvidenceRuleEvaluator()
            evidence_profile = evaluator.evaluate_temporal_sequence(
                host_id=target_host,
                temporal_features=batch.X[0],
                feature_names=[f"feature_{i}" for i in range(num_features)],
                base_window_id=0,
            )

            mapper = MitreMapper()
            mitre_interpretation = mapper.map_prediction_to_mitre(
                predicted_stage="Discovery",
                forecast_risk=max(sample_risk_timeline),
                evidence_profile=evidence_profile,
                top_features=xai_result["top_features"],
            )

            assert mitre_interpretation.candidate_technique_id in ("T1046", "T1498", "T1071", "T1048", "T1059", "UNCERTAIN", None)
            assert mitre_interpretation.mapping_confidence >= 0.0
            assert mitre_interpretation.mapping_confidence <= 1.0

            # Recommendation engine
            recommender = RecommendationEngine()
            recs = recommender.get_recommendations(
                mitre_technique=mitre_interpretation.candidate_technique_id,
                risk_score=max(sample_risk_timeline),
            )
            assert isinstance(recs, list)
            assert len(recs) > 0
            assert "action_type" in recs[0]
            assert "priority" in recs[0]

            # -------------------------------------------------------------
            # Stage 11: Multi-Horizon Evaluation Records (Aryan / Ankit)
            # -------------------------------------------------------------
            pred_records = batch.to_prediction_records(risk_probs)
            assert len(pred_records) == batch.X.shape[0] * 3

            for pr in pred_records:
                assert isinstance(pr, PredictionRecord)
                assert pr.forecast_horizon in (10, 20, 30)
                assert pr.target_time == pr.prediction_time + timedelta(seconds=pr.forecast_horizon)
                assert 0.0 <= pr.predicted_risk <= 1.0

            # Multi-horizon evaluation metrics
            eval_by_horizon = evaluate_prediction_records(pred_records, threshold=0.50)
            assert 10 in eval_by_horizon
            assert 20 in eval_by_horizon
            assert 30 in eval_by_horizon

            for h in (10, 20, 30):
                res = eval_by_horizon[h]
                assert 0.0 <= res.precision <= 1.0
                assert 0.0 <= res.recall <= 1.0
                assert 0.0 <= res.f1 <= 1.0
                assert 0.0 <= res.fpr <= 1.0

            # -------------------------------------------------------------
            # Stage 12: Baselines Comparison (Ankit / Aryan)
            # -------------------------------------------------------------
            # Aryan Persistence Baseline
            aryan_persistence = AryanPersistenceBaseline(forecast_horizons=(1, 2, 3))
            current_states = train_feats[-len(batch.X):, 0]  # last known feature
            aryan_pred = aryan_persistence.predict(current_states)
            assert aryan_pred.shape == (len(current_states), 3)

            # Ankit Persistence Baseline (expects current state in [0, 1])
            state_in_01 = np.clip(train_feats[-len(batch.X):, [0]] / 100.0, 0.0, 1.0)
            ankit_persistence = AnkitPersistenceBaseline(current_state_column=0)
            ankit_persistence.fit(state_in_01, train_targets[-len(batch.X):])
            ankit_pred = ankit_persistence.predict_risk(state_in_01)
            assert len(ankit_pred) == len(current_states)
            assert np.all((ankit_pred >= 0.0) & (ankit_pred <= 1.0))

        finally:
            if os.path.exists(pcap_path):
                os.unlink(pcap_path)

    def test_forbidden_target_leakage_rejection(self):
        """Verify that forbidden labels, risk_scores, or targets are never allowed in features."""
        for forbidden in FORBIDDEN_FEATURE_NAMES:
            with pytest.raises(ValueError):
                validate_feature_names(["valid_feature_1", forbidden])

    def test_cyber_forecast_pipeline_unified_execution(self):
        """Verify the high-level CyberForecastPipeline orchestration wrapper."""
        pipeline = CyberForecastPipeline()

        telemetry = np.ones((10, len(CANONICAL_MODEL_FEATURE_NAMES))) * 5.0
        attributions = np.zeros((3, 10, len(CANONICAL_MODEL_FEATURE_NAMES)))
        attributions[0, -1, 0] = 0.40

        payload = pipeline.process_prediction(
            host_id="192.168.1.10",
            window_id=100,
            timestamp="2026-09-14T10:00:00Z",
            current_risk=0.30,
            risk_timeline=[0.40, 0.65, 0.85],
            predicted_stage="Reconnaissance",
            temporal_features=telemetry,
            feature_attributions=attributions,
            temporal_attribution_tensor=attributions,
        )

        assert payload["host_id"] == "192.168.1.10"
        assert len(payload["forecast"]["risk_timeline"]) == 3
        assert payload["forecast"]["risk_timeline"][0]["offset_seconds"] == 10
        assert payload["forecast"]["risk_timeline"][1]["offset_seconds"] == 20
        assert payload["forecast"]["risk_timeline"][2]["offset_seconds"] == 30
        assert "mitre_attack" in payload
        assert "evidence" in payload
        assert "recommendations" in payload
