"""
Integration tests for CyberForecastPipeline.
Validates end-to-end pipeline execution from model output to final payload.
"""

import json
import unittest
import numpy as np

from src.mitre.evidence import CANONICAL_FEATURES
from src.pipeline import CyberForecastPipeline


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.pipeline = CyberForecastPipeline()

    def test_full_pipeline_with_real_arrays(self):
        """Tests pipeline execution with realistic numpy arrays (10 windows x F)."""
        num_windows = 10
        num_features = len(CANONICAL_FEATURES)

        # 1. Temporal feature input sequence (10 x F)
        telemetry = np.zeros((num_windows, num_features))
        syn_idx = CANONICAL_FEATURES.index("syn_count")
        port_idx = CANONICAL_FEATURES.index("unique_dst_port_count")
        seq_idx = CANONICAL_FEATURES.index("sequential_port_ratio")
        ack_idx = CANONICAL_FEATURES.index("ack_count")

        telemetry[:, ack_idx] = 2.0
        # Surge in reconnaissance in latest windows (indices 7, 8, 9)
        telemetry[7:, syn_idx] = [25.0, 45.0, 55.0]
        telemetry[7:, port_idx] = [20.0, 40.0, 60.0]
        telemetry[7:, seq_idx] = [0.70, 0.85, 0.90]

        # 2. Multi-horizon model forecast (+10s, +20s, +30s)
        risk_timeline = [0.60, 0.75, 0.88]

        # 3. Temporal attribution tensor [3, 10, F]
        attributions = np.zeros((3, num_windows, num_features))
        attributions[0, 7, syn_idx] = 0.20
        attributions[1, 8, syn_idx] = 0.30
        attributions[2, 9, syn_idx] = 0.35
        attributions[2, 9, port_idx] = 0.25

        payload = self.pipeline.process_prediction(
            host_id="host-192-168-1-105",
            window_id=42,
            timestamp="2026-09-10T22:50:00Z",
            current_risk=0.45,
            risk_timeline=risk_timeline,
            predicted_stage="Discovery",
            temporal_features=telemetry,
            feature_attributions=attributions,
            temporal_attribution_tensor=attributions,
        )

        # Check required schema keys
        self.assertIn("host_id", payload)
        self.assertIn("forecast", payload)
        self.assertIn("mitre_attack", payload)
        self.assertIn("evidence", payload)
        self.assertIn("explainability", payload)
        self.assertIn("recommendations", payload)

        # Check forecast details
        self.assertEqual(payload["forecast"]["predicted_risk_30s"], 0.88)
        self.assertEqual(payload["forecast"]["urgency_level"], "HIGH")

        # Check MITRE mapping details
        mitre = payload["mitre_attack"]
        self.assertTrue(mitre["is_mapped"])
        self.assertEqual(mitre["candidate_technique_id"], "T1046")
        self.assertEqual(mitre["candidate_technique_name"], "Network Service Discovery")
        self.assertGreater(mitre["mapping_confidence"], 0.5)

        # Check JSON serializability
        json_str = json.dumps(payload, indent=2)
        self.assertIsInstance(json_str, str)
        self.assertIn("T1046", json_str)


if __name__ == "__main__":
    unittest.main()
