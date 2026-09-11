"""
Unit tests for MITRE Evidence Extraction Engine.
"""

import math
import unittest
import numpy as np

from src.mitre.evidence import (
    CANONICAL_FEATURES,
    EvidenceItem,
    EvidenceRuleEvaluator,
    HostEvidenceProfile,
)


class TestEvidenceExtraction(unittest.TestCase):
    def setUp(self):
        self.evaluator = EvidenceRuleEvaluator()

    def test_empty_and_missing_features(self):
        """Tests that evaluator does not crash on empty or missing feature dicts."""
        profile = self.evaluator.evaluate_state("host-1", 10, {})
        self.assertFalse(profile.is_sufficient_evidence)
        self.assertEqual(len(profile.evidence_items), 0)
        self.assertIn("nominal baseline bounds", profile.behavior_summary[0])

    def test_port_scan_evidence(self):
        """Tests triggering of port scanning and sequential access heuristics."""
        features = {
            "unique_dst_port_count": 65,
            "sequential_port_ratio": 0.85,
            "syn_count": 50,
            "ack_count": 5,
        }
        profile = self.evaluator.evaluate_state("host-101", 1, features)
        self.assertTrue(profile.is_sufficient_evidence)
        self.assertEqual(len(profile.evidence_items), 3)

        rule_ids = [item.rule_id for item in profile.evidence_items]
        self.assertIn("RULE-DISC-01", rule_ids)
        self.assertIn("RULE-DISC-02", rule_ids)
        self.assertIn("RULE-RECON-03", rule_ids)

    def test_dos_flood_evidence(self):
        """Tests triggering of volumetric flood heuristic."""
        features = {
            "packets_total": 5000,
            "syn_count": 1000,
            "ack_count": 10,
        }
        profile = self.evaluator.evaluate_state("host-102", 2, features)
        rule_ids = [item.rule_id for item in profile.evidence_items]
        self.assertIn("RULE-DOS-01", rule_ids)

    def test_exfiltration_evidence(self):
        """Tests triggering of asymmetric data exfiltration heuristic."""
        features = {
            "bidirectional_ratio": 0.95,
            "bytes_total": 2000000.0,
        }
        profile = self.evaluator.evaluate_state("host-103", 3, features)
        rule_ids = [item.rule_id for item in profile.evidence_items]
        self.assertIn("RULE-EXFIL-01", rule_ids)

    def test_malformed_and_nan_features(self):
        """Tests resilience to NaN, Inf, string, None values (XAI-17)."""
        malformed = {
            "unique_dst_port_count": float("nan"),
            "syn_count": float("inf"),
            "sequential_port_ratio": "invalid_string",
            "bidirectional_ratio": None,
        }
        profile = self.evaluator.evaluate_state("host-104", 4, malformed)
        self.assertIsInstance(profile, HostEvidenceProfile)
        self.assertFalse(profile.is_sufficient_evidence)

    def test_temporal_sequence_evaluation(self):
        """Tests multi-window temporal sequence evaluation ([10, F]) (XAI-09)."""
        num_windows = 10
        num_features = len(CANONICAL_FEATURES)
        matrix = np.zeros((num_windows, num_features))

        # Inject port scan burst at window index 7
        syn_idx = CANONICAL_FEATURES.index("syn_count")
        port_idx = CANONICAL_FEATURES.index("unique_dst_port_count")
        matrix[7, syn_idx] = 40.0
        matrix[7, port_idx] = 30.0

        profile = self.evaluator.evaluate_temporal_sequence(
            host_id="host-105",
            temporal_features=matrix,
            feature_names=CANONICAL_FEATURES,
            base_window_id=100,
        )

        self.assertTrue(profile.is_sufficient_evidence)
        self.assertGreaterEqual(len(profile.evidence_items), 2)
        offsets = [item.window_offset for item in profile.evidence_items]
        self.assertIn(7, offsets)


if __name__ == "__main__":
    unittest.main()
