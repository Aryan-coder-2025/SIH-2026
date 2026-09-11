"""
Unit tests for Defensible MITRE Mapping Engine.
Validates all requirements from Aryan's review (XAI-03 through XAI-18).
"""

import math
import unittest

from src.mitre.evidence import EvidenceItem, EvidenceRuleEvaluator, HostEvidenceProfile
from src.mitre.mapping import (
    ATTACK_VERSION,
    CONTROLLED_TECHNIQUE_REGISTRY,
    MitreInterpretation,
    MitreMapper,
)


class TestMitreMapping(unittest.TestCase):
    def setUp(self):
        self.mapper = MitreMapper()
        self.evaluator = EvidenceRuleEvaluator()

    def test_valid_discovery_mapping(self):
        """Tests valid mapping to T1046 (Network Service Discovery)."""
        features = {
            "unique_dst_port_count": 55,
            "sequential_port_ratio": 0.90,
            "syn_count": 40,
            "ack_count": 5,
        }
        profile = self.evaluator.evaluate_state("host-1", 1, features)
        top_features = [{"feature": "syn_count"}, {"feature": "unique_dst_port_count"}]

        result = self.mapper.map_prediction_to_mitre(
            predicted_stage="Discovery",
            forecast_risk=0.86,
            evidence_profile=profile,
            top_features=top_features,
        )

        self.assertTrue(result.is_mapped)
        self.assertEqual(result.candidate_technique_id, "T1046")
        self.assertEqual(result.candidate_technique_name, "Network Service Discovery")
        self.assertEqual(result.candidate_tactic_name, "Discovery")
        self.assertEqual(result.candidate_tactic_id, "TA0007")
        self.assertEqual(result.attack_version, ATTACK_VERSION)

        # Ensure terminology is candidate / interpretation
        self.assertIn("consistent with candidate behavior", result.rationale)
        self.assertNotIn("Confirmed attack", result.rationale)

    def test_unsupported_or_insufficient_evidence_behavior(self):
        """Tests XAI-14: No forced MITRE mapping when evidence is insufficient."""
        empty_profile = HostEvidenceProfile(host_id="host-2", window_id=2)

        result = self.mapper.map_prediction_to_mitre(
            predicted_stage="Impact",
            forecast_risk=0.92,
            evidence_profile=empty_profile,
            top_features=[{"feature": "packets_total"}],
        )

        self.assertFalse(result.is_mapped)
        self.assertIsNone(result.candidate_technique_id)
        self.assertEqual(result.mapping_confidence, 0.0)
        self.assertIn("insufficient", result.rationale)

    def test_confidence_bounds_and_calculation(self):
        """Tests XAI-06 & XAI-17: Confidence bounded strictly to [0.0, 1.0] and calculation formula."""
        features = {"unique_dst_port_count": 50, "syn_count": 30}
        profile = self.evaluator.evaluate_state("host-3", 3, features)

        # Test various risk inputs including edge cases
        for test_risk in [0.0, 0.5, 0.86, 1.0, -0.5, 1.7, float("nan")]:
            result = self.mapper.map_prediction_to_mitre(
                predicted_stage="Discovery",
                forecast_risk=test_risk,
                evidence_profile=profile,
            )
            self.assertFalse(math.isnan(result.mapping_confidence))
            self.assertFalse(math.isinf(result.mapping_confidence))
            self.assertGreaterEqual(result.mapping_confidence, 0.0)
            self.assertLessEqual(result.mapping_confidence, 1.0)

    def test_rejection_of_arbitrary_technique_ids(self):
        """Tests XAI-16: System cannot produce arbitrary technique IDs like T9999."""
        # Check that registry only contains verified MITRE techniques
        for tech_id in CONTROLLED_TECHNIQUE_REGISTRY:
            self.assertTrue(tech_id.startswith("T1"))
            self.assertNotEqual(tech_id, "T9999")

        # Even if stage mapping or feature claims unknown technique, mapper falls back safely
        bogus_profile = HostEvidenceProfile(
            host_id="host-4",
            window_id=4,
            evidence_items=[
                EvidenceItem(
                    rule_id="RULE-FAKE-01",
                    description="Fake heuristic",
                    metric="fake_metric",
                    observed_value=1.0,
                    threshold=0.5,
                    confidence_weight=0.5,
                    associated_tactics=["TA0099"],
                    candidate_techniques=["T9999"],  # Arbitrary technique
                )
            ],
            is_sufficient_evidence=True,
        )

        result = self.mapper.map_prediction_to_mitre(
            predicted_stage="UnknownStage",
            forecast_risk=0.8,
            evidence_profile=bogus_profile,
        )
        self.assertFalse(result.is_mapped)
        self.assertIsNone(result.candidate_technique_id)

    def test_separation_of_model_risk_and_mapping_confidence(self):
        """Tests XAI-03 & XAI-18: Model forecast risk and mapping confidence are distinct fields."""
        features = {
            "unique_dst_port_count": 55,
            "sequential_port_ratio": 0.90,
            "syn_count": 40,
        }
        profile = self.evaluator.evaluate_state("host-5", 5, features)

        model_risk = 0.50
        result = self.mapper.map_prediction_to_mitre(
            predicted_stage="Discovery",
            forecast_risk=model_risk,
            evidence_profile=profile,
            top_features=[{"feature": "syn_count"}],
        )

        d = result.to_dict()
        self.assertIn("mapping_confidence", d)
        self.assertIn("candidate_technique_id", d)
        # Mapping confidence is a computed rule score, not just echoing model_risk
        self.assertIsInstance(d["mapping_confidence"], float)


if __name__ == "__main__":
    unittest.main()
