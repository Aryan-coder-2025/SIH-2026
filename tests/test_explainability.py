"""
Unit tests for Temporal Explainability & SHAP Translation Engine.
Validates requirements from Aryan's review (XAI-08, XAI-09, XAI-10, XAI-17).
"""

import math
import unittest
import numpy as np

from src.explain.shap_explain import (
    ExplainedFeature,
    HorizonExplanation,
    ShapExplainer,
)
from src.mitre.evidence import CANONICAL_FEATURES


class TestExplainability(unittest.TestCase):
    def setUp(self):
        self.explainer = ShapExplainer()

    def test_single_horizon_ranking_and_direction(self):
        """Tests that features are ranked by absolute magnitude with proper directionality."""
        attributions = {
            "syn_count": 0.35,
            "unique_dst_port_count": 0.25,
            "flow_count": -0.15,
            "bytes_total": 0.05,
            "ttl_std": 0.10,
        }

        h_expl = self.explainer.explain_horizon(
            attributions=attributions,
            horizon_seconds=30,
            predicted_risk=0.85,
            top_k=3,
        )

        self.assertEqual(h_expl.horizon_seconds, 30)
        self.assertEqual(h_expl.urgency_level, "HIGH")
        self.assertEqual(len(h_expl.top_features), 3)

        # Ranked: syn_count (0.35), unique_dst_port_count (0.25), flow_count (-0.15)
        self.assertEqual(h_expl.top_features[0].feature, "syn_count")
        self.assertEqual(h_expl.top_features[0].direction, "up")
        self.assertEqual(h_expl.top_features[2].feature, "flow_count")
        self.assertEqual(h_expl.top_features[2].direction, "down")

        # Check bullet points format
        self.assertIn("TCP SYN Activity ↑ (+0.35)", h_expl.readable_bullet_points[0])

    def test_multi_horizon_explanation(self):
        """Tests XAI-10: Three forecast horizons (+10s, +20s, +30s) separate interpretation."""
        risk_timeline = [
            {"offset_seconds": 10, "risk": 0.55},
            {"offset_seconds": 20, "risk": 0.72},
            {"offset_seconds": 30, "risk": 0.88},
        ]

        # Different attributions for each horizon
        attrs_by_horizon = {
            10: {"syn_count": 0.15, "flow_count": 0.10},
            20: {"syn_count": 0.25, "unique_dst_port_count": 0.20},
            30: {"syn_count": 0.35, "unique_dst_port_count": 0.30, "sequential_port_ratio": 0.20},
        }

        result = self.explainer.explain_multi_horizon(
            risk_timeline=risk_timeline,
            attributions_by_horizon=attrs_by_horizon,
        )

        self.assertIn("horizons", result)
        self.assertEqual(len(result["horizons"]), 3)
        self.assertEqual(result["horizons"][0]["horizon_seconds"], 10)
        self.assertEqual(result["horizons"][0]["urgency_level"], "MEDIUM")
        self.assertEqual(result["horizons"][2]["horizon_seconds"], 30)
        self.assertEqual(result["horizons"][2]["urgency_level"], "HIGH")

    def test_preservation_of_temporal_context(self):
        """Tests XAI-09: Temporal window context preservation (10 windows x F)."""
        num_windows = 10
        num_features = len(CANONICAL_FEATURES)
        # Synthetic temporal attribution matrix [10, F]
        temp_attr_matrix = np.zeros((num_windows, num_features))
        syn_idx = CANONICAL_FEATURES.index("syn_count")
        # Peak attribution occurred at window index 8 (which is 10s ago)
        temp_attr_matrix[8, syn_idx] = 0.42

        h_expl = self.explainer.explain_horizon(
            attributions={"syn_count": 0.42},
            horizon_seconds=30,
            predicted_risk=0.86,
            feature_names=CANONICAL_FEATURES,
            temporal_window_attributions=temp_attr_matrix,
        )

        self.assertEqual(len(h_expl.top_features), 1)
        feat = h_expl.top_features[0]
        self.assertEqual(feat.most_active_window, 8)
        self.assertEqual(feat.relative_time_offset, "-10s")

    def test_resilience_to_nan_inf_and_out_of_bounds(self):
        """Tests XAI-17: Numeric validation for NaN, Inf, and out-of-bounds risks."""
        attributions = {
            "syn_count": float("nan"),
            "flow_count": float("inf"),
            "unique_dst_port_count": 0.30,
        }

        h_expl = self.explainer.explain_horizon(
            attributions=attributions,
            horizon_seconds=30,
            predicted_risk=float("nan"),  # Invalid risk
        )

        self.assertFalse(math.isnan(h_expl.predicted_risk))
        self.assertEqual(h_expl.predicted_risk, 0.0)
        self.assertEqual(len(h_expl.top_features), 1)
        self.assertEqual(h_expl.top_features[0].feature, "unique_dst_port_count")


if __name__ == "__main__":
    unittest.main()
