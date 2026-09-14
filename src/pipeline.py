"""
End-to-End Cyber Forecast, XAI & MITRE Attribution Pipeline.
Owner: Srijani (Cybersecurity + MITRE + Explainability)

This module defines the integration interface connecting:
Model Output (Sohini) -> XAI Explainer -> Evidence Extraction -> MITRE Candidate Mapping -> Recommendations
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from src.explain.shap_explain import ShapExplainer
from src.mitre.evidence import CANONICAL_FEATURES, EvidenceRuleEvaluator, HostEvidenceProfile
from src.mitre.mapping import MitreInterpretation, MitreMapper
from src.mitre.recommendations import RecommendationEngine


class CyberForecastPipeline:
    """
    Orchestrates real model outputs into defensible, multi-horizon XAI and MITRE interpretations.
    """

    def __init__(
        self,
        evidence_evaluator: Optional[EvidenceRuleEvaluator] = None,
        mitre_mapper: Optional[MitreMapper] = None,
        shap_explainer: Optional[ShapExplainer] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
    ):
        self.evaluator = evidence_evaluator or EvidenceRuleEvaluator()
        self.mapper = mitre_mapper or MitreMapper()
        self.explainer = shap_explainer or ShapExplainer()
        self.recommender = recommendation_engine or RecommendationEngine()

    def process_prediction(
        self,
        host_id: str,
        window_id: int,
        timestamp: Optional[str],
        current_risk: float,
        risk_timeline: Sequence[Union[float, Dict[str, float]]],
        predicted_stage: str,
        temporal_features: Union[List[Dict[str, float]], np.ndarray],
        feature_attributions: Union[Dict[int, Any], np.ndarray, List[Any]],
        feature_names: Optional[List[str]] = None,
        temporal_attribution_tensor: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end integration pipeline:
        1. XAI translation across 3 forecast horizons (+10s, +20s, +30s)
        2. Evidence heuristic extraction across temporal sequence
        3. Candidate MITRE mapping with explicit confidence & rationale
        4. Actionable defender recommendations
        """
        names = feature_names or CANONICAL_FEATURES

        # 1. Multi-horizon Explainability
        xai_result = self.explainer.explain_multi_horizon(
            risk_timeline=risk_timeline,
            attributions_by_horizon=feature_attributions,
            feature_names=names,
            top_k=5,
            temporal_attribution_tensor=temporal_attribution_tensor,
        )

        # 2. Evidence extraction across temporal input
        evidence_profile = self.evaluator.evaluate_temporal_sequence(
            host_id=host_id,
            temporal_features=temporal_features,
            feature_names=names,
            base_window_id=window_id,
            timestamp=timestamp,
        )

        # 3. Maximum forecasted risk across horizons
        risks = [h["predicted_risk"] for h in xai_result["horizons"]]
        max_forecast_risk = max(risks) if risks else current_risk

        # 4. MITRE Candidate Mapping
        mitre_result: MitreInterpretation = self.mapper.map_prediction_to_mitre(
            predicted_stage=predicted_stage,
            forecast_risk=max_forecast_risk,
            evidence_profile=evidence_profile,
            top_features=xai_result["top_features"],
        )

        # 5. SOC Recommendations
        recs = self.recommender.get_recommendations(
            mitre_technique=mitre_result.candidate_technique_id,
            risk_score=max_forecast_risk,
        )

        # 6. Assemble standardized output schema
        timeline_formatted = [
            {"offset_seconds": (idx + 1) * 10, "risk": round(float(r), 4)}
            for idx, r in enumerate(risks)
        ]

        # Explicitly separate model risk from mapping confidence
        return {
            "host_id": host_id,
            "window_id": window_id,
            "timestamp": timestamp,
            "forecast": {
                "current_risk": round(float(current_risk), 4),
                "predicted_risk_30s": round(float(risks[-1] if risks else current_risk), 4),
                "urgency_level": xai_result["primary_horizon"]["urgency_level"],
                "predicted_stage": predicted_stage,
                "risk_timeline": timeline_formatted,
            },
            "mitre_attack": mitre_result.to_dict(),
            "evidence": {
                "observed_heuristics": evidence_profile.behavior_summary,
                "is_sufficient_evidence": evidence_profile.is_sufficient_evidence,
                "rules_matched": len(evidence_profile.evidence_items),
                "evidence_details": [item.to_dict() for item in evidence_profile.evidence_items],
            },
            "explainability": xai_result,
            "recommendations": recs,
        }
