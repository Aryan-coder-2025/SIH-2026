"""
Evidence-Grounded MITRE ATT&CK Mapping Engine.
Owner: Srijani (Cybersecurity + MITRE + Explainability)

This module maps observed evidence, predicted stage, and SHAP feature attributions
to defensible MITRE ATT&CK candidate techniques with quantitative rule confidence and rationales.

Architecture Principle:
Traffic -> Risk Forecasting Model -> Elevated Future Risk -> XAI Attribution -> Observed Evidence -> Candidate MITRE Mapping
(Interpretation layer, NOT an attack classifier).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from src.mitre.evidence import HostEvidenceProfile


ATTACK_VERSION: str = "v15.1"
MAPPING_VERSION: str = "2026.09.10-r1"
MAPPING_SOURCE: str = "MITRE Enterprise ATT&CK / Project Telemetry Heuristics"


@dataclass(frozen=True)
class TechniqueMetadata:
    technique_id: str
    technique_name: str
    tactic_id: str
    tactic_name: str
    description: str
    primary_indicators: List[str]
    default_rationale_template: str


# Strictly controlled MITRE ATT&CK Technique Registry (Prevents arbitrary technique injection like T9999)
CONTROLLED_TECHNIQUE_REGISTRY: Dict[str, TechniqueMetadata] = {
    "T1046": TechniqueMetadata(
        technique_id="T1046",
        technique_name="Network Service Discovery",
        tactic_id="TA0007",
        tactic_name="Discovery",
        description="Adversaries may attempt to get a listing of services running on remote hosts to identify vulnerable services or attack surface.",
        primary_indicators=["unique_dst_port_count", "sequential_port_ratio", "syn_count", "unique_dst_ports"],
        default_rationale_template="Traffic pattern shows systematic multi-port probing and elevated SYN activity consistent with port/service scanning.",
    ),
    "T1110": TechniqueMetadata(
        technique_id="T1110",
        technique_name="Brute Force",
        tactic_id="TA0006",
        tactic_name="Credential Access",
        description="Adversaries may use brute force techniques to attempt authentication on network services.",
        primary_indicators=["flow_count", "retransmission_count", "duration_std"],
        default_rationale_template="High-frequency connection attempts targeting standard authentication endpoints with rapid retry cycles.",
    ),
    "T1498": TechniqueMetadata(
        technique_id="T1498",
        technique_name="Network Denial of Service",
        tactic_id="TA0040",
        tactic_name="Impact",
        description="Adversaries may perform Network Denial of Service attacks to degrade or block the availability of targeted services.",
        primary_indicators=["packets_total", "syn_count", "bytes_total", "packet_count"],
        default_rationale_template="Severe volumetric surge in packet rate and high SYN ratio indicating an active or impending network flood.",
    ),
    "T1499": TechniqueMetadata(
        technique_id="T1499",
        technique_name="Endpoint Denial of Service",
        tactic_id="TA0040",
        tactic_name="Impact",
        description="Adversaries may target specific endpoint application services to exhaust resources.",
        primary_indicators=["rst_count", "flow_count", "duration_mean"],
        default_rationale_template="Abnormal connection termination flags and persistent incomplete sessions targeting host services.",
    ),
    "T1071": TechniqueMetadata(
        technique_id="T1071",
        technique_name="Application Layer Protocol (C2)",
        tactic_id="TA0011",
        tactic_name="Command and Control",
        description="Adversaries may communicate using application layer protocols to avoid detection and maintain persistent control.",
        primary_indicators=["iat_std", "duration_mean", "bidirectional_ratio", "packet_iat_std"],
        default_rationale_template="Low inter-arrival time jitter combined with periodic persistent sessions indicative of automated C2 beaconing.",
    ),
    "T1048": TechniqueMetadata(
        technique_id="T1048",
        technique_name="Exfiltration Over Alternative Protocol",
        tactic_id="TA0010",
        tactic_name="Exfiltration",
        description="Adversaries may steal data by transferring it over network protocols to external repositories.",
        primary_indicators=["bytes_total", "bidirectional_ratio", "payload_max"],
        default_rationale_template="Asymmetric outbound data transfer volume heavily exceeding normal baseline ratios.",
    ),
    "T1190": TechniqueMetadata(
        technique_id="T1190",
        technique_name="Exploit Public-Facing Application",
        tactic_id="TA0001",
        tactic_name="Initial Access",
        description="Adversaries may attempt to take advantage of a weakness in an Internet-facing computer or program.",
        primary_indicators=["payload_mean", "payload_max", "fragment_count"],
        default_rationale_template="Anomalous payload size distributions and fragmented packet sequences targeting public server endpoints.",
    ),
}

STAGE_TO_TECHNIQUE_MAP: Dict[str, List[str]] = {
    "Reconnaissance": ["T1046"],
    "Discovery": ["T1046"],
    "Initial Access": ["T1190", "T1110"],
    "Credential Access": ["T1110"],
    "Command and Control": ["T1071"],
    "Exfiltration": ["T1048"],
    "Impact": ["T1498", "T1499"],
    "Denial of Service": ["T1498", "T1499"],
}


@dataclass
class MitreInterpretation:
    is_mapped: bool
    candidate_technique_id: Optional[str]
    candidate_technique_name: Optional[str]
    candidate_tactic_name: Optional[str]
    candidate_tactic_id: Optional[str]
    observed_behaviour: str
    mapping_confidence: float  # Rule/evidence confidence in [0.0, 1.0], NOT model probability
    observed_evidence: List[str]
    rationale: str
    attack_version: str = ATTACK_VERSION
    mapping_version: str = MAPPING_VERSION

    # Aliases for backward-compatibility with dashboards
    @property
    def mitre_technique(self) -> str:
        return self.candidate_technique_id or "UNMAPPED"

    @property
    def technique_name(self) -> str:
        return self.candidate_technique_name or "Unmapped Anomalous Activity"

    @property
    def tactic(self) -> str:
        return self.candidate_tactic_name or "Unspecified"

    @property
    def tactic_id(self) -> str:
        return self.candidate_tactic_id or "TA0000"

    @property
    def behaviour(self) -> str:
        return self.observed_behaviour

    @property
    def confidence(self) -> float:
        return self.mapping_confidence

    @property
    def evidence(self) -> List[str]:
        return self.observed_evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_mapped": self.is_mapped,
            "candidate_technique_id": self.candidate_technique_id,
            "candidate_technique_name": self.candidate_technique_name,
            "candidate_tactic_name": self.candidate_tactic_name,
            "candidate_tactic_id": self.candidate_tactic_id,
            "observed_behaviour": self.observed_behaviour,
            "mapping_confidence": round(self.mapping_confidence, 4),
            "observed_evidence": self.observed_evidence,
            "rationale": self.rationale,
            "attack_version": self.attack_version,
            "mapping_version": self.mapping_version,
            # Legacy fields for UI compatibility
            "mitre": self.candidate_technique_id or "UNMAPPED",
            "technique_name": self.candidate_technique_name or "Unmapped Anomalous Activity",
            "tactic": self.candidate_tactic_name or "Unspecified",
            "tactic_id": self.candidate_tactic_id or "TA0000",
            "behaviour": self.observed_behaviour,
            "confidence": round(self.mapping_confidence, 2),
            "evidence": self.observed_evidence,
        }


class MitreMapper:
    """
    Translates model predictions, SHAP attributions, and extracted evidence into defensible MITRE mappings.
    Enforces candidate terminology and handles unknown/unsupported behaviors safely without forcing mappings.
    """

    def __init__(self, registry: Optional[Dict[str, TechniqueMetadata]] = None):
        self.registry = registry or CONTROLLED_TECHNIQUE_REGISTRY

    def _validate_risk(self, risk: Any) -> float:
        """Validates that risk is a finite float in [0.0, 1.0]."""
        if risk is None:
            return 0.0
        try:
            val = float(risk)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

    def map_prediction_to_mitre(
        self,
        predicted_stage: str,
        forecast_risk: float,
        evidence_profile: HostEvidenceProfile,
        top_features: Optional[List[Dict[str, Any]]] = None,
        min_evidence_rules: int = 1,
    ) -> MitreInterpretation:
        """
        Executes the defensible MITRE attribution pipeline:
        Observed Evidence -> Behaviour Interpretation -> Candidate ATT&CK Technique -> Mapping Confidence -> Rationale
        """
        clean_risk = self._validate_risk(forecast_risk)
        top_features = top_features or []
        top_feature_names: Set[str] = {f.get("feature", "") for f in top_features if isinstance(f, dict)}

        # Handle Unknown / Unsupported Behavior (XAI-14)
        # If no evidence heuristics triggered, do NOT force a MITRE technique!
        if not evidence_profile.evidence_items or len(evidence_profile.evidence_items) < min_evidence_rules:
            return MitreInterpretation(
                is_mapped=False,
                candidate_technique_id=None,
                candidate_technique_name="Unmapped Anomalous Telemetry",
                candidate_tactic_name=None,
                candidate_tactic_id=None,
                observed_behaviour="unclassified anomalous network telemetry",
                mapping_confidence=0.0,
                observed_evidence=[item.description for item in evidence_profile.evidence_items]
                if evidence_profile.evidence_items
                else ["Observed feature deviation did not satisfy heuristic threshold criteria for a specific technique."],
                rationale="Elevated malicious risk forecasted, but observed evidence is insufficient to defensibly map to a specific MITRE ATT&CK technique.",
            )

        # Step 1: Count candidate techniques proposed by evidence items
        technique_scores: Dict[str, float] = {}
        for item in evidence_profile.evidence_items:
            for tech in item.candidate_techniques:
                if tech in self.registry:
                    # Weight by rule confidence weight
                    technique_scores[tech] = technique_scores.get(tech, 0.0) + (item.confidence_weight * 2.0)

        # Consider stage candidates
        stage_candidates = STAGE_TO_TECHNIQUE_MAP.get(predicted_stage, [])
        for tech in stage_candidates:
            if tech in self.registry:
                technique_scores[tech] = technique_scores.get(tech, 0.0) + 1.0

        # Step 2: Select candidate with highest evidence & SHAP alignment
        best_tech_id: Optional[str] = None
        best_score = -1.0

        for tech_id, base_score in technique_scores.items():
            meta = self.registry[tech_id]
            # SHAP alignment bonus
            shap_matches = sum(1 for feat in meta.primary_indicators if feat in top_feature_names)
            total_score = base_score + (shap_matches * 1.5)
            if total_score > best_score:
                best_score = total_score
                best_tech_id = tech_id

        if best_tech_id is None or best_tech_id not in self.registry:
            return MitreInterpretation(
                is_mapped=False,
                candidate_technique_id=None,
                candidate_technique_name="Unsupported Network Behavior",
                candidate_tactic_name=None,
                candidate_tactic_id=None,
                observed_behaviour="unsupported anomalous pattern",
                mapping_confidence=0.0,
                observed_evidence=[item.description for item in evidence_profile.evidence_items],
                rationale="Telemetry does not match any registered MITRE ATT&CK technique signatures.",
            )

        meta = self.registry[best_tech_id]

        # Step 3: Compute deterministic rule/mapping confidence score
        # Formula: C_mapping = w_model * P(risk) + w_rule * (matched_rules / 3.0) + w_shap * (shap_aligned / 2.0)
        # Bounded strictly in [0.0, 1.0]
        matched_rules = min(len(evidence_profile.evidence_items), 3)
        shap_aligned = min(sum(1 for feat in meta.primary_indicators if feat in top_feature_names), 2)
        shap_score = (shap_aligned / 2.0) if top_feature_names else 0.5

        mapping_confidence = (0.40 * clean_risk) + (0.35 * (matched_rules / 3.0)) + (0.25 * shap_score)
        mapping_confidence = max(0.05, min(0.99, mapping_confidence))

        # Step 4: Behavior formulation (Candidate, not Confirmed)
        behaviour_map = {
            "T1046": "network service scanning",
            "T1110": "credential brute-forcing",
            "T1498": "volumetric network denial-of-service",
            "T1499": "application-layer resource starvation",
            "T1071": "command-and-control beaconing",
            "T1048": "asymmetric data exfiltration",
            "T1190": "public service exploit attempt",
        }
        behaviour = behaviour_map.get(meta.technique_id, "suspicious network activity")

        # Step 5: Construct defensible rationale
        evidence_list = [item.description for item in evidence_profile.evidence_items]
        rationale = (
            f"Observed network telemetry exhibits patterns consistent with candidate behavior '{behaviour}' "
            f"based on {len(evidence_list)} heuristic evidence rule(s). Primary indicators align with "
            f"MITRE ATT&CK {meta.technique_id} ({meta.technique_name}) in tactic {meta.tactic_name}."
        )

        return MitreInterpretation(
            is_mapped=True,
            candidate_technique_id=meta.technique_id,
            candidate_technique_name=meta.technique_name,
            candidate_tactic_name=meta.tactic_name,
            candidate_tactic_id=meta.tactic_id,
            observed_behaviour=behaviour,
            mapping_confidence=mapping_confidence,
            observed_evidence=evidence_list,
            rationale=rationale,
            attack_version=ATTACK_VERSION,
            mapping_version=MAPPING_VERSION,
        )
