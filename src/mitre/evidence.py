"""
MITRE Evidence Rules & Extraction Engine.
Owner: Srijani (Cybersecurity + MITRE + Explainability)

This module extracts structured cybersecurity evidence from host state vectors
and temporal sequences (e.g. 10 temporal windows x canonical features) following
defensible, evidence-grounded rules.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


# Canonical feature names aligning with upstream pipeline (Aman / Shaurya)
CANONICAL_FEATURES: List[str] = [
    "syn_count",
    "ack_count",
    "unique_dst_port_count",
    "unique_dst_ip_count",
    "sequential_port_ratio",
    "flow_count",
    "packets_total",
    "packet_count",
    "bytes_total",
    "bytes_mean",
    "duration_mean",
    "iat_mean",
    "iat_std",
    "packet_iat_std",
    "ttl_mean",
    "ttl_std",
    "tcp_window_mean",
    "tcp_window_std",
    "fragment_count",
    "retransmission_count",
    "bidirectional_ratio",
]


@dataclass
class EvidenceItem:
    rule_id: str
    description: str
    metric: str
    observed_value: float
    threshold: float
    confidence_weight: float
    associated_tactics: List[str]
    candidate_techniques: List[str]
    window_offset: Optional[int] = None  # Offset in temporal sequence (e.g. window index 0-9)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "metric": self.metric,
            "observed_value": round(self.observed_value, 4),
            "threshold": self.threshold,
            "confidence_weight": round(self.confidence_weight, 4),
            "associated_tactics": self.associated_tactics,
            "candidate_techniques": self.candidate_techniques,
            "window_offset": self.window_offset,
        }


@dataclass
class HostEvidenceProfile:
    host_id: str
    window_id: int
    timestamp: Optional[str] = None
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    behavior_summary: List[str] = field(default_factory=list)
    is_sufficient_evidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_id": self.host_id,
            "window_id": self.window_id,
            "timestamp": self.timestamp,
            "evidence": [item.description for item in self.evidence_items],
            "evidence_details": [item.to_dict() for item in self.evidence_items],
            "behavior_summary": self.behavior_summary,
            "is_sufficient_evidence": self.is_sufficient_evidence,
            "rules_matched_count": len(self.evidence_items),
        }


class EvidenceRuleEvaluator:
    """
    Evaluates host state features and temporal windows against rigorous cybersecurity evidence rules.
    Prevents arbitrary labeling by requiring concrete, measurable indicators.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            "syn_count_high": 20.0,
            "syn_ack_ratio_high": 4.0,
            "unique_dst_ports_high": 15.0,
            "sequential_port_ratio_high": 0.60,
            "unique_dst_ip_high": 10.0,
            "packets_total_flood": 1000.0,
            "bytes_total_burst": 500000.0,
            "ttl_std_spoofing": 15.0,
            "retransmission_high": 10.0,
            "outbound_ratio_high": 0.85,
            "iat_std_beaconing_low": 0.10,
            "rst_count_high": 25.0,
            "duration_mean_short": 0.05,
        }

    def _sanitize_float(self, val: Any, default: float = 0.0) -> float:
        """Sanitizes inputs, guarding against NaN, Inf, None, or string types."""
        if val is None:
            return default
        try:
            num = float(val)
            if math.isnan(num) or math.isinf(num):
                return default
            return num
        except (ValueError, TypeError):
            return default

    def evaluate_state(
        self,
        host_id: str,
        window_id: int,
        state_features: Dict[str, Any],
        timestamp: Optional[str] = None,
        window_offset: Optional[int] = None,
    ) -> HostEvidenceProfile:
        """
        Extracts evidence items from a single host state dictionary.
        Handles missing keys and invalid numeric values safely.
        """
        profile = HostEvidenceProfile(
            host_id=str(host_id) if host_id else "unknown-host",
            window_id=int(window_id) if window_id is not None else 0,
            timestamp=timestamp,
        )

        # Rule 1: High Destination Port Diversity (Port Scanning)
        unique_ports = self._sanitize_float(
            state_features.get("unique_dst_port_count", state_features.get("unique_dst_ports", 0.0))
        )
        if unique_ports >= self.thresholds["unique_dst_ports_high"]:
            profile.evidence_items.append(
                EvidenceItem(
                    rule_id="RULE-DISC-01",
                    description=f"High unique destination-port count ({int(unique_ports)} ports in window)",
                    metric="unique_dst_port_count",
                    observed_value=unique_ports,
                    threshold=self.thresholds["unique_dst_ports_high"],
                    confidence_weight=0.35,
                    associated_tactics=["TA0007", "TA0043"],  # Discovery, Reconnaissance
                    candidate_techniques=["T1046"],  # Network Service Discovery
                    window_offset=window_offset,
                )
            )

        # Rule 2: Sequential Destination Port Access Pattern
        seq_port_ratio = self._sanitize_float(state_features.get("sequential_port_ratio", 0.0))
        if seq_port_ratio >= self.thresholds["sequential_port_ratio_high"]:
            profile.evidence_items.append(
                EvidenceItem(
                    rule_id="RULE-DISC-02",
                    description=f"Sequential destination-port access pattern (score: {seq_port_ratio:.2f})",
                    metric="sequential_port_ratio",
                    observed_value=seq_port_ratio,
                    threshold=self.thresholds["sequential_port_ratio_high"],
                    confidence_weight=0.30,
                    associated_tactics=["TA0007", "TA0043"],
                    candidate_techniques=["T1046"],
                    window_offset=window_offset,
                )
            )

        # Rule 3: Elevated SYN Packet Activity (Unanswered Probes / SYN Flood)
        syn_count = self._sanitize_float(state_features.get("syn_count", 0.0))
        ack_count = self._sanitize_float(state_features.get("ack_count", 1.0), default=1.0)
        syn_ack_ratio = syn_count / max(ack_count, 1.0)

        if syn_count >= self.thresholds["syn_count_high"]:
            profile.evidence_items.append(
                EvidenceItem(
                    rule_id="RULE-RECON-03",
                    description=f"Elevated SYN packet ratio without handshake completion (SYN: {int(syn_count)}, SYN/ACK: {syn_ack_ratio:.1f})",
                    metric="syn_count",
                    observed_value=syn_count,
                    threshold=self.thresholds["syn_count_high"],
                    confidence_weight=0.25,
                    associated_tactics=["TA0043", "TA0040"],  # Reconnaissance / Impact
                    candidate_techniques=["T1046", "T1498"],  # Discovery or DoS
                    window_offset=window_offset,
                )
            )

        # Rule 4: Volumetric Flood / DoS Indicators
        packets_total = self._sanitize_float(
            state_features.get("packets_total", state_features.get("packet_count", 0.0))
        )
        if packets_total >= self.thresholds["packets_total_flood"] and syn_ack_ratio > self.thresholds["syn_ack_ratio_high"]:
            profile.evidence_items.append(
                EvidenceItem(
                    rule_id="RULE-DOS-01",
                    description=f"Volumetric packet surge ({int(packets_total)} packets/10s) with disproportionate SYN ratio",
                    metric="packets_total",
                    observed_value=packets_total,
                    threshold=self.thresholds["packets_total_flood"],
                    confidence_weight=0.40,
                    associated_tactics=["TA0040"],  # Impact
                    candidate_techniques=["T1498"],  # Network Denial of Service
                    window_offset=window_offset,
                )
            )

        # Rule 5: High-Frequency Connection Attempts (Brute Force Probing)
        flow_count = self._sanitize_float(state_features.get("flow_count", 0.0))
        retrans = self._sanitize_float(state_features.get("retransmission_count", 0.0))
        if flow_count >= 80.0 and retrans >= self.thresholds["retransmission_high"]:
            profile.evidence_items.append(
                EvidenceItem(
                    rule_id="RULE-AUTH-01",
                    description=f"High-frequency connection retries ({int(flow_count)} flows, {int(retrans)} retransmissions)",
                    metric="flow_count",
                    observed_value=flow_count,
                    threshold=80.0,
                    confidence_weight=0.35,
                    associated_tactics=["TA0006", "TA0001"],  # Credential Access / Initial Access
                    candidate_techniques=["T1110"],  # Brute Force
                    window_offset=window_offset,
                )
            )

        # Rule 6: TTL Variance & Potential IP Spoofing
        ttl_std = self._sanitize_float(state_features.get("ttl_std", 0.0))
        if ttl_std >= self.thresholds["ttl_std_spoofing"]:
            profile.evidence_items.append(
                EvidenceItem(
                    rule_id="RULE-EVAS-01",
                    description=f"High TTL variance (std: {ttl_std:.2f}) indicating multi-hop route anomaly or source spoofing",
                    metric="ttl_std",
                    observed_value=ttl_std,
                    threshold=self.thresholds["ttl_std_spoofing"],
                    confidence_weight=0.20,
                    associated_tactics=["TA0005"],  # Defense Evasion
                    candidate_techniques=["T1498", "T1046"],
                    window_offset=window_offset,
                )
            )

        # Rule 7: Asymmetric Outbound Exfiltration
        outbound_ratio = self._sanitize_float(state_features.get("bidirectional_ratio", 0.5), default=0.5)
        bytes_total = self._sanitize_float(state_features.get("bytes_total", 0.0))
        if outbound_ratio >= self.thresholds["outbound_ratio_high"] and bytes_total >= self.thresholds["bytes_total_burst"]:
            profile.evidence_items.append(
                EvidenceItem(
                    rule_id="RULE-EXFIL-01",
                    description=f"Asymmetric outbound byte flow ({bytes_total:,.0f} bytes, outbound ratio {outbound_ratio:.2f})",
                    metric="bidirectional_ratio",
                    observed_value=outbound_ratio,
                    threshold=self.thresholds["outbound_ratio_high"],
                    confidence_weight=0.35,
                    associated_tactics=["TA0010"],  # Exfiltration
                    candidate_techniques=["T1048"],  # Exfiltration Over Alternative Protocol
                    window_offset=window_offset,
                )
            )

        # Rule 8: Low IAT Jitter / Automated Beaconing
        iat_std = self._sanitize_float(
            state_features.get("iat_std", state_features.get("packet_iat_std", 1.0)), default=1.0
        )
        duration_mean = self._sanitize_float(state_features.get("duration_mean", 0.0))
        if iat_std <= self.thresholds["iat_std_beaconing_low"] and flow_count >= 15.0 and duration_mean > 1.0:
            profile.evidence_items.append(
                EvidenceItem(
                    rule_id="RULE-C2-01",
                    description=f"Strict periodic inter-arrival timing (IAT std: {iat_std:.3f}s) consistent with automated beaconing",
                    metric="iat_std",
                    observed_value=iat_std,
                    threshold=self.thresholds["iat_std_beaconing_low"],
                    confidence_weight=0.35,
                    associated_tactics=["TA0011"],  # Command and Control
                    candidate_techniques=["T1071"],  # Application Layer Protocol
                    window_offset=window_offset,
                )
            )

        # Build behavior summary & sufficiency check
        if profile.evidence_items:
            profile.behavior_summary = [item.description for item in profile.evidence_items]
            profile.is_sufficient_evidence = len(profile.evidence_items) >= 1
        else:
            profile.behavior_summary = ["Observed telemetry features remain within nominal baseline bounds."]
            profile.is_sufficient_evidence = False

        return profile

    def evaluate_temporal_sequence(
        self,
        host_id: str,
        temporal_features: Union[List[Dict[str, float]], np.ndarray],
        feature_names: Optional[List[str]] = None,
        base_window_id: int = 0,
        timestamp: Optional[str] = None,
    ) -> HostEvidenceProfile:
        """
        Evaluates a 10-window temporal sequence (shape: [10, num_features] or list of 10 dicts).
        Preserves temporal context by recording the specific window offset where each heuristic triggered.
        """
        combined_profile = HostEvidenceProfile(
            host_id=str(host_id) if host_id else "host-unknown",
            window_id=base_window_id,
            timestamp=timestamp,
        )

        windows: List[Dict[str, Any]] = []
        if isinstance(temporal_features, np.ndarray):
            names = feature_names or CANONICAL_FEATURES[: temporal_features.shape[-1]]
            for t_idx in range(temporal_features.shape[0]):
                w_dict = {names[i]: float(temporal_features[t_idx, i]) for i in range(min(len(names), temporal_features.shape[1]))}
                windows.append(w_dict)
        elif isinstance(temporal_features, list):
            for item in temporal_features:
                if isinstance(item, dict):
                    windows.append(item)
                elif isinstance(item, (list, tuple, np.ndarray)):
                    names = feature_names or CANONICAL_FEATURES[: len(item)]
                    w_dict = {names[i]: float(item[i]) for i in range(min(len(names), len(item)))}
                    windows.append(w_dict)

        # Evaluate across temporal sequence
        seen_rules = set()
        for idx, w_features in enumerate(windows):
            single_profile = self.evaluate_state(
                host_id=host_id,
                window_id=base_window_id + idx,
                state_features=w_features,
                timestamp=timestamp,
                window_offset=idx,
            )
            for item in single_profile.evidence_items:
                rule_key = (item.rule_id, item.metric)
                # Avoid duplicate identical rule logging unless from latest or highest intensity window
                if rule_key not in seen_rules:
                    seen_rules.add(rule_key)
                    combined_profile.evidence_items.append(item)

        if combined_profile.evidence_items:
            combined_profile.behavior_summary = [item.description for item in combined_profile.evidence_items]
            combined_profile.is_sufficient_evidence = True
        else:
            combined_profile.behavior_summary = ["No evidence heuristics triggered across the temporal observation window."]
            combined_profile.is_sufficient_evidence = False

        return combined_profile
