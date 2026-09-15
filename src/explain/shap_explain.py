"""
Temporal Feature Attribution & XAI Translation Engine.
Owner: Srijani (Cybersecurity + MITRE + Explainability)

This module acts as an attribution consumer and translation layer. It translates
raw upstream model attributions (e.g. SHAP values, Integrated Gradients, or Temporal Attention)
and temporal sequence representations (10 windows x canonical features) into structured,
multi-horizon, human-readable SOC explanations.

Scientific & Architectural Boundary:
- This module consumes and interprets model attributions; it does not claim to compute raw SHAP.
- Feature attributions represent model association with predicted future risk, not direct causal proof.
- Strictly validates temporal invariants (10 windows x 41 canonical features, 3 forecast horizons).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from src.mitre.evidence import CANONICAL_FEATURES


# Canonical analyst terminology display mapping (scientifically framed as potential behavioral indicators)
FEATURE_DISPLAY_MAP: Dict[str, Dict[str, str]] = {
    "syn_count": {
        "display_name": "TCP SYN Activity",
        "up_desc": "Elevated unanswered connection initiation attempts (probing / SYN flood indicator)",
        "down_desc": "Low connection initiation rate",
    },
    "ack_count": {
        "display_name": "TCP ACK Activity",
        "up_desc": "High acknowledged session traffic",
        "down_desc": "Low completed connection handshake rate",
    },
    "unique_dst_port_count": {
        "display_name": "Destination Port Diversity",
        "up_desc": "Systematic probing pattern across multiple target destination ports",
        "down_desc": "Traffic focused on a single destination port",
    },
    "unique_dst_ports": {
        "display_name": "Unique Destination Ports",
        "up_desc": "Multi-port reconnaissance pattern observed in telemetry",
        "down_desc": "Target port distribution is narrow",
    },
    "unique_dst_ip_count": {
        "display_name": "Destination IP Breadth",
        "up_desc": "Horizontal network subnet scanning across multiple targets",
        "down_desc": "Traffic focused on a single target IP",
    },
    "sequential_port_ratio": {
        "display_name": "Sequential Port Scanning Pattern",
        "up_desc": "High ratio of ordered port probing detected in connection stream",
        "down_desc": "Ports accessed in non-sequential order",
    },
    "flow_count": {
        "display_name": "Connection Frequency",
        "up_desc": "Rapid surge in total connections per window",
        "down_desc": "Steady, nominal connection rate",
    },
    "packet_count": {
        "display_name": "Packet Volume",
        "up_desc": "High packet volume transmitting across interface",
        "down_desc": "Low packet density in window",
    },
    "packets_total": {
        "display_name": "Total Packets",
        "up_desc": "Volumetric burst of network packets",
        "down_desc": "Nominal packet volume",
    },
    "bytes_total": {
        "display_name": "Total Byte Volume",
        "up_desc": "Elevated data transfer volume",
        "down_desc": "Low data payload footprint",
    },
    "bytes_mean": {
        "display_name": "Mean Byte Size",
        "up_desc": "Elevated payload size per packet",
        "down_desc": "Minimal header-only packet sizes",
    },
    "ttl_mean": {
        "display_name": "Average Time-To-Live (TTL)",
        "up_desc": "Anomalous hop distance or external OS fingerprint",
        "down_desc": "Standard expected network TTL",
    },
    "ttl_std": {
        "display_name": "TTL Variation",
        "up_desc": "Unstable TTL variance indicative of potential multi-path routing / spoofing",
        "down_desc": "Consistent routing path observed",
    },
    "tcp_window_mean": {
        "display_name": "TCP Window Size",
        "up_desc": "Custom stack parameters observed in TCP options",
        "down_desc": "Standard operating system TCP window",
    },
    "tcp_window_std": {
        "display_name": "TCP Window Jitter",
        "up_desc": "Variable socket buffer sizing",
        "down_desc": "Uniform TCP window",
    },
    "duration_mean": {
        "display_name": "Flow Duration",
        "up_desc": "Long-lived persistent connection streams",
        "down_desc": "Short, ephemeral connection durations",
    },
    "iat_mean": {
        "display_name": "Packet Inter-Arrival Time",
        "up_desc": "Extended delay between packet bursts",
        "down_desc": "Rapid, low-latency packet transmission",
    },
    "iat_std": {
        "display_name": "Inter-Arrival Jitter",
        "up_desc": "Irregular burst patterns in packet timing",
        "down_desc": "Strictly periodic packet pacing (potential beaconing indicator)",
    },
    "packet_iat_std": {
        "display_name": "Packet IAT Jitter",
        "up_desc": "Irregular packet timing variance",
        "down_desc": "Uniform packet timing",
    },
    "bidirectional_ratio": {
        "display_name": "Outbound-to-Inbound Ratio",
        "up_desc": "Asymmetric outbound data transfer (potential exfiltration indicator)",
        "down_desc": "Balanced symmetric request-response traffic",
    },
    "retransmission_count": {
        "display_name": "TCP Retransmissions",
        "up_desc": "Elevated packet loss / reset activity observed",
        "down_desc": "Clean packet delivery with no retransmissions",
    },
    "fragment_count": {
        "display_name": "Fragmented Packets",
        "up_desc": "Packet fragmentation observed (potential evasion / MTU mismatch)",
        "down_desc": "Standard non-fragmented frames",
    },
}


@dataclass
class ExplainedFeature:
    feature: str
    display_name: str
    contribution: float
    direction: str  # "up" (↑) or "down" (↓)
    human_text: str
    most_active_window: Optional[int] = None  # Historical window index (0 to T-1)
    relative_time_offset: Optional[str] = None  # e.g., "-20s", "-10s", "0s (current)"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "feature": self.feature,
            "display_name": self.display_name,
            "contribution": round(self.contribution, 4),
            "direction": self.direction,
            "human_text": self.human_text,
        }
        if self.most_active_window is not None:
            d["most_active_window"] = self.most_active_window
        if self.relative_time_offset is not None:
            d["relative_time_offset"] = self.relative_time_offset
        return d


@dataclass
class HorizonExplanation:
    horizon_seconds: int
    predicted_risk: float
    urgency_level: str
    top_features: List[ExplainedFeature]
    readable_bullet_points: List[str]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "predicted_risk": round(self.predicted_risk, 4),
            "urgency_level": self.urgency_level,
            "top_features": [f.to_dict() for f in self.top_features],
            "readable_bullet_points": self.readable_bullet_points,
            "summary_text": self.summary_text,
        }


class AttributionExplainer:
    """
    Translates raw model feature attributions and multi-horizon predictions into
    temporal, human-interpretable SOC explanations.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        # Default strictly to the canonical model feature schema
        self.feature_names = feature_names or list(CANONICAL_FEATURES)

    def _sanitize_risk(self, risk: Any) -> float:
        """Sanitizes risk, preventing NaN / Inf / negative values."""
        if risk is None:
            return 0.0
        try:
            val = float(risk)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

    def _get_urgency_level(self, risk: float) -> str:
        if risk >= 0.70:
            return "HIGH"
        if risk >= 0.40:
            return "MEDIUM"
        return "LOW"

    def explain_horizon(
        self,
        attributions: Union[Dict[str, float], List[float], np.ndarray],
        horizon_seconds: int,
        predicted_risk: float,
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
        temporal_window_attributions: Optional[np.ndarray] = None,  # Shape: [T, F]
        strict_shape: bool = False,
    ) -> HorizonExplanation:
        """
        Generates explanation for a single prediction horizon (e.g. +10s, +20s, or +30s).
        Preserves temporal context if temporal_window_attributions is provided.
        """
        risk = self._sanitize_risk(predicted_risk)
        urgency = self._get_urgency_level(risk)
        names = feature_names or self.feature_names

        feature_val_map: Dict[str, float] = {}
        if isinstance(attributions, dict):
            for k, v in attributions.items():
                if not (math.isnan(v) or math.isinf(v)):
                    feature_val_map[k] = float(v)
        else:
            arr_np = np.asarray(attributions, dtype=float)
            if arr_np.ndim == 2:
                # 2D matrix [T, F]: aggregate across temporal windows to get feature attribution
                if temporal_window_attributions is None:
                    temporal_window_attributions = arr_np
                arr = np.sum(arr_np, axis=0)
            else:
                arr = arr_np.flatten()

            if strict_shape and len(arr) != len(names):
                raise ValueError(
                    f"Attribution vector length ({len(arr)}) does not match canonical feature count ({len(names)})"
                )
            for i, val in enumerate(arr[: len(names)]):
                if not (math.isnan(val) or math.isinf(val)):
                    feature_val_map[names[i]] = float(val)

        # Validate temporal window attribution dimensions if supplied
        if temporal_window_attributions is not None:
            if temporal_window_attributions.ndim != 2:
                raise ValueError(
                    f"Expected 2D temporal attribution matrix [T, F], got shape {temporal_window_attributions.shape}"
                )
            if strict_shape and temporal_window_attributions.shape[1] != len(names):
                raise ValueError(
                    f"Temporal attribution matrix features ({temporal_window_attributions.shape[1]}) does not match feature names ({len(names)})"
                )

        # Sort by absolute impact
        sorted_features = sorted(
            feature_val_map.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_k]

        explained_list: List[ExplainedFeature] = []
        readable_bullet_points: List[str] = []

        for rank, (feat, val) in enumerate(sorted_features, 1):
            mapping = FEATURE_DISPLAY_MAP.get(
                feat,
                {
                    "display_name": feat.replace("_", " ").title(),
                    "up_desc": f"Elevated {feat.replace('_', ' ')} value observed",
                    "down_desc": f"Reduced {feat.replace('_', ' ')} value observed",
                },
            )

            direction = "up" if val >= 0 else "down"
            dir_sym = "↑" if val >= 0 else "↓"
            human_text = mapping["up_desc"] if val >= 0 else mapping["down_desc"]

            most_active_w = None
            rel_offset = None
            if temporal_window_attributions is not None and feat in names:
                feat_idx = names.index(feat)
                if feat_idx < temporal_window_attributions.shape[1]:
                    t_magnitudes = np.abs(temporal_window_attributions[:, feat_idx])
                    most_active_w = int(np.argmax(t_magnitudes))
                    num_windows = temporal_window_attributions.shape[0]
                    sec_ago = (num_windows - 1 - most_active_w) * 10
                    rel_offset = f"-{sec_ago}s" if sec_ago > 0 else "0s (current window)"

            explained_feat = ExplainedFeature(
                feature=feat,
                display_name=mapping["display_name"],
                contribution=val,
                direction=direction,
                human_text=human_text,
                most_active_window=most_active_w,
                relative_time_offset=rel_offset,
            )
            explained_list.append(explained_feat)

            time_str = f" [window: {rel_offset}]" if rel_offset else ""
            readable_bullet_points.append(
                f"{rank}. {mapping['display_name']} {dir_sym} ({'+' if val >= 0 else ''}{val:.2f}){time_str}"
            )

        # Summary text
        top_names = [f.display_name for f in explained_list[:2]]
        top_str = " and ".join(top_names) if top_names else "general telemetry patterns"
        summary_text = (
            f"Forecasted risk at +{horizon_seconds}s ({urgency} {int(risk * 100)}%) is primarily driven by {top_str}."
        )

        return HorizonExplanation(
            horizon_seconds=horizon_seconds,
            predicted_risk=risk,
            urgency_level=urgency,
            top_features=explained_list,
            readable_bullet_points=readable_bullet_points,
            summary_text=summary_text,
        )

    def explain_multi_horizon(
        self,
        risk_timeline: Sequence[Union[float, Dict[str, float]]],
        attributions_by_horizon: Union[Dict[int, Any], np.ndarray, List[Any]],
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
        temporal_attribution_tensor: Optional[np.ndarray] = None,  # Shape: [3, T, F]
        strict_contract: bool = True,
    ) -> Dict[str, Any]:
        """
        Processes all 3 forecast horizons (+10s, +20s, +30s) and generates per-horizon
        explanations as well as a consolidated summary.
        
        Strict Contract:
        - Rejects missing horizons (no silent padding/manufactured forecasts).
        - Rejects missing horizon attributions (no dangerous fallback across horizons).
        """
        names = feature_names or self.feature_names
        horizon_seconds_list = [10, 20, 30]

        # Extract risks with strict length checking
        risks: List[float] = []
        for item in risk_timeline:
            if isinstance(item, dict):
                risks.append(self._sanitize_risk(item.get("risk", 0.0)))
            else:
                risks.append(self._sanitize_risk(item))

        if strict_contract and len(risks) < 3:
            raise ValueError(
                f"Multi-horizon forecast contract violation: expected 3 horizons (+10s, +20s, +30s), got {len(risks)}. "
                f"Silent padding of missing horizons is prohibited."
            )

        # Validate temporal attribution tensor if provided
        if temporal_attribution_tensor is not None:
            if temporal_attribution_tensor.ndim != 3:
                raise ValueError(
                    f"Expected 3D temporal attribution tensor [3, T, F], got shape {temporal_attribution_tensor.shape}"
                )
            if strict_contract and temporal_attribution_tensor.shape[0] != 3:
                raise ValueError(
                    f"Temporal attribution tensor first dimension must equal 3 (horizons), got {temporal_attribution_tensor.shape[0]}"
                )

        horizon_explanations: List[HorizonExplanation] = []

        for h_idx, h_sec in enumerate(horizon_seconds_list):
            h_attr = None
            if isinstance(attributions_by_horizon, dict):
                h_attr = attributions_by_horizon.get(h_sec, attributions_by_horizon.get(str(h_sec)))
            elif isinstance(attributions_by_horizon, (list, tuple, np.ndarray)):
                if len(attributions_by_horizon) > h_idx:
                    h_attr = attributions_by_horizon[h_idx]

            if h_attr is None:
                if strict_contract:
                    raise ValueError(
                        f"Missing attribution for forecast horizon +{h_sec}s. "
                        f"Silent reuse of other horizon attributions is prohibited."
                    )
                h_attr = attributions_by_horizon

            temp_win_attr = None
            if temporal_attribution_tensor is not None:
                if h_idx < temporal_attribution_tensor.shape[0]:
                    temp_win_attr = temporal_attribution_tensor[h_idx]

            h_expl = self.explain_horizon(
                attributions=h_attr,
                horizon_seconds=h_sec,
                predicted_risk=risks[h_idx] if h_idx < len(risks) else 0.0,
                feature_names=names,
                top_k=top_k,
                temporal_window_attributions=temp_win_attr,
                strict_shape=strict_contract,
            )
            horizon_explanations.append(h_expl)

        # Build primary (+30s or maximum risk) consolidated explanation
        max_h_expl = max(horizon_explanations, key=lambda x: x.predicted_risk)

        return {
            "primary_horizon": max_h_expl.to_dict(),
            "horizons": [h.to_dict() for h in horizon_explanations],
            "top_features": [f.to_dict() for f in max_h_expl.top_features],
            "readable_bullet_points": max_h_expl.readable_bullet_points,
            "summary_text": max_h_expl.summary_text,
        }

    # Backward-compatible wrapper
    def explain_prediction(
        self,
        shap_values: Union[List[float], np.ndarray, Dict[str, float]],
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
        risk_score: float = 0.86,
    ) -> Dict[str, Any]:
        """Backward-compatible single horizon explanation method."""
        h_expl = self.explain_horizon(
            attributions=shap_values,
            horizon_seconds=30,
            predicted_risk=risk_score,
            feature_names=feature_names,
            top_k=top_k,
        )
        return {
            "top_features": [f.to_dict() for f in h_expl.top_features],
            "readable_bullet_points": h_expl.readable_bullet_points,
            "summary_text": h_expl.summary_text,
        }


# Backward-compatibility alias
ShapExplainer = AttributionExplainer
