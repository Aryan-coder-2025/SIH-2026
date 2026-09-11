# 08 — MITRE ATT&CK Mapping Specification & Evidence Rules

## Overview
This document defines the scientific, evidence-grounded framework for translating forecasted network states and temporal telemetry into candidate **MITRE ATT&CK Enterprise Framework** techniques.

- **MITRE ATT&CK Framework Version:** `v15.1 Enterprise`
- **Mapping Specification Version:** `2026.09.10-r1`
- **Component Owner:** Srijani (Cybersecurity + MITRE + Explainability)

> [!IMPORTANT]
> **Core Architectural Principle (Interpretation Layer, NOT Attack Classifier):**
> $\text{Network Telemetry} \rightarrow \text{Risk Forecasting Model} \rightarrow \text{Elevated Future Risk} \rightarrow \text{XAI Feature Attributions} \rightarrow \text{Observed Evidence} \rightarrow \text{Candidate MITRE Technique}$
> 
> The system maps evidence to **Candidate Techniques** with explicit **Mapping Confidence** and **Defensible Rationales**. We never output unsupported claims like "Confirmed Attack" or "Attacker definitely performed X".

---

## 1. Controlled Technique Registry

The mapping engine enforces a strictly controlled registry of verified MITRE Enterprise techniques (arbitrary IDs like `T9999` are rejected):

| Technique ID | Technique Name | Tactic | Primary Indicators | Behavioral Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **T1046** | Network Service Discovery | Discovery (TA0007) | `unique_dst_port_count`, `sequential_port_ratio`, `syn_count` | Systematic sequential or distributed port probing |
| **T1110** | Brute Force | Credential Access (TA0006) | `flow_count`, `retransmission_count`, `duration_std` | Rapid repeated authentication attempts with high retry volume |
| **T1498** | Network Denial of Service | Impact (TA0040) | `packets_total`, `syn_count`, `bytes_total` | Severe volumetric packet surge attempting link saturation |
| **T1499** | Endpoint Denial of Service | Impact (TA0040) | `rst_count`, `flow_count`, `duration_mean` | Abnormal TCP connection resets / target resource starvation |
| **T1071** | Application Layer Protocol (C2) | Command and Control (TA0011) | `iat_std`, `duration_mean`, `bidirectional_ratio` | Strict periodic inter-arrival timing (automated C2 beaconing) |
| **T1048** | Exfiltration Over Alternative Protocol | Exfiltration (TA0010) | `bytes_total`, `bidirectional_ratio` | High outbound byte transfer volume exceeding baseline ratios |
| **T1190** | Exploit Public-Facing Application | Initial Access (TA0001) | `payload_mean`, `payload_max`, `fragment_count` | Anomalous payload sizes and packet fragmentation on public ports |

---

## 2. Canonical Feature Alignment

The mapping engine operates strictly on the canonical feature matrix produced upstream (Aman & Shaurya):
- **Port & IP Breadth:** `unique_dst_port_count`, `unique_dst_ip_count`, `sequential_port_ratio`
- **TCP Flag & Flow Dynamics:** `syn_count`, `ack_count`, `rst_count`, `flow_count`, `retransmission_count`
- **Volume & Timing:** `packets_total`, `bytes_total`, `duration_mean`, `iat_mean`, `iat_std`, `ttl_mean`, `ttl_std`, `bidirectional_ratio`

---

## 3. Deterministic Mapping Confidence Scoring

Mapping confidence $C_{\text{mapping}} \in [0.0, 1.0]$ is computed deterministically from observed evidence strength and SHAP feature alignment:

$$C_{\text{mapping}} = w_{\text{model}} \cdot P(\text{forecast\_risk}) + w_{\text{rule}} \cdot \left(\frac{\min(N_{\text{matched}}, 3)}{3}\right) + w_{\text{shap}} \cdot S_{\text{alignment}}$$

Where:
- $P(\text{forecast\_risk}) \in [0.0, 1.0]$: Maximum forecasted risk from the World Model.
- $N_{\text{matched}}$: Number of active heuristic evidence rules satisfied.
- $S_{\text{alignment}} \in [0.0, 1.0]$: Fraction of top SHAP features matching candidate technique indicators.
- Weights: $w_{\text{model}} = 0.40, w_{\text{rule}} = 0.35, w_{\text{shap}} = 0.25$.
- Result is bounded strictly: $C_{\text{mapping}} \in [0.05, 0.99]$.

> [!NOTE]
> **Semantic Separation:**
> - `forecast_risk`: Model prediction of future malicious risk escalation.
> - `mapping_confidence`: Rule-based attribution confidence that observed telemetry matches the candidate technique's behavioral signature.

---

## 4. Handling Unsupported / Unmapped Telemetry

If the forecast model predicts elevated risk, but observed telemetry does NOT satisfy minimum heuristic criteria for any technique:
1. `is_mapped = False`
2. `candidate_technique_id = None` (`UNMAPPED`)
3. `candidate_technique_name = "Unmapped Anomalous Telemetry"`
4. `mapping_confidence = 0.0`
5. `rationale = "Elevated malicious risk forecasted, but observed evidence is insufficient to defensibly map to a specific MITRE ATT&CK technique."`
6. Generic SOC investigation recommendations are provided rather than forcing an inaccurate MITRE technique.
