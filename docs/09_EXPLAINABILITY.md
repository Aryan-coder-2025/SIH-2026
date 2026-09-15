# 09 — Explainability & Attribution Translation Protocol

## Overview
This document defines the protocol for consuming and translating **model feature attributions** (e.g. SHAP values, Integrated Gradients, or Temporal Attention weights) and temporal telemetry into **human-interpretable, multi-horizon SOC explanations**.

- **Protocol Version:** `2026.09.15-r2`
- **Component Owner:** Srijani (Cybersecurity + MITRE + Explainability)
- **Primary Engine Class:** `AttributionExplainer` (with `ShapExplainer` alias for backwards compatibility)

> [!CAUTION]
> **Anti-Pattern & Scientific Transparency Guidelines:**
> 1. NEVER expose raw floating-point arrays (e.g. `[-0.043, 0.281, ...]`) directly to SOC dashboards.
> 2. **Attribution Interpretation:** The module acts as an attribution *consumer and translation layer*. It translates upstream attributions into ranked, directional, human-readable insights.
> 3. **Non-Causal Guardrail:** A feature attribution indicates that the model *associated* the feature with its risk prediction; it does not constitute causal proof that network behavior was malicious.

---

## 1. Multi-Horizon Explainability Architecture

Our temporal forecasting model outputs predictions across three distinct horizons:
- **$+10\text{s}$ Forecast:** Near-term initiation risk
- **$+20\text{s}$ Forecast:** Intermediate propagation risk
- **$+30\text{s}$ Forecast:** Maximum forecasted escalation risk

The explainability engine supports **per-horizon explanations** ($+10\text{s} \rightarrow$ risk + top features, $+20\text{s} \rightarrow$ risk + top features, $+30\text{s} \rightarrow$ risk + top features) as well as primary aggregate summaries.

> [!IMPORTANT]
> **Strict Multi-Horizon Contract (Auditor Requirements S4 & S5):**
> - All 3 horizons ($+10\text{s}, +20\text{s}, +30\text{s}$) must be explicitly supplied.
> - **No Silent Padding:** Missing horizons are rejected; the system never manufactures artificial forecasts.
> - **No Cross-Horizon Substitution:** Each horizon requires its own attribution vector; silent fallback across horizons is prohibited.

---

## 2. Temporal Context Preservation ($10\text{ Windows} \times F$)

Inputs to the forecasting model consist of temporal sequences ($10\text{ windows} \times 41\text{ canonical features}$).

The XAI engine computes the most impactful historical window for each top contributing feature:
- Window index ($0 \dots 9$)
- Relative time offset (e.g. `-20s`, `-10s`, `0s (current window)`)

Example analyst bullet point:
`1. TCP SYN Activity ↑ (+0.35) [window: -10s]`

---

## 3. Analyst Display Mapping & Directionality

| Canonical Feature Name | Display Name | Positive Direction ($\uparrow$) | Negative Direction ($\downarrow$) |
| :--- | :--- | :--- | :--- |
| `syn_count` | TCP SYN Activity | Elevated unanswered connection initiation attempts (probing / SYN flood indicator) | Low connection initiation rate |
| `unique_dst_port_count` | Destination Port Diversity | Systematic probing pattern across multiple target destination ports | Traffic focused on single port |
| `sequential_port_ratio` | Sequential Port Scanning Pattern | High ratio of ordered port probing detected in connection stream | Ports accessed in non-sequential order |
| `flow_count` | Connection Frequency | Rapid surge in total connections per window | Steady, nominal connection rate |
| `packets_total` | Total Packets | Volumetric burst of network packets | Nominal packet volume |
| `bytes_total` | Total Byte Volume | Elevated data transfer volume | Low data payload footprint |
| `ttl_std` | TTL Variation | Unstable TTL variance indicative of potential multi-path routing / spoofing | Consistent routing path observed |
| `iat_std` | Inter-Arrival Jitter | Irregular burst patterns in packet timing | Strictly periodic pacing (potential beaconing indicator) |
| `bidirectional_ratio` | Outbound-to-Inbound Ratio | Asymmetric outbound data transfer (potential exfiltration indicator) | Balanced symmetric request-response traffic |
| `retransmission_count` | TCP Retransmissions | Elevated packet loss / reset activity observed | Clean packet delivery |

---

## 4. Input Validation & Numeric Sanitization

The XAI engine enforces strict validation:
- Risks are bounded strictly to $[0.0, 1.0]$.
- `NaN`, `Inf`, and out-of-bounds numeric inputs are sanitized for dashboard stability while preserving strict contract enforcement for missing horizons.
- Top-$K$ ranking is evaluated by absolute attribution magnitude $|\phi_i|$.
