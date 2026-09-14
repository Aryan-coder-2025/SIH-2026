# 09 — Explainability & SHAP Translation Protocol

## Overview
This document defines the protocol for translating **SHAP attributions**, **gradient saliency**, and **temporal attention weights** into **human-interpretable, multi-horizon SOC explanations**.

- **Protocol Version:** `2026.09.10-r1`
- **Component Owner:** Srijani (Cybersecurity + MITRE + Explainability)

> [!CAUTION]
> **Anti-Pattern Warning:**
> NEVER expose raw floating-point arrays (e.g. `[-0.043, 0.281, ...]`) directly to SOC dashboards. All attributions must be resolved into:
> 1. Human-readable feature names
> 2. Impact directionality ($\uparrow$ drives risk higher / $\downarrow$ mitigates risk)
> 3. Temporal window context (which historical observation window contributed most)
> 4. Per-horizon breakdowns ($+10\text{s}$, $+20\text{s}$, $+30\text{s}$)

---

## 1. Multi-Horizon Explainability Architecture

Our temporal forecasting model outputs predictions across three separate horizons:
- **$+10\text{s}$ Forecast:** Near-term initiation risk
- **$+20\text{s}$ Forecast:** Intermediate propagation risk
- **$+30\text{s}$ Forecast:** Maximum forecasted escalation risk

The explainability engine supports **per-horizon explanations** ($+10\text{s} \rightarrow$ risk + top features, $+20\text{s} \rightarrow$ risk + top features, $+30\text{s} \rightarrow$ risk + top features) as well as primary aggregate summaries.

---

## 2. Temporal Context Preservation ($10\text{ Windows} \times F$)

Inputs to the forecasting model consist of temporal sequences ($10\text{ windows} \times F\text{ canonical features}$).

The XAI engine computes the most impactful historical window for each top contributing feature:
- Window index ($0 \dots 9$)
- Relative time offset (e.g. `-20s`, `-10s`, `0s (current window)`)

Example analyst bullet point:
`1. TCP SYN Activity ↑ (+0.35) [window: -10s]`

---

## 3. Analyst Display Mapping & Directionality

| Canonical Feature Name | Display Name | Positive Direction ($\uparrow$) | Negative Direction ($\downarrow$) |
| :--- | :--- | :--- | :--- |
| `syn_count` | TCP SYN Activity | Elevated unanswered connection attempts (probing / SYN flood) | Low connection initiation rate |
| `unique_dst_port_count` | Destination Port Diversity | Systematic probing across multiple target destination ports | Traffic focused on single port |
| `sequential_port_ratio` | Sequential Port Scanning Pattern | High ratio of ordered port probing detected | Ports accessed in non-sequential order |
| `flow_count` | Connection Frequency | Rapid surge in total connections per window | Steady, normal connection rate |
| `packets_total` | Total Packets | Volumetric burst of network packets | Nominal packet volume |
| `bytes_total` | Total Byte Volume | Large data transfer volume | Low data payload footprint |
| `ttl_std` | TTL Variation | Unstable TTL variance indicative of potential source spoofing | Consistent routing path observed |
| `iat_std` | Inter-Arrival Jitter | Irregular burst patterns | Strictly periodic pacing (potential C2 beacon) |
| `bidirectional_ratio` | Outbound-to-Inbound Ratio | Asymmetric outbound transfer (possible exfiltration) | Balanced symmetric request-response traffic |
| `retransmission_count` | TCP Retransmissions | Elevated packet loss / stealth reset attempts | Clean packet delivery |

---

## 4. Input Validation & Numeric Sanitization

The XAI engine enforces strict validation:
- Risks are bounded strictly to $[0.0, 1.0]$.
- `NaN`, `Inf`, `None`, and out-of-bounds numeric inputs are automatically sanitized to prevent dashboard failures.
- Top-$K$ ranking is evaluated by absolute attribution magnitude $|\phi_i|$.
