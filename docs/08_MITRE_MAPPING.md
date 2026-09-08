# 08_MITRE_MAPPING.md

## Purpose

This document defines how our system converts observed network evidence into
candidate MITRE ATT&CK technique mappings. We do **not** claim a technique is
"proven" — every mapping is evidence-based, confidence-scored, and accompanied
by a rationale a defender/evaluator can independently check.

Reference: [MITRE ATT&CK](https://attack.mitre.org/) | [MITRE Enterprise Matrix](https://attack.mitre.org/matrices/enterprise/)

MITRE defines **tactics** as the adversary's objective ("why") and
**techniques** as the "how." Our mapping pipeline follows this chain:

```
Observed evidence
      ↓
Behaviour interpretation
      ↓
Candidate ATT&CK technique
      ↓
Confidence score
      ↓
Rationale
```

---

## Mapping Rule Format

Every mapping rule in this table follows this schema:

| Field | Description |
|---|---|
| `evidence` | List of concrete, measurable signals from flow/packet features |
| `behaviour` | Plain-language interpretation of what the host is doing |
| `mitre_tactic` | The ATT&CK tactic (the "why") |
| `mitre_technique` | The ATT&CK technique ID + name (the "how") |
| `confidence` | 0.0–1.0, based on how many/how strong the supporting evidence signals are |
| `rationale` | One-sentence justification a human reviewer can sanity-check |

---

## Candidate Mapping Table (Day 1 draft — to be expanded/refined once real feature data is available)

| ID | Evidence (feature signals) | Behaviour | Tactic | Technique | Confidence (example) | Rationale |
|---|---|---|---|---|---|---|
| MAP-001 | high `unique_dst_port_count`, sequential `dst_port` access pattern, elevated `syn_count` | Network service scanning | Discovery | T1046 — Network Service Discovery | 0.86 | Traffic pattern is consistent with systematic probing across multiple ports on the same/similar hosts |
| MAP-002 | high `unique_dst_ip_count` from single `src_ip`, low `bytes_mean` per connection, short `duration_mean` | Network sweep / host discovery | Discovery | T1018 — Remote System Discovery | 0.75 | Broad, shallow connections to many hosts suggest reconnaissance rather than data transfer |
| MAP-003 | elevated `retransmission_count`, abnormal `tcp_window_std`, high `packet_iat_std` | Possible evasion / unstable connection behaviour | Defense Evasion | T1562 (candidate — needs stronger evidence before inclusion) | 0.40 | Weak/ambiguous signal on its own; flagged for review, not a standalone claim |
| MAP-004 | high `bytes_total` outbound, sustained `duration`, low request/response symmetry (`bidirectional_ratio`) | Possible data exfiltration | Exfiltration | T1041 — Exfiltration Over C2 Channel | 0.55 | Sustained one-directional high-volume transfer is consistent with exfiltration but needs destination reputation context to strengthen |
| MAP-005 | repeated failed connection attempts (`rst_count` high), narrow `dst_port` range, short `iat_mean` | Possible brute-force attempt | Credential Access | T1110 — Brute Force | 0.70 | Rapid repeated connection attempts to a narrow service set is characteristic of credential guessing |

> ⚠️ **Note:** These are placeholder/example mappings for Day 1 based on the
> canonical feature names in `02_DATA_DICTIONARY.md`. They must be validated
> and re-scored once real `feature_matrix.parquet` data and SHAP outputs are
> available (Checkpoint 3).

---

## Confidence Scoring Guidance

Confidence is **not** a model-calibrated probability at this stage — it's a
rule-based heuristic reflecting evidence strength:

- **0.8–1.0** — Multiple independent strong signals align with the behaviour
- **0.5–0.79** — Some supporting signals, but missing corroborating context
- **0.0–0.49** — Weak/ambiguous signal; should not be surfaced as a primary claim in the dashboard without more evidence

---

## What We Will NOT Do

- Will not state a MITRE technique as fact without listing the supporting evidence
- Will not map a single weak feature spike directly to a technique
- Will not claim classification of an attack type the model was never trained on — only that elevated risk was detected on excluded categories (see `13_KNOWN_LIMITATIONS.md`)

---

## Open Items / TODO

- [ ] Validate mapping rules against real flow+packet feature distributions once `feature_matrix.parquet` exists
- [ ] Cross-reference confidence scores with SHAP-derived feature importances (see `09_EXPLAINABILITY.md`)
- [ ] Add tactic/technique coverage for additional CSE-CIC-IDS2018 attack categories (Botnet, DDoS, Web Attack, Infiltration, Heartbleed)
- [ ] Get team sign-off (Aryan + Sohini) on final mapping table before freeze

---

**Owner:** Srijani

