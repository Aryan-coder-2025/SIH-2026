# SIH26153 — Data Contract



## 1. Purpose



This document defines the canonical data representation used throughout

the SIH26153 network attack forecasting pipeline.



All data-processing, feature-engineering, model-training, evaluation,

and inference components must conform to this contract.



The purpose is to prevent:



\- inconsistent column names

\- inconsistent timestamps

\- accidental future-information leakage

\- inconsistent host/entity definitions

\- incompatible flow and packet feature representations

\- feature-order mismatches between training and inference



---



## 2. Canonical Temporal Contract



| Parameter | Value |

|---|---:|

| Window size | 10 seconds |

| Sequence length | 10 windows |

| Historical context | 100 seconds |

| Forecast horizon | 3 windows |

| Forecast offsets | +10s, +20s, +30s |

| Entity | Source host |



These values are defined centrally in:



`configs/project.yaml`



No individual module may redefine these values independently.



---



## 3. Definition of a Network State



A network state represents the observed behaviour of one source host

during one 10-second time window.



The canonical state unit is:



`one source\_host × one 10-second window`



Conceptually:



```text

State(t) = Features(source\_host, 10-second window t)

---



## 4. Entity Definition



The MVP entity is the source host.



Canonical entity definition:



`host\_id = src\_ip`



Every state must identify the source host responsible for the observed traffic.



---



## 5. Canonical Identifiers



The following fields may exist in raw or intermediate data:



| Field | Meaning |

|---|---|

| `src\_ip` | Source IP address |

| `dst\_ip` | Destination IP address |

| `src\_port` | Source port |

| `dst\_port` | Destination port |

| `protocol` | Network protocol |

| `flow\_id` | Raw flow identifier |

| `host\_id` | Canonical source-host entity |

| `timestamp` | Original observation timestamp |

| `window\_start` | Start of the 10-second window |

| `window\_end` | End of the 10-second window |

| `window\_id` | Deterministic window identifier |



These identifiers are used for grouping, joining, auditing, traceability, and visualization.



They are not automatically valid ML features.



---



## 6. Identifier Leakage Policy



Raw identifiers must not automatically be included in the core ML feature vector.



Identifier-sensitive fields include:



```text

src\_ip

dst\_ip

src\_port

dst\_port

flow\_id

