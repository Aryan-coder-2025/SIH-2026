# SIH26153 — Temporal Architecture Specification

## 1. Scope & Objective
This specification formalizes the temporal sequence extraction, host isolation, continuity validation, and leakage-safe temporal partitioning rules implemented in the Aryan architecture branch.

It defines the foundation against which upstream data processing (Shaurya, Aman) and downstream model training (Sohini), XAI (Srijani), and evaluation (Ankit) integrate.

---

## 2. Canonical Temporal Parameters
The single source of truth is `configs/project.yaml`.

| Parameter | Symbol / Field | Canonical Value | Meaning |
| :--- | :--- | :--- | :--- |
| **Window Size** | `window_seconds` | `10` seconds | Duration of one network traffic state |
| **Sequence Length** | `sequence_length_windows` | `10` windows | Historical observation context (100 seconds total) |
| **Forecast Horizons** | `forecast_horizon_windows` | `3` steps | Multi-step forecast steps (1, 2, 3) |
| **Forecast Offsets** | `forecast_offsets_seconds` | `+10s, +20s, +30s` | Physical future lead times predicted |
| **Entity Key** | `src_ip` / `source_host` | Canonical string | Host-level boundary for temporal grouping |

---

## 3. Host Isolation Invariant
- **Rule**: Every sequence $X_i$ of shape `(10, num_features)` and target vector $y_i$ of shape `(3,)` belongs strictly to **one** `source_host`.
- **Enforcement**:
  - Raw records are grouped by `source_host`.
  - Sliding windows never cross host boundaries.
  - A sequence never contains observations from multiple hosts.
  - Host targets never contaminate other hosts (e.g., Host A malicious activity never assigns targets to Host B).

---

## 4. Physical Timestamp & Continuity Policy
Real-world network monitoring cannot assume sequential rows represent contiguous 10-second intervals.

### 4.1 Chronological Sorting
- Within each host group, records are sorted deterministically in strictly ascending chronological order:
  $$\forall i < j: \quad t_i \le t_j$$

### 4.2 Duplicate Window Policy
- If multiple records exist for the identical `(source_host, timestamp)`, the pipeline raises an explicit `ValueError`.
- Silent overwrite or arbitrary deduplication is strictly prohibited.

### 4.3 Discontinuities & Gap Handling
- The physical time delta between consecutive observations for a host is:
  $$\Delta t = t_i - t_{i-1}$$
- If $\Delta t = 10\text{s}$: windows are contiguous.
- If $\Delta t > 10\text{s}$: a temporal gap exists (e.g., host was idle, network paused).
- **Enforcement**:
  - Gaps are **never silently compressed** into consecutive steps.
  - The host timeline is segmented into strictly contiguous sub-blocks where every consecutive step is exactly $10\text{s}$.
  - Sequences are extracted only within contiguous sub-blocks.
  - If a sub-block has length $< \text{history\_length} + \text{max\_horizon}$ (i.e. $< 13$ windows), no sequence is formed; windows are accounted for and no invalid sequences or fabricated targets are created.
  - In `strict_continuity=True` mode, any discontinuity immediately raises `ValueError`.

---

## 5. Temporal Splitting & Leakage Prevention Policy

### 5.1 Split Before Sequence Construction
- **Principle**: Train / validation / test splits must occur **before** sequence construction.
- Random train/test splitting (such as `sklearn.model_selection.train_test_split`) is **strictly forbidden**.

### 5.2 Global Synchronized Timestamp Splitting
- All hosts are partitioned at the exact same physical time cutoff:
  - Train Partition: $\{ \text{record} \mid t < T_{\text{train\_cutoff}} \}$
  - Validation Partition: $\{ \text{record} \mid T_{\text{train\_cutoff}} \le t < T_{\text{val\_cutoff}} \}$
  - Test Partition: $\{ \text{record} \mid t \ge T_{\text{val\_cutoff}} \}$
- Synchronous splitting ensures that evaluation strictly models forecasting into the future across the entire monitored network.

### 5.3 Boundary Leakage & History Policy
- **Train Partition**:
  - In the training set, sequences are constructed such that all target horizons ($T + 10s, T + 20s, T + 30s$) fall strictly **before** $T_{\text{train\_cutoff}}$.
  - No training sequence can observe or predict across the train boundary into validation/test.
- **Evaluation / Test Partition**:
  - *Strict Partition Policy*: Test sequences use only observations $\ge T_{\text{train\_cutoff}}$.
  - *Causal Boundary History Policy* (Under scientific review): To evaluate the very first test window ($T_{\text{train\_cutoff}} + 10s$), the last 10 historical windows prior to $T_{\text{train\_cutoff}}$ provide historical input context. Because these historical windows occurred in the past, no future target leaks into history.

---

## 6. Numerical & Schema Integrity
- Features and targets are validated for finiteness:
  $$\text{NaN} \notin \{X, y\}, \quad \pm\infty \notin \{X, y\}$$
- Empty datasets produce controlled empty sequence arrays with correct tensor dimensions (`(0, 10, F)` and `(0, 3)`).
- Input sorting and host grouping are deterministic, guaranteeing identical outputs across repeated executions.
