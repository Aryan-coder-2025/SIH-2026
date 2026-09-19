# Scientific Validation Report: SIH-26153

**Evaluation Notice:** `SYNTHETIC DEMONSTRATION ONLY`  
**Disclaimer:** `REAL DATA EXPERIMENT NOT EXECUTED - RAW CSE-CIC-IDS2018 TELEMETRY PENDING EXTRACTION`  
**Execution Timestamp:** 2026-09-18  
**Environment:** Python 3.11.9, PyTorch 2.11.0+cpu, Windows 11  

---

## 1. Executive Summary

This report documents the empirical evaluation of the **SIH-26153 Bi-Head Temporal WorldModel** against three competitive baselines across multi-horizon forecast windows (+10s, +20s, +30s).

All metrics in this report were generated using the **canonical integration fixture** (`data/processed/feature_matrix.parquet`), consisting of 1,440 continuous 10-second windows across 6 network hosts. Because raw multi-gigabyte CSE-CIC-IDS2018 telemetry is pending external extraction, these results validate **system correctness, pipeline mechanics, and baseline fairness**, but must **NOT** be cited as real-world benchmark metrics.

---

## 2. Dataset & Split Parameters

- **Total Windows:** 1,440 temporal windows
- **Temporal Splitting Protocol:** Chronological partition by physical timestamp (`train < val < test`)
  - **Train Split:** 1,008 windows (70%) $\to$ 936 temporal sequences of shape `(10, 41)`
  - **Validation Split:** 216 windows (15%) $\to$ 144 temporal sequences of shape `(10, 41)`
  - **Test Split:** 216 windows (15%) $\to$ 144 temporal sequences of shape `(10, 41)`
- **Class Distribution (Test Split):**
  - Positive (Malicious): 22 sequences (15.28%)
  - Negative (Benign): 122 sequences (84.72%)
  - **All-Negative Baseline Accuracy:** **84.72%** (Any model reporting ~85% accuracy without F1 is non-informative).

---

## 3. WorldModel vs. Fair Baselines Test Performance

All models were evaluated on the exact same held-out test split (144 sequences). Decision thresholds were tuned **strictly on the validation partition** and remained frozen during test evaluation.

### Multi-Horizon Test Metrics Table

| Forecast Horizon | Model Architecture | Inputs / History | Threshold (Val-Tuned) | Test Precision | Test Recall | Test F1 | Test FPR | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **+10s** | **WorldModel (Bi-Head LSTM)** | $10 \times 41$ sequence | 0.05 | 0.9130 | 0.9545 | 0.9333 | 0.0164 | **0.9974** | 0.9854 |
| | Logistic Regression | $410$ features (flattened) | 0.05 | 0.8333 | 0.9091 | 0.8696 | 0.0328 | 0.9970 | **0.9857** |
| | Random Forest | $410$ features (flattened) | 0.10 | **0.9167** | **1.0000** | **0.9565** | 0.0164 | 0.9944 | 0.9593 |
| | Persistence Baseline | State $y_{\text{curr}}$ at time $t$ | 0.05 | 0.9091 | 0.9091 | 0.9091 | 0.0164 | 0.9463 | 0.8403 |
| **+20s** | **WorldModel (Bi-Head LSTM)** | $10 \times 41$ sequence | 0.90 | **0.9412** | 0.7273 | 0.8205 | **0.0082** | 0.9885 | **0.9517** |
| | Logistic Regression | $410$ features (flattened) | 0.05 | 0.8333 | **0.9091** | **0.8696** | 0.0328 | **0.9892** | 0.9411 |
| | Random Forest | $410$ features (flattened) | 0.60 | 0.8333 | **0.9091** | **0.8696** | 0.0328 | 0.9424 | 0.8301 |
| | Persistence Baseline | State $y_{\text{curr}}$ at time $t$ | 0.05 | 0.8182 | 0.8182 | 0.8182 | 0.0328 | 0.8927 | 0.6972 |
| **+30s** | **WorldModel (Bi-Head LSTM)** | $10 \times 41$ sequence | 0.75 | **0.8947** | 0.7727 | **0.8293** | **0.0164** | **0.9881** | **0.9409** |
| | Logistic Regression | $410$ features (flattened) | 0.05 | 0.7500 | **0.8182** | 0.7826 | 0.0492 | 0.9817 | 0.8976 |
| | Random Forest | $410$ features (flattened) | 0.85 | 0.8095 | 0.7727 | 0.7907 | 0.0328 | 0.9311 | 0.7317 |
| | Persistence Baseline | State $y_{\text{curr}}$ at time $t$ | 0.05 | 0.7273 | 0.7273 | 0.7273 | 0.0492 | 0.8390 | 0.5706 |

### Baseline Fairness Observations
1. **History Fairness:** When classical models (LR and RF) receive equivalent flattened 10-window history ($10 \times 41 = 410$ features), their competitive performance increases significantly compared to single-window evaluation.
2. **Horizon Degradation:** All models experience decay in F1 score as the horizon increases from +10s to +30s. The Persistence baseline drops from F1 = 0.9091 at +10s to F1 = 0.7273 at +30s, demonstrating that static state broadcasting rapidly degrades into the future.
3. **WorldModel Lead:** At the farthest horizon (+30s), the LSTM WorldModel retains higher ROC-AUC (0.9881) and PR-AUC (0.9409) than Random Forest (PR-AUC: 0.7317) and Persistence (PR-AUC: 0.5706).

---

## 4. Controlled Modality & History Depth Ablation Study

Evaluated under identical chronological train/val/test splits using 8 training epochs and validation-tuned thresholds. Logged in `artifacts/ablation_results.json`.

### Modality Ablation (History = 10 Windows)
- **Flow-Only (22 features):** Mean Test F1 = **0.8097**
- **Packet-Only (19 features):** Mean Test F1 = **0.7915**
- **Multimodal Early Fusion (41 features):** Mean Test F1 = **0.8100**

*Finding:* Fusing flow dynamics and packet headers provides a marginal improvement (+0.0003 F1 over flow-only, +0.0185 F1 over packet-only) on the synthetic integration fixture.

### History Depth Ablation (Features = 41)
- **1 Window (10s history):** Mean Test F1 = **0.8614**
- **5 Windows (50s history):** Mean Test F1 = **0.8276**
- **10 Windows (100s history):** Mean Test F1 = **0.8100**

*Finding:* 10 windows is the frozen architectural contract for the temporal forecasting system; the current synthetic ablation does not establish that it is optimal. Specifically, the current synthetic ablation does not establish optimal history depth, as the synthetic demonstration fixture does not model complex multi-scale temporal dependencies.

---

## 5. Explainability (Integrated Gradients) Validation

- **Implemented Method:** Path-Integrated Gradients (Sundararajan et al., 2017) over 50 interpolation steps.
- **Completeness Sanity Check:**
  $$\sum_{i=1}^{41 \times 10} \text{Attribution}_i \approx F(x) - F(x')$$
  - The implementation validates completeness via `check_completeness(delta_target, attribution_sum, relative_tolerance=0.20)` against a 0.20 tolerance threshold (20% relative Riemann approximation bound). Across test sequences with 50 interpolation steps, relative completeness error is consistently $< 0.05$ (well below the 0.20 tolerance threshold).
- **Attribution Shape:** Preserves `(10, 41)` exactly, allowing per-window and per-feature attribution without dimensionality collapse.
- **SHAP Clarification:** Because the Python `shap` package is uninstalled in the host environment, SHAP sampling was **NOT executed** and is not claimed.

---

## 6. Cryptographic Audit Ledger Validation

- **Implemented Architecture:** SHA-256 block hash chaining for tamper-evident event logging.
- **Integrity Verification:**
  - Valid chain verification: **PASSED**
  - Modified payload injection: **DETECTED & REJECTED**
  - Modified previous-hash injection: **DETECTED & REJECTED**
  - Reordered blocks injection: **DETECTED & REJECTED**
  - Deleted block injection: **DETECTED & REJECTED**
  - Duplicated block injection: **DETECTED & REJECTED**
- **Blockchain Clarification:** SHA-256 cryptographic chaining provides tamper evidence on a single ledger. It does **NOT** constitute a decentralized peer-to-peer blockchain consensus mechanism.
