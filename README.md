# SIH-26153: Temporal Multi-Horizon Cyber Attack Forecasting & Tamper-Evident Audit Ledger

> **Important Provenance Notice:**
> **Status:** `SYNTHETIC DEMONSTRATION ONLY`  
> **Disclaimer:** `REAL DATA EXPERIMENT NOT EXECUTED - RAW CSE-CIC-IDS2018 TELEMETRY PENDING EXTRACTION`  
> All benchmarks, trajectories, and evaluations in this repository demonstrate architectural correctness, fair baseline comparisons, and pipeline mechanics using the canonical integration dataset. They must not be cited as real-world benchmark metrics.

---

## 1. Overview

**SIH-26153** is an end-to-end cybersecurity analytics platform for predictive threat forecasting. Rather than relying on purely reactive intrusion detection, the system forecasts upcoming network attack risk across multiple discrete horizons:
- **$+10\text{s}$ Forecast** (Immediate operational triage)
- **$+20\text{s}$ Forecast** (Automated rule preparation)
- **$+30\text{s}$ Forecast** (Defensive perimeter reconfiguration)

### Core Architectural Components
1. **Canonical 41-Feature Ingestion:** Strictly locks 22 NetFlow features (Flow duration, byte rates, TCP flags) and 19 raw packet header features (TTL variance, TCP window sizes, payload entropy) with SHA-256 schema hash enforcement.
2. **Bi-Head LSTM Temporal Forecaster:** Consumes 10 historical windows ($10 \times 41$) to simultaneously forecast continuous multi-horizon risk probabilities and classify auxiliary cyber attack stages.
3. **Memory-Bounded Chunked Streaming:** Streams large CSV/PCAP telemetry files into disk-backed Parquet without loading full datasets into GPU VRAM.
4. **Axiomatic Explainability (Integrated Gradients):** Computes Path-Integrated Gradients feature attributions satisfying mathematical completeness ($\sum \text{Attr} \approx \Delta F$), preserving temporal dimensions without collapsing history.
5. **MITRE ATT&CK & SOC Playbooks:** Translates top-attributed features into candidate MITRE ATT&CK techniques (e.g. T1046, T1498) and actionable containment recommendations.
6. **Tamper-Evident Audit Ledger:** Cryptographically hash-chains forecasting decisions, MITRE findings, and timestamps using SHA-256 blocks to provide tamper detection.

---

## 2. Repository Structure

```
SIH-2026-INTEGRATION/
├── artifacts/                  # Trained checkpoints, evaluation metrics, and metadata
│   ├── world_model.pt          # Canonical 41-feature trained WorldModel checkpoint
│   ├── metadata.json           # Software versions, schema hash, split provenance
│   ├── evaluation_metrics.json # Full benchmark metrics and baseline comparisons
│   ├── ablation_results.json   # Modality and history depth ablation study results
│   ├── audit_ledger.json       # Tamper-evident SHA-256 audit ledger
│   └── demo_output.json        # End-to-end demo inference record
├── data/
│   ├── mock/                   # Raw canonical integration CSV fixtures
│   └── processed/              # Processed canonical feature matrix (Parquet)
├── docs/                       # Technical protocol specifications
├── src/
│   ├── audit/                  # Tamper-evident hash-chained audit ledger
│   ├── baseline/               # Fair baselines (LR, RF receiving 410 inputs; Persistence)
│   ├── data/                   # Chunked ingestion pipeline for large telemetry
│   ├── eval/                   # Metric calculation, evaluation suite, and ablations
│   ├── explain/                # Path-Integrated Gradients explainability engine
│   ├── flow/                   # Flow feature extraction modules
│   ├── fusion/                 # Multimodal early fusion layer
│   ├── mitre/                  # MITRE ATT&CK mapping & SOC playbook engine
│   ├── schemas/                # Authoritative 41-feature schema and anti-leakage gates
│   ├── temporal/               # Chronological splitting & sequence generators
│   ├── dashboard.py            # Streamlit SOC forecaster dashboard
│   ├── demo.py                 # Core demonstration runner
│   ├── model.py                # Production 41-feature WorldModel architecture
│   ├── pipeline.py             # CyberForecastPipeline end-to-end integration
│   └── train.py                # Checkpoint trainer with AMP and OOM fallback
├── tests/                      # Acceptance test suite (all 211 automated tests passed)
│   └── test_final_acceptance_gates.py  # Behavioral acceptance and failure-injection tests
├── run_demo.py                 # Single-command demonstration entrypoint
├── requirements.txt            # Python dependencies
├── IMPLEMENTATION_STATUS.md    # Subsystem implementation taxonomy
├── SCIENTIFIC_VALIDATION_REPORT.md  # Comprehensive benchmark report
├── DATA_PROVENANCE.md          # Data lineage and anti-leakage proof
├── KNOWN_LIMITATIONS.md        # Explicit technical disclosures
└── REPRODUCIBILITY.md          # Exact reproduction steps and environment metadata
```

---

## 3. Quick Start & Execution

### Installation
```bash
# 1. Clone repository
git clone https://github.com/Aryan-coder-2025/SIH-2026.git
cd SIH-2026-INTEGRATION

# 2. Install dependencies
pip install -r requirements.txt
```

### Verification & Automated Testing
```bash
# Run full test suite (211 tests covering schema, leakage, baselines, ledger, and XAI)
pytest -v
```
All 211 automated tests passed. The repository is ready for GitHub publication with documented limitations around real CSE-CIC-IDS2018 validation and CUDA execution.

### Run Model Training
```bash
# Trains canonical 41-feature WorldModel on CPU or GPU with automatic batch fallback
python src/train.py
```

### Run Baseline & System Evaluation
```bash
# Evaluates WorldModel against Logistic Regression, Random Forest, and Persistence
python src/eval/evaluate_system.py
```

### Run Controlled Ablation Study
```bash
# Evaluates Flow-Only (22), Packet-Only (19), Fusion (41), and History Depths (1, 5, 10)
python src/eval/ablation.py
```

### Run End-to-End Demonstration
```bash
# Executes inference, Integrated Gradients attribution, MITRE mapping, and ledger logging
python run_demo.py --scenario attack
```

### Launch Interactive SOC Dashboard

Evaluators can launch the interactive console using any of the following three options:

- **Option 1 (VS Code Run & Debug):** Open the project root in VS Code, navigate to the Run & Debug panel (`Ctrl+Shift+D`), select **"SIH26153 Dashboard"**, and press `F5`.
- **Option 2 (One-Click Windows Batch):** Double-click or execute [`RUN_DASHBOARD.bat`](RUN_DASHBOARD.bat) from the repository root. It automatically resolves the repository path and starts the dashboard.
- **Option 3 (Terminal Fallback):**
  ```bash
  python -m streamlit run src/dashboard.py
  ```

---

## 4. Key Scientific Results (Synthetic Demonstration Only)

*Evaluated on the held-out test split of the canonical integration fixture with validation-frozen decision thresholds:*

| Forecast Horizon | Model | Inputs | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$+10\text{s}$** | **WorldModel (Bi-Head LSTM)** | $10 \times 41$ | 0.9130 | 0.9545 | 0.9333 | **0.9974** | 0.9854 |
| | Logistic Regression | $410$ features | 0.8333 | 0.9091 | 0.8696 | 0.9970 | **0.9857** |
| | Random Forest | $410$ features | **0.9167** | **1.0000** | **0.9565** | 0.9944 | 0.9593 |
| | Persistence ($y_{\text{curr}}$ at $t$) | State at $t$ | 0.9091 | 0.9091 | 0.9091 | 0.9463 | 0.8403 |
| **$+20\text{s}$** | **WorldModel (Bi-Head LSTM)** | $10 \times 41$ | **0.9412** | 0.7273 | 0.8205 | 0.9885 | **0.9517** |
| | Logistic Regression | $410$ features | 0.8333 | 0.9091 | **0.8696** | **0.9892** | 0.9411 |
| | Random Forest | $410$ features | 0.8333 | **0.9091** | **0.8696** | 0.9424 | 0.8301 |
| | Persistence ($y_{\text{curr}}$ at $t$) | State at $t$ | 0.8182 | 0.8182 | 0.8182 | 0.8927 | 0.6972 |
| **$+30\text{s}$** | **WorldModel (Bi-Head LSTM)** | $10 \times 41$ | **0.8947** | 0.7727 | **0.8293** | **0.9881** | **0.9409** |
| | Logistic Regression | $410$ features | 0.7500 | **0.8182** | 0.7826 | 0.9817 | 0.8976 |
| | Random Forest | $410$ features | 0.8095 | 0.7727 | 0.7907 | 0.9311 | 0.7317 |
| | Persistence ($y_{\text{curr}}$ at $t$) | State at $t$ | 0.7273 | 0.7273 | 0.7273 | 0.8390 | 0.5706 |

*Note: All-negative baseline test accuracy is 84.72% due to class distribution. Baselines are evaluated fairly with 410 flattened features (10 historical windows × 41 features). Random Forest achieves higher F1 at immediate horizons (+10s, +20s), while WorldModel provides temporal multi-horizon joint sequence forecasting with superior long-range precision and discrimination (+30s F1: 0.8293, ROC-AUC: 0.9881).*

---

## 5. Architectural Invariants & Hard Gates

1. **Production Contract:** The production model input size is strictly locked to 41 features. Dynamic schema mutation is prohibited in production.
2. **Anti-Leakage Enforcement:** Chronological splitting occurs on raw windows before sequence generation. Scalers are fitted strictly on the training partition.
3. **Axiomatic Attribution:** Integrated Gradients is implemented and mathematically verified for completeness. Kernel SHAP is not claimed due to library absence.
4. **Audit Integrity:** Cryptographically chained SHA-256 ledger verifies record immutability. Distributed blockchain consensus is explicitly not claimed.

---

## 6. License & Provenance

Developed for **Smart India Hackathon (SIH-2026)** under Problem Statement **SIH-26153**.  
For detailed scientific validation, see [SCIENTIFIC_VALIDATION_REPORT.md](SCIENTIFIC_VALIDATION_REPORT.md).  
For implementation taxonomy, see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
