# Reproducibility Guide: SIH-26153

**Evaluation Policy:** Deterministic, Defensible, Re-executable Pipeline  

---

## 1. System & Software Metadata

The environment configuration captured during the pre-GitHub freeze validation run:

- **OS:** Windows 11 (win32)
- **Python Version:** 3.11.9
- **PyTorch Version:** 2.11.0+cpu (CUDA Unavailable — CPU Execution)
- **Scikit-Learn Version:** 1.5.0
- **Pandas Version:** 2.3.3
- **PyArrow Version:** 24.0.0
- **PyTest Version:** 9.1.1
- **Random Seed:** 42 (Enforced across Python `random`, `numpy.random`, `torch.manual_seed`)
- **Canonical Schema Hash:** `7dc05760abf2b5e881e51f9bd464f45b89ae35b6ec67af4032a8e6235390e6dd`

---

## 2. Step-by-Step Reproduction Commands

All commands below have been tested and verified to execute without errors.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Compilation & Code Integrity
```bash
python -m compileall .
```

### Step 3: Run Full Verification & Acceptance Test Suite
```bash
pytest -v
```
*Expected Result:* All 235 automated tests passed. The repository is ready for GitHub publication with documented limitations around real CSE-CIC-IDS2018 validation and CUDA execution.

### Step 4: Train Temporal WorldModel Checkpoint
```bash
python src/train.py
```
*Outputs Created:*
- `artifacts/world_model.pt` (Trained model checkpoint with schema hash)
- `artifacts/scaler.pkl` (Train-only fitted StandardScaler)
- `artifacts/metadata.json` (Full provenance metadata)
- `artifacts/feature_order.json` (Canonical 41-feature order)
- `artifacts/stage_classes.json` (Attack stage classification categories)

### Step 5: Run Comprehensive Evaluation & Fair Baselines
```bash
python src/eval/evaluate_system.py
```
*Outputs Created:*
- `artifacts/evaluation_metrics.json` (Multi-horizon F1, Precision, Recall, FPR, ROC-AUC, PR-AUC, Confusion Matrices, and baseline comparisons).

### Step 6: Run Controlled Modality & History Depth Ablations
```bash
python src/eval/ablation.py
```
*Outputs Created:*
- `artifacts/ablation_results.json` (Controlled experiments for Flow-Only 22, Packet-Only 19, Fusion 41, and history depths 1, 5, 10).

### Step 7: Run End-to-End Inference Demo
```bash
python run_demo.py --scenario attack
```
*Outputs Created:*
- `artifacts/demo_output.json` (Multi-horizon forecast, Integrated Gradients attribution, MITRE technique mapping, SOC recommendations, and Tamper-Evident Ledger block entry).

### Step 8: Launch Interactive SOC Forecaster Dashboard
```bash
streamlit run src/dashboard.py
```
*Displays:* Real-time forecast trajectory, Integrated Gradients feature importance chart, candidate MITRE playbook, and cryptographic audit ledger integrity verification status.
