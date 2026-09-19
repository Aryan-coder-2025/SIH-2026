# Known Limitations: SIH-26153

**Evaluation Standard:** Evaluator Defensibility & Scientific Honesty  

---

## 1. Data Limitations

### Synthetic Integration Fixture vs. Real CSE-CIC-IDS2018
- **Limitation:** The raw multi-gigabyte CSE-CIC-IDS2018 PCAP/CSV files are pending extraction from external sources and are not present on the evaluation machine.
- **Impact:** All reported numerical metrics (F1, ROC-AUC, PR-AUC) reflect model performance on the canonical synthetic integration fixture (`data/processed/feature_matrix.parquet`).
- **Required Disclosure:** Metrics must not be cited as empirical benchmarks for real-world network attack forecasting.

### Unseen-Attack & Unseen-Host Generalization
- **Limitation:** In the 1,440-row canonical fixture, attack classes and host IPs are distributed across all temporal partitions.
- **Impact:** True zero-shot generalization to unseen attack categories or unseen network topologies has **not been experimentally validated**.

---

## 2. Hardware & Environment Limitations

### PyTorch CPU Execution
- **Limitation:** The host environment is running PyTorch CPU-only (`2.11.0+cpu`). Dedicated GPU acceleration (NVIDIA RTX 5050 Laptop GPU) was unavailable in this Python runtime.
- **Impact:** Model training and evaluation were executed on CPU. While throughput reached ~3,170 sequences/second, large-scale training over ~2.8M rows will require GPU acceleration.
- **Remediation Implemented:** Automatic batch fallback (64 $\to$ 32 $\to$ 16 $\to$ 8) and AMP gradient scaling are architected and ready for CUDA environments.

---

## 3. Explainability & Algorithm Disclosures

### Integrated Gradients vs. SHAP
- **Limitation:** The Python `shap` package is uninstalled in the host environment.
- **Impact:** Kernel SHAP and Tree SHAP computations are **not implemented and not validated**.
- **Alternative Implemented:** Axiomatic Path-Integrated Gradients (Sundararajan et al., 2017) is implemented, preserving `(10, 41)` shape and satisfying mathematical completeness ($\sum \text{Attr} \approx \Delta F$). It must strictly be referred to as Integrated Gradients, not SHAP.

### Cryptographic Audit Ledger vs. Decentralized Blockchain
- **Limitation:** Decentralized consensus protocols (Proof-of-Work, Proof-of-Stake, Raft, Paxos) and peer-to-peer gossip networking are **not implemented**.
- **Alternative Implemented:** A cryptographically hash-chained tamper-evident audit ledger using SHA-256. It provides verifiable append-only tamper detection on single nodes, but does not constitute a distributed blockchain.
