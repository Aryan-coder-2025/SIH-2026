# Data Provenance: SIH-26153

**Canonical Policy:** Zero-Fabrication Audit Standard  
**Data Status:** `SYNTHETIC DEMONSTRATION ONLY`  
**Notice:** `REAL DATA EXPERIMENT NOT EXECUTED - RAW CSE-CIC-IDS2018 TELEMETRY PENDING EXTRACTION`  

---

## 1. Primary Data Source Summary

| Property | Status / Value |
| :--- | :--- |
| **Active Dataset** | Canonical Integration Fixture (`data/processed/feature_matrix.parquet`) |
| **Raw Telemetry Source** | `data/mock/canonical_feature_matrix.csv` |
| **Total Processed Rows** | 1,440 temporal windows |
| **Duration Covered** | 4.0 continuous hours (14,400 seconds) at 10-second window step |
| **Network Hosts** | 6 distinct IP entities (`192.168.10.10`, `192.168.10.11`, `192.168.10.12`, `192.168.10.15`, `192.168.10.20`, `192.168.10.25`) |
| **Feature Dimensionality** | Exactly 41 numerical features (22 NetFlow/binetflow + 19 raw PCAP packet metrics) |
| **Schema Hash** | `7dc05760abf2b5e881e51f9bd464f45b89ae35b6ec67af4032a8e6235390e6dd` |

---

## 2. Ingestion Pipeline & Real-Data Readiness

Although raw CSE-CIC-IDS2018 PCAP/CSV files (~2.8M rows) are pending external extraction, the end-to-end ingestion architecture is fully implemented and tested in `src/data/chunked_ingestion.py`:

```
Raw Telemetry (CSV / PCAP stream)
             │
             ▼
[ChunkedIngestionPipeline]  <-- Memory-bounded chunk reader (chunk_size=50,000)
             │
      ┌──────┴─────────────────────────────────┐
      │                                        │
      ▼                                        ▼
Alias Normalization (FEATURE_ALIASES)    Zero-Fill / Clamp (NaNs, Infs)
      │                                        │
      └──────┬─────────────────────────────────┘
             │
             ▼
PyArrow ParquetWriter (Disk-backed Parquet streaming)
             │
             ▼
`data/processed/feature_matrix.parquet` + `artifacts/data_provenance_report.json`
```

### Verified Smoke Test Metrics
- Input: 1,440 raw rows
- Output: 1,440 clean Parquet rows
- Dropped Rows: 0
- NaN / Inf Violations in Output: 0
- Extraction Duration: < 1.5 seconds

---

## 3. Strict Anti-Leakage Controls

To guarantee that future information does not corrupt historical models:
1. **Timestamped Physical Split:** Chronological split occurs on raw windows **before** sequence construction.
2. **Scaler Isolation:** `StandardScaler` is fitted **strictly on the training split**. Test and validation sequences are transformed using the train-fitted mean and variance.
3. **Forbidden Columns:** Columns matching `FORBIDDEN_FEATURE_NAMES` (e.g., `src_ip`, `dst_ip`, `flow_id`, `window_id`, `is_malicious`, `stage`, `target_*`) are rejected at schema validation and cannot enter feature matrices.
4. **Horizon Separation:** Multi-horizon future targets (+10s, +20s, +30s) are constructed strictly from time steps $t+1$, $t+2$, $t+3$ relative to the sequence history ending at time $t$.
