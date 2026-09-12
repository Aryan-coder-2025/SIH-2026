#!/usr/bin/env python3
"""
build_dataset.py
----------------
End-to-end pipeline step: raw .binetflow CSV → processed Parquet.

Usage (from project root or directly):
    python src/features/build_dataset.py
    python -m src.features.build_dataset
"""
import logging
import sys
from pathlib import Path

# Add project root to sys.path so direct invocation works
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.flow_features import aggregate_flow_features
from src.features.windowing import add_time_columns, add_window_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/CTU-13/sample_100k.binetflow")
OUTPUT_PATH = Path("data/processed/ctu13_scenario1_windows.parquet")

REQUIRED_COLUMNS = [
    "StartTime", "SrcAddr", "DstAddr", "Dport",
    "TotBytes", "TotPkts", "Dur", "Proto", "Dir", "State", "Label",
]


def build_dataset(raw_path: Path = RAW_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Load raw flows, engineer features, attach labels, save Parquet."""

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    if not raw_path.is_file():
        log.error("Raw data not found at %s. Run generate_synthetic_data.py first.", raw_path)
        sys.exit(1)

    log.info("Loading raw data from %s …", raw_path)
    df = pd.read_csv(raw_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        log.error("CSV is missing required columns: %s", missing)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Windowing & aggregation
    # ------------------------------------------------------------------
    log.info("Adding time windows and host identifiers …")
    df = add_time_columns(df)

    log.info("Computing flow-level aggregate features …")
    features = aggregate_flow_features(df)

    log.info("Computing window labels …")
    labels = add_window_labels(df)

    # ------------------------------------------------------------------
    # 3. Merge & save
    # ------------------------------------------------------------------
    log.info("Merging features and labels …")
    merged = pd.merge(features, labels, on=["host_id", "window_start"], how="inner")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    log.info("Saved %d rows to %s", len(merged), output_path)

    return merged


if __name__ == "__main__":
    build_dataset()
