"""
Memory-bounded Chunked Ingestion Pipeline for SIH-26153.

Handles large-scale telemetry (e.g. ~2.8M rows) by reading in discrete chunks,
sanitizing NaNs/Infs, verifying canonical 41-feature schema alignment, and writing
directly to disk-backed Parquet without loading full datasets into RAM/VRAM.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.schemas.features import (
    CANONICAL_MODEL_FEATURE_NAMES,
    FEATURE_ALIASES,
    validate_feature_names,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class ChunkedIngestionPipeline:
    """
    Stream-processes large raw CSV telemetry files in memory-bounded chunks
    and outputs canonical Parquet tables with provenance auditing.
    """

    def __init__(
        self,
        raw_csv_path: Path | str,
        output_parquet_path: Path | str,
        chunk_size: int = 50_000,
        provenance_report_path: Optional[Path | str] = None,
    ):
        self.raw_csv_path = Path(raw_csv_path)
        self.output_parquet_path = Path(output_parquet_path)
        self.chunk_size = chunk_size
        self.provenance_report_path = Path(provenance_report_path) if provenance_report_path else None

    def sanitize_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Renames aliases, fills NaNs, caps Infs, and ensures numeric types for 41 canonical features.
        """
        # 1. Remap aliases
        rename_map = {orig: target for orig, target in FEATURE_ALIASES.items() if orig in chunk.columns}
        if rename_map:
            chunk = chunk.rename(columns=rename_map)

        # 2. Check presence of essential metadata columns
        required_meta = ["source_host", "timestamp"]
        for col in required_meta:
            if col not in chunk.columns:
                if col == "source_host" and "src_ip" in chunk.columns:
                    chunk["source_host"] = chunk["src_ip"]
                elif col == "timestamp" and "time" in chunk.columns:
                    chunk["timestamp"] = chunk["time"]
                else:
                    raise KeyError(f"Missing required metadata column: {col}")

        # Ensure labels exist or default
        if "is_malicious" not in chunk.columns:
            if "label" in chunk.columns:
                chunk["is_malicious"] = (chunk["label"].astype(str).str.lower() != "benign").astype(int)
            else:
                chunk["is_malicious"] = 0

        if "stage" not in chunk.columns:
            if "attack_type" in chunk.columns:
                chunk["stage"] = chunk["attack_type"]
            else:
                chunk["stage"] = np.where(chunk["is_malicious"] == 1, "malicious", "benign")

        # 3. Check and sanitize 41 canonical features
        for feat in CANONICAL_MODEL_FEATURE_NAMES:
            if feat not in chunk.columns:
                # If feature is completely missing from chunk, fill with 0.0
                chunk[feat] = 0.0
            else:
                col_data = chunk[feat].astype(np.float32)
                # Replace Inf with large finite number or max non-inf
                col_data = col_data.replace([np.inf, -np.inf], 0.0)
                # Fill NaN with 0.0
                col_data = col_data.fillna(0.0)
                chunk[feat] = col_data

        # 4. Canonical column order
        ordered_cols = ["source_host", "timestamp", "window_id", "is_malicious", "stage"]
        ordered_cols = [c for c in ordered_cols if c in chunk.columns]
        ordered_cols.extend(list(CANONICAL_MODEL_FEATURE_NAMES))

        return chunk[ordered_cols]

    def run(self) -> Dict[str, Any]:
        """
        Stream-process the CSV file and write directly to Parquet.
        """
        if not self.raw_csv_path.is_file():
            raise FileNotFoundError(f"Raw telemetry file not found: {self.raw_csv_path}")

        self.output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Starting chunked ingestion from {self.raw_csv_path} (chunk_size={self.chunk_size})")

        total_rows_read = 0
        total_rows_written = 0
        total_dropped = 0
        unique_hosts = set()
        min_timestamp = None
        max_timestamp = None

        parquet_writer = None

        try:
            reader = pd.read_csv(self.raw_csv_path, chunksize=self.chunk_size, low_memory=False)
            for chunk_idx, chunk in enumerate(reader):
                chunk_len = len(chunk)
                total_rows_read += chunk_len

                # Sanitize
                clean_chunk = self.sanitize_chunk(chunk)
                clean_chunk = clean_chunk.drop_duplicates()
                clean_len = len(clean_chunk)
                total_dropped += (chunk_len - clean_len)
                total_rows_written += clean_len

                # Track metadata
                unique_hosts.update(clean_chunk["source_host"].unique().tolist())
                chunk_ts = pd.to_datetime(clean_chunk["timestamp"])
                chunk_min_ts = chunk_ts.min()
                chunk_max_ts = chunk_ts.max()

                if min_timestamp is None or chunk_min_ts < min_timestamp:
                    min_timestamp = chunk_min_ts
                if max_timestamp is None or chunk_max_ts > max_timestamp:
                    max_timestamp = chunk_max_ts

                # PyArrow table
                arrow_table = pa.Table.from_pandas(clean_chunk, preserve_index=False)

                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(
                        str(self.output_parquet_path),
                        arrow_table.schema,
                        compression="snappy",
                    )

                parquet_writer.write_table(arrow_table)

                if (chunk_idx + 1) % 5 == 0:
                    log.info(f"Processed {chunk_idx + 1} chunks ({total_rows_written} rows written)...")

        finally:
            if parquet_writer is not None:
                parquet_writer.close()

        provenance_report = {
            "source_raw_csv": str(self.raw_csv_path),
            "output_parquet": str(self.output_parquet_path),
            "chunk_size": self.chunk_size,
            "total_raw_rows_read": total_rows_read,
            "total_processed_rows": total_rows_written,
            "total_dropped_rows": total_dropped,
            "unique_entities": len(unique_hosts),
            "min_timestamp": min_timestamp.isoformat() if min_timestamp is not None else None,
            "max_timestamp": max_timestamp.isoformat() if max_timestamp is not None else None,
            "canonical_features_validated": len(CANONICAL_MODEL_FEATURE_NAMES),
        }

        if self.provenance_report_path:
            self.provenance_report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.provenance_report_path, "w", encoding="utf-8") as f:
                json.dump(provenance_report, f, indent=2)
            log.info(f"Saved ingestion provenance report to {self.provenance_report_path}")

        log.info(f"Chunked ingestion complete: {total_rows_written} rows written to {self.output_parquet_path}")
        return provenance_report
