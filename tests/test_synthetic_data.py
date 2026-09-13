"""
tests/test_synthetic_data.py
----------------------------
Verifies that the synthetic data generator produces a well-formed CSV.
"""
import sys
import unittest
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.features.generate_synthetic_data import generate, REQUIRED_COLUMNS


class TestSyntheticData(unittest.TestCase):
    def test_output_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.binetflow"
            generate(rows=200, seed=0, out_path=str(out))
            self.assertTrue(out.exists())
            df = pd.read_csv(out)
            self.assertEqual(len(df), 200)

    def test_required_columns_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.binetflow"
            generate(rows=200, seed=1, out_path=str(out))
            df = pd.read_csv(out)
            for col in REQUIRED_COLUMNS:
                self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_labels_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.binetflow"
            generate(rows=1000, seed=99, out_path=str(out))
            df = pd.read_csv(out)
            labels = set(df["Label"].unique())
            self.assertTrue("botnet" in labels or "normal" in labels)


if __name__ == "__main__":
    unittest.main()
