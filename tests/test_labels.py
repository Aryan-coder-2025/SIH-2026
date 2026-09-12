"""
tests/test_labels.py
--------------------
Unit tests for src/features/labels.py
"""
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.labels import classify_label


class TestLabels(unittest.TestCase):
    def test_botnet(self):
        self.assertEqual(classify_label("flow=From-Botnet-V42-TCP"), "botnet")
        self.assertEqual(classify_label("botnet"), "botnet")

    def test_normal(self):
        self.assertEqual(classify_label("flow=From-Normal-V42-Jist"), "normal")
        self.assertEqual(classify_label("normal"), "normal")
        self.assertEqual(classify_label("flow=To-Botnet-Something"), "normal")

    def test_background(self):
        self.assertEqual(classify_label("flow=Background-UDP"), "background")
        self.assertEqual(classify_label("unknown_label"), "background")

    def test_none_safe(self):
        self.assertEqual(classify_label(None), "background")

    def test_whitespace(self):
        self.assertEqual(classify_label("  botnet  "), "botnet")
        self.assertEqual(classify_label("  flow=From-Botnet-X  "), "botnet")


if __name__ == "__main__":
    unittest.main()
