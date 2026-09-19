#!/usr/bin/env python3
"""
Single-Command Demonstration Entry Point for SIH26153.
Project: SIH26153 - AI-Based Network Attack Forecasting from Network Traffic Data

Usage:
    python run_demo.py
    python run_demo.py --scenario attack
    python run_demo.py --scenario benign
    python run_demo.py --host 192.168.10.10
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo import main

if __name__ == "__main__":
    main()
