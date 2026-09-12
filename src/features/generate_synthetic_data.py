#!/usr/bin/env python3
"""
generate_synthetic_data.py
--------------------------
Creates a small synthetic CTU-13-style binetflow CSV for pipeline testing.
Columns match exactly what build_dataset.py / flow_features.py expect.
"""
import argparse
import logging
import pathlib
import random

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "StartTime", "SrcAddr", "DstAddr", "Dport",
    "TotBytes", "TotPkts", "Dur", "Proto", "Dir", "State", "Label",
]


def generate(rows: int = 5000, seed: int = 42, out_path: str = "data/raw/CTU-13/sample_100k.binetflow") -> pathlib.Path:
    """Generate ``rows`` synthetic flow records and write them to *out_path*."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # --- timestamps spread over 1 hour ---
    start = pd.Timestamp("2023-01-01 09:00:00")
    offsets_s = rng.integers(0, 3600, size=rows)
    times = [start + pd.Timedelta(seconds=int(o)) for o in offsets_s]
    time_strs = [t.strftime("%Y/%m/%d %H:%M:%S.%f") for t in times]

    # --- IPs: 10 fixed source IPs so we get real host groups ---
    fixed_srcs = [f"10.0.0.{i}" for i in range(1, 11)]
    src_ips = [random.choice(fixed_srcs) for _ in range(rows)]
    dst_ips = [
        f"192.168.{rng.integers(0, 4)}.{rng.integers(1, 255)}"
        for _ in range(rows)
    ]

    dports = rng.integers(1, 65535, size=rows)
    totbytes = rng.exponential(scale=5000, size=rows).astype(int).clip(min=64)
    totpkts = rng.poisson(lam=10, size=rows).clip(min=1)
    dur = rng.exponential(scale=0.5, size=rows).clip(min=0.001)
    proto = rng.choice(["tcp", "udp"], size=rows, p=[0.7, 0.3])
    direction = rng.choice(["<->", "->", "<-"], size=rows, p=[0.5, 0.3, 0.2])
    states = ["S", "A", "FA", "FRA", "CON", "INT", ""]
    state = [random.choice(states) for _ in range(rows)]

    # 3 % botnet – realistic imbalance for quick test
    is_botnet = rng.random(size=rows) < 0.03
    label = np.where(is_botnet, "botnet", "normal")

    df = pd.DataFrame({
        "StartTime": time_strs,
        "SrcAddr": src_ips,
        "DstAddr": dst_ips,
        "Dport": dports,
        "TotBytes": totbytes,
        "TotPkts": totpkts,
        "Dur": dur,
        "Proto": proto,
        "Dir": direction,
        "State": state,
        "Label": label,
    })

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Generated %d rows → %s", rows, out)
    log.info("Label distribution:\n%s", df["Label"].value_counts().to_string())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a tiny synthetic CTU-13 binetflow file for testing."
    )
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="data/raw/CTU-13/sample_100k.binetflow", help="Output path")
    args = parser.parse_args()
    generate(args.rows, args.seed, args.out)
