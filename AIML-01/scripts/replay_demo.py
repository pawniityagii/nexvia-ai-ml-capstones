"""No-broker demo: replay transactions through the Scorer directly.

Lets you populate the case store and see the pipeline work WITHOUT running
Kafka - useful for the screen-recorded demo and for CI. Uses the identical
Scorer the Kafka consumer uses.

Run (after training + export_ring_cache):
    python scripts/replay_demo.py --limit 5000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as C
from src.common import data_loader
from src.serving.scorer import Scorer


def main(limit: int):
    df = data_loader.load_raw().sort_values(C.TIME_COL)
    if limit:
        df = df.head(limit)
    scorer = Scorer(write_cases=True)
    n = flagged = 0
    lat = []
    t0 = time.perf_counter()
    for _, row in df.iterrows():
        rec = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        res = scorer.score(rec)
        lat.append(res["latency_ms"])
        n += 1
        flagged += res["decision"] == "flag"
    dt = time.perf_counter() - t0
    lat.sort()
    print(f"scored {n:,} in {dt:.1f}s ({n/dt:,.0f}/s), {flagged:,} flagged")
    print(f"latency p50={lat[len(lat)//2]:.2f}ms  p99={lat[int(len(lat)*0.99)]:.2f}ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=5000)
    a = p.parse_args()
    main(a.limit)
