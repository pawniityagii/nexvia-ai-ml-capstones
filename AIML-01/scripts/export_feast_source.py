"""Build the Feast offline source parquet: per-card feature snapshots.

Writes feature_repo/data/feast_card_features.parquet with one row per card per
observed time, plus an event_timestamp Feast needs for point-in-time joins.
Run before `feast apply` / `feast materialize`.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config as C
from src.common import data_loader
from src.features import feature_engineering as fe
from src.models import graph_builder

EPOCH = datetime(2026, 1, 1)


def main():
    df = data_loader.load_raw()
    feats = fe.engineer(df)
    feats = graph_builder.attach_ring_features(feats)

    # event timestamp from TransactionDT seconds offset
    feats["event_timestamp"] = feats[C.TIME_COL].apply(
        lambda s: EPOCH + timedelta(seconds=float(s)))

    agg = (feats.sort_values("event_timestamp")
           .groupby("card_uid")
           .agg(txn_count=(C.AMOUNT_COL, "size"),
                amt_mean=(C.AMOUNT_COL, "mean"),
                amt_std=(C.AMOUNT_COL, "std"),
                ring_degree=("ring_degree", "max"),
                ring_fraud_rate=("ring_fraud_rate", "last"),
                event_timestamp=("event_timestamp", "last"))
           .reset_index())
    agg["amt_std"] = agg["amt_std"].fillna(0.0)
    for c in ["amt_mean", "amt_std", "ring_degree", "ring_fraud_rate"]:
        agg[c] = agg[c].astype("float32")
    agg["txn_count"] = agg["txn_count"].astype("int64")

    out = C.ROOT / "feature_repo" / "data" / "feast_card_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out, index=False)
    print(f"wrote {len(agg):,} card rows -> {out}")


if __name__ == "__main__":
    main()
