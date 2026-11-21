"""Load raw transactions - real IEEE-CIS if present, else synthetic.

If data/train_transaction.csv (+ optional data/train_identity.csv) exist, they
are used and merged on TransactionID. Otherwise a synthetic set is generated so
the pipeline always runs. The column set is normalised to what the feature
layer expects; missing identity columns are filled with "missing".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import synthetic


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = (C.NUMERIC_RAW + C.CATEGORICAL_RAW + [C.ID_COL, C.TIME_COL])
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    for col in C.CATEGORICAL_RAW:
        df[col] = df[col].astype("object").where(df[col].notna(), "missing")
    return df


def load_raw(prefer_synthetic: bool = False, n_synthetic: int = 60_000) -> pd.DataFrame:
    if not prefer_synthetic and C.RAW_TRANSACTION_CSV.exists():
        df = pd.read_csv(C.RAW_TRANSACTION_CSV)
        if C.RAW_IDENTITY_CSV.exists():
            ident = pd.read_csv(C.RAW_IDENTITY_CSV)
            # IEEE-CIS identity columns are named id-01..id-38 in some dumps
            ident.columns = [c.replace("id-", "id_") for c in ident.columns]
            df = df.merge(ident, on=C.ID_COL, how="left")
        source = "ieee-cis"
    else:
        if C.SYNTHETIC_CSV.exists():
            df = pd.read_csv(C.SYNTHETIC_CSV)
        else:
            df = synthetic.generate(n_transactions=n_synthetic)
            df.to_csv(C.SYNTHETIC_CSV, index=False)
        source = "synthetic"
    df = _ensure_columns(df)
    df.attrs["source"] = source
    return df


def time_split(df: pd.DataFrame, fracs=(0.7, 0.15, 0.15)) -> dict[str, pd.DataFrame]:
    """Chronological split - no future transaction leaks into training."""
    d = df.sort_values(C.TIME_COL).reset_index(drop=True)
    n = len(d)
    i1 = int(n * fracs[0])
    i2 = int(n * (fracs[0] + fracs[1]))
    return {"train": d.iloc[:i1].copy(),
            "val": d.iloc[i1:i2].copy(),
            "test": d.iloc[i2:].copy()}
