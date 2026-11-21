"""Feature engineering for the fraud model.

All time-dependent features are LEAK-FREE: every rolling statistic for a
transaction uses only transactions that happened strictly before it, computed
with shifted cumulative sums and ``searchsorted`` over per-card time windows.
Categoricals use frequency encoding fitted on training data only, so the exact
same transform runs at serving time (train/serve parity).

Public API
----------
build_identities(df)      -> adds card_uid / device_uid / email_key columns
fit_transform(df)         -> (features_df, preprocessor)  [training]
transform(df, preproc)    -> features_df                  [serving / eval]
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common import config as C


# --------------------------------------------------------------------------- #
# Identity columns (the nodes of the fraud-ring graph)
# --------------------------------------------------------------------------- #
def _hash_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Stable string id from the concatenation of several columns."""
    joined = df[cols].astype(str).agg("|".join, axis=1)
    return joined.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:16])


def build_identities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["card_uid"] = _hash_cols(df, C.CARD_KEYS)
    df["device_uid"] = _hash_cols(df, C.DEVICE_KEYS)
    df["email_key"] = df[C.EMAIL_KEYS[0]].astype(str)
    return df


# --------------------------------------------------------------------------- #
# Preprocessor (fitted on train, reused at serve)
# --------------------------------------------------------------------------- #
@dataclass
class Preprocessor:
    freq_maps: dict[str, dict] = field(default_factory=dict)
    numeric_fill: dict[str, float] = field(default_factory=dict)
    default_freq: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "freq_maps": self.freq_maps,
            "numeric_fill": self.numeric_fill,
            "default_freq": self.default_freq,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(d["freq_maps"], d["numeric_fill"], d["default_freq"])


# --------------------------------------------------------------------------- #
# Leak-free time features, computed per card_uid
# --------------------------------------------------------------------------- #
def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return engineered time/velocity columns aligned to df.index."""
    d = df[[C.TIME_COL, C.AMOUNT_COL, "card_uid", "device_uid", "addr1"]].copy()
    d = d.sort_values([C.TIME_COL]).reset_index()            # keep original index
    amt = d[C.AMOUNT_COL].to_numpy(dtype=float)
    d["_amt"] = amt
    d["_amt_sq"] = amt * amt
    out = pd.DataFrame(index=d["index"])

    g = d.groupby("card_uid", sort=False)
    # running count of PRIOR transactions on this card
    prior_cnt = g.cumcount()
    # shifted cumulative sums => uses strictly-past rows only
    cs = g["_amt"].cumsum().to_numpy() - amt                 # sum of prior amts
    css = g["_amt_sq"].cumsum().to_numpy() - amt * amt       # sum of prior amts^2
    cnt = prior_cnt.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        past_mean = np.where(cnt > 0, cs / np.maximum(cnt, 1), np.nan)
        past_var = np.where(cnt > 0, css / np.maximum(cnt, 1) - past_mean ** 2, np.nan)
        past_std = np.sqrt(np.clip(past_var, 0, None))

    out["amt_to_card_mean"] = np.where(cnt > 0, amt / np.where(past_mean > 0, past_mean, np.nan), 1.0)
    out["amt_to_card_std"] = np.where(
        (cnt > 0) & (past_std > 0), (amt - past_mean) / np.where(past_std > 0, past_std, np.nan), 0.0)
    out["amt_to_card_mean"] = pd.Series(out["amt_to_card_mean"].values, index=d["index"]).fillna(1.0)
    out["amt_to_card_std"] = pd.Series(out["amt_to_card_std"].values, index=d["index"]).fillna(0.0)

    # recency + velocity via per-card sorted time windows (strictly past)
    secs_prev = np.full(len(d), np.nan)
    vel_1h = np.zeros(len(d), dtype=int)
    vel_24h = np.zeros(len(d), dtype=int)
    t_all = d[C.TIME_COL].to_numpy()
    for _, idx in g.groups.items():
        pos = d.index.get_indexer(idx)
        times = t_all[pos]
        order = np.argsort(times, kind="stable")
        pos_sorted = pos[order]
        ts = times[order]
        for k in range(len(ts)):
            if k > 0:
                secs_prev[pos_sorted[k]] = ts[k] - ts[k - 1]
            left_1h = np.searchsorted(ts, ts[k] - 3600, side="left")
            left_24h = np.searchsorted(ts, ts[k] - 86400, side="left")
            vel_1h[pos_sorted[k]] = k - left_1h
            vel_24h[pos_sorted[k]] = k - left_24h

    out["secs_since_prev_card_txn"] = pd.Series(secs_prev, index=d["index"]).fillna(1e7)
    out["txn_velocity_card_1h"] = pd.Series(vel_1h, index=d["index"])
    out["txn_velocity_card_24h"] = pd.Series(vel_24h, index=d["index"])

    # device / region change vs the card's previous transaction (past only)
    prev_device = g["device_uid"].shift(1)
    prev_addr = g["addr1"].shift(1)
    dev_change = ((d["device_uid"] != prev_device) & prev_device.notna()).astype(int)
    addr_change = ((d["addr1"] != prev_addr) & prev_addr.notna()).astype(int)
    out["device_change_flag"] = pd.Series(dev_change.values, index=d["index"])
    out["addr_change_flag"] = pd.Series(addr_change.values, index=d["index"])

    out = out.reindex(df.index)
    return out


def _static_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["amt_log"] = np.log1p(df[C.AMOUNT_COL].clip(lower=0))
    out["hour_of_day"] = (df[C.TIME_COL] // 3600) % 24
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-free feature engineering (no encoding). Run over full history."""
    df = build_identities(df)
    feats = pd.concat([_static_features(df), _time_features(df)], axis=1)
    # ring_* are filled by models/graph_builder; default to 0 so this module
    # is runnable standalone and the column order stays stable.
    feats["ring_degree"] = 0.0
    feats["ring_fraud_rate"] = 0.0
    # carry identity + raw + time/id columns the downstream steps need
    carry = (C.NUMERIC_RAW + C.CATEGORICAL_RAW
             + ["card_uid", "device_uid", "email_key", C.TIME_COL])
    for col in carry:
        if col in df.columns:
            feats[col] = df[col].values
    if C.TARGET in df.columns:
        feats[C.TARGET] = df[C.TARGET].values
    if C.ID_COL in df.columns:
        feats[C.ID_COL] = df[C.ID_COL].values
    return feats


def fit_encoders(feats: pd.DataFrame) -> Preprocessor:
    """Fit frequency encoders + numeric imputation on a (train-only) frame."""
    pre = Preprocessor()
    for col in C.CATEGORICAL_RAW:
        vc = feats[col].astype(str).value_counts(normalize=True)
        pre.freq_maps[col] = vc.to_dict()
        pre.default_freq[col] = float(vc.min()) if len(vc) else 0.0
    for col in C.NUMERIC_RAW:
        med = float(pd.to_numeric(feats[col], errors="coerce").median())
        pre.numeric_fill[col] = med if np.isfinite(med) else 0.0
    return pre


def apply_encoders(feats: pd.DataFrame, pre: Preprocessor) -> pd.DataFrame:
    """Apply fitted encoders + imputation. Safe for unseen categories."""
    feats = feats.copy()
    for col in C.CATEGORICAL_RAW:
        fmap = pre.freq_maps.get(col, {})
        default = pre.default_freq.get(col, 0.0)
        feats[f"{col}__enc"] = feats[col].astype(str).map(fmap).fillna(default)
    for col in C.NUMERIC_RAW:
        feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(
            pre.numeric_fill.get(col, 0.0))
    return feats


def fit_transform(df: pd.DataFrame) -> tuple[pd.DataFrame, Preprocessor]:
    """Convenience single-shot: engineer + fit + apply on one frame."""
    feats = engineer(df)
    pre = fit_encoders(feats)
    return apply_encoders(feats, pre), pre


def transform(df: pd.DataFrame, pre: Preprocessor) -> pd.DataFrame:
    """Convenience: engineer + apply fitted encoders."""
    return apply_encoders(engineer(df), pre)


def to_model_matrix(feats: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the ordered MODEL_FEATURES the estimator consumes."""
    missing = [c for c in C.MODEL_FEATURES if c not in feats.columns]
    for c in missing:
        feats[c] = 0.0
    return feats[C.MODEL_FEATURES].astype(float)
