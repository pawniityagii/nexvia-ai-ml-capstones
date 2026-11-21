"""Online feature computation for a single incoming transaction.

At serving time we cannot recompute history from a batch frame, so we keep a
small rolling state per card (count, running mean/var of amount, recent
timestamps, last device/region). This reproduces the SAME engineered features
the training pipeline builds, so train/serve stay consistent.

In production this state lives in the feature store (Feast online store backed
by Redis) - the brief's "fetch features" step. Here it is an in-process dict so
the demo runs with no external services. Ring features come from a cache written
at training time (card_uid -> ring_degree, ring_fraud_rate); unseen cards get 0.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque, defaultdict

from ..common import config as C
from ..features.feature_engineering import Preprocessor


def _hash(values) -> str:
    return hashlib.md5("|".join(str(v) for v in values).encode()).hexdigest()[:16]


class _CardState:
    __slots__ = ("count", "sum", "sumsq", "times", "last_device", "last_addr")

    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.times = deque(maxlen=500)          # recent timestamps for velocity
        self.last_device = None
        self.last_addr = None


class OnlineFeatureStore:
    def __init__(self, preprocessor: Preprocessor, ring_cache: dict | None = None):
        self.pre = preprocessor
        self.ring_cache = ring_cache or {}
        self._state: dict[str, _CardState] = defaultdict(_CardState)

    def _ids(self, rec: dict):
        card_uid = _hash(rec.get(k, "") for k in C.CARD_KEYS)
        device_uid = _hash(rec.get(k, "") for k in C.DEVICE_KEYS)
        return card_uid, device_uid

    def build(self, rec: dict, update_state: bool = True) -> dict:
        """Return the ordered MODEL_FEATURES dict for one raw transaction."""
        card_uid, device_uid = self._ids(rec)
        st = self._state[card_uid]
        t = float(rec.get(C.TIME_COL, 0) or 0)
        amt = float(rec.get(C.AMOUNT_COL, 0) or 0)

        # --- time / velocity features (prior state only) ---
        if st.count > 0:
            mean = st.sum / st.count
            var = max(0.0, st.sumsq / st.count - mean * mean)
            std = math.sqrt(var)
            amt_to_mean = amt / mean if mean > 0 else 1.0
            amt_to_std = (amt - mean) / std if std > 0 else 0.0
            secs_prev = t - st.times[-1] if st.times else 1e7
        else:
            amt_to_mean, amt_to_std, secs_prev = 1.0, 0.0, 1e7
        vel_1h = sum(1 for tt in st.times if t - tt <= 3600)
        vel_24h = sum(1 for tt in st.times if t - tt <= 86400)
        device_change = int(st.last_device is not None and device_uid != st.last_device)
        addr_change = int(st.last_addr is not None and rec.get("addr1") != st.last_addr)

        ring = self.ring_cache.get(card_uid, {"ring_degree": 1.0, "ring_fraud_rate": 0.0})

        feat = {
            "amt_log": math.log1p(max(0.0, amt)),
            "amt_to_card_mean": amt_to_mean,
            "amt_to_card_std": amt_to_std,
            "txn_velocity_card_1h": vel_1h,
            "txn_velocity_card_24h": vel_24h,
            "secs_since_prev_card_txn": secs_prev,
            "device_change_flag": device_change,
            "addr_change_flag": addr_change,
            "hour_of_day": int((t // 3600) % 24),
            "ring_degree": float(ring["ring_degree"]),
            "ring_fraud_rate": float(ring["ring_fraud_rate"]),
        }
        # raw numerics (imputed) + categorical freq-encodings
        for col in C.NUMERIC_RAW:
            v = rec.get(col)
            feat[col] = float(v) if v not in (None, "") and _isnum(v) \
                else self.pre.numeric_fill.get(col, 0.0)
        for col in C.CATEGORICAL_RAW:
            fmap = self.pre.freq_maps.get(col, {})
            feat[f"{col}__enc"] = fmap.get(str(rec.get(col, "missing")),
                                           self.pre.default_freq.get(col, 0.0))

        if update_state:
            st.count += 1
            st.sum += amt
            st.sumsq += amt * amt
            st.times.append(t)
            st.last_device = device_uid
            st.last_addr = rec.get("addr1")

        return {k: feat.get(k, 0.0) for k in C.MODEL_FEATURES}


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def load_ring_cache() -> dict:
    """card_uid -> ring features, written by scripts/export_ring_cache.py."""
    path = C.ARTIFACTS_DIR / "ring_cache.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}
