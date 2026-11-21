"""Standalone scoring core, independent of FastAPI.

Both the HTTP service (app.py) and the Kafka consumer (streaming/consumer.py)
use one Scorer so batch replay and the online API score identically.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np

from ..common import config as C
from ..features.feature_engineering import Preprocessor
from . import case_store
from .online_features import OnlineFeatureStore, load_ring_cache


class Scorer:
    def __init__(self, write_cases: bool = True):
        if not C.XGB_MODEL_PATH.exists():
            raise RuntimeError(
                "No trained model. Run: python -m src.models.train_xgboost")
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(C.XGB_MODEL_PATH)
        pre = Preprocessor.from_dict(joblib.load(C.PREPROCESSOR_PATH))
        meta = json.loads(C.FEATURE_META_PATH.read_text())
        self.threshold = float(meta.get("threshold", C.DEFAULT_THRESHOLD))
        self.features = OnlineFeatureStore(pre, load_ring_cache())
        self.db = case_store.connect() if write_cases else None

    def score(self, rec: dict) -> dict:
        t0 = time.perf_counter()
        feats = self.features.build(rec)
        x = np.array([[feats[k] for k in C.MODEL_FEATURES]], dtype=float)
        proba = float(self.model.predict_proba(x)[:, 1][0])
        decision = "flag" if proba >= self.threshold else "pass"
        if self.db is not None:
            case_store.upsert_case(
                self.db, transaction_id=rec["TransactionID"],
                ts=rec.get("TransactionDT", 0), amount=rec.get("TransactionAmt", 0),
                card_uid=self.features._ids(rec)[0], score=proba, decision=decision)
        return {"transaction_id": int(rec["TransactionID"]),
                "fraud_probability": round(proba, 6), "decision": decision,
                "threshold": self.threshold,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}
