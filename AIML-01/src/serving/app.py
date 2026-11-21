"""FastAPI real-time scoring service.

POST /score with a raw transaction -> fraud probability + decision in a few ms.
The service loads the trained XGBoost model, the fitted preprocessor and the
ring-feature cache, builds online features (online_features.py) and writes every
scored transaction to the case store for the analyst dashboard.

This is the brief's "FastAPI consumer service that fetches features and returns
a fraud score". The Kafka consumer (streaming/consumer.py) calls this same
scoring core, so batch replay and the HTTP API share one code path.

Run:
    uvicorn src.serving.app:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..common import config as C
from . import case_store
from .scorer import Scorer

_STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _STATE["scorer"] = Scorer(write_cases=True)
    yield
    _STATE.clear()


app = FastAPI(title="Real-Time Fraud Scoring", version="1.0.0", lifespan=lifespan)


class Transaction(BaseModel):
    TransactionID: int
    TransactionDT: float = Field(..., description="seconds offset")
    TransactionAmt: float
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card5: float | None = None
    card4: str | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    C1: float | None = None
    C2: float | None = None
    C13: float | None = None
    C14: float | None = None
    D1: float | None = None
    D4: float | None = None
    D10: float | None = None
    D15: float | None = None
    ProductCD: str | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    id_30: str | None = None
    id_31: str | None = None

    model_config = {"extra": "allow"}


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    decision: str
    threshold: float
    latency_ms: float


@app.get("/health")
def health():
    s: Scorer | None = _STATE.get("scorer")
    return {"status": "ok", "model_loaded": s is not None,
            "threshold": s.threshold if s else None}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    try:
        return _STATE["scorer"].score(txn.model_dump())
    except Exception as e:                       # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cases")
def cases(status: str | None = None, limit: int = 50):
    return case_store.list_cases(_STATE["scorer"].db, status=status, limit=limit)


@app.get("/stats")
def stats():
    return case_store.stats(_STATE["scorer"].db)
