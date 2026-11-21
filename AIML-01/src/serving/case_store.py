"""SQLite-backed case store shared by the scoring service and the dashboard.

Chosen over a heavier database so the project runs anywhere with zero setup.
The schema is the analyst's work queue: each scored transaction becomes a case
with a status and an optional feedback label (the loop the brief asks for).
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from ..common import config as C

DB_PATH = C.ARTIFACTS_DIR / "cases.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    transaction_id INTEGER PRIMARY KEY,
    ts             REAL,
    amount         REAL,
    card_uid       TEXT,
    score          REAL,
    decision       TEXT,      -- 'flag' | 'pass'
    status         TEXT,      -- 'open' | 'confirmed_fraud' | 'cleared'
    feedback       INTEGER,   -- analyst label: 1 fraud, 0 legit, NULL none
    created_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_cases_score ON cases(score DESC);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
"""


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def upsert_case(conn, *, transaction_id, ts, amount, card_uid, score, decision):
    conn.execute(
        """INSERT INTO cases (transaction_id, ts, amount, card_uid, score,
                              decision, status, feedback, created_at)
           VALUES (?,?,?,?,?,?, 'open', NULL, ?)
           ON CONFLICT(transaction_id) DO UPDATE SET
               score=excluded.score, decision=excluded.decision""",
        (int(transaction_id), float(ts), float(amount), str(card_uid),
         float(score), decision, time.time()))
    conn.commit()


def list_cases(conn, status: str | None = None, limit: int = 100):
    q = "SELECT * FROM cases"
    args: list = []
    if status:
        q += " WHERE status = ?"
        args.append(status)
    q += " ORDER BY score DESC LIMIT ?"
    args.append(limit)
    return [dict(r) for r in conn.execute(q, args).fetchall()]


def set_status(conn, transaction_id: int, status: str):
    conn.execute("UPDATE cases SET status=? WHERE transaction_id=?",
                 (status, int(transaction_id)))
    conn.commit()


def set_feedback(conn, transaction_id: int, label: int):
    conn.execute("UPDATE cases SET feedback=? WHERE transaction_id=?",
                 (int(label), int(transaction_id)))
    conn.commit()


def stats(conn) -> dict:
    row = conn.execute(
        """SELECT COUNT(*) n,
                  SUM(CASE WHEN decision='flag' THEN 1 ELSE 0 END) flagged,
                  SUM(CASE WHEN status='confirmed_fraud' THEN 1 ELSE 0 END) confirmed,
                  SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) open_cases
           FROM cases""").fetchone()
    return {k: (row[k] or 0) for k in ("n", "flagged", "confirmed", "open_cases")}
