"""SQLite store for studies, predictions, radiologist decisions and the audit log.

Every decision (approve/reject) writes an immutable audit row - the brief
requires "an audit log of every decision". Chosen over a heavier DB so the
project runs anywhere with no setup.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from ..common import config as C

DB_PATH = C.ARTIFACTS_DIR / "cases.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT,
    patient_id    TEXT,
    modality      TEXT,
    probability   REAL,
    label         TEXT,
    model_mode    TEXT,          -- 'model' | 'demo-heuristic'
    status        TEXT,          -- 'pending' | 'approved' | 'rejected'
    overlay_path  TEXT,
    created_at    REAL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    INTEGER,
    action     TEXT,
    actor      TEXT,
    note       TEXT,
    ts         REAL,
    FOREIGN KEY(case_id) REFERENCES cases(id)
);
"""


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def add_case(conn, *, filename, patient_id, modality, probability, label,
             model_mode, overlay_path) -> int:
    cur = conn.execute(
        """INSERT INTO cases (filename, patient_id, modality, probability, label,
                              model_mode, status, overlay_path, created_at)
           VALUES (?,?,?,?,?,?, 'pending', ?, ?)""",
        (filename, patient_id, modality, float(probability), label,
         model_mode, overlay_path, time.time()))
    conn.commit()
    case_id = int(cur.lastrowid)
    _audit(conn, case_id, "created", "system",
           f"auto-scored label={label} p={probability:.3f} mode={model_mode}")
    return case_id


def _audit(conn, case_id, action, actor, note):
    conn.execute(
        "INSERT INTO audit_log (case_id, action, actor, note, ts) VALUES (?,?,?,?,?)",
        (case_id, action, actor, note, time.time()))
    conn.commit()


def set_decision(conn, case_id: int, decision: str, actor: str, note: str = ""):
    if decision not in ("approve", "reject"):
        raise ValueError("decision must be 'approve' or 'reject'")
    status = "approved" if decision == "approve" else "rejected"
    conn.execute("UPDATE cases SET status=? WHERE id=?", (status, int(case_id)))
    conn.commit()
    _audit(conn, case_id, decision, actor, note)
    return status


def get_case(conn, case_id: int):
    row = conn.execute("SELECT * FROM cases WHERE id=?", (int(case_id),)).fetchone()
    return dict(row) if row else None


def list_cases(conn, status: str | None = None, limit: int = 100):
    q = "SELECT * FROM cases"
    args: list = []
    if status:
        q += " WHERE status=?"
        args.append(status)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    return [dict(r) for r in conn.execute(q, args).fetchall()]


def list_audit(conn, case_id: int):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM audit_log WHERE case_id=? ORDER BY ts", (int(case_id),)).fetchall()]


def stats(conn) -> dict:
    row = conn.execute(
        """SELECT COUNT(*) n,
                  SUM(status='pending') pending,
                  SUM(status='approved') approved,
                  SUM(status='rejected') rejected,
                  SUM(label='abnormal') abnormal
           FROM cases""").fetchone()
    return {k: (row[k] or 0) for k in ("n", "pending", "approved", "rejected", "abnormal")}
