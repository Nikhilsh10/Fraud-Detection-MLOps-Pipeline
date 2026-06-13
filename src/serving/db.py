"""
src/serving/db.py
------------------
Thin SQLite wrapper for the predictions log.

Table: predictions_log
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  ts              TEXT     — ISO-8601 UTC timestamp
  fraud_prob      REAL     — raw model probability
  is_fraud        INTEGER  — 0 or 1
  model_version   TEXT
  amount          REAL     — echoed from input for drift monitoring
  time_feature    REAL     — echoed from input

The DB file path defaults to "predictions.db" (relative to CWD),
which works both locally (project root) and inside Docker (WORKDIR=/app).
Override via the PREDICTIONS_DB_PATH environment variable.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

_DB_PATH = Path(os.getenv("PREDICTIONS_DB_PATH", "predictions.db"))

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    fraud_prob    REAL    NOT NULL,
    is_fraud      INTEGER NOT NULL,
    model_version TEXT    NOT NULL,
    amount        REAL,
    time_feature  REAL
);
"""


def init_db(path: Path | None = None) -> Path:
    """
    Create the database file and ensure the predictions_log table exists.
    Returns the resolved path to the DB file.
    """
    db_path = path or _DB_PATH
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_CREATE_TABLE)
        conn.commit()
    finally:
        conn.close()
    return db_path


@contextmanager
def get_conn(path: Path | None = None):
    """Context manager yielding an open sqlite3.Connection (auto-closed)."""
    db_path = path or _DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_prediction(
    *,
    ts: str,
    fraud_prob: float,
    is_fraud: bool,
    model_version: str,
    amount: float | None = None,
    time_feature: float | None = None,
    path: Path | None = None,
) -> int:
    """
    Write one prediction row and return its auto-incremented id.
    """
    sql = """
    INSERT INTO predictions_log
        (ts, fraud_prob, is_fraud, model_version, amount, time_feature)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_conn(path) as conn:
        cur = conn.execute(
            sql,
            (ts, fraud_prob, int(is_fraud), model_version, amount, time_feature),
        )
        conn.commit()
        return cur.lastrowid


def count_predictions(path: Path | None = None) -> tuple[int, int]:
    """
    Returns (total_count, fraud_count) from predictions_log.
    """
    with get_conn(path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0]
        fraud = conn.execute(
            "SELECT COUNT(*) FROM predictions_log WHERE is_fraud = 1"
        ).fetchone()[0]
    return int(total), int(fraud)
