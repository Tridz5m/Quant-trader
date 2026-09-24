"""SQLite trade journal: trades, signals, equity, model history and bot state."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    ticket INTEGER PRIMARY KEY,
    symbol TEXT,
    direction INTEGER,
    volume REAL,
    entry_time TEXT,
    bar_time TEXT,
    entry_price REAL,
    sl REAL,
    tp REAL,
    initial_risk REAL,
    risk_money REAL,
    atr REAL,
    p_long REAL,
    p_short REAL,
    model_version TEXT,
    features TEXT,
    status TEXT DEFAULT 'open',
    exit_time TEXT,
    exit_price REAL,
    profit REAL,
    r_multiple REAL,
    close_reason TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    time TEXT,
    bar_time TEXT,
    price REAL,
    spread REAL,
    atr REAL,
    p_long REAL,
    p_short REAL,
    thr_long REAL,
    thr_short REAL,
    decision INTEGER,
    action TEXT,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS equity (time TEXT, balance REAL, equity REAL);
CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    created_at TEXT,
    trained_until TEXT,
    passed INTEGER,
    promoted INTEGER,
    reason TEXT,
    metrics TEXT
);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
"""


def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- trades -----------------------------------------------------------
    def record_entry(self, **fields: Any) -> None:
        if isinstance(fields.get("features"), dict):
            fields["features"] = json.dumps({k: _clean(v) for k, v in fields["features"].items()})
        fields = {k: _clean(v) for k, v in fields.items()}
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        self.conn.execute(f"INSERT OR REPLACE INTO trades ({cols}) VALUES ({marks})", list(fields.values()))
        self.conn.commit()

    def get_trade(self, ticket: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,)).fetchone()
        return dict(row) if row else None

    def open_trades(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time").fetchall()
        return [dict(r) for r in rows]

    def update_sl(self, ticket: int, sl: float) -> None:
        self.conn.execute("UPDATE trades SET sl = ? WHERE ticket = ?", (sl, ticket))
        self.conn.commit()

    def close_trade(self, ticket: int, exit_time: pd.Timestamp, exit_price: float, profit: float, reason: str) -> float | None:
        row = self.get_trade(ticket)
        r_mult = None
        if row and row.get("risk_money"):
            r_mult = profit / row["risk_money"]
        self.conn.execute(
            "UPDATE trades SET status='closed', exit_time=?, exit_price=?, profit=?, r_multiple=?, close_reason=? WHERE ticket=?",
            (_clean(exit_time), exit_price, profit, r_mult, reason, ticket),
        )
        self.conn.commit()
        return r_mult

    def closed_trades(self, limit: int | None = None) -> list[dict]:
        q = "SELECT * FROM trades WHERE status = 'closed' ORDER BY exit_time DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        rows = [dict(r) for r in self.conn.execute(q).fetchall()]
        return rows[::-1]  # oldest first

    def recent_r(self, n: int = 50) -> list[float]:
        return [t["r_multiple"] for t in self.closed_trades(n) if t["r_multiple"] is not None]

    # --- signals / equity -------------------------------------------------
    def log_signal(self, **fields: Any) -> None:
        fields = {k: _clean(v) for k, v in fields.items()}
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        self.conn.execute(f"INSERT INTO signals ({cols}) VALUES ({marks})", list(fields.values()))
        self.conn.commit()

    def log_equity(self, time: pd.Timestamp, balance: float, equity: float) -> None:
        self.conn.execute("INSERT INTO equity VALUES (?, ?, ?)", (_clean(time), balance, equity))
        self.conn.commit()

    # --- models -----------------------------------------------------------
    def log_model(self, version: str, created_at: str, trained_until: str, passed: bool, promoted: bool, reason: str, metrics: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?)",
            (version, created_at, trained_until, int(passed), int(promoted), reason, json.dumps(metrics, default=str)),
        )
        self.conn.commit()

    def models(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # --- key/value state --------------------------------------------------
    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO state VALUES (?, ?)", (key, json.dumps(value, default=str)))
        self.conn.commit()
