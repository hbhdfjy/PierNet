"""SQLite-backed audit event log."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any

from PierNet.shared.db.migrations import Migration, ensure_sqlite_schema
from PierNet.shared.runtime.paths import RUNLOG_ROOT

AUDIT_STORE_PATH = Path(os.getenv("PierNet_AUDIT_STORE_PATH", RUNLOG_ROOT / "audit_events.sqlite"))

_LOCK = RLock()
_INITIALIZED = False


def _connect() -> sqlite3.Connection:
    AUDIT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUDIT_STORE_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT NOT NULL,
            method TEXT,
            path TEXT,
            status_code INTEGER,
            request_id TEXT,
            client TEXT,
            details_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_ts ON audit_events(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_action ON audit_events(action)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_target ON audit_events(target)")


def init_store() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _LOCK, _connect() as conn:
        ensure_sqlite_schema(conn, "audit_events", [Migration(1, "initial audit events", _create_schema)])
        _INITIALIZED = True


def append_event(
    *,
    actor: str,
    action: str,
    target: str,
    method: str | None = None,
    path: str | None = None,
    status_code: int | None = None,
    request_id: str | None = None,
    client: str | None = None,
    details: dict[str, Any] | None = None,
    ts: float | None = None,
) -> None:
    init_store()
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO audit_events(
                ts, actor, action, target, method, path, status_code, request_id, client, details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time() if ts is None else ts,
                actor,
                action,
                target,
                method,
                path,
                status_code,
                request_id,
                client,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True, default=str),
            ),
        )


def list_events(*, limit: int = 200, action: str | None = None, target: str | None = None) -> list[dict[str, Any]]:
    init_store()
    clauses: list[str] = []
    params: list[Any] = []
    if action:
        clauses.append("action=?")
        params.append(action)
    if target:
        clauses.append("target LIKE ?")
        params.append(f"{target}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 1000)))
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, ts, actor, action, target, method, path, status_code, request_id, client, details_json
            FROM audit_events
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        try:
            details = json.loads(row["details_json"] or "{}")
        except json.JSONDecodeError:
            details = {}
        events.append(
            {
                "id": row["id"],
                "ts": row["ts"],
                "actor": row["actor"],
                "action": row["action"],
                "target": row["target"],
                "method": row["method"],
                "path": row["path"],
                "status_code": row["status_code"],
                "request_id": row["request_id"],
                "client": row["client"],
                "details": details,
            }
        )
    return events
