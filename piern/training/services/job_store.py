"""SQLite task store and event audit for training jobs.

This is the primary runtime store for training job metadata. The previous JSON
registry is intentionally no longer used as the source of truth.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any

from piern.shared.db.migrations import Migration, ensure_sqlite_schema
from piern.shared.runtime.paths import RUNLOG_ROOT
from piern.shared.tasks.state import normalize_status

TRAINING_JOB_STORE_PATH = Path(os.getenv("PIERN_TRAINING_JOB_STORE_PATH", RUNLOG_ROOT / "training_jobs.sqlite"))

_LOCK = RLock()
_INITIALIZED = False


def _connect() -> sqlite3.Connection:
    TRAINING_JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(TRAINING_JOB_STORE_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_jobs(
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            simulator TEXT,
            gpu_id INTEGER,
            pid INTEGER,
            created_at REAL,
            started_at REAL,
            ended_at REAL,
            request_json TEXT,
            snapshot_json TEXT NOT NULL,
            error_message TEXT,
            deleted INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_job_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            ts REAL NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES training_jobs(job_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_created ON training_jobs(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_training_events_job_id_id ON training_job_events(job_id, id)")


def init_store() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _LOCK, _connect() as conn:
        ensure_sqlite_schema(conn, "training_jobs", [Migration(1, "initial training jobs", _create_schema)])
        _INITIALIZED = True


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _append_event_locked(conn: sqlite3.Connection, job_id: str, event_type: str, payload: dict[str, Any]) -> None:
    ts = float(payload.get("ts") or time.time())
    conn.execute(
        """
        INSERT INTO training_job_events(job_id, event_type, ts, payload_json)
        VALUES (?, ?, ?, ?)
        """,
        (job_id, event_type, ts, _json(payload)),
    )


def upsert_job(entry: dict[str, Any]) -> None:
    init_store()
    job_id = str(entry["job_id"])
    status = normalize_status(entry.get("status"), fallback="external_terminated")
    now = time.time()
    request_json = {
        "config": entry.get("config") or {},
        "command": entry.get("command") or [],
        "scenarios": entry.get("scenarios") or [],
    }
    with _LOCK, _connect() as conn:
        previous = conn.execute(
            "SELECT status, deleted FROM training_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO training_jobs(
                job_id, status, simulator, gpu_id, pid, created_at, started_at, ended_at,
                request_json, snapshot_json, error_message, deleted, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status=excluded.status,
                simulator=excluded.simulator,
                gpu_id=excluded.gpu_id,
                pid=excluded.pid,
                created_at=COALESCE(training_jobs.created_at, excluded.created_at),
                started_at=excluded.started_at,
                ended_at=excluded.ended_at,
                request_json=excluded.request_json,
                snapshot_json=excluded.snapshot_json,
                error_message=excluded.error_message,
                deleted=0,
                updated_at=excluded.updated_at
            """,
            (
                job_id,
                status,
                entry.get("simulator"),
                entry.get("gpu_id"),
                entry.get("pid"),
                entry.get("created_at"),
                entry.get("started_at"),
                entry.get("ended_at"),
                _json(request_json),
                _json(entry),
                entry.get("error_message"),
                now,
            ),
        )
        if previous is None:
            _append_event_locked(conn, job_id, "created", {"ts": now, "status": status})
        elif previous["status"] != status:
            _append_event_locked(
                conn,
                job_id,
                "status_changed",
                {"ts": now, "from": previous["status"], "to": status},
            )


def save_jobs(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        upsert_job(entry)


def mark_deleted(job_id: str) -> None:
    init_store()
    now = time.time()
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE training_jobs SET deleted=1, status='deleted', updated_at=? WHERE job_id=?",
            (now, job_id),
        )
        _append_event_locked(conn, job_id, "deleted", {"ts": now, "status": "deleted"})


def list_job_snapshots(include_deleted: bool = False) -> list[dict[str, Any]]:
    init_store()
    where = "" if include_deleted else "WHERE deleted=0"
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            f"SELECT snapshot_json FROM training_jobs {where} ORDER BY created_at DESC, updated_at DESC"
        ).fetchall()
    snapshots = []
    for row in rows:
        snapshot = _loads(row["snapshot_json"], None)
        if isinstance(snapshot, dict):
            snapshots.append(snapshot)
    return snapshots


def list_events(job_id: str) -> list[dict[str, Any]]:
    init_store()
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT event_type, ts, payload_json FROM training_job_events WHERE job_id=? ORDER BY id ASC",
            (job_id,),
        ).fetchall()
    return [
        {"event_type": row["event_type"], "ts": row["ts"], "payload": _loads(row["payload_json"], {})}
        for row in rows
    ]
