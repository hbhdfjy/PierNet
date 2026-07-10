"""SQLite-backed worker heartbeat registry."""

from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Iterator
from threading import Event, RLock, Thread
from typing import Any

from PierNet.shared.db.migrations import Migration, ensure_sqlite_schema
from PierNet.shared.runtime.paths import RUNLOG_ROOT

WORKER_STORE_PATH = Path(os.getenv("PierNet_WORKER_STORE_PATH", RUNLOG_ROOT / "worker_heartbeats.sqlite"))
DEFAULT_STALE_AFTER_SECONDS = float(os.getenv("PierNet_WORKER_STALE_AFTER_SECONDS", "90"))

LOGGER = logging.getLogger(__name__)

_LOCK = RLock()
_INITIALIZED = False


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


@contextmanager
def heartbeat_while(
    *,
    worker_id: str | None = None,
    kind: str = "PierNet-worker",
    status: str = "running",
    current_job_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    interval: float = 10.0,
) -> Iterator[str]:
    """Refresh worker heartbeat while a blocking job is running."""

    wid = worker_id or default_worker_id()
    refresh_interval = max(0.5, float(interval))
    stop_event = Event()
    upsert_worker(
        worker_id=wid,
        kind=kind,
        status=status,
        current_job_id=current_job_id,
        metadata=metadata,
    )

    def _refresh_loop() -> None:
        while not stop_event.wait(refresh_interval):
            try:
                upsert_worker(
                    worker_id=wid,
                    kind=kind,
                    status=status,
                    current_job_id=current_job_id,
                    metadata=metadata,
                )
            except Exception:
                LOGGER.exception("worker heartbeat refresh failed worker_id=%s", wid)

    thread = Thread(target=_refresh_loop, name=f"PierNet-heartbeat-{wid}", daemon=True)
    thread.start()
    try:
        yield wid
    finally:
        stop_event.set()
        thread.join(timeout=min(2.0, refresh_interval))
        try:
            upsert_worker(worker_id=wid, kind=kind, status=status, current_job_id=None, metadata=metadata)
        except Exception:
            LOGGER.exception("worker heartbeat clear failed worker_id=%s", wid)


def _connect() -> sqlite3.Connection:
    WORKER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(WORKER_STORE_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS worker_heartbeats(
            worker_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            host TEXT NOT NULL,
            pid INTEGER NOT NULL,
            started_at REAL NOT NULL,
            heartbeat_at REAL NOT NULL,
            status TEXT NOT NULL,
            current_job_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_worker_heartbeats_kind ON worker_heartbeats(kind)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_worker_heartbeats_heartbeat ON worker_heartbeats(heartbeat_at)")


def init_store() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _LOCK, _connection() as conn:
        ensure_sqlite_schema(conn, "worker_heartbeats", [Migration(1, "initial worker heartbeats", _create_schema)])
        _INITIALIZED = True


def upsert_worker(
    *,
    worker_id: str | None = None,
    kind: str = "maintenance",
    status: str = "running",
    current_job_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    init_store()
    wid = worker_id or default_worker_id()
    now = time.time()
    host = socket.gethostname()
    pid = os.getpid()
    with _LOCK, _connection() as conn:
        conn.execute(
            """
            INSERT INTO worker_heartbeats(
                worker_id, kind, host, pid, started_at, heartbeat_at, status, current_job_id, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                kind=excluded.kind,
                host=excluded.host,
                pid=excluded.pid,
                heartbeat_at=excluded.heartbeat_at,
                status=excluded.status,
                current_job_id=excluded.current_job_id,
                metadata_json=excluded.metadata_json
            """,
            (
                wid,
                kind,
                host,
                pid,
                now,
                now,
                status,
                current_job_id,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, default=str),
            ),
        )
    return wid


def mark_worker_stopped(worker_id: str | None = None) -> None:
    init_store()
    wid = worker_id or default_worker_id()
    with _LOCK, _connection() as conn:
        conn.execute(
            "UPDATE worker_heartbeats SET status='stopped', heartbeat_at=?, current_job_id=NULL WHERE worker_id=?",
            (time.time(), wid),
        )


def list_workers(*, stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS) -> list[dict[str, Any]]:
    init_store()
    now = time.time()
    with _LOCK, _connection() as conn:
        rows = conn.execute(
            """
            SELECT worker_id, kind, host, pid, started_at, heartbeat_at, status, current_job_id, metadata_json
            FROM worker_heartbeats
            ORDER BY heartbeat_at DESC
            """
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        age = max(0.0, now - float(row["heartbeat_at"] or 0.0))
        status = row["status"]
        if status == "running" and age > stale_after_seconds:
            status = "stale"
        result.append(
            {
                "worker_id": row["worker_id"],
                "kind": row["kind"],
                "host": row["host"],
                "pid": row["pid"],
                "started_at": row["started_at"],
                "heartbeat_at": row["heartbeat_at"],
                "heartbeat_age_seconds": age,
                "status": status,
                "current_job_id": row["current_job_id"],
                "metadata": metadata,
            }
        )
    return result
