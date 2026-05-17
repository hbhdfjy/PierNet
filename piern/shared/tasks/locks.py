"""SQLite-backed cooperative task locks."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from piern.shared.db.migrations import Migration, ensure_sqlite_schema
from piern.shared.runtime.paths import RUNLOG_ROOT

LOCK_STORE_PATH = Path(os.getenv("PIERN_LOCK_STORE_PATH", RUNLOG_ROOT / "job_locks.sqlite"))
DEFAULT_TTL_SECONDS = float(os.getenv("PIERN_LOCK_TTL_SECONDS", str(24 * 3600)))

_LOCK = RLock()
_INITIALIZED = False


def _connect() -> sqlite3.Connection:
    LOCK_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LOCK_STORE_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_locks(
            lock_key TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            acquired_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_job_locks_expires_at ON job_locks(expires_at)")


def init_store() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _LOCK, _connect() as conn:
        ensure_sqlite_schema(conn, "shared_job_locks", [Migration(1, "initial job locks", _create_schema)])
        _INITIALIZED = True


def _json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, default=str)


def acquire_lock(
    lock_key: str,
    owner: str,
    *,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    metadata: dict[str, Any] | None = None,
) -> bool:
    init_store()
    now = time.time()
    expires_at = now + max(1.0, float(ttl_seconds))
    with _LOCK, _connect() as conn:
        row = conn.execute("SELECT owner, expires_at FROM job_locks WHERE lock_key=?", (lock_key,)).fetchone()
        if row is not None and row["owner"] != owner and float(row["expires_at"]) > now:
            return False
        conn.execute(
            """
            INSERT INTO job_locks(lock_key, owner, acquired_at, expires_at, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(lock_key) DO UPDATE SET
                owner=excluded.owner,
                acquired_at=excluded.acquired_at,
                expires_at=excluded.expires_at,
                metadata_json=excluded.metadata_json
            """,
            (lock_key, owner, now, expires_at, _json(metadata)),
        )
    return True


def refresh_lock(lock_key: str, owner: str, *, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> bool:
    init_store()
    expires_at = time.time() + max(1.0, float(ttl_seconds))
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "UPDATE job_locks SET expires_at=? WHERE lock_key=? AND owner=?",
            (expires_at, lock_key, owner),
        )
    return cur.rowcount > 0


def release_lock(lock_key: str, owner: str | None = None) -> bool:
    init_store()
    query = "DELETE FROM job_locks WHERE lock_key=?"
    params: tuple[Any, ...] = (lock_key,)
    if owner is not None:
        query += " AND owner=?"
        params = (lock_key, owner)
    with _LOCK, _connect() as conn:
        cur = conn.execute(query, params)
    return cur.rowcount > 0


def release_owner(owner: str) -> int:
    init_store()
    with _LOCK, _connect() as conn:
        cur = conn.execute("DELETE FROM job_locks WHERE owner=?", (owner,))
    return int(cur.rowcount or 0)


def cleanup_expired(now: float | None = None) -> int:
    init_store()
    cutoff = time.time() if now is None else float(now)
    with _LOCK, _connect() as conn:
        cur = conn.execute("DELETE FROM job_locks WHERE expires_at<=?", (cutoff,))
    return int(cur.rowcount or 0)


def list_locks(*, prefix: str | None = None, include_expired: bool = False) -> list[dict[str, Any]]:
    init_store()
    cleanup_expired()
    clauses: list[str] = []
    params: list[Any] = []
    if prefix:
        clauses.append("lock_key LIKE ?")
        params.append(f"{prefix}%")
    if not include_expired:
        clauses.append("expires_at > ?")
        params.append(time.time())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            f"SELECT lock_key, owner, acquired_at, expires_at, metadata_json FROM job_locks {where} ORDER BY lock_key",
            params,
        ).fetchall()
    result = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        result.append(
            {
                "lock_key": row["lock_key"],
                "owner": row["owner"],
                "acquired_at": row["acquired_at"],
                "expires_at": row["expires_at"],
                "metadata": metadata,
            }
        )
    return result


@contextmanager
def task_lock(
    lock_key: str,
    owner: str,
    *,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    if not acquire_lock(lock_key, owner, ttl_seconds=ttl_seconds, metadata=metadata):
        raise RuntimeError(f"resource is locked: {lock_key}")
    try:
        yield
    finally:
        release_lock(lock_key, owner)
