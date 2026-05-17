"""SQLite-backed persistence for synthesis platform background jobs."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from piern.shared.db.migrations import Migration, ensure_sqlite_schema
from piern.shared.runtime.paths import RUNLOG_ROOT
from piern.shared.tasks.state import (
    ACTIVE_STATUSES as SHARED_ACTIVE_STATUSES,
    TERMINAL_STATUSES as SHARED_TERMINAL_STATUSES,
    normalize_status,
)

JOB_STORE_PATH = Path(os.getenv("PIERN_JOB_STORE_PATH", RUNLOG_ROOT / "jobs.sqlite"))
ACTIVE_STATUSES = set(SHARED_ACTIVE_STATUSES)
TERMINAL_STATUSES = set(SHARED_TERMINAL_STATUSES)

_LOCK = RLock()
_INITIALIZED = False


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _connect() -> sqlite3.Connection:
    JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(JOB_STORE_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs(
            job_id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at REAL,
            finished_at REAL,
            pid INTEGER,
            request_json TEXT,
            scenario_totals_json TEXT NOT NULL DEFAULT '{}',
            progress_json TEXT NOT NULL DEFAULT '{}',
            stats_json TEXT NOT NULL DEFAULT '{}',
            error_message TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            ts REAL NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_started_at ON jobs(started_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_job_events_job_id_id ON job_events(job_id, id)")


def init_store() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _LOCK, _connect() as conn:
        ensure_sqlite_schema(conn, "synth_jobs", [Migration(1, "initial synth jobs", _create_schema)])
        _INITIALIZED = True


def upsert_job(
    *,
    job_id: str,
    job_type: str,
    status: str,
    started_at: float,
    scenario_totals: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
    finished_at: float | None = None,
    pid: int | None = None,
    request_json: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    init_store()
    status = normalize_status(status)
    now = time.time()
    request_payload = json.dumps(request_json, ensure_ascii=False, sort_keys=True) if request_json is not None else None
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs(
                job_id, job_type, status, started_at, finished_at, pid, request_json,
                scenario_totals_json, progress_json, stats_json, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                job_type=excluded.job_type,
                status=excluded.status,
                started_at=COALESCE(jobs.started_at, excluded.started_at),
                finished_at=excluded.finished_at,
                pid=COALESCE(excluded.pid, jobs.pid),
                request_json=COALESCE(excluded.request_json, jobs.request_json),
                scenario_totals_json=excluded.scenario_totals_json,
                progress_json=excluded.progress_json,
                stats_json=excluded.stats_json,
                error_message=excluded.error_message,
                updated_at=excluded.updated_at
            """,
            (
                job_id,
                job_type,
                status,
                started_at,
                finished_at,
                pid,
                request_payload,
                _json_dumps(scenario_totals),
                _json_dumps(progress),
                _json_dumps(stats),
                error_message,
                now,
                now,
            ),
        )


def append_event(job_id: str, event: dict[str, Any]) -> None:
    init_store()
    event_type = str(event.get("type") or "event")
    ts = float(event.get("ts") or time.time())
    with _LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO job_events(job_id, event_type, ts, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, event_type, ts, json.dumps(event, ensure_ascii=False, sort_keys=True)),
        )


def _event_rows(conn: sqlite3.Connection, job_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT payload_json FROM job_events WHERE job_id=? ORDER BY id ASC",
        (job_id,),
    ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        payload = _json_loads(row["payload_json"], None)
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _row_to_job(row: sqlite3.Row, *, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "job_type": row["job_type"],
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "pid": row["pid"],
        "request_json": _json_loads(row["request_json"], {}),
        "scenario_totals": _json_loads(row["scenario_totals_json"], {}),
        "progress": _json_loads(row["progress_json"], {}),
        "stats": _json_loads(row["stats_json"], {}),
        "error_message": row["error_message"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "events": events or [],
    }


def load_job(job_id: str) -> dict[str, Any] | None:
    init_store()
    with _LOCK, _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return _row_to_job(row, events=_event_rows(conn, job_id))


def list_jobs(
    *,
    job_type: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    init_store()
    clauses: list[str] = []
    params: list[Any] = []
    if job_type:
        clauses.append("job_type=?")
        params.append(job_type)
    if status:
        clauses.append("status=?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, int(limit)))
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM jobs {where} ORDER BY started_at DESC, created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_job(row, events=_event_rows(conn, row["job_id"])) for row in rows]


def mark_incomplete_external_terminated(active_job_ids: Iterable[str] = ()) -> list[str]:
    init_store()
    active_ids = set(active_job_ids)
    now = time.time()
    message = "服务重启或任务执行器消失，任务已标记为外部终止。"
    updated: list[str] = []
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT job_id FROM jobs WHERE status IN ('running', 'starting', 'stopping')"
        ).fetchall()
        for row in rows:
            job_id = str(row["job_id"])
            if job_id in active_ids:
                continue
            event = {
                "type": "external_terminated",
                "ts": now,
                "message": message,
            }
            conn.execute(
                """
                UPDATE jobs
                SET status='external_terminated', finished_at=?, error_message=?, updated_at=?
                WHERE job_id=?
                """,
                (now, message, now, job_id),
            )
            conn.execute(
                """
                INSERT INTO job_events(job_id, event_type, ts, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, "external_terminated", now, json.dumps(event, ensure_ascii=False, sort_keys=True)),
            )
            updated.append(job_id)
    return updated
