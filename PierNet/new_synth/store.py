from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterator

from PierNet.new_synth.paths import NEW_SYNTH_DB_PATH, ensure_roots

_LOCK = RLock()
_INITIALIZED = False


def _dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load(value: str | None) -> Any:
    return json.loads(value) if value else None


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    NEW_SYNTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(NEW_SYNTH_DB_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_store() -> None:
    global _INITIALIZED
    with _LOCK:
        if _INITIALIZED:
            return
        ensure_roots()
        with _connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflows(
                    workflow_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_step TEXT NOT NULL,
                    source_json TEXT,
                    definition_json TEXT,
                    artifacts_json TEXT,
                    error_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_new_synth_workflows_owner
                    ON workflows(owner_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS workflow_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_new_synth_events_workflow
                    ON workflow_events(workflow_id, id);

                CREATE TABLE IF NOT EXISTS datasets(
                    dataset_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    simulator TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    path TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_new_synth_datasets_kind
                    ON datasets(kind, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_new_synth_datasets_workflow
                    ON datasets(workflow_id, kind);
                """
            )
            running = connection.execute(
                "SELECT workflow_id FROM workflows WHERE status='running'"
            ).fetchall()
            now = time.time()
            for row in running:
                error = {
                    "code": "service_restarted",
                    "message": "服务重启中断了生成任务，可以安全重试。",
                }
                connection.execute(
                    """
                    UPDATE workflows
                    SET status='failed', error_json=?, cancel_requested=0, updated_at=?
                    WHERE workflow_id=?
                    """,
                    (_dump(error), now, row["workflow_id"]),
                )
        _INITIALIZED = True


def _workflow_from_row(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    for key in ("source_json", "definition_json", "artifacts_json", "error_json"):
        value[key.removesuffix("_json")] = _load(value.pop(key))
    value["cancel_requested"] = bool(value["cancel_requested"])
    return value


def create_workflow(owner_id: str, name: str) -> dict[str, Any]:
    init_store()
    workflow_id = f"new-synth-{uuid.uuid4().hex[:12]}"
    now = time.time()
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO workflows(
                workflow_id, owner_id, name, status, current_step, created_at, updated_at
            ) VALUES(?, ?, ?, 'draft', 'source', ?, ?)
            """,
            (workflow_id, owner_id, name.strip(), now, now),
        )
    append_event(workflow_id, "workflow_created", {"message": "数据合成任务已创建"})
    return get_workflow(owner_id, workflow_id)


def list_workflows(owner_id: str) -> list[dict[str, Any]]:
    init_store()
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM workflows WHERE owner_id=? ORDER BY updated_at DESC",
            (owner_id,),
        ).fetchall()
    return [_workflow_from_row(row) for row in rows]


def get_workflow(owner_id: str, workflow_id: str) -> dict[str, Any]:
    init_store()
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM workflows WHERE owner_id=? AND workflow_id=?",
            (owner_id, workflow_id),
        ).fetchone()
    if row is None:
        raise KeyError(workflow_id)
    return _workflow_from_row(row)


def get_workflow_unscoped(workflow_id: str) -> dict[str, Any]:
    init_store()
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,)
        ).fetchone()
    if row is None:
        raise KeyError(workflow_id)
    return _workflow_from_row(row)


def update_workflow(workflow_id: str, **fields: Any) -> dict[str, Any]:
    init_store()
    allowed = {
        "name",
        "status",
        "current_step",
        "source",
        "definition",
        "artifacts",
        "error",
        "cancel_requested",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"unsupported workflow fields: {sorted(unknown)}")
    assignments: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        if key in {"source", "definition", "artifacts", "error"}:
            assignments.append(f"{key}_json=?")
            values.append(_dump(value))
        elif key == "cancel_requested":
            assignments.append("cancel_requested=?")
            values.append(int(bool(value)))
        else:
            assignments.append(f"{key}=?")
            values.append(value)
    assignments.append("updated_at=?")
    values.extend([time.time(), workflow_id])
    with _LOCK, _connection() as connection:
        connection.execute(
            f"UPDATE workflows SET {', '.join(assignments)} WHERE workflow_id=?",
            values,
        )
    return get_workflow_unscoped(workflow_id)


def append_event(workflow_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    init_store()
    now = time.time()
    with _LOCK, _connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO workflow_events(workflow_id, event_type, payload_json, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (workflow_id, event_type, _dump(payload) or "{}", now),
        )
        event_id = int(cursor.lastrowid)
    return {
        "id": event_id,
        "event_type": event_type,
        "payload": payload,
        "created_at": now,
    }


def list_events(workflow_id: str, *, after_id: int = 0) -> list[dict[str, Any]]:
    init_store()
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, event_type, payload_json, created_at
            FROM workflow_events WHERE workflow_id=? AND id>?
            ORDER BY id ASC
            """,
            (workflow_id, after_id),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "event_type": row["event_type"],
            "payload": _load(row["payload_json"]) or {},
            "created_at": float(row["created_at"]),
        }
        for row in rows
    ]


def register_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    init_store()
    now = float(payload.get("created_at") or time.time())
    with _LOCK, _connection() as connection:
        connection.execute(
            """
            INSERT INTO datasets(
                dataset_id, workflow_id, owner_id, kind, simulator, scenario,
                path, payload_json, created_at, last_accessed_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                path=excluded.path,
                payload_json=excluded.payload_json,
                last_accessed_at=excluded.last_accessed_at
            """,
            (
                payload["dataset_id"],
                payload["workflow_id"],
                payload["owner_id"],
                payload["kind"],
                payload["simulator"],
                payload["scenario"],
                payload["path"],
                _dump(payload) or "{}",
                now,
                now,
            ),
        )
    return get_dataset(str(payload["dataset_id"]), touch=False)


def get_dataset(dataset_id: str, *, touch: bool = True) -> dict[str, Any]:
    init_store()
    with _LOCK, _connection() as connection:
        row = connection.execute(
            "SELECT payload_json FROM datasets WHERE dataset_id=?", (dataset_id,)
        ).fetchone()
        if row is None:
            raise KeyError(dataset_id)
        payload = _load(row["payload_json"]) or {}
        if touch:
            now = time.time()
            connection.execute(
                "UPDATE datasets SET last_accessed_at=? WHERE dataset_id=?",
                (now, dataset_id),
            )
            payload["last_accessed_at"] = now
    return payload


def list_datasets(
    *,
    kind: str | None = None,
    workflow_id: str | None = None,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    init_store()
    clauses: list[str] = []
    values: list[Any] = []
    for column, value in (("kind", kind), ("workflow_id", workflow_id), ("owner_id", owner_id)):
        if value is not None:
            clauses.append(f"{column}=?")
            values.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection() as connection:
        rows = connection.execute(
            f"SELECT payload_json FROM datasets{where} ORDER BY created_at DESC",
            values,
        ).fetchall()
    return [_load(row["payload_json"]) or {} for row in rows]
