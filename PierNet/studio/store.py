from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterator

from PierNet.studio.paths import STUDIO_DB_PATH, ensure_studio_roots

_LOCK = RLock()
_INITIALIZED = False

STAGE_DEFINITIONS = (
    ("resources", "准备自己的资源"),
    ("inspection", "理解数据与模型"),
    ("compatibility", "检查资源匹配"),
    ("preparation", "准备训练内容"),
    ("training", "训练 Demo"),
    ("assembly", "创建可对话 Demo"),
    ("validation", "验证计算结果"),
)


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_load(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    STUDIO_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(STUDIO_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_store() -> None:
    global _INITIALIZED
    with _LOCK:
        if _INITIALIZED:
            return
        ensure_studio_roots()
        with _connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects(
                    project_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    stages_json TEXT NOT NULL,
                    data_json TEXT,
                    expert_json TEXT,
                    inspection_json TEXT,
                    compatibility_json TEXT,
                    artifacts_json TEXT,
                    result_json TEXT,
                    error_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_studio_projects_owner
                    ON projects(owner_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS project_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_studio_events_project
                    ON project_events(project_id, id);
                CREATE TABLE IF NOT EXISTS project_chats(
                    chat_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS audit_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    project_id TEXT,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_studio_audit_owner
                    ON audit_events(owner_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_studio_audit_project
                    ON audit_events(project_id, created_at DESC);
                """
            )
            rows = conn.execute("SELECT project_id, stages_json FROM projects WHERE status='running'").fetchall()
            now = time.time()
            for row in rows:
                stages = _json_load(row["stages_json"]) or {}
                for stage in stages.values():
                    if stage.get("status") == "running":
                        stage.update(
                            {
                                "status": "failed",
                                "message": "服务重启中断了此阶段，可以安全重试。",
                                "retryable": True,
                                "finished_at": now,
                            }
                        )
                conn.execute(
                    """
                    UPDATE projects
                    SET status='failed', stages_json=?, error_json=?, updated_at=?
                    WHERE project_id=?
                    """,
                    (
                        _json_dump(stages),
                        _json_dump(
                            {
                                "code": "service_restarted",
                                "message": "服务重启中断了任务，请点击重试。",
                            }
                        ),
                        now,
                        row["project_id"],
                    ),
                )
        _INITIALIZED = True


def _initial_stages() -> dict[str, dict[str, Any]]:
    return {
        stage_id: {
            "id": stage_id,
            "title": title,
            "status": "waiting",
            "progress": None,
            "message": "等待开始",
            "retryable": False,
            "started_at": None,
            "finished_at": None,
        }
        for stage_id, title in STAGE_DEFINITIONS
    }


def create_project(owner_id: str, name: str, goal: str) -> dict[str, Any]:
    init_store()
    project_id = f"studio-{uuid.uuid4().hex[:12]}"
    now = time.time()
    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO projects(
                project_id, owner_id, name, goal, status, current_stage,
                stages_json, created_at, updated_at
            ) VALUES(?, ?, ?, ?, 'draft', 'resources', ?, ?, ?)
            """,
            (project_id, owner_id, name.strip(), goal.strip(), _json_dump(_initial_stages()), now, now),
        )
    append_event(project_id, "project_created", {"message": "项目已创建"})
    return get_project(owner_id, project_id)


def _row_to_project(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in (
        "stages_json",
        "data_json",
        "expert_json",
        "inspection_json",
        "compatibility_json",
        "artifacts_json",
        "result_json",
        "error_json",
    ):
        data[key.removesuffix("_json")] = _json_load(data.pop(key))
    data["cancel_requested"] = bool(data["cancel_requested"])
    return data


def list_projects(owner_id: str) -> list[dict[str, Any]]:
    init_store()
    with _connection() as conn:
        rows = conn.execute("SELECT * FROM projects WHERE owner_id=? ORDER BY updated_at DESC", (owner_id,)).fetchall()
    return [_row_to_project(row) for row in rows]


def get_project(owner_id: str, project_id: str) -> dict[str, Any]:
    init_store()
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE owner_id=? AND project_id=?",
            (owner_id, project_id),
        ).fetchone()
    if row is None:
        raise KeyError(project_id)
    return _row_to_project(row)


def get_project_unscoped(project_id: str) -> dict[str, Any]:
    init_store()
    with _connection() as conn:
        row = conn.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
    if row is None:
        raise KeyError(project_id)
    return _row_to_project(row)


def delete_project(owner_id: str, project_id: str) -> dict[str, Any]:
    init_store()
    with _LOCK, _connection() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE owner_id=? AND project_id=?",
            (owner_id, project_id),
        ).fetchone()
        if row is None:
            raise KeyError(project_id)
        conn.execute(
            "DELETE FROM projects WHERE owner_id=? AND project_id=?",
            (owner_id, project_id),
        )
    return _row_to_project(row)


def update_project(project_id: str, **fields: Any) -> dict[str, Any]:
    init_store()
    allowed = {
        "name",
        "goal",
        "status",
        "current_stage",
        "stages",
        "data",
        "expert",
        "inspection",
        "compatibility",
        "artifacts",
        "result",
        "error",
        "cancel_requested",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unsupported project fields: {sorted(unknown)}")
    assignments: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        column = (
            f"{key}_json"
            if key
            in {
                "stages",
                "data",
                "expert",
                "inspection",
                "compatibility",
                "artifacts",
                "result",
                "error",
            }
            else key
        )
        if column.endswith("_json"):
            value = _json_dump(value)
        elif key == "cancel_requested":
            value = int(bool(value))
        assignments.append(f"{column}=?")
        values.append(value)
    assignments.append("updated_at=?")
    values.append(time.time())
    values.append(project_id)
    with _LOCK, _connection() as conn:
        conn.execute(
            f"UPDATE projects SET {', '.join(assignments)} WHERE project_id=?",
            values,
        )
    return get_project_unscoped(project_id)


def update_stage(
    project_id: str,
    stage_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    retryable: bool | None = None,
) -> dict[str, Any]:
    with _LOCK:
        project = get_project_unscoped(project_id)
        stages = project["stages"]
        if stage_id not in stages:
            raise KeyError(stage_id)
        stage = stages[stage_id]
        now = time.time()
        if status is not None:
            stage["status"] = status
            if status == "running" and stage.get("started_at") is None:
                stage["started_at"] = now
            if status in {"succeeded", "failed", "cancelled"}:
                stage["finished_at"] = now
        if progress is not None:
            stage["progress"] = max(0.0, min(1.0, float(progress)))
        if message is not None:
            stage["message"] = message
        if retryable is not None:
            stage["retryable"] = retryable
        return update_project(project_id, stages=stages, current_stage=stage_id)


def append_event(project_id: str, event_type: str, payload: dict[str, Any]) -> int:
    init_store()
    now = time.time()
    with _connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO project_events(project_id, event_type, payload_json, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (project_id, event_type, _json_dump(payload) or "{}", now),
        )
        event_id = int(cursor.lastrowid)
    return event_id


def list_events(project_id: str, after_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
    init_store()
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT id, event_type, payload_json, created_at
            FROM project_events
            WHERE project_id=? AND id>?
            ORDER BY id ASC LIMIT ?
            """,
            (project_id, after_id, limit),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "event_type": row["event_type"],
            "payload": _json_load(row["payload_json"]) or {},
            "created_at": float(row["created_at"]),
        }
        for row in rows
    ]


def save_chat(project_id: str, request: dict[str, Any], response: dict[str, Any]) -> str:
    chat_id = f"chat-{uuid.uuid4().hex[:12]}"
    now = time.time()
    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO project_chats(chat_id, project_id, request_json, response_json, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (chat_id, project_id, _json_dump(request), _json_dump(response), now),
        )
    return chat_id


def append_audit(
    owner_id: str,
    action: str,
    *,
    project_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> int:
    init_store()
    now = time.time()
    with _connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO audit_events(
                owner_id, project_id, action, payload_json, created_at
            ) VALUES(?, ?, ?, ?, ?)
            """,
            (
                owner_id,
                project_id,
                action,
                _json_dump(payload or {}) or "{}",
                now,
            ),
        )
        return int(cursor.lastrowid)


def list_audit_events(
    owner_id: str,
    *,
    project_id: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    init_store()
    query = """
        SELECT id, owner_id, project_id, action, payload_json, created_at
        FROM audit_events
        WHERE owner_id=?
    """
    values: list[Any] = [owner_id]
    if project_id is not None:
        query += " AND project_id=?"
        values.append(project_id)
    query += " ORDER BY id DESC LIMIT ?"
    values.append(max(1, min(int(limit), 1000)))
    with _connection() as conn:
        rows = conn.execute(query, values).fetchall()
    return [
        {
            "id": int(row["id"]),
            "owner_id": row["owner_id"],
            "project_id": row["project_id"],
            "action": row["action"],
            "payload": _json_load(row["payload_json"]) or {},
            "created_at": float(row["created_at"]),
        }
        for row in rows
    ]
