import sqlite3

from PierNet.shared.db.migrations import Migration, ensure_sqlite_schema
from PierNet.shared.tasks import locks
from PierNet.shared.tasks.state import IllegalStatusTransition, normalize_status, validate_transition


def test_task_state_normalizes_aliases_and_rejects_illegal_transition():
    assert normalize_status("succeeded") == "done"
    assert validate_transition("starting", "running") == "running"
    try:
        validate_transition("done", "running")
    except IllegalStatusTransition:
        pass
    else:
        raise AssertionError("expected illegal transition")


def test_sqlite_migration_helper_applies_versions_once():
    calls = []
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    def migration(connection):
        calls.append("v1")
        connection.execute("CREATE TABLE demo(id INTEGER PRIMARY KEY)")

    assert ensure_sqlite_schema(conn, "demo", [Migration(1, "initial", migration)]) == 1
    assert ensure_sqlite_schema(conn, "demo", [Migration(1, "initial", migration)]) == 1
    assert calls == ["v1"]


def test_task_locks_acquire_release_and_expire(monkeypatch, tmp_path):
    monkeypatch.setattr(locks, "LOCK_STORE_PATH", tmp_path / "locks.sqlite")
    monkeypatch.setattr(locks, "_INITIALIZED", False)

    assert locks.acquire_lock("gpu:0", "train-a", ttl_seconds=10)
    assert not locks.acquire_lock("gpu:0", "train-b", ttl_seconds=10)
    assert locks.release_lock("gpu:0", "train-a")
    assert locks.acquire_lock("gpu:0", "train-b", ttl_seconds=0.01)
    assert locks.cleanup_expired(now=10**12) >= 1
    assert locks.list_locks() == []
