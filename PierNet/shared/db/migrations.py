"""Small SQLite schema migration helper."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations(
            namespace TEXT NOT NULL,
            version INTEGER NOT NULL,
            name TEXT NOT NULL,
            applied_at REAL NOT NULL,
            PRIMARY KEY(namespace, version)
        )
        """
    )


def current_version(conn: sqlite3.Connection, namespace: str) -> int:
    _ensure_table(conn)
    row = conn.execute(
        "SELECT MAX(version) AS version FROM schema_migrations WHERE namespace=?",
        (namespace,),
    ).fetchone()
    if row is None:
        return 0
    value = row["version"] if isinstance(row, sqlite3.Row) else row[0]
    return int(value or 0)


def ensure_sqlite_schema(conn: sqlite3.Connection, namespace: str, migrations: Iterable[Migration]) -> int:
    _ensure_table(conn)
    applied = current_version(conn, namespace)
    for migration in sorted(migrations, key=lambda item: item.version):
        if migration.version <= applied:
            continue
        migration.apply(conn)
        conn.execute(
            """
            INSERT INTO schema_migrations(namespace, version, name, applied_at)
            VALUES (?, ?, ?, ?)
            """,
            (namespace, migration.version, migration.name, time.time()),
        )
        applied = migration.version
    return applied
