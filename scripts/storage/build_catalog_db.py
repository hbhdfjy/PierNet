#!/usr/bin/env python3
"""Build a portable SQLite catalog for PiERN data assets.

The catalog is derived state. It is safe to delete and rebuild after moving the
project to a new server.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from piern.shared.runtime.paths import ARTIFACT_ROOT, DATA_ROOT  # noqa: E402
from piern.shared.storage import portable  # noqa: E402

CATALOG_PATH = portable.CATALOG_DB_PATH


def main() -> None:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CATALOG_PATH.with_suffix(".sqlite.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    con = sqlite3.connect(tmp_path)
    try:
        con.execute(
            """
            CREATE TABLE assets (
                id TEXT PRIMARY KEY,
                storage TEXT NOT NULL,
                kind TEXT NOT NULL,
                simulator TEXT,
                scenario TEXT,
                path TEXT NOT NULL,
                row_count INTEGER,
                file_size_bytes INTEGER NOT NULL,
                mtime REAL NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        con.execute("CREATE INDEX idx_assets_kind ON assets(kind)")
        con.execute("CREATE INDEX idx_assets_scenario ON assets(simulator, scenario)")
        for row in collect_assets():
            con.execute(
                """
                INSERT OR REPLACE INTO assets
                (id, storage, kind, simulator, scenario, path, row_count, file_size_bytes, mtime, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["storage"],
                    row["kind"],
                    row.get("simulator"),
                    row.get("scenario"),
                    row["path"],
                    row.get("row_count"),
                    row["file_size_bytes"],
                    row["mtime"],
                    json.dumps(row.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                ),
            )
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute("INSERT INTO meta VALUES (?, ?)", ("generated_at", str(time.time())))
        con.execute("INSERT INTO meta VALUES (?, ?)", ("project_root", str(PROJECT_ROOT)))
        con.commit()
    finally:
        con.close()
    tmp_path.replace(CATALOG_PATH)
    print(json.dumps({"ok": True, "path": str(CATALOG_PATH), "assets": count_assets()}, ensure_ascii=False, indent=2))


def collect_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    assets.extend(parquet_assets())
    assets.extend(legacy_jsonl_assets())
    assets.extend(hdf5_assets())
    assets.extend(training_artifact_assets())
    return assets


def parquet_assets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("text2comp", "router"):
        for part in portable.discover_partitions(kind):
            rows.append(
                {
                    "id": f"parquet:{kind}:{part.simulator}:{part.scenario}",
                    "storage": "parquet",
                    "kind": kind,
                    "simulator": part.simulator,
                    "scenario": part.scenario,
                    "path": str(part.path),
                    "row_count": part.row_count,
                    "file_size_bytes": part.file_size_bytes,
                    "mtime": part.mtime,
                    "metadata": part.metadata,
                }
            )
    return rows


def legacy_jsonl_assets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ("text2comp_jsonl", DATA_ROOT / "text2comp", "*.jsonl"),
        ("router_jsonl", DATA_ROOT / "router" / "by_scenario", "*.jsonl"),
    ]
    for kind, directory, pattern in sources:
        if not directory.exists():
            continue
        for path in sorted(directory.glob(pattern)):
            if path.name == "all_training_data.jsonl":
                continue
            stat = path.stat()
            rows.append(
                {
                    "id": f"jsonl:{kind}:{path.stem}",
                    "storage": "jsonl",
                    "kind": kind,
                    "simulator": None,
                    "scenario": path.stem,
                    "path": str(path),
                    "row_count": count_jsonl(path),
                    "file_size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                    "metadata": {},
                }
            )
    return rows


def hdf5_assets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data_root = DATA_ROOT
    if not data_root.exists():
        return rows
    for path in sorted([*data_root.glob("*/*.h5"), *data_root.glob("*/*.hdf5")]):
        stat = path.stat()
        simulator = path.parent.name
        rows.append(
            {
                "id": f"hdf5:{simulator}:{path.stem}",
                "storage": "hdf5",
                "kind": "hdf5",
                "simulator": simulator,
                "scenario": path.stem,
                "path": str(path),
                "row_count": None,
                "file_size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "metadata": {},
            }
        )
    return rows


def training_artifact_assets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = ARTIFACT_ROOT
    if not root.exists():
        return rows
    for path in sorted(root.glob("token_router/*/runs/*")):
        if not path.is_dir():
            continue
        rows.append(
            {
                "id": f"artifact:run:{path.name}",
                "storage": "filesystem",
                "kind": "training_run",
                "simulator": path.parent.parent.name,
                "scenario": None,
                "path": str(path),
                "row_count": None,
                "file_size_bytes": dir_size(path),
                "mtime": path.stat().st_mtime,
                "metadata": {},
            }
        )
    return rows


def count_jsonl(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def count_assets() -> int:
    con = sqlite3.connect(CATALOG_PATH)
    try:
        return int(con.execute("SELECT count(*) FROM assets").fetchone()[0])
    finally:
        con.close()


if __name__ == "__main__":
    main()
