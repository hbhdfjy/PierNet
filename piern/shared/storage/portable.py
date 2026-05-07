"""Portable Parquet/DuckDB/SQLite storage helpers.

The project still supports legacy JSONL files. This module adds a file-based
storage layer that is easier to migrate between servers: large records live in
partitioned Parquet directories, while generated SQLite catalogs can be rebuilt.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

from piern.shared.runtime.paths import PROJECT_ROOT

TEXT2COMP_PARQUET_DIR = PROJECT_ROOT / "data" / "text2comp_parquet"
ROUTER_PARQUET_DIR = PROJECT_ROOT / "data" / "router_parquet"
CATALOG_DB_PATH = PROJECT_ROOT / "data" / "catalog.sqlite"
SUPPORTED_KINDS = {"text2comp", "router"}


@dataclass(frozen=True)
class PartitionInfo:
    kind: str
    simulator: str
    scenario: str
    path: Path
    row_count: int
    file_size_bytes: int
    mtime: float
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


def parquet_root(kind: str) -> Path:
    if kind == "text2comp":
        return TEXT2COMP_PARQUET_DIR
    if kind == "router":
        return ROUTER_PARQUET_DIR
    raise ValueError(f"unsupported parquet dataset kind: {kind}")


def parquet_available() -> bool:
    return _module_available("pyarrow")


def duckdb_available() -> bool:
    return _module_available("duckdb")


def has_partitions(kind: str) -> bool:
    root = parquet_root(kind)
    return root.exists() and any(root.glob("simulator=*/scenario=*/*.parquet"))


def discover_partitions(kind: str) -> list[PartitionInfo]:
    root = parquet_root(kind)
    if not root.exists():
        return []

    partitions: list[PartitionInfo] = []
    for scenario_dir in sorted(root.glob("simulator=*/scenario=*")):
        if not scenario_dir.is_dir():
            continue
        simulator = _decode_partition_value(scenario_dir.parent.name, "simulator")
        scenario = _decode_partition_value(scenario_dir.name, "scenario")
        parquet_files = sorted(scenario_dir.glob("*.parquet"))
        if not parquet_files:
            continue
        manifest = _read_json(scenario_dir / "_manifest.json") or {}
        row_count = int(manifest.get("row_count") or _parquet_row_count(parquet_files))
        file_size = sum(_safe_stat(path)[0] for path in parquet_files)
        mtime = max((_safe_stat(path)[1] for path in parquet_files), default=0.0)
        partitions.append(
            PartitionInfo(
                kind=kind,
                simulator=str(manifest.get("simulator") or simulator),
                scenario=str(manifest.get("scenario") or scenario),
                path=scenario_dir,
                row_count=row_count,
                file_size_bytes=file_size,
                mtime=mtime,
                metadata=manifest,
            )
        )
    return partitions


def text2comp_manifest_like() -> dict[str, Any] | None:
    partitions = discover_partitions("text2comp")
    if not partitions:
        return None

    summary = text2comp_stats()
    items = []
    for part in partitions:
        meta = part.metadata
        items.append(
            {
                "scenario": part.scenario,
                "simulator": part.simulator,
                "sample_count": part.row_count,
                "file_size_bytes": part.file_size_bytes,
                "mtime": part.mtime,
                "path": str(part.path),
                "storage": "parquet",
                "by_language": meta.get("by_language", {}),
                "by_style": meta.get("by_style", {}),
                "by_time_mode": meta.get("by_time_mode", {}),
                "timeseries_shape_obs": meta.get("timeseries_shape_obs"),
            }
        )
    return {
        "version": 2,
        "kind": "sample_manifest",
        "storage": "parquet",
        "generated_at": time.time(),
        "items": sorted(items, key=lambda item: item["scenario"]),
        "summary": summary,
    }


def router_manifest_like() -> dict[str, Any] | None:
    partitions = discover_partitions("router")
    if not partitions:
        return None

    label_counts = _group_counts("router", "label")
    scenarios = [
        {
            "scenario": part.scenario,
            "simulator": part.simulator,
            "router_count": part.row_count,
            "file_size_bytes": part.file_size_bytes,
            "mtime": part.mtime,
            "path": str(part.path),
            "storage": "parquet",
        }
        for part in partitions
    ]
    total = sum(part.row_count for part in partitions)
    mtime = max((part.mtime for part in partitions), default=0.0)
    size = sum(part.file_size_bytes for part in partitions)
    return {
        "version": 2,
        "kind": "router_manifest",
        "storage": "parquet",
        "generated_at": time.time(),
        "splits": {
            "train": {
                "exists": bool(partitions),
                "count": total,
                "file_size_bytes": size,
                "mtime": mtime,
                "storage": "parquet",
            }
        },
        "total": total,
        "label_counts": {"0": int(label_counts.get("0", 0)), "1": int(label_counts.get("1", 0))},
        "scenarios": sorted(scenarios, key=lambda item: item["scenario"]),
    }


def text2comp_stats() -> dict[str, Any]:
    fallback = {
        "total_samples": 0,
        "by_simulator": {},
        "by_scenario": {},
        "by_language": {},
        "by_style": {},
        "by_time_mode": {},
        "timeseries_shapes": {},
        "storage": "parquet",
    }
    if not has_partitions("text2comp"):
        return fallback
    if not duckdb_available():
        return _manifest_stats_fallback()

    con = _duckdb_connect()
    paths_expr = _paths_expr(_parquet_files("text2comp"))
    try:
        total = int(con.execute(f"SELECT count(*) FROM read_parquet({paths_expr})").fetchone()[0])
        return {
            "total_samples": total,
            "by_simulator": _query_group_counts(con, paths_expr, "simulator"),
            "by_scenario": _query_group_counts(con, paths_expr, "scenario"),
            "by_language": _query_group_counts(con, paths_expr, "language"),
            "by_style": _query_group_counts(con, paths_expr, "style"),
            "by_time_mode": _query_group_counts(con, paths_expr, "time_mode"),
            "timeseries_shapes": _timeseries_shapes_from_manifests(),
            "storage": "parquet",
        }
    finally:
        con.close()


def read_text2comp_page(
    *,
    scenario: str,
    page: int,
    page_size: int,
    language: str | None = None,
    style: str | None = None,
) -> tuple[int, list[dict[str, Any]]] | None:
    filters: list[tuple[str, Any]] = [("scenario", scenario)]
    if language:
        filters.append(("language", language))
    if style:
        filters.append(("style", style))
    return read_records_page("text2comp", page=page, page_size=page_size, filters=filters)


def read_router_page(
    *,
    scenario: str = "",
    page: int,
    page_size: int,
    label: int = -1,
) -> tuple[int, list[dict[str, Any]]] | None:
    filters: list[tuple[str, Any]] = []
    if scenario:
        filters.append(("scenario", scenario))
    if label in (0, 1):
        filters.append(("label", label))
    return read_records_page("router", page=page, page_size=page_size, filters=filters)


def read_records_page(
    kind: str,
    *,
    page: int,
    page_size: int,
    filters: Iterable[tuple[str, Any]] = (),
) -> tuple[int, list[dict[str, Any]]] | None:
    if not has_partitions(kind):
        return None
    if not duckdb_available():
        raise RuntimeError("DuckDB is required to read Parquet dataset pages; install duckdb>=1.0.0")

    paths = _parquet_files(kind)
    if not paths:
        return None
    paths_expr = _paths_expr(paths)
    where_sql, params = _where_clause(filters)
    offset = page * page_size
    con = _duckdb_connect()
    try:
        total = int(con.execute(f"SELECT count(*) FROM read_parquet({paths_expr}) {where_sql}", params).fetchone()[0])
        rows = con.execute(
            f"""
            SELECT record_json
            FROM read_parquet({paths_expr})
            {where_sql}
            ORDER BY scenario, row_index
            LIMIT ? OFFSET ?
            """,
            [*params, page_size, offset],
        ).fetchall()
    finally:
        con.close()
    items: list[dict[str, Any]] = []
    for (raw,) in rows:
        try:
            value = json.loads(raw)
        except Exception:
            continue
        if isinstance(value, dict):
            items.append(value)
    return total, items


def export_records_to_jsonl(
    kind: str,
    output_path: Path,
    *,
    simulator: str | None = None,
    scenario: str | None = None,
) -> int:
    """Materialize Parquet records to JSONL for legacy training code paths."""

    if not has_partitions(kind):
        raise FileNotFoundError(f"no Parquet partitions found for {kind}")
    if not duckdb_available():
        raise RuntimeError("DuckDB is required to export Parquet records to JSONL")

    filters: list[tuple[str, Any]] = []
    if simulator:
        filters.append(("simulator", simulator))
    if scenario:
        filters.append(("scenario", scenario))
    paths_expr = _paths_expr(_parquet_files(kind))
    where_sql, params = _where_clause(filters)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    con = _duckdb_connect()
    count = 0
    try:
        cursor = con.execute(
            f"""
            SELECT record_json
            FROM read_parquet({paths_expr})
            {where_sql}
            ORDER BY scenario, row_index
            """,
            params,
        )
        with tmp_path.open("w", encoding="utf-8") as handle:
            while True:
                rows = cursor.fetchmany(8192)
                if not rows:
                    break
                for (raw,) in rows:
                    if raw:
                        handle.write(str(raw).rstrip("\n") + "\n")
                        count += 1
        tmp_path.replace(output_path)
    finally:
        con.close()
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return count


def partition_for(kind: str, scenario: str, simulator: str | None = None) -> PartitionInfo | None:
    for part in discover_partitions(kind):
        if part.scenario != scenario:
            continue
        if simulator and part.simulator != simulator:
            continue
        return part
    return None


def delete_partition(kind: str, scenario: str, simulator: str | None = None) -> bool:
    part = partition_for(kind, scenario, simulator=simulator)
    if part is None:
        return False
    root = parquet_root(kind).resolve()
    target = part.path.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"refusing to delete path outside Parquet root: {target}") from exc
    shutil.rmtree(target)
    return True


def safe_partition_value(value: str) -> str:
    cleaned = str(value or "unknown").strip() or "unknown"
    return cleaned.replace("/", "_").replace("\\", "_").replace("\0", "_")


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "file_size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def current_storage_summary() -> dict[str, Any]:
    return {
        "text2comp_parquet": [part.to_json() for part in discover_partitions("text2comp")],
        "router_parquet": [part.to_json() for part in discover_partitions("router")],
        "dependencies": {"pyarrow": parquet_available(), "duckdb": duckdb_available()},
    }


def _module_available(name: str) -> bool:
    try:
        import_module(name)
    except Exception:
        return False
    return True


def _duckdb_connect():
    duckdb = import_module("duckdb")
    return duckdb.connect(database=":memory:")


def _parquet_row_count(paths: list[Path]) -> int:
    if not parquet_available():
        return 0
    pq = import_module("pyarrow.parquet")
    total = 0
    for path in paths:
        try:
            total += int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            continue
    return total


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_stat(path: Path) -> tuple[int, float]:
    try:
        stat = path.stat()
    except OSError:
        return 0, 0.0
    return int(stat.st_size), float(stat.st_mtime)


def _decode_partition_value(name: str, key: str) -> str:
    prefix = f"{key}="
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _parquet_files(kind: str) -> list[Path]:
    return sorted(parquet_root(kind).glob("simulator=*/scenario=*/*.parquet"))


def _paths_expr(paths: list[Path]) -> str:
    if not paths:
        raise FileNotFoundError("no parquet files found")
    escaped = [str(path).replace("'", "''") for path in paths]
    return "[" + ", ".join(f"'{path}'" for path in escaped) + "]"


def _where_clause(filters: Iterable[tuple[str, Any]]) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    for column, value in filters:
        if value in (None, ""):
            continue
        if column not in {"simulator", "scenario", "language", "style", "time_mode", "label"}:
            raise ValueError(f"unsupported filter column: {column}")
        clauses.append(f"{column} = ?")
        params.append(value)
    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


def _query_group_counts(con, paths_expr: str, column: str) -> dict[str, int]:
    rows = con.execute(
        f"""
        SELECT coalesce(cast({column} as varchar), 'unknown') AS key, count(*) AS n
        FROM read_parquet({paths_expr})
        GROUP BY key
        ORDER BY key
        """
    ).fetchall()
    return {str(key): int(count) for key, count in rows if key not in (None, "")}


def _group_counts(kind: str, column: str) -> dict[str, int]:
    if not has_partitions(kind):
        return {}
    if not duckdb_available():
        counts: Counter[str] = Counter()
        for part in discover_partitions(kind):
            key = f"by_{column}"
            counts.update({str(k): int(v) for k, v in part.metadata.get(key, {}).items()})
        return dict(counts)
    con = _duckdb_connect()
    try:
        return _query_group_counts(con, _paths_expr(_parquet_files(kind)), column)
    finally:
        con.close()


def _manifest_stats_fallback() -> dict[str, Any]:
    by_simulator: Counter[str] = Counter()
    by_scenario: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_style: Counter[str] = Counter()
    by_time_mode: Counter[str] = Counter()
    total = 0
    for part in discover_partitions("text2comp"):
        total += part.row_count
        by_simulator[part.simulator] += part.row_count
        by_scenario[part.scenario] += part.row_count
        by_language.update({str(k): int(v) for k, v in part.metadata.get("by_language", {}).items()})
        by_style.update({str(k): int(v) for k, v in part.metadata.get("by_style", {}).items()})
        by_time_mode.update({str(k): int(v) for k, v in part.metadata.get("by_time_mode", {}).items()})
    return {
        "total_samples": total,
        "by_simulator": dict(by_simulator),
        "by_scenario": dict(by_scenario),
        "by_language": dict(by_language),
        "by_style": dict(by_style),
        "by_time_mode": dict(by_time_mode),
        "timeseries_shapes": _timeseries_shapes_from_manifests(),
        "storage": "parquet",
    }


def _timeseries_shapes_from_manifests() -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    for part in discover_partitions("text2comp"):
        shape = part.metadata.get("timeseries_shape_obs")
        if shape is not None:
            shapes.setdefault(part.simulator, shape)
    return shapes
