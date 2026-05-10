"""Portable Parquet/DuckDB/SQLite storage helpers.

The project still supports legacy JSONL files. This module adds a file-based
storage layer that is easier to migrate between servers: large records live in
partitioned Parquet directories, while generated SQLite catalogs can be rebuilt.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

from piern.shared.runtime.paths import DATA_ROOT

TEXT2COMP_PARQUET_DIR = DATA_ROOT / "text2comp_parquet"
ROUTER_PARQUET_DIR = DATA_ROOT / "router_parquet"
CATALOG_DB_PATH = DATA_ROOT / "catalog.sqlite"
SUPPORTED_KINDS = {"text2comp", "router"}
PARQUET_SCHEMA_VERSION = 1



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


def require_parquet_modules():
    try:
        pa = import_module("pyarrow")
        pq = import_module("pyarrow.parquet")
    except Exception as exc:
        raise RuntimeError("pyarrow is required for Parquet storage; install pyarrow>=15.0.0") from exc
    return pa, pq


def parquet_schema():
    pa, _ = require_parquet_modules()
    return pa.schema(
        [
            pa.field("row_index", pa.int64()),
            pa.field("simulator", pa.string()),
            pa.field("scenario", pa.string()),
            pa.field("language", pa.string()),
            pa.field("style", pa.string()),
            pa.field("time_mode", pa.string()),
            pa.field("label", pa.int8()),
            pa.field("context", pa.string()),
            pa.field("input", pa.string()),
            pa.field("output", pa.string()),
            pa.field("metadata_json", pa.string()),
            pa.field("record_json", pa.string()),
        ]
    )


def partition_dir_for(kind: str, simulator: str, scenario: str, output_root: Path | None = None) -> Path:
    root = Path(output_root) if output_root is not None else parquet_root(kind)
    return (
        root
        / f"simulator={safe_partition_value(simulator)}"
        / f"scenario={safe_partition_value(scenario)}"
    )


def record_to_parquet_row(
    kind: str,
    scenario_hint: str,
    row_index: int,
    record: dict[str, Any],
    *,
    simulator_hint: str | None = None,
) -> dict[str, Any]:
    metadata = _record_metadata(record)
    observation = metadata.get("observation", {})
    if not isinstance(observation, dict):
        observation = {}
    simulator = str(metadata.get("simulator") or simulator_hint or "unknown")
    scenario = str(metadata.get("scenario") or scenario_hint)
    label_value = record.get("label") if kind == "router" else None
    try:
        label = int(label_value) if label_value is not None else None
    except (TypeError, ValueError):
        label = None
    return {
        "row_index": row_index,
        "simulator": simulator,
        "scenario": scenario,
        "language": _optional_str(metadata.get("language")),
        "style": _optional_str(metadata.get("style")),
        "time_mode": _optional_str(observation.get("time_mode")),
        "label": label,
        "context": _optional_str(record.get("context")),
        "input": _optional_str(record.get("input")),
        "output": _optional_str(record.get("output")),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        "record_json": json.dumps(record, ensure_ascii=False, separators=(",", ":")),
    }


def write_records_partition(
    kind: str,
    records: Iterable[dict[str, Any]],
    *,
    simulator: str,
    scenario: str,
    source: dict[str, Any] | None = None,
    output_root: Path | None = None,
    batch_size: int = 8192,
    compression: str = "zstd",
    overwrite: bool = True,
    extra_manifest: dict[str, Any] | None = None,
    progress_callback=None,
    validate: bool = True,
) -> dict[str, Any]:
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported parquet dataset kind: {kind}")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    pa, pq = require_parquet_modules()
    schema = parquet_schema()
    root = Path(output_root) if output_root is not None else parquet_root(kind)
    partition_dir = partition_dir_for(kind, simulator, scenario, root)
    if partition_dir.exists() and not overwrite:
        return {"kind": kind, "scenario": scenario, "simulator": simulator, "status": "skipped", "reason": "partition exists", "path": str(partition_dir)}

    tmp_dir = root / f".tmp-{kind}-{safe_partition_value(simulator)}-{safe_partition_value(scenario)}-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_path = tmp_dir / "part-00000.parquet"

    writer = None
    rows: list[dict[str, Any]] = []
    counters = {
        "by_language": Counter(),
        "by_style": Counter(),
        "by_time_mode": Counter(),
        "by_label": Counter(),
    }
    timeseries_shape_obs = None
    row_count = 0

    def flush() -> None:
        nonlocal writer, rows
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(
                part_path,
                schema,
                compression=None if compression == "none" else compression,
                use_dictionary=True,
            )
        writer.write_table(table)
        rows = []

    try:
        for row_index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            row = record_to_parquet_row(kind, scenario, row_index, record, simulator_hint=simulator)
            rows.append(row)
            row_count += 1
            if row["language"]:
                counters["by_language"][row["language"]] += 1
            if row["style"]:
                counters["by_style"][row["style"]] += 1
            if row["time_mode"]:
                counters["by_time_mode"][row["time_mode"]] += 1
            if row["label"] is not None:
                counters["by_label"][str(row["label"])] += 1
            metadata = _record_metadata(record)
            if timeseries_shape_obs is None:
                shape = metadata.get("timeseries_shape_obs")
                if isinstance(shape, (list, tuple)):
                    timeseries_shape_obs = list(shape)
            if progress_callback:
                progress_callback(row_count)
            if len(rows) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    if row_count == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"kind": kind, "scenario": scenario, "simulator": simulator, "status": "empty", "rows": 0}

    if validate:
        actual = pq.ParquetFile(part_path).metadata.num_rows
        if int(actual) != row_count:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"row count mismatch for {kind}/{scenario}: generated={row_count}, parquet={actual}")

    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    partition_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.replace(partition_dir)

    manifest = {
        "version": PARQUET_SCHEMA_VERSION,
        "kind": kind,
        "storage": "parquet",
        "generated_at": time.time(),
        "source": source or {},
        "simulator": simulator,
        "scenario": scenario,
        "row_count": row_count,
        "by_language": dict(counters["by_language"]),
        "by_style": dict(counters["by_style"]),
        "by_time_mode": dict(counters["by_time_mode"]),
        "by_label": dict(counters["by_label"]),
        "timeseries_shape_obs": timeseries_shape_obs,
        "schema": [field.name for field in schema],
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (partition_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "kind": kind,
        "scenario": scenario,
        "simulator": simulator,
        "status": "written",
        "rows": row_count,
        "path": str(partition_dir),
    }


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


def iter_records(
    kind: str,
    *,
    filters: Iterable[tuple[str, Any]] = (),
) -> Iterable[dict[str, Any]]:
    if not has_partitions(kind):
        return
    if not duckdb_available():
        raise RuntimeError("DuckDB is required to stream Parquet records; install duckdb>=1.0.0")

    paths_expr = _paths_expr(_parquet_files(kind))
    where_sql, params = _where_clause(filters)
    con = _duckdb_connect()
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
        while True:
            rows = cursor.fetchmany(8192)
            if not rows:
                break
            for (raw,) in rows:
                try:
                    value = json.loads(raw)
                except Exception:
                    continue
                if isinstance(value, dict):
                    yield value
    finally:
        con.close()


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


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


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
