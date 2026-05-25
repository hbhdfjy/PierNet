#!/usr/bin/env python3
"""Migrate legacy PierNet JSONL datasets to partitioned Parquet.

The migration is side-by-side: existing JSONL files are not deleted.  For low
free-disk environments, migrate one kind or scenario at a time and delete legacy
files only after validation and a successful backup/DVC push.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PierNet.shared.runtime.paths import DATA_ROOT  # noqa: E402
from PierNet.shared.storage import portable  # noqa: E402

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - user-facing runtime guard
    pa = None
    pq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None

SCHEMA_VERSION = 1
_SCHEMA = None


def require_pyarrow():
    if pa is None or pq is None:
        raise SystemExit("pyarrow is required for JSONL -> Parquet migration. Install pyarrow>=15.0.0") from _PYARROW_IMPORT_ERROR


def parquet_schema():
    global _SCHEMA
    require_pyarrow()
    if _SCHEMA is None:
        _SCHEMA = pa.schema(
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
    return _SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate PierNet JSONL data to partitioned Parquet.")
    parser.add_argument("--kind", choices=["text2comp", "router", "all"], default="all")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Only migrate these scenario names.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Parquet partitions.")
    parser.add_argument("--validate", action="store_true", help="Validate Parquet row count after each partition is written.")
    parser.add_argument("--dvc-add", action="store_true", help="Run 'dvc add' for migrated roots when dvc is installed.")
    return parser.parse_args()


def iter_sources(kind: str, scenarios: set[str] | None) -> list[Path]:
    if kind == "text2comp":
        source_dir = DATA_ROOT / "text2comp"
        paths = sorted(path for path in source_dir.glob("*.jsonl") if path.name != "all_training_data.jsonl")
    elif kind == "router":
        source_dir = DATA_ROOT / "router" / "by_scenario"
        paths = sorted(source_dir.glob("*.jsonl"))
    else:
        raise ValueError(kind)
    return [path for path in paths if source_matches_scenarios(kind, path, scenarios)]



def source_matches_scenarios(kind: str, path: Path, scenarios: set[str] | None) -> bool:
    if not scenarios:
        return True
    simulator, scenario = detect_simulator_scenario(kind, path)
    return scenario in scenarios or f"{simulator}/{scenario}" in scenarios or path.stem in scenarios

def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def first_record(path: Path) -> dict[str, Any] | None:
    return next(iter_jsonl(path), None)


def metadata_for(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def detect_simulator_scenario(kind: str, path: Path) -> tuple[str, str]:
    record = first_record(path)
    metadata = metadata_for(record or {})
    simulator = str(metadata.get("simulator") or "unknown")
    scenario = str(metadata.get("scenario") or path.stem)
    return simulator, scenario


def to_row(kind: str, scenario_hint: str, row_index: int, record: dict[str, Any]) -> dict[str, Any]:
    metadata = metadata_for(record)
    observation = metadata.get("observation", {})
    if not isinstance(observation, dict):
        observation = {}
    simulator = str(metadata.get("simulator") or "unknown")
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


def convert_file(kind: str, source_path: Path, *, batch_size: int, compression: str, overwrite: bool, validate: bool) -> dict[str, Any]:
    simulator, scenario = detect_simulator_scenario(kind, source_path)
    output_root = portable.parquet_root(kind)
    partition_dir = output_root / f"simulator={portable.safe_partition_value(simulator)}" / f"scenario={portable.safe_partition_value(scenario)}"
    if partition_dir.exists() and not overwrite:
        return {"source": str(source_path), "scenario": scenario, "status": "skipped", "reason": "partition exists"}

    tmp_dir = output_root / f".tmp-{kind}-{portable.safe_partition_value(scenario)}-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_path = tmp_dir / "part-00000.parquet"

    require_pyarrow()
    schema = parquet_schema()
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
        for row_index, record in enumerate(iter_jsonl(source_path)):
            row = to_row(kind, scenario, row_index, record)
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
            metadata = metadata_for(record)
            if timeseries_shape_obs is None:
                shape = metadata.get("timeseries_shape_obs")
                if isinstance(shape, (list, tuple)):
                    timeseries_shape_obs = list(shape)
            if len(rows) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    if row_count == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"source": str(source_path), "scenario": scenario, "status": "empty"}

    if validate:
        actual = pq.ParquetFile(part_path).metadata.num_rows
        if int(actual) != row_count:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"row count mismatch for {source_path}: jsonl={row_count}, parquet={actual}")

    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    partition_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.replace(partition_dir)
    manifest = {
        "version": SCHEMA_VERSION,
        "kind": kind,
        "storage": "parquet",
        "generated_at": time.time(),
        "source": portable.source_signature(source_path),
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
    (partition_dir / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "source": str(source_path),
        "scenario": scenario,
        "simulator": simulator,
        "status": "written",
        "rows": row_count,
        "path": str(partition_dir),
    }


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def maybe_dvc_add(kinds: list[str]) -> None:
    dvc = shutil.which("dvc")
    if not dvc:
        print("[warn] --dvc-add requested but dvc is not installed; skipping", file=sys.stderr)
        return
    for kind in kinds:
        root = portable.parquet_root(kind)
        if root.exists():
            subprocess.run([dvc, "add", str(root)], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    kinds = ["text2comp", "router"] if args.kind == "all" else [args.kind]
    scenarios = set(args.scenarios) if args.scenarios else None
    results = []
    for kind in kinds:
        sources = iter_sources(kind, scenarios)
        if not sources:
            print(f"[warn] no {kind} JSONL sources found")
            continue
        print(f"[migrate] kind={kind} files={len(sources)}")
        for source in sources:
            result = convert_file(
                kind,
                source,
                batch_size=max(1, args.batch_size),
                compression=args.compression,
                overwrite=args.overwrite,
                validate=args.validate,
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    if args.dvc_add:
        maybe_dvc_add(kinds)
    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
