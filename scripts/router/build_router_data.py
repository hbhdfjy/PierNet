"""
Build Stage 4 Token Router training data from Stage 3 samples.

The default path reads Stage 3 Parquet partitions and writes Router Parquet
partitions directly. JSONL input/output remains available for compatibility via
--input-format jsonl and --output-format jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from piern.shared.storage import portable  # noqa: E402

DEFAULT_LOCAL_QWEN_DIR = str(Path.home() / "Qwen" / "Qwen2.5-0.5B-Instruct")
DEFAULT_QWEN_EMBEDDING_MODEL = os.getenv("PIERN_QWEN_EMBEDDING_MODEL", DEFAULT_LOCAL_QWEN_DIR)
DEFAULT_QWEN_EMBEDDING_TOKENIZER = os.getenv("PIERN_QWEN_EMBEDDING_TOKENIZER", DEFAULT_QWEN_EMBEDDING_MODEL)

QWEN_TEMPLATE = {
    "_name": "qwen",
    "user_prefix": "<|im_start|>user\n",
    "user_suffix": "<|im_end|>\n",
    "assistant_prefix": "<|im_start|>assistant\n",
}


@dataclass(frozen=True)
class Stage3Source:
    name: str
    simulator: str
    scenario: str
    row_count: int
    storage: str
    source_signature: dict[str, object]
    records: Callable[[], Iterable[dict[str, object]]]
    part_paths: tuple[Path, ...] = ()


def _duplicate_jsonl_output_scenarios(sources: list[Stage3Source]) -> list[str]:
    seen: dict[str, str] = {}
    duplicates: set[str] = set()
    for source in sources:
        previous = seen.setdefault(source.scenario, source.simulator)
        if previous != source.simulator:
            duplicates.add(source.scenario)
    return sorted(duplicates)


def _source_selector(source: Stage3Source) -> str:
    return f"{source.simulator}/{source.scenario}"


def _source_matches_selector(source: Stage3Source, selector: str) -> bool:
    selector = selector.strip()
    if not selector:
        return False
    if "::" in selector:
        simulator, scenario = selector.split("::", 1)
        return source.simulator == simulator and source.scenario == scenario
    if "/" in selector:
        simulator, scenario = selector.split("/", 1)
        return source.simulator == simulator and source.scenario == scenario
    return source.scenario == selector


def _apply_chat_template_pos(input_text: str, trigger_prefix: str, tmpl: dict[str, str]) -> str:
    return tmpl["user_prefix"] + input_text + tmpl["user_suffix"] + tmpl["assistant_prefix"] + trigger_prefix


def _apply_chat_template_neg(
    input_text: str,
    trigger_prefix: str,
    tmpl: dict[str, str],
    rng: random.Random,
) -> str:
    trigger_length = len(trigger_prefix)
    if trigger_length <= 1:
        return tmpl["user_prefix"] + input_text + tmpl["user_suffix"] + tmpl["assistant_prefix"]
    pos = rng.randint(0, trigger_length - 1)
    return tmpl["user_prefix"] + input_text + tmpl["user_suffix"] + tmpl["assistant_prefix"] + trigger_prefix[:pos]


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _load_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _iter_text2comp_parquet_records(paths: tuple[Path, ...]) -> Iterable[dict[str, object]]:
    for path in paths:
        yield from _iter_parquet_row_group_records(
            path, 0, portable.require_parquet_modules()[1].ParquetFile(path).num_row_groups
        )


def _first_record(path: Path) -> dict[str, object] | None:
    return next(_load_jsonl(path), None)


def _metadata_for(record: dict[str, object] | None) -> dict[str, object]:
    metadata = (record or {}).get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _build_jsonl_signature(jsonl_path: Path) -> dict[str, object]:
    stat = jsonl_path.stat()
    return {
        "name": jsonl_path.name,
        "path": str(jsonl_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "row_count": _count_nonempty_lines(jsonl_path),
    }


def _discover_jsonl_sources(data_dir: Path) -> list[Stage3Source]:
    if not data_dir.exists():
        return []
    paths = sorted(path for path in data_dir.glob("*.jsonl") if path.name != "all_training_data.jsonl")
    sources: list[Stage3Source] = []
    for path in paths:
        record = _first_record(path)
        metadata = _metadata_for(record)
        simulator = str(metadata.get("simulator") or "unknown")
        scenario = str(metadata.get("scenario") or path.stem)
        signature = _build_jsonl_signature(path)
        sources.append(
            Stage3Source(
                name=path.name,
                simulator=simulator,
                scenario=scenario,
                row_count=int(signature["row_count"]),
                storage="jsonl",
                source_signature=signature,
                records=lambda p=path: _load_jsonl(p),
            )
        )
    return sources


def _safe_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    return path.name[len(prefix) :] if path.name.startswith(prefix) else path.name


def _parquet_row_count(paths: tuple[Path, ...]) -> int:
    _, pq = portable.require_parquet_modules()
    return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in paths)


def _discover_parquet_sources(data_dir: Path) -> list[Stage3Source]:
    sources: list[Stage3Source] = []
    if not data_dir.exists():
        return []
    for part_dir in sorted(data_dir.glob("simulator=*/scenario=*")):
        if not part_dir.is_dir():
            continue
        part_paths = tuple(sorted(part_dir.glob("*.parquet")))
        if not part_paths:
            continue
        manifest = _safe_json(part_dir / "_manifest.json")
        simulator = str(manifest.get("simulator") or _partition_value(part_dir.parent, "simulator"))
        scenario = str(manifest.get("scenario") or _partition_value(part_dir, "scenario"))
        row_count = int(manifest.get("row_count") or _parquet_row_count(part_paths))
        file_size_bytes = sum(path.stat().st_size for path in part_paths if path.exists())
        mtime = max((path.stat().st_mtime for path in part_paths if path.exists()), default=0.0)
        signature = {
            "storage": "parquet",
            "path": str(part_dir),
            "row_count": row_count,
            "file_size_bytes": file_size_bytes,
            "mtime": mtime,
            "stage3_source": manifest.get("source", {}),
        }
        sources.append(
            Stage3Source(
                name=f"{scenario}.parquet",
                simulator=simulator,
                scenario=scenario,
                row_count=row_count,
                storage="parquet",
                source_signature=signature,
                records=lambda paths=part_paths: _iter_text2comp_parquet_records(paths),
                part_paths=part_paths,
            )
        )
    return sorted(sources, key=lambda item: (item.simulator, item.scenario))


def _source_key(source: Stage3Source) -> tuple[str, str]:
    return (source.simulator, source.scenario)


def _auto_source_dirs(data_dir: Path) -> tuple[list[Path], list[Path]]:
    parquet_dirs = [data_dir]
    jsonl_dirs = [data_dir]
    if (data_dir / "text2comp_parquet").exists():
        parquet_dirs.append(data_dir / "text2comp_parquet")
    if (data_dir / "text2comp").exists():
        jsonl_dirs.append(data_dir / "text2comp")
    return parquet_dirs, jsonl_dirs


def _dedupe_sources(sources: list[Stage3Source]) -> list[Stage3Source]:
    deduped: dict[tuple[str, str], Stage3Source] = {}
    for source in sorted(sources, key=lambda item: (item.simulator, item.scenario, item.storage)):
        deduped.setdefault(_source_key(source), source)
    return sorted(deduped.values(), key=lambda item: (item.simulator, item.scenario))


def _merge_auto_sources(
    parquet_sources: list[Stage3Source],
    jsonl_sources: list[Stage3Source],
) -> tuple[str, list[Stage3Source]]:
    parquet_sources = _dedupe_sources(parquet_sources)
    parquet_keys = {_source_key(source) for source in parquet_sources}
    jsonl_only = [source for source in _dedupe_sources(jsonl_sources) if _source_key(source) not in parquet_keys]
    sources = sorted([*parquet_sources, *jsonl_only], key=lambda item: (item.simulator, item.scenario))
    if parquet_sources and jsonl_only:
        return "mixed", sources
    if parquet_sources:
        return "parquet", sources
    return "jsonl", sources


def _select_sources(data_dir: Path, input_format: str) -> tuple[str, list[Stage3Source]]:
    input_format = input_format.lower()
    if input_format == "jsonl":
        return "jsonl", _discover_jsonl_sources(data_dir)
    if input_format == "parquet":
        return "parquet", _discover_parquet_sources(data_dir)

    parquet_dirs, jsonl_dirs = _auto_source_dirs(data_dir)
    parquet_sources = _dedupe_sources([source for root in parquet_dirs for source in _discover_parquet_sources(root)])
    jsonl_sources = _dedupe_sources([source for root in jsonl_dirs for source in _discover_jsonl_sources(root)])
    return _merge_auto_sources(parquet_sources, jsonl_sources)


def _build_embedding_metadata() -> dict[str, str]:
    return {
        "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
        "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_TOKENIZER,
    }


def _extract_trigger_prefix(target_template: str) -> str:
    marker = "{output_0}"
    pos = target_template.find(marker)
    if pos == -1:
        return ""
    return target_template[:pos]


def _router_samples_from_record(
    record: dict[str, object],
    *,
    simulator: str,
    scenario: str,
    rng: random.Random,
    neg_ratio: int,
) -> Iterable[dict[str, object]]:
    input_text = str(record.get("input") or "")
    metadata = record.get("metadata", {})
    if not input_text or not isinstance(metadata, dict):
        return
    target_template = str(metadata.get("target_template") or "")
    trigger_prefix = _extract_trigger_prefix(target_template)
    if not trigger_prefix:
        return

    base_meta = {
        "simulator": str(metadata.get("simulator") or simulator),
        "scenario": str(metadata.get("scenario") or scenario),
        "language": str(metadata.get("language") or "unknown"),
    }
    yield {
        "context": _apply_chat_template_pos(input_text, trigger_prefix, QWEN_TEMPLATE),
        "label": 1,
        "metadata": base_meta,
    }
    for _ in range(neg_ratio):
        yield {
            "context": _apply_chat_template_neg(input_text, trigger_prefix, QWEN_TEMPLATE, rng),
            "label": 0,
            "metadata": base_meta,
        }


def _iter_router_samples(
    source: Stage3Source,
    rng: random.Random,
    *,
    neg_ratio: int,
    progress_callback=None,
    progress_interval: int = 10000,
) -> Iterable[dict[str, object]]:
    source_processed = 0
    emitted = 0
    for record in source.records():
        produced = 0
        for sample in _router_samples_from_record(
            record,
            simulator=source.simulator,
            scenario=source.scenario,
            rng=rng,
            neg_ratio=neg_ratio,
        ):
            yield sample
            produced += 1
            emitted += 1
        if produced == 0:
            continue

        source_processed += 1
        if progress_callback and source_processed % progress_interval == 0:
            progress_callback(emitted)


def _iter_parquet_row_group_records(
    path: Path, row_group_start: int, row_group_stop: int
) -> Iterable[dict[str, object]]:
    _, pq = portable.require_parquet_modules()
    parquet_file = pq.ParquetFile(path)
    for row_group in range(row_group_start, row_group_stop):
        table = parquet_file.read_row_group(row_group, columns=["record_json"])
        for raw in table.column("record_json").to_pylist():
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _router_part_worker(args: tuple) -> dict[str, object]:
    (
        part_index,
        input_path_raw,
        row_group_start,
        row_group_stop,
        output_path_raw,
        simulator,
        scenario,
        neg_ratio,
        seed,
        compression,
        batch_size,
        row_index_base,
    ) = args
    pa, pq = portable.require_parquet_modules()
    schema = portable.parquet_schema()
    rng = random.Random(int(seed))
    input_path = Path(str(input_path_raw))
    output_path = Path(str(output_path_raw))
    writer = None
    rows: list[dict[str, object]] = []
    counters = {
        "by_language": Counter(),
        "by_style": Counter(),
        "by_time_mode": Counter(),
        "by_label": Counter(),
    }
    row_count = 0
    pos_count = 0

    def flush() -> None:
        nonlocal writer, rows
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema,
                compression=None if compression == "none" else compression,
                use_dictionary=True,
            )
        writer.write_table(table)
        rows = []

    try:
        for record in _iter_parquet_row_group_records(input_path, int(row_group_start), int(row_group_stop)):
            for sample in _router_samples_from_record(
                record,
                simulator=str(simulator),
                scenario=str(scenario),
                rng=rng,
                neg_ratio=int(neg_ratio),
            ):
                row = portable.record_to_parquet_row(
                    "router",
                    str(scenario),
                    int(row_index_base) + row_count,
                    sample,
                    simulator_hint=str(simulator),
                )
                rows.append(row)
                row_count += 1
                if sample.get("label") == 1:
                    pos_count += 1
                if row["language"]:
                    counters["by_language"][row["language"]] += 1
                if row["style"]:
                    counters["by_style"][row["style"]] += 1
                if row["time_mode"]:
                    counters["by_time_mode"][row["time_mode"]] += 1
                if row["label"] is not None:
                    counters["by_label"][str(row["label"])] += 1
                if len(rows) >= batch_size:
                    flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    return {
        "part_index": part_index,
        "path": str(output_path),
        "rows": row_count,
        "pos": pos_count,
        "counters": {key: dict(value) for key, value in counters.items()},
    }


def _build_router_chunks(
    source: Stage3Source,
    output_dir: Path,
    *,
    neg_ratio: int,
    seed: int,
    compression: str,
    batch_size: int,
) -> list[tuple]:
    _, pq = portable.require_parquet_modules()
    chunks: list[tuple] = []
    row_index_base = 0
    part_index = 0
    source_rows_per_chunk = max(10000, min(50000, int(batch_size) * 2))
    for input_path in source.part_paths:
        parquet_file = pq.ParquetFile(input_path)
        start_group = 0
        group_rows = 0
        for group_index in range(parquet_file.num_row_groups):
            group_rows += int(parquet_file.metadata.row_group(group_index).num_rows)
            is_last = group_index == parquet_file.num_row_groups - 1
            if group_rows < source_rows_per_chunk and not is_last:
                continue
            row_group_stop = group_index + 1
            output_path = output_dir / f"part-{part_index:05d}.parquet"
            chunks.append(
                (
                    part_index,
                    str(input_path),
                    start_group,
                    row_group_stop,
                    str(output_path),
                    source.simulator,
                    source.scenario,
                    neg_ratio,
                    seed + part_index,
                    compression,
                    batch_size,
                    row_index_base,
                )
            )
            row_index_base += group_rows * (1 + neg_ratio)
            part_index += 1
            start_group = row_group_stop
            group_rows = 0
    return chunks


def _write_router_parquet_parallel(
    source: Stage3Source,
    *,
    output_root: Path,
    seed: int,
    neg_ratio: int,
    compression: str,
    batch_size: int,
    max_workers: int,
    progress_callback=None,
    extra_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    _, pq = portable.require_parquet_modules()
    schema = portable.parquet_schema()
    partition_dir = portable.partition_dir_for("router", source.simulator, source.scenario, output_root)
    tmp_dir = output_root / (
        f".tmp-router-{portable.safe_partition_value(source.simulator)}-"
        f"{portable.safe_partition_value(source.scenario)}-{os.getpid()}-{time.time_ns()}"
    )
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunks = _build_router_chunks(
        source,
        tmp_dir,
        neg_ratio=neg_ratio,
        seed=seed,
        compression=compression,
        batch_size=batch_size,
    )
    if not chunks:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {
            "kind": "router",
            "scenario": source.scenario,
            "simulator": source.simulator,
            "status": "empty",
            "rows": 0,
            "pos": 0,
        }

    counters = {
        "by_language": Counter(),
        "by_style": Counter(),
        "by_time_mode": Counter(),
        "by_label": Counter(),
    }
    row_count = 0
    pos_count = 0
    part_paths: list[str] = []
    try:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
            futures = [executor.submit(_router_part_worker, chunk) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                rows = int(result.get("rows") or 0)
                row_count += rows
                pos_count += int(result.get("pos") or 0)
                if rows > 0:
                    part_paths.append(str(result["path"]))
                for key, values in (result.get("counters") or {}).items():
                    if key in counters and isinstance(values, dict):
                        counters[key].update({str(k): int(v) for k, v in values.items()})
                if progress_callback:
                    progress_callback(row_count)

        if row_count == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {
                "kind": "router",
                "scenario": source.scenario,
                "simulator": source.simulator,
                "status": "empty",
                "rows": 0,
                "pos": 0,
            }

        actual = 0
        for path in sorted(Path(item) for item in part_paths):
            actual += int(pq.ParquetFile(path).metadata.num_rows)
        if actual != row_count:
            raise RuntimeError(
                f"row count mismatch for router/{source.scenario}: generated={row_count}, parquet={actual}"
            )

        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        partition_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.replace(partition_dir)

        manifest = {
            "version": portable.PARQUET_SCHEMA_VERSION,
            "kind": "router",
            "storage": "parquet",
            "generated_at": time.time(),
            "source": {**source.source_signature, "router_workers": max_workers},
            "simulator": source.simulator,
            "scenario": source.scenario,
            "row_count": row_count,
            "by_language": dict(counters["by_language"]),
            "by_style": dict(counters["by_style"]),
            "by_time_mode": dict(counters["by_time_mode"]),
            "by_label": dict(counters["by_label"]),
            "timeseries_shape_obs": None,
            "schema": [field.name for field in schema],
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        (partition_dir / "_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "kind": "router",
            "scenario": source.scenario,
            "simulator": source.simulator,
            "status": "written",
            "rows": row_count,
            "pos": pos_count,
            "path": str(partition_dir),
        }
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _write_jsonl(samples: Iterable[dict[str, object]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_all_jsonl(samples: list[dict[str, object]], output_dir: Path, seed: int) -> int:
    rng = random.Random(seed)
    rng.shuffle(samples)
    out_path = output_dir / "train.jsonl"
    count = _write_jsonl(samples, out_path)
    print(f"  [train] {count} rows -> {out_path}", flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Token Router training data.")
    parser.add_argument("--data-dir", type=str, default=str(portable.TEXT2COMP_PARQUET_DIR))
    parser.add_argument("--output-dir", type=str, default=str(portable.ROUTER_PARQUET_DIR))
    parser.add_argument("--input-format", choices=["auto", "parquet", "jsonl"], default="auto")
    parser.add_argument("--output-format", choices=["parquet", "jsonl", "both"], default="parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--chat-template", type=str, default="qwen", choices=["qwen"])
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("PIERN_ROUTER_BUILD_WORKERS", "8")),
        help="Parquet 路由数据构建的场景内并行进程数",
    )
    args = parser.parse_args()

    if args.chat_template != "qwen":
        print("[error] only qwen chat template is supported", flush=True)
        raise SystemExit(1)

    embedding_metadata = _build_embedding_metadata()
    data_dir = PROJECT_ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    output_dir = PROJECT_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    scenario_jsonl_dir = output_dir / "by_scenario"
    max_workers = max(1, int(args.max_workers))

    selected_format, sources = _select_sources(data_dir, args.input_format)
    if args.scenarios:
        wanted = {selector.strip() for selector in args.scenarios if selector.strip()}
        sources = [
            source for source in sources if any(_source_matches_selector(source, selector) for selector in wanted)
        ]
        if not sources:
            print(f"[error] scenarios not found: {sorted(wanted)}", flush=True)
            raise SystemExit(1)
    if not sources:
        print(f"[error] no Stage 3 samples found for input_format={args.input_format} data_dir={data_dir}", flush=True)
        raise SystemExit(1)

    if args.output_format in {"jsonl", "both"}:
        duplicate_scenarios = _duplicate_jsonl_output_scenarios(sources)
        if duplicate_scenarios:
            print(
                "[error] JSONL router output cannot represent duplicate scenario names across simulators: "
                f"{duplicate_scenarios}. Use --output-format parquet instead.",
                flush=True,
            )
            raise SystemExit(1)

    print(f"[router-build] input_storage={selected_format}", flush=True)
    print(f"[router-build] output_format={args.output_format}", flush=True)
    print(f"[router-build] chat_template={QWEN_TEMPLATE['_name']}", flush=True)
    print(
        "[router-build] embedding "
        f"model={embedding_metadata['embedding_model']} "
        f"tokenizer={embedding_metadata['embedding_tokenizer']}",
        flush=True,
    )
    print(f"[router-build] scenarios={len(sources)}", flush=True)
    print(f"[router-build] workers={max_workers} batch_size={max(1, args.batch_size)}", flush=True)

    for source in sources:
        expected = source.row_count * (1 + args.neg_ratio)
        print(f"PROGRESS_INIT:{_source_selector(source)}:{expected}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_format in {"jsonl", "both"}:
        scenario_jsonl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    all_jsonl_samples: list[dict[str, object]] = []
    total_rows = 0
    total_pos = 0

    for source in sources:
        expected = source.row_count * (1 + args.neg_ratio)
        print(f"  processing {_source_selector(source)} ({source.storage}) ...", flush=True)

        def _progress(done_samples: int, sc: str = _source_selector(source), exp: int = expected) -> None:
            print(f"PROGRESS_UPDATE:{sc}:{done_samples}:{exp}", flush=True)

        counters = {"rows": 0, "pos": 0}

        def generated_samples() -> Iterable[dict[str, object]]:
            for sample in _iter_router_samples(source, rng, neg_ratio=args.neg_ratio, progress_callback=_progress):
                counters["rows"] += 1
                if sample.get("label") == 1:
                    counters["pos"] += 1
                yield sample

        if args.output_format == "parquet":
            extra_manifest = {
                "chat_template": QWEN_TEMPLATE["_name"],
                "neg_ratio": args.neg_ratio,
                "source_signature": source.source_signature,
                **embedding_metadata,
            }
            if source.storage == "parquet" and source.part_paths and max_workers > 1:
                result = _write_router_parquet_parallel(
                    source,
                    output_root=output_dir,
                    seed=args.seed,
                    neg_ratio=args.neg_ratio,
                    compression=args.compression,
                    batch_size=max(1, args.batch_size),
                    max_workers=max_workers,
                    progress_callback=_progress,
                    extra_manifest=extra_manifest,
                )
                rows = int(result.get("rows") or 0)
                counters["rows"] = rows
                counters["pos"] = int(result.get("pos") or 0)
            else:
                result = portable.write_records_partition(
                    "router",
                    generated_samples(),
                    simulator=source.simulator,
                    scenario=source.scenario,
                    source=source.source_signature,
                    output_root=output_dir,
                    batch_size=max(1, args.batch_size),
                    compression=args.compression,
                    overwrite=True,
                    extra_manifest=extra_manifest,
                )
                rows = int(result.get("rows") or counters["rows"])
        else:
            scenario_samples = list(generated_samples())
            rows = len(scenario_samples)
            if args.output_format in {"jsonl", "both"}:
                scenario_path = scenario_jsonl_dir / f"{source.scenario}.jsonl"
                _write_jsonl(scenario_samples, scenario_path)
                meta_payload = {
                    "scenario": source.scenario,
                    "source_file": source.name,
                    "source_storage": source.storage,
                    "chat_template": QWEN_TEMPLATE["_name"],
                    "neg_ratio": args.neg_ratio,
                    "source_signature": source.source_signature,
                    "output_count": rows,
                    **embedding_metadata,
                }
                scenario_path.with_suffix(".meta.json").write_text(
                    json.dumps(meta_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                all_jsonl_samples.extend(scenario_samples)
            if args.output_format == "both":
                portable.write_records_partition(
                    "router",
                    scenario_samples,
                    simulator=source.simulator,
                    scenario=source.scenario,
                    source=source.source_signature,
                    output_root=output_dir,
                    batch_size=max(1, args.batch_size),
                    compression=args.compression,
                    overwrite=True,
                    extra_manifest={
                        "chat_template": QWEN_TEMPLATE["_name"],
                        "neg_ratio": args.neg_ratio,
                        "source_signature": source.source_signature,
                        **embedding_metadata,
                    },
                )

        if rows == 0:
            print(f"  [warn] no router samples generated for {_source_selector(source)}", flush=True)
            print(f"PROGRESS_DONE:{_source_selector(source)}:0:{expected}", flush=True)
            continue

        n_pos = counters["pos"]
        if args.output_format != "parquet":
            n_pos = sum(1 for sample in all_jsonl_samples[-rows:] if sample.get("label") == 1)
        total_rows += rows
        total_pos += n_pos
        print(f"  done: {rows} rows (pos={n_pos}, neg={rows - n_pos})", flush=True)
        print(f"PROGRESS_DONE:{_source_selector(source)}:{rows}:{expected}", flush=True)

    if total_rows == 0:
        print(
            "[error] no router samples were generated; make sure Stage 3 data exists and target_template contains {output_0}",
            flush=True,
        )
        raise SystemExit(1)

    if args.output_format in {"jsonl", "both"}:
        total_rows = _write_all_jsonl(all_jsonl_samples, output_dir, args.seed)
        total_pos = sum(1 for sample in all_jsonl_samples if sample.get("label") == 1)

    print(f"\n  total={total_rows} rows (pos={total_pos}, neg={total_rows - total_pos})", flush=True)
    print(f"\n[done] total={total_rows} -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
