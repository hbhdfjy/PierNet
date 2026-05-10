"""
Build Stage 4 Token Router training data from Stage 3 samples.

The default path reads Stage 3 Parquet partitions and writes Router Parquet
partitions directly. JSONL input/output remains available for compatibility via
--input-format jsonl and --output-format jsonl.
"""

from __future__ import annotations

import argparse
import os
import json
import random
import sys
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


def _discover_parquet_sources() -> list[Stage3Source]:
    sources: list[Stage3Source] = []
    for part in portable.discover_partitions("text2comp"):
        signature = {
            "storage": "parquet",
            "path": str(part.path),
            "row_count": part.row_count,
            "file_size_bytes": part.file_size_bytes,
            "mtime": part.mtime,
            "stage3_source": part.metadata.get("source", {}),
        }
        sources.append(
            Stage3Source(
                name=f"{part.scenario}.parquet",
                simulator=part.simulator,
                scenario=part.scenario,
                row_count=part.row_count,
                storage="parquet",
                source_signature=signature,
                records=lambda sim=part.simulator, sc=part.scenario: portable.iter_records(
                    "text2comp",
                    filters=[("simulator", sim), ("scenario", sc)],
                ),
            )
        )
    return sorted(sources, key=lambda item: item.scenario)


def _select_sources(data_dir: Path, input_format: str) -> tuple[str, list[Stage3Source]]:
    input_format = input_format.lower()
    jsonl_sources = _discover_jsonl_sources(data_dir)
    parquet_sources = _discover_parquet_sources()
    if input_format == "jsonl":
        return "jsonl", jsonl_sources
    if input_format == "parquet":
        return "parquet", parquet_sources
    if parquet_sources and (data_dir.name.endswith("_parquet") or not jsonl_sources):
        return "parquet", parquet_sources
    if jsonl_sources:
        return "jsonl", jsonl_sources
    return "parquet", parquet_sources


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


def _iter_router_samples(
    source: Stage3Source,
    rng: random.Random,
    *,
    neg_ratio: int,
    progress_callback=None,
    progress_interval: int = 500,
) -> Iterable[dict[str, object]]:
    source_processed = 0
    emitted = 0
    for record in source.records():
        input_text = str(record.get("input") or "")
        metadata = record.get("metadata", {})
        if not input_text or not isinstance(metadata, dict):
            continue
        target_template = str(metadata.get("target_template") or "")
        trigger_prefix = _extract_trigger_prefix(target_template)
        if not trigger_prefix:
            continue

        base_meta = {
            "simulator": str(metadata.get("simulator") or source.simulator),
            "scenario": str(metadata.get("scenario") or source.scenario),
            "language": str(metadata.get("language") or "unknown"),
        }
        yield {
            "context": _apply_chat_template_pos(input_text, trigger_prefix, QWEN_TEMPLATE),
            "label": 1,
            "metadata": base_meta,
        }
        emitted += 1
        for _ in range(neg_ratio):
            yield {
                "context": _apply_chat_template_neg(input_text, trigger_prefix, QWEN_TEMPLATE, rng),
                "label": 0,
                "metadata": base_meta,
            }
            emitted += 1

        source_processed += 1
        if progress_callback and source_processed % progress_interval == 0:
            progress_callback(emitted)


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
    parser.add_argument("--data-dir", type=str, default="data/text2comp_parquet")
    parser.add_argument("--output-dir", type=str, default="data/router_parquet")
    parser.add_argument("--input-format", choices=["auto", "parquet", "jsonl"], default="auto")
    parser.add_argument("--output-format", choices=["parquet", "jsonl", "both"], default="parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--chat-template", type=str, default="qwen", choices=["qwen"])
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    if args.chat_template != "qwen":
        print("[error] only qwen chat template is supported", flush=True)
        raise SystemExit(1)

    embedding_metadata = _build_embedding_metadata()
    data_dir = PROJECT_ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    output_dir = PROJECT_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    scenario_jsonl_dir = output_dir / "by_scenario"

    selected_format, sources = _select_sources(data_dir, args.input_format)
    if args.scenarios:
        wanted = set(args.scenarios)
        sources = [source for source in sources if source.scenario in wanted]
        if not sources:
            print(f"[error] scenarios not found: {sorted(wanted)}", flush=True)
            raise SystemExit(1)
    if not sources:
        print(f"[error] no Stage 3 samples found for input_format={args.input_format} data_dir={data_dir}", flush=True)
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

    for source in sources:
        expected = source.row_count * (1 + args.neg_ratio)
        print(f"PROGRESS_INIT:{source.scenario}:{expected}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_format in {"jsonl", "both"}:
        scenario_jsonl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    all_jsonl_samples: list[dict[str, object]] = []
    total_rows = 0
    total_pos = 0

    for source in sources:
        expected = source.row_count * (1 + args.neg_ratio)
        print(f"  processing {source.name} ({source.storage}) ...", flush=True)

        def _progress(done_samples: int, sc: str = source.scenario, exp: int = expected) -> None:
            print(f"PROGRESS_UPDATE:{sc}:{done_samples}:{exp}", flush=True)

        counters = {"rows": 0, "pos": 0}

        def generated_samples() -> Iterable[dict[str, object]]:
            for sample in _iter_router_samples(source, rng, neg_ratio=args.neg_ratio, progress_callback=_progress):
                counters["rows"] += 1
                if sample.get("label") == 1:
                    counters["pos"] += 1
                yield sample

        if args.output_format == "parquet":
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
                extra_manifest={
                    "chat_template": QWEN_TEMPLATE["_name"],
                    "neg_ratio": args.neg_ratio,
                    "source_signature": source.source_signature,
                    **embedding_metadata,
                },
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
            print(f"  [warn] no router samples generated for {source.scenario}", flush=True)
            print(f"PROGRESS_DONE:{source.scenario}:0:{expected}", flush=True)
            continue

        n_pos = counters["pos"]
        if args.output_format != "parquet":
            n_pos = sum(1 for sample in all_jsonl_samples[-rows:] if sample.get("label") == 1)
        total_rows += rows
        total_pos += n_pos
        print(f"  done: {rows} rows (pos={n_pos}, neg={rows - n_pos})", flush=True)
        print(f"PROGRESS_DONE:{source.scenario}:{rows}:{expected}", flush=True)

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
