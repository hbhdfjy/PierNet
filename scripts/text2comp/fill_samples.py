"""
阶段二批量脚本：将模板库与 HDF5 数值结合，生成最终训练样本。

默认直接写入便携式 Parquet 分区 data/text2comp_parquet/，不再先落 JSONL。
如需兼容旧流程，可显式传 --output-format jsonl 或 --output-format both。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PierNet.core.storage import load_dataset
from PierNet.shared.storage import portable
from PierNet.synth.text2comp.pipeline import (
    _resolve_data_path,
    _scan_h5_files,
    _scenario_name_from_path,
    assert_unique_stage_scenarios,
    load_config,
)
from PierNet.synth.text2comp.template_store import fill_sample, load_templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PARALLEL_FILL_CONTEXT: dict[str, object] = {}
_PARALLEL_FILL_STOP_EVENT = None


def _init_parallel_fill_worker(
    timeseries,
    params,
    template_specs,
    t_order,
    d_order,
    output_info_list,
    precision: int,
    simulator: str,
    scenario: str,
    compression: str,
    batch_size: int,
    stop_event=None,
):
    global _PARALLEL_FILL_CONTEXT, _PARALLEL_FILL_STOP_EVENT
    _PARALLEL_FILL_CONTEXT = {
        "timeseries": timeseries,
        "params": params,
        "template_specs": template_specs,
        "t_order": t_order,
        "d_order": d_order,
        "output_info_list": output_info_list,
        "precision": precision,
        "simulator": simulator,
        "scenario": scenario,
        "compression": compression,
        "batch_size": batch_size,
    }
    _PARALLEL_FILL_STOP_EVENT = stop_event


def _parallel_fill_part_worker(args: tuple[int, int, int, str]) -> dict[str, object]:
    part_index, start, stop, part_path_raw = args
    ctx = _PARALLEL_FILL_CONTEXT
    timeseries = ctx["timeseries"]
    params = ctx["params"]
    template_specs = ctx["template_specs"]
    t_order = ctx["t_order"]
    d_order = ctx["d_order"]
    output_info_list = ctx["output_info_list"]
    precision = int(ctx["precision"])
    simulator = str(ctx["simulator"])
    scenario = str(ctx["scenario"])
    compression = str(ctx["compression"])
    batch_size = int(ctx["batch_size"])

    pa, pq = portable.require_parquet_modules()
    schema = portable.parquet_schema()
    part_path = Path(part_path_raw)
    writer = None
    rows: list[dict[str, object]] = []
    counters = {
        "by_language": Counter(),
        "by_style": Counter(),
        "by_time_mode": Counter(),
        "by_label": Counter(),
    }
    row_count = 0
    skipped = 0
    nan_skipped = 0
    timeseries_shape_obs = None
    n_avail_t = len(t_order)
    n_avail_d = len(d_order)

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
        for i in range(start, stop):
            if _PARALLEL_FILL_STOP_EVENT is not None and (i - start) % 256 == 0 and _PARALLEL_FILL_STOP_EVENT.is_set():
                raise InterruptedError("任务已终止")
            t_idx = t_order[i % n_avail_t]
            d_idx = d_order[i % n_avail_d]
            template, valid_time_idx, valid_ch = template_specs[t_idx]
            ts = timeseries[d_idx]
            p = params[d_idx]

            if len(valid_time_idx) == 0:
                skipped += 1
                continue

            ts_obs = ts[:, valid_time_idx]
            if valid_ch is not None:
                if len(valid_ch) == 0:
                    skipped += 1
                    continue
                ts_obs = ts_obs[valid_ch, :]

            if not np.isfinite(ts_obs).all():
                nan_skipped += 1
                skipped += 1
                continue

            try:
                sample = fill_sample(
                    template,
                    p,
                    ts_obs,
                    sample_idx=d_idx,
                    output_info=output_info_list,
                    precision=precision,
                )
            except Exception:
                skipped += 1
                continue

            metadata = sample.get("metadata", {})
            if isinstance(metadata, dict) and timeseries_shape_obs is None:
                shape = metadata.get("timeseries_shape_obs")
                if isinstance(shape, (list, tuple)):
                    timeseries_shape_obs = list(shape)
            row = portable.record_to_parquet_row(
                "text2comp",
                scenario,
                row_count + start,
                sample,
                simulator_hint=simulator,
            )
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
            if len(rows) >= batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    return {
        "part_index": part_index,
        "path": str(part_path),
        "row_count": row_count,
        "skipped": skipped,
        "nan_skipped": nan_skipped,
        "counters": {key: dict(value) for key, value in counters.items()},
        "timeseries_shape_obs": timeseries_shape_obs,
    }


def _jsonl_row_count(path: Path) -> int:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _partition_row_count(root: Path, simulator: str, scenario: str) -> int:
    part_dir = portable.partition_dir_for("text2comp", simulator, scenario, root)
    manifest_path = part_dir / "_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            return max(0, int(payload.get("row_count") or 0))
        except Exception:
            pass

    parquet_files = sorted(part_dir.glob("*.parquet"))
    if not parquet_files:
        return 0
    try:
        _, pq = portable.require_parquet_modules()
        return sum(int(pq.ParquetFile(str(path)).metadata.num_rows) for path in parquet_files)
    except Exception:
        # Existing legacy partitions without readable metadata are treated as
        # non-empty so skip-existing remains conservative.
        return 1


def _source_signature(path: Path) -> dict[str, object]:
    try:
        return portable.source_signature(path)
    except FileNotFoundError:
        return {"path": str(path), "missing": True}


def _valid_time_indices(indices: Iterable[int], n_timesteps: int) -> np.ndarray:
    arr = np.asarray(list(indices), dtype=np.int64)
    return arr[(arr >= 0) & (arr < n_timesteps)]


def _valid_channel_indices(indices: Iterable[int] | None, n_channels: int) -> np.ndarray | None:
    if indices is None:
        return None
    arr = np.asarray(list(indices), dtype=np.int64)
    return arr[(arr >= 0) & (arr < n_channels)]


def run_fill_samples(
    cfg_path: str,
    n_samples: int = None,
    scenarios: list = None,
    templates_dir: str = None,
    output_dir: str = None,
    skip_existing: bool = False,
    seed: int = None,
    precision: int = 4,
    output_format: str = "parquet",
    compression: str = "zstd",
    batch_size: int = 8192,
    max_workers: int | None = None,
    on_scenario_start=None,  # (scenario: str, total: int) -> None
    on_progress=None,        # (scenario: str, done: int) -> None
    on_log=None,             # (line: str) -> None
    should_stop=None,        # () -> bool
) -> None:
    output_format = (output_format or "parquet").lower()
    if output_format not in {"parquet", "jsonl", "both"}:
        raise ValueError("output_format must be one of: parquet, jsonl, both")

    cfg_path = Path(cfg_path)
    cfg = load_config(cfg_path)
    gen_cfg = cfg.get("generation", {})

    n_per_scenario = n_samples if n_samples is not None else gen_cfg.get("n_samples_per_scenario", 1000)
    _seed = seed if seed is not None else cfg.get("seed", 42)
    scenario_workers = max(1, int(max_workers if max_workers is not None else (gen_cfg.get("max_workers") or 1)))

    base_dir = cfg_path.parent.parent.parent
    if not (base_dir / "data").exists():
        base_dir = Path.cwd()

    registry: dict = {}
    registry_path_str = cfg.get("registry", "configs/text2comp/registry.yaml")
    registry_path = base_dir / registry_path_str if not Path(registry_path_str).is_absolute() else Path(registry_path_str)
    if registry_path.exists():
        try:
            registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass

    tmpl_dir = Path(templates_dir) if templates_dir else _resolve_data_path("data/templates", base_dir)
    parquet_dir = None
    jsonl_dir = None
    if output_format in {"parquet", "both"}:
        default_parquet = cfg.get("parquet_output_dir", "data/text2comp_parquet")
        parquet_dir = Path(output_dir) if output_dir else _resolve_data_path(default_parquet, base_dir)
        parquet_dir.mkdir(parents=True, exist_ok=True)
    if output_format in {"jsonl", "both"}:
        jsonl_dir = Path(output_dir) if output_dir else _resolve_data_path(cfg.get("output_dir", "data/text2comp"), base_dir)
        jsonl_dir.mkdir(parents=True, exist_ok=True)

    h5_files = _scan_h5_files(cfg, base_dir)
    if not h5_files:
        raise RuntimeError("未找到任何 HDF5 文件，请检查 data_root 配置")

    if scenarios is not None:
        scenarios_set = set(scenarios)
        h5_files = [
            (p, s, sfx) for p, s, sfx in h5_files
            if _scenario_name_from_path(p, sfx) in scenarios_set
        ]
        if not h5_files:
            raise RuntimeError(f"未找到所选场景: {', '.join(sorted(scenarios_set))}")

    assert_unique_stage_scenarios(h5_files)

    log_lock = threading.Lock()

    def _log(line: str) -> None:
        with log_lock:
            print(line)
            if on_log:
                on_log(line)

    failures: list[str] = []
    failures_lock = threading.Lock()

    def _raise_if_stopped() -> None:
        if should_stop is not None and should_stop():
            raise InterruptedError("任务已终止")

    def _record_failure(scenario: str, message: str) -> None:
        _log(f"[错误] {scenario}: {message}")
        with failures_lock:
            failures.append(f"{scenario}: {message}")

    _log(f"\n找到 {len(h5_files)} 个场景，每场景生成 {n_per_scenario} 条样本")
    _log(f"输出格式: {output_format}")
    if output_format == "parquet" and scenario_workers > 1:
        _log(f"场景内样本并行: {scenario_workers}")
    else:
        _log(f"场景并行: {min(scenario_workers, max(len(h5_files), 1))}")
    if parquet_dir is not None:
        _log(f"Parquet目录: {parquet_dir}")
    if jsonl_dir is not None:
        _log(f"JSONL目录: {jsonl_dir}")
    _log("")

    stats = Counter()
    written_outputs: list[Path] = []

    stats_lock = threading.Lock()

    def _process_scenario(item: tuple[Path, str, str]) -> None:
        _raise_if_stopped()
        h5_path, simulator, file_suffix = item
        scenario_name = _scenario_name_from_path(h5_path, file_suffix)
        tmpl_path = tmpl_dir / f"{scenario_name}_templates.jsonl"
        jsonl_out_path = jsonl_dir / f"{scenario_name}.jsonl" if jsonl_dir is not None else None
        local_outputs: list[Path] = []

        if skip_existing:
            existing_outputs: list[tuple[str, int]] = []
            if parquet_dir is not None:
                existing_outputs.append(("Parquet", _partition_row_count(parquet_dir, simulator, scenario_name)))
            if jsonl_out_path is not None:
                existing_outputs.append(("JSONL", _jsonl_row_count(jsonl_out_path)))

            if existing_outputs and all(count >= n_per_scenario for _label, count in existing_outputs):
                counts = ", ".join(f"{label}={count}/{n_per_scenario}" for label, count in existing_outputs)
                _log(f"[跳过] {scenario_name} 已达到目标（{counts}）")
                if on_progress:
                    on_progress(scenario_name, n_per_scenario)
                if jsonl_out_path is not None and _jsonl_row_count(jsonl_out_path) > 0:
                    local_outputs.append(jsonl_out_path)
                    with stats_lock:
                        written_outputs.extend(local_outputs)
                return

            partial_outputs = [(label, count) for label, count in existing_outputs if count > 0]
            if partial_outputs:
                counts = ", ".join(f"{label}={count}/{n_per_scenario}" for label, count in partial_outputs)
                _log(f"[重建] {scenario_name} 已有输出未达到目标（{counts}）")

        if not tmpl_path.exists():
            _record_failure(scenario_name, f"模板文件不存在: {tmpl_path}")
            return

        _log(f"\n[处理] {scenario_name}")
        _log(f"  模板: {tmpl_path.name}")
        _log(f"  数据: {h5_path.name}")

        output_info_list = None
        sim_entry = registry.get(simulator, {})
        if isinstance(sim_entry, dict):
            output_info_list = sim_entry.get("output_info")
            sc_entry = sim_entry.get("scenarios", {}).get(scenario_name, {})
            if isinstance(sc_entry, dict) and "output_info" in sc_entry:
                output_info_list = sc_entry["output_info"]

        templates = load_templates(tmpl_path)
        if not templates:
            _record_failure(scenario_name, f"模板文件为空: {tmpl_path}")
            return

        try:
            timeseries, params, param_names = load_dataset(str(h5_path))
        except Exception as exc:
            _record_failure(scenario_name, f"加载 HDF5 失败: {exc}")
            return

        n_avail_t = len(templates)
        n_avail_d = len(params)
        n = n_per_scenario
        _log(f"  模板数: {n_avail_t}，HDF5样本数: {n_avail_d}，目标: {n_per_scenario}，实际: {n}")

        if on_scenario_start:
            on_scenario_start(scenario_name, n)

        ts_len = int(timeseries.shape[2])
        n_channels = int(timeseries.shape[1])
        template_specs = []
        for template in templates:
            valid_time_idx = _valid_time_indices(template.time_indices, ts_len)
            valid_ch = _valid_channel_indices(template.channel_indices, n_channels)
            template_specs.append((template, valid_time_idx, valid_ch))

        rng = np.random.default_rng(_seed)
        t_order = rng.permutation(n_avail_t)
        d_order = rng.permutation(n_avail_d)
        written = 0
        skipped = 0
        nan_skipped = 0
        last_reported = -1
        progress_every = max(10000, min(50000, max(1, n // 100)))

        def iter_samples(jsonl_handle=None) -> Iterable[dict[str, object]]:
            nonlocal written, skipped, nan_skipped, last_reported
            for i in range(n):
                if i % 256 == 0:
                    _raise_if_stopped()
                t_idx = t_order[i % n_avail_t]
                d_idx = d_order[i % n_avail_d]
                template, valid_time_idx, valid_ch = template_specs[t_idx]
                ts = timeseries[d_idx]
                p = params[d_idx]

                if len(valid_time_idx) == 0:
                    skipped += 1
                    continue

                ts_obs = ts[:, valid_time_idx]
                if valid_ch is not None:
                    if len(valid_ch) == 0:
                        skipped += 1
                        continue
                    ts_obs = ts_obs[valid_ch, :]

                if not np.isfinite(ts_obs).all():
                    nan_skipped += 1
                    skipped += 1
                    continue

                try:
                    sample = fill_sample(template, p, ts_obs, sample_idx=d_idx, output_info=output_info_list, precision=precision)
                except Exception as exc:
                    logger.warning(f"样本 {i} 填充失败: {exc}")
                    skipped += 1
                    continue

                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
                if on_progress and written % progress_every == 0 and written != last_reported:
                    on_progress(scenario_name, written)
                    last_reported = written
                yield sample

        source = {
            "hdf5": _source_signature(h5_path),
            "templates": _source_signature(tmpl_path),
            "requested_samples": n,
            "seed": _seed,
            "precision": precision,
            "max_workers": scenario_workers,
        }

        def write_parallel_partition(sample_workers: int) -> dict[str, object]:
            nonlocal written, skipped, nan_skipped, last_reported
            _, pq = portable.require_parquet_modules()
            schema = portable.parquet_schema()
            root = Path(parquet_dir)
            partition_dir = portable.partition_dir_for("text2comp", simulator, scenario_name, root)
            tmp_dir = root / (
                f".tmp-text2comp-{portable.safe_partition_value(simulator)}-"
                f"{portable.safe_partition_value(scenario_name)}-{os.getpid()}-{time.time_ns()}"
            )
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            target_parts = max(1, min(sample_workers, n))
            chunk_size = max(10000, min(50000, (n + target_parts - 1) // target_parts))
            chunks = [
                (part_index, start, min(start + chunk_size, n), str(tmp_dir / f"part-{part_index:05d}.parquet"))
                for part_index, start in enumerate(range(0, n, chunk_size))
            ]
            counters = {
                "by_language": Counter(),
                "by_style": Counter(),
                "by_time_mode": Counter(),
                "by_label": Counter(),
            }
            timeseries_shape_obs = None
            part_paths: list[str] = []
            try:
                mp_context = get_context("fork")
                stop_event = mp_context.Event()
                with ProcessPoolExecutor(
                    max_workers=min(sample_workers, len(chunks)),
                    mp_context=mp_context,
                    initializer=_init_parallel_fill_worker,
                    initargs=(
                        timeseries,
                        params,
                        template_specs,
                        t_order,
                        d_order,
                        output_info_list,
                        precision,
                        simulator,
                        scenario_name,
                        compression,
                        batch_size,
                        stop_event,
                    ),
                ) as executor:
                    pending = {executor.submit(_parallel_fill_part_worker, chunk) for chunk in chunks}
                    while pending:
                        if should_stop is not None and should_stop():
                            stop_event.set()
                            for future in pending:
                                future.cancel()
                            raise InterruptedError("任务已终止")
                        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        if not done:
                            continue
                        for future in done:
                            if stop_event.is_set():
                                raise InterruptedError("任务已终止")
                            result_part = future.result()
                            rows = int(result_part.get("row_count") or 0)
                            written += rows
                            skipped += int(result_part.get("skipped") or 0)
                            nan_skipped += int(result_part.get("nan_skipped") or 0)
                            if result_part.get("timeseries_shape_obs") is not None:
                                timeseries_shape_obs = result_part["timeseries_shape_obs"]
                            if rows > 0:
                                part_paths.append(str(result_part["path"]))
                            for key, values in (result_part.get("counters") or {}).items():
                                if key in counters and isinstance(values, dict):
                                    counters[key].update({str(k): int(v) for k, v in values.items()})
                            if on_progress and written - last_reported >= progress_every:
                                on_progress(scenario_name, min(written, n))
                                last_reported = written
                    _raise_if_stopped()

                if written == 0:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return {"kind": "text2comp", "scenario": scenario_name, "simulator": simulator, "status": "empty", "rows": 0}

                actual = 0
                for part_path in sorted(Path(path) for path in part_paths):
                    actual += int(pq.ParquetFile(part_path).metadata.num_rows)
                if actual != written:
                    raise RuntimeError(f"row count mismatch for text2comp/{scenario_name}: generated={written}, parquet={actual}")

                if partition_dir.exists():
                    shutil.rmtree(partition_dir)
                partition_dir.parent.mkdir(parents=True, exist_ok=True)
                tmp_dir.replace(partition_dir)

                manifest = {
                    "version": portable.PARQUET_SCHEMA_VERSION,
                    "kind": "text2comp",
                    "storage": "parquet",
                    "generated_at": time.time(),
                    "source": {**source, "sample_workers": sample_workers, "part_chunk_size": chunk_size},
                    "simulator": simulator,
                    "scenario": scenario_name,
                    "row_count": written,
                    "by_language": dict(counters["by_language"]),
                    "by_style": dict(counters["by_style"]),
                    "by_time_mode": dict(counters["by_time_mode"]),
                    "by_label": dict(counters["by_label"]),
                    "timeseries_shape_obs": timeseries_shape_obs,
                    "schema": [field.name for field in schema],
                }
                (partition_dir / "_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                return {
                    "kind": "text2comp",
                    "scenario": scenario_name,
                    "simulator": simulator,
                    "status": "written",
                    "rows": written,
                    "path": str(partition_dir),
                }
            except Exception:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        result = None
        with ExitStack() as stack:
            jsonl_handle = None
            if jsonl_out_path is not None:
                jsonl_out_path.parent.mkdir(parents=True, exist_ok=True)
                jsonl_handle = stack.enter_context(jsonl_out_path.open("w", encoding="utf-8"))
            records = iter_samples(jsonl_handle=jsonl_handle)
            if parquet_dir is not None:
                if jsonl_handle is None and scenario_workers > 1:
                    result = write_parallel_partition(scenario_workers)
                else:
                    result = portable.write_records_partition(
                        "text2comp",
                        records,
                        simulator=simulator,
                        scenario=scenario_name,
                        source=source,
                        output_root=parquet_dir,
                        batch_size=batch_size,
                        compression=compression,
                        overwrite=True,
                    )
                if result.get("status") == "written":
                    local_outputs.append(Path(str(result["path"])))
            else:
                for _ in records:
                    pass
                if jsonl_out_path is not None:
                    local_outputs.append(jsonl_out_path)

        if written == 0:
            _record_failure(scenario_name, f"没有写出任何样本（跳过 {skipped} 条）")
            return

        if nan_skipped > 0:
            nan_ratio = nan_skipped / max(n, 1)
            if nan_ratio > 0.1:
                logger.warning(
                    f"  {scenario_name}: NaN/Inf 跳过率 {nan_ratio*100:.1f}%"
                    f"（{nan_skipped}/{n}），请检查仿真数据质量"
                )
            else:
                logger.info(f"  {scenario_name}: 跳过 {nan_skipped} 个 NaN/Inf 样本")

        if on_progress:
            on_progress(scenario_name, written)
        target_desc = result.get("path") if result else (str(jsonl_out_path) if jsonl_out_path else "")
        _log(f"  已写入 {written} 条，跳过 {skipped} 条 → {target_desc}")
        with stats_lock:
            written_outputs.extend(local_outputs)
            stats["total"] += written
            stats[simulator] += written

    if output_format == "parquet" and scenario_workers > 1:
        for item in h5_files:
            _process_scenario(item)
    elif scenario_workers <= 1 or len(h5_files) <= 1:
        for item in h5_files:
            _process_scenario(item)
    else:
        with ThreadPoolExecutor(max_workers=min(scenario_workers, len(h5_files))) as executor:
            futures = {executor.submit(_process_scenario, item): item for item in h5_files}
            for future in as_completed(futures):
                try:
                    future.result()
                except InterruptedError:
                    for pending in futures:
                        pending.cancel()
                    raise

    if failures:
        preview = "; ".join(failures[:5])
        suffix = f"；另有 {len(failures) - 5} 个失败" if len(failures) > 5 else ""
        raise RuntimeError(f"样本填充失败: {preview}{suffix}")

    total_merged = 0
    merged_path = None
    if jsonl_dir is not None:
        output_file = cfg.get("output_file", "all_training_data.jsonl")
        merged_path = jsonl_dir / output_file
        scenario_files = [p for p in sorted(jsonl_dir.glob("*.jsonl")) if p.name != output_file]
        _log(f"\n[合并] 准备合并 {len(scenario_files)} 个场景文件 -> {merged_path.name}")
        with merged_path.open("w", encoding="utf-8") as fout:
            for jl_path in scenario_files:
                with jl_path.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            fout.write(line + "\n")
                            total_merged += 1
        _log(f"[完成] 共合并 {total_merged} 条")

    _log("\n" + "=" * 60)
    _log("样本填充完成")
    _log("=" * 60)
    _log(f"写入总数:    {stats['total']}")
    if parquet_dir is not None:
        _log(f"Parquet目录: {parquet_dir}")
    if merged_path is not None:
        _log(f"合并文件:    {merged_path}  ({total_merged} 条)")
    _log("\n按 simulator 统计:")
    for sim, cnt in sorted(stats.items()):
        if sim != "total":
            _log(f"  {sim:20s}: {cnt}")
    _log("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 阶段二：批量填充数值生成训练样本（不调 LLM）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/text2comp/default.yaml")
    parser.add_argument("--n-samples", type=int, default=None, help="每场景样本数")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--templates-dir", default=None, help="模板目录（默认 data/templates/）")
    parser.add_argument("--output-dir", default=None, help="输出目录；Parquet 默认 data/text2comp_parquet/，JSONL 默认 data/text2comp/")
    parser.add_argument("--output-format", choices=["parquet", "jsonl", "both"], default="parquet")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--precision", type=int, default=4, help="数值小数位数（默认4位）")
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    parser.add_argument("--batch-size", type=int, default=8192, help="Parquet 写入批大小")
    parser.add_argument("--max-workers", type=int, default=None, help="场景级并行数（默认 1；线程并行不一定更快，建议实测后调高）")
    args = parser.parse_args()

    run_fill_samples(
        cfg_path=args.config,
        n_samples=args.n_samples,
        scenarios=args.scenarios,
        templates_dir=args.templates_dir,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        seed=args.seed,
        precision=args.precision,
        output_format=args.output_format,
        compression=args.compression,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
