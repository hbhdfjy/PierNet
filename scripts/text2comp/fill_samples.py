"""
阶段二批量脚本：将模板库与 HDF5 数值结合，生成最终训练样本。

默认直接写入便携式 Parquet 分区 data/text2comp_parquet/，不再先落 JSONL。
如需兼容旧流程，可显式传 --output-format jsonl 或 --output-format both。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from piern.core.storage import load_dataset
from piern.shared.storage import portable
from piern.synth.text2comp.pipeline import _scan_h5_files, _scenario_name_from_path, load_config
from piern.synth.text2comp.template_store import fill_sample, load_templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _has_nonempty_jsonl(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open("r", encoding="utf-8") as handle:
            return any(line.strip() for line in handle)
    except OSError:
        return False


def _has_nonempty_partition(root: Path, simulator: str, scenario: str) -> bool:
    part_dir = portable.partition_dir_for("text2comp", simulator, scenario, root)
    manifest_path = part_dir / "_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            return int(payload.get("row_count") or 0) > 0
        except Exception:
            pass
    return any(part_dir.glob("*.parquet"))


def _source_signature(path: Path) -> dict[str, object]:
    try:
        return portable.source_signature(path)
    except FileNotFoundError:
        return {"path": str(path), "missing": True}


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
) -> None:
    output_format = (output_format or "parquet").lower()
    if output_format not in {"parquet", "jsonl", "both"}:
        raise ValueError("output_format must be one of: parquet, jsonl, both")

    cfg_path = Path(cfg_path)
    cfg = load_config(cfg_path)
    gen_cfg = cfg.get("generation", {})

    n_per_scenario = n_samples if n_samples is not None else gen_cfg.get("n_samples_per_scenario", 1000)
    _seed = seed if seed is not None else cfg.get("seed", 42)
    scenario_workers = max(1, int(max_workers if max_workers is not None else 1))

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

    tmpl_dir = Path(templates_dir) if templates_dir else base_dir / "data" / "templates"
    parquet_dir = None
    jsonl_dir = None
    if output_format in {"parquet", "both"}:
        default_parquet = cfg.get("parquet_output_dir", "data/text2comp_parquet")
        parquet_dir = Path(output_dir) if output_dir else base_dir / default_parquet
        parquet_dir.mkdir(parents=True, exist_ok=True)
    if output_format in {"jsonl", "both"}:
        jsonl_dir = Path(output_dir) if output_dir else base_dir / cfg.get("output_dir", "data/text2comp")
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

    log_lock = threading.Lock()

    def _log(line: str) -> None:
        with log_lock:
            print(line)
            if on_log:
                on_log(line)

    _log(f"\n找到 {len(h5_files)} 个场景，每场景生成 {n_per_scenario} 条样本")
    _log(f"输出格式: {output_format}")
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
        h5_path, simulator, file_suffix = item
        scenario_name = _scenario_name_from_path(h5_path, file_suffix)
        tmpl_path = tmpl_dir / f"{scenario_name}_templates.jsonl"
        jsonl_out_path = jsonl_dir / f"{scenario_name}.jsonl" if jsonl_dir is not None else None
        local_outputs: list[Path] = []

        if skip_existing:
            if parquet_dir is not None and _has_nonempty_partition(parquet_dir, simulator, scenario_name):
                _log(f"[跳过] {scenario_name} Parquet 分区已存在")
                return
            if jsonl_out_path is not None and _has_nonempty_jsonl(jsonl_out_path):
                _log(f"[跳过] {jsonl_out_path.name} 已存在")
                local_outputs.append(jsonl_out_path)
                with stats_lock:
                    written_outputs.extend(local_outputs)
                return

        if not tmpl_path.exists():
            logger.warning(f"模板文件不存在，跳过: {tmpl_path}")
            logger.warning(f"  → 请先运行 generate_templates.py 为 {scenario_name} 生成模板")
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
            logger.warning(f"模板文件为空，跳过: {tmpl_path}")
            return

        try:
            timeseries, params, param_names = load_dataset(str(h5_path))
        except Exception as exc:
            logger.error(f"加载 {h5_path} 失败: {exc}")
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
            time_idx = np.asarray(template.time_indices, dtype=np.int64)
            valid_time_idx = time_idx[time_idx < ts_len]
            if template.channel_indices is None:
                valid_ch = None
            else:
                ch_arr = np.asarray(template.channel_indices, dtype=np.int64)
                valid_ch = ch_arr[ch_arr < n_channels]
            template_specs.append((template, valid_time_idx, valid_ch))

        rng = np.random.default_rng(_seed)
        t_order = rng.permutation(n_avail_t)
        d_order = rng.permutation(n_avail_d)
        written = 0
        skipped = 0
        nan_skipped = 0
        last_reported = -1

        def iter_samples(jsonl_handle=None) -> Iterable[dict[str, object]]:
            nonlocal written, skipped, nan_skipped, last_reported
            for i in range(n):
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
                if on_progress and written % 50 == 0 and written != last_reported:
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
        result = None
        with ExitStack() as stack:
            jsonl_handle = None
            if jsonl_out_path is not None:
                jsonl_out_path.parent.mkdir(parents=True, exist_ok=True)
                jsonl_handle = stack.enter_context(jsonl_out_path.open("w", encoding="utf-8"))
            records = iter_samples(jsonl_handle=jsonl_handle)
            if parquet_dir is not None:
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

    if scenario_workers <= 1 or len(h5_files) <= 1:
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
