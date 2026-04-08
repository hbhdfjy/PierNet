"""
阶段二批量脚本：将模板库与 HDF5 数值结合，生成最终训练样本（JSONL）。

纯本地计算，不调用任何 LLM。
可对同一套模板反复填充不同的 HDF5 数据集。

用法：
  # 为所有场景填充（模板目录 data/templates/，HDF5 目录来自 data_dirs 配置）
  python scripts/text2comp/fill_samples.py \
      --config configs/text2comp/default.yaml

  # 只填充指定场景
  python scripts/text2comp/fill_samples.py \
      --config configs/text2comp/default.yaml \
      --scenarios unified_aquifer ieee14_baseload \
      --n-samples 500

  # 指定模板目录
  python scripts/text2comp/fill_samples.py \
      --config configs/text2comp/default.yaml \
      --templates-dir data/templates

  # 跳过已存在的输出文件
  python scripts/text2comp/fill_samples.py \
      --config configs/text2comp/default.yaml \
      --skip-existing
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from piern.core.storage import load_dataset
from piern.text2comp.template_store import TemplateRecord, fill_sample, load_templates
from piern.text2comp.pipeline import _scan_h5_files, _scenario_name_from_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_fill_samples(
    cfg_path: str,
    n_samples: int = None,
    scenarios: list = None,
    templates_dir: str = None,
    output_dir: str = None,
    skip_existing: bool = False,
    seed: int = None,
    on_scenario_start=None,  # (scenario: str, total: int) -> None
    on_progress=None,        # (scenario: str, done: int) -> None
    on_log=None,             # (line: str) -> None
) -> None:
    cfg_path = Path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 合并 generation_config
    gen_cfg_path = cfg.get("generation_config")
    if gen_cfg_path:
        gen_file = cfg_path.parent.parent.parent / gen_cfg_path
        if not gen_file.exists():
            gen_file = Path.cwd() / gen_cfg_path
        if gen_file.exists():
            with open(gen_file, "r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f) or {}
            for k, v in base_cfg.items():
                if k not in cfg:
                    cfg[k] = v
                elif isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k] = {**v, **cfg[k]}
        else:
            logger.warning(f"generation_config 文件未找到：{gen_cfg_path}")

    gen_cfg = cfg.get("generation", {})
    data_dirs = cfg.get("data_dirs", {})

    n_per_scenario = n_samples if n_samples is not None else gen_cfg.get("n_samples_per_scenario", 1000)
    _seed = seed if seed is not None else cfg.get("seed", 42)

    base_dir = cfg_path.parent.parent.parent
    if not (base_dir / "data").exists():
        base_dir = Path.cwd()

    # 加载 registry，用于从中提取完整 output_info（含 name_zh、unit）
    registry: dict = {}
    registry_path_str = cfg.get("registry", "configs/text2comp/registry.yaml")
    registry_path = base_dir / registry_path_str if not Path(registry_path_str).is_absolute() else Path(registry_path_str)
    if registry_path.exists():
        try:
            registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass

    tmpl_dir = Path(templates_dir) if templates_dir else base_dir / "data" / "templates"
    out_dir = Path(output_dir) if output_dir else base_dir / cfg.get("output_dir", "data/text2comp")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 扫描 HDF5 文件
    h5_files = _scan_h5_files(data_dirs, base_dir)
    if not h5_files:
        raise RuntimeError("未找到任何 HDF5 文件，请检查 data_dirs 配置")

    if scenarios is not None:
        scenarios_set = set(scenarios)
        h5_files = [
            (p, s, sfx) for p, s, sfx in h5_files
            if _scenario_name_from_path(p, sfx) in scenarios_set
        ]

    def _log(line: str):
        print(line)
        if on_log:
            on_log(line)

    _log(f"\n找到 {len(h5_files)} 个场景，每场景生成 {n_per_scenario} 条样本\n")

    stats = Counter()
    all_jsonl: list[Path] = []

    for h5_path, simulator, file_suffix in h5_files:
        scenario_name = _scenario_name_from_path(h5_path, file_suffix)
        tmpl_path = tmpl_dir / f"{scenario_name}_templates.jsonl"
        out_path = out_dir / f"{scenario_name}.jsonl"

        if skip_existing and out_path.exists():
            _log(f"[跳过] {out_path.name} 已存在")
            all_jsonl.append(out_path)
            continue

        if not tmpl_path.exists():
            logger.warning(f"模板文件不存在，跳过: {tmpl_path}")
            logger.warning(f"  → 请先运行 generate_templates.py 为 {scenario_name} 生成模板")
            continue

        _log(f"\n[处理] {scenario_name}")
        _log(f"  模板: {tmpl_path.name}")
        _log(f"  数据: {h5_path.name}")
        if on_scenario_start:
            on_scenario_start(scenario_name, n_per_scenario)

        # 从 registry 提取完整 output_info（含 name_zh、unit）
        output_info_list = None
        sim_entry = registry.get(simulator, {})
        if isinstance(sim_entry, dict):
            output_info_list = sim_entry.get("output_info")
            # 场景级覆盖
            sc_entry = sim_entry.get("scenarios", {}).get(scenario_name, {})
            if isinstance(sc_entry, dict) and "output_info" in sc_entry:
                output_info_list = sc_entry["output_info"]

        # 加载模板
        templates = load_templates(tmpl_path)
        if not templates:
            logger.warning(f"模板文件为空，跳过: {tmpl_path}")
            continue

        # 加载 HDF5 数值
        try:
            timeseries, params, param_names = load_dataset(str(h5_path))
        except Exception as e:
            logger.error(f"加载 {h5_path} 失败: {e}")
            continue

        n_avail_t = len(templates)
        n_avail_d = len(params)
        n = min(n_per_scenario, n_avail_t * n_avail_d)  # 不超过可组合数量上限
        n = max(n, min(n_per_scenario, n_avail_t))       # 至少尽量达到目标数

        _log(f"  模板数: {n_avail_t}，HDF5样本数: {n_avail_d}，目标: {n_per_scenario}，实际: {n}")

        rng = np.random.default_rng(_seed)
        written = 0
        skipped = 0
        nan_skipped = 0
        last_reported = -1   # 去重：避免进度回调在同一值上触发两次

        with open(out_path, "w", encoding="utf-8") as fout:
            for i in range(n):
                t_idx = i % n_avail_t
                d_idx = i % n_avail_d
                template = templates[t_idx]

                ts = timeseries[d_idx]   # (ch_orig, ts_orig)
                p = params[d_idx]        # (n_params,)

                # 按模板记录的 time_indices 和 channel_indices 切片
                time_idx = np.array(template.time_indices)
                # 防止索引越界（模板生成时的时序长度可能与当前 HDF5 不同）
                ts_len = ts.shape[1]
                valid_time_idx = time_idx[time_idx < ts_len]
                if len(valid_time_idx) == 0:
                    skipped += 1
                    continue

                ts_time = ts[:, valid_time_idx]

                ch_idx = template.channel_indices
                if ch_idx is not None:
                    ch_arr = np.array(ch_idx)
                    # 防止通道索引越界：用 ts_time 的通道数（切片后），而非原始 ts
                    valid_ch = ch_arr[ch_arr < ts_time.shape[0]]
                    if len(valid_ch) == 0:
                        skipped += 1
                        continue
                    ts_obs = ts_time[valid_ch, :]
                else:
                    ts_obs = ts_time

                # NaN/Inf 检查
                if not np.isfinite(ts_obs).all():
                    nan_skipped += 1
                    skipped += 1
                    continue

                try:
                    sample = fill_sample(template, p, ts_obs, sample_idx=d_idx, output_info=output_info_list)
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
                    # 每 50 条上报一次进度，避免事件洪泛
                    if on_progress and written % 50 == 0 and written != last_reported:
                        on_progress(scenario_name, written)
                        last_reported = written
                except Exception as e:
                    logger.warning(f"样本 {i} 填充失败: {e}")
                    skipped += 1

        # NaN 跳过率超过 10% 时发出警告
        if nan_skipped > 0:
            nan_ratio = nan_skipped / max(n, 1)
            if nan_ratio > 0.1:
                logger.warning(
                    f"  {scenario_name}: NaN/Inf 跳过率 {nan_ratio*100:.1f}%"
                    f"（{nan_skipped}/{n}），请检查仿真数据质量"
                )
            else:
                logger.info(f"  {scenario_name}: 跳过 {nan_skipped} 个 NaN/Inf 样本")

        # 最终进度（去重：避免与循环内最后一次回调重复）
        if on_progress and written > 0 and written != last_reported:
            on_progress(scenario_name, written)
        _log(f"  已写入 {written} 条，跳过 {skipped} 条 → {out_path.name}")
        stats["total"] += written
        stats[simulator] += written
        all_jsonl.append(out_path)

    # 合并所有 JSONL
    output_file = cfg.get("output_file", "all_training_data.jsonl")
    merged_path = out_dir / output_file
    total_merged = 0
    with open(merged_path, "w", encoding="utf-8") as fout:
        for jl_path in all_jsonl:
            if not jl_path.exists():
                continue
            with open(jl_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
                        total_merged += 1

    _log("\n" + "=" * 60)
    _log("阶段二完成")
    _log("=" * 60)
    _log(f"总样本数:    {stats['total']}")
    _log(f"合并文件:    {merged_path}  ({total_merged} 条)")
    _log(f"\n按 simulator:")
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
    parser.add_argument("--output-dir", default=None, help="输出目录（默认 data/text2comp/）")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_fill_samples(
        cfg_path=args.config,
        n_samples=args.n_samples,
        scenarios=args.scenarios,
        templates_dir=args.templates_dir,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
