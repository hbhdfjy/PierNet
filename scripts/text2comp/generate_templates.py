"""
阶段一批量脚本：生成语言模板库（templates.jsonl）。

只调用 LLM，不接触任何实际数值或时序数据。
输出的 templates.jsonl 可被 fill_samples.py 复用任意次。

用法：
  # 为所有场景生成模板（每场景 1000 条）
  python scripts/text2comp/generate_templates.py \
      --config configs/text2comp/default.yaml

  # 只为指定场景生成
  python scripts/text2comp/generate_templates.py \
      --config configs/text2comp/default.yaml \
      --scenarios unified_aquifer ieee14_baseload \
      --n-templates 200

  # 跳过已存在的模板文件
  python scripts/text2comp/generate_templates.py \
      --config configs/text2comp/default.yaml \
      --skip-existing
"""

import argparse
import logging
import os
import sys
import threading
from collections import Counter
from pathlib import Path

import yaml

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from piern.core.llm_client import LLMClient
from piern.core.storage import load_dataset
from piern.text2comp.generator import DOMAIN_REGISTRY, LLMTextGenerator
from piern.text2comp.template_store import save_templates
from piern.text2comp.pipeline import _scan_h5_files, _scenario_name_from_path, _load_registry, _resolve_domain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_generate_templates(
    cfg_path: str,
    n_templates: int = None,
    scenarios: list = None,
    skip_existing: bool = False,
    append_existing: bool = False,
    language_mix: float = None,
    transform_prob: float = None,
    max_workers: int = None,
    on_scenario_start=None,   # (scenario: str, total: int) -> None
    on_progress=None,         # (scenario: str, done: int) -> None
    on_log=None,              # (line: str) -> None
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

    llm_cfg = cfg.get("llm", {})
    gen_cfg = cfg.get("generation", {})
    data_dirs = cfg.get("data_dirs", {})
    registry_path = cfg.get("registry", "configs/text2comp/registry.yaml")

    # CLI 参数覆盖 generation.yaml 的值
    if language_mix is not None:
        gen_cfg["language_mix"] = language_mix
    if transform_prob is not None:
        gen_cfg["transform_prob"] = transform_prob
    if max_workers is not None:
        gen_cfg["max_workers"] = max_workers

    n_per_scenario = n_templates if n_templates is not None else gen_cfg.get("n_samples_per_scenario", 1000)
    seed = cfg.get("seed", 42)

    base_dir = cfg_path.parent.parent.parent
    if not (base_dir / "data").exists():
        base_dir = Path.cwd()

    # 模板输出目录
    templates_dir = base_dir / "data" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    # 加载 registry
    registry_path_full = base_dir / registry_path if not Path(registry_path).is_absolute() else Path(registry_path)
    registry = _load_registry(registry_path_full)

    # 初始化 LLM 客户端
    provider = llm_cfg.get("provider", "siliconflow")
    model = llm_cfg.get("model", "Qwen/Qwen2.5-72B-Instruct")
    api_key = llm_cfg.get("api_key") or os.getenv(
        {"siliconflow": "SILICONFLOW_API_KEY",
         "openai": "OPENAI_API_KEY",
         "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "SILICONFLOW_API_KEY")
    )
    llm_client = LLMClient(
        provider=provider, model=model, api_key=api_key,
        max_retries=llm_cfg.get("max_retries", 3),
        timeout=llm_cfg.get("timeout", 60),
    )

    max_workers = gen_cfg.get("max_workers", 1)
    generator = LLMTextGenerator(
        llm_client=llm_client,
        temperature=llm_cfg.get("temperature", 0.8),
        max_tokens=llm_cfg.get("max_tokens", 600),
        language_mix=gen_cfg.get("language_mix", 0.5),
        transform_prob=gen_cfg.get("transform_prob", 0.1),
        styles=gen_cfg.get("styles", ["technical", "popular", "concise"]),
        style_weights=gen_cfg.get("style_weights", [0.4, 0.3, 0.3]),
        max_workers=max_workers,
    )
    def _log(line: str):
        print(line)
        if on_log:
            on_log(line)

    _log(f"并发线程数: {max_workers}")

    # 扫描 HDF5 文件（只需要获取 param_names 和 timeseries_shape）
    h5_files = _scan_h5_files(data_dirs, base_dir)
    if not h5_files:
        raise RuntimeError("未找到任何 HDF5 文件，请检查 data_dirs 配置")

    if scenarios is not None:
        scenarios_set = set(scenarios)
        h5_files = [
            (p, s, sfx) for p, s, sfx in h5_files
            if _scenario_name_from_path(p, sfx) in scenarios_set
        ]

    _log(f"\n找到 {len(h5_files)} 个场景，每场景生成 {n_per_scenario} 条模板\n")

    stats = Counter()
    for h5_path, simulator, file_suffix in h5_files:
        scenario_name = _scenario_name_from_path(h5_path, file_suffix)
        out_path = templates_dir / f"{scenario_name}_templates.jsonl"

        if skip_existing and out_path.exists():
            _log(f"[跳过] {out_path.name} 已存在")
            continue

        # 追加模式：补齐到目标数量
        already_have = 0
        if append_existing and out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    already_have = sum(1 for l in f if l.strip())
            except Exception:
                already_have = 0
            if already_have >= n_per_scenario:
                _log(f"[跳过] {out_path.name} 已有 {already_have} 条，已达目标 {n_per_scenario} 条")
                continue
            need = n_per_scenario - already_have
            _log(f"\n[追加] {out_path.name}（已有 {already_have} 条，补充 {need} 条至 {n_per_scenario} 条）")
        else:
            need = n_per_scenario
            _log(f"\n[处理] {h5_path.name}")

        _log(f"  simulator={simulator}, scenario={scenario_name}")
        if on_scenario_start:
            on_scenario_start(scenario_name, n_per_scenario)

        # 解析 domain 元数据
        try:
            domain = _resolve_domain(simulator, scenario_name, registry)
        except ValueError as e:
            logger.error(str(e))
            continue

        # 从 HDF5 读取 param_names 和 timeseries_shape（不读取实际数值）
        try:
            ts_arr, _, param_names = load_dataset(str(h5_path))
            timeseries_shape = ts_arr.shape[1:]  # (ch, ts)，去掉 N 维
            _log(f"  timeseries_shape: {timeseries_shape}, params: {len(param_names)}")
        except Exception as e:
            logger.error(f"加载 {h5_path} 失败: {e}")
            continue

        # 生成模板（追加模式用不同 seed 偏移避免重复）
        file_lock = threading.Lock()
        effective_seed = seed + already_have  # 追加时偏移 seed，避免生成重复模板

        def _make_progress_cb(sc_name, offset):
            def _cb(done: int):
                if on_progress:
                    on_progress(sc_name, offset + done)
            return _cb

        write_mode = "a" if (append_existing and out_path.exists()) else "w"
        with open(out_path, write_mode, encoding="utf-8") as fout:
            templates = generator.make_template_batch(
                simulator=simulator,
                scenario_name=scenario_name,
                param_names=list(param_names),
                timeseries_shape=timeseries_shape,
                n_templates=need,
                domain=domain,
                seed=effective_seed,
                output_file=fout,
                file_lock=file_lock,
                progress_callback=_make_progress_cb(scenario_name, already_have),
            )

        # 截断到精确目标行数（并发写入可能超出）
        with open(out_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        actual = len(lines)
        if actual > n_per_scenario:
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(lines[:n_per_scenario])
            _log(f"  截断 {actual} → {n_per_scenario} 条")
            actual = n_per_scenario

        total_now = actual
        _log(f"  已保存 {len(templates)} 条模板 → {out_path.name}（共 {total_now} 条）")
        stats["total"] += len(templates)
        stats[simulator] += len(templates)

    _log("\n" + "=" * 60)
    _log(f"阶段一完成：共生成 {stats['total']} 条模板")
    _log(f"模板目录: {templates_dir}")
    for sim, cnt in sorted(stats.items()):
        if sim != "total":
            _log(f"  {sim:20s}: {cnt}")
    _log("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 阶段一：批量生成语言模板库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/text2comp/default.yaml")
    parser.add_argument("--n-templates", type=int, default=None, help="每场景模板数")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有模板文件的场景")
    parser.add_argument("--append-existing", action="store_true", help="补齐到目标数量（追加模式）")
    parser.add_argument("--language-mix", type=float, default=None, help="中文比例 0-1（覆盖 generation.yaml）")
    parser.add_argument("--transform-prob", type=float, default=None, help="参数变换概率 0-1（覆盖 generation.yaml）")
    parser.add_argument("--max-workers", type=int, default=None, help="并发线程数（覆盖 generation.yaml）")
    args = parser.parse_args()

    run_generate_templates(
        cfg_path=args.config,
        n_templates=args.n_templates,
        scenarios=args.scenarios,
        skip_existing=args.skip_existing,
        append_existing=args.append_existing,
        language_mix=args.language_mix,
        transform_prob=args.transform_prob,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
