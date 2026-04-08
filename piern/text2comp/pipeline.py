"""
Stage 2 工具函数：HDF5 文件扫描、场景名解析、registry 加载、domain 解析。

这些函数被 generate_templates.py、fill_samples.py、auto_register.py 共用。
"""

import logging
from pathlib import Path

import yaml

from piern.text2comp.generator import DOMAIN_REGISTRY

logger = logging.getLogger(__name__)


# ── 文件扫描（配置驱动，无任务知识）──────────────────────────────

def _scan_h5_files(data_dirs: dict, base_dir: Path) -> list:
    """
    扫描所有 HDF5 文件，返回 (h5_path, simulator_type) 列表。

    data_dirs 支持两种格式：
      旧格式（向后兼容）: {"modflow": "data/modflow"}
      新格式:            {"modflow": {"path": "data/modflow", "simulator": "modflow",
                                      "file_suffix": "_groundwater_timeseries",
                                      "transient_simulator": "power_transient",
                                      "transient_keywords": ["fault", "trip"]}}
    """
    found = []
    for dir_key, dir_cfg in data_dirs.items():
        # 兼容旧格式（纯字符串）
        if isinstance(dir_cfg, str):
            dir_path = base_dir / dir_cfg
            simulator = dir_key
            file_suffix = None
            transient_simulator = None
            transient_keywords = []
            include_keywords = []
        else:
            dir_path = base_dir / dir_cfg["path"]
            simulator = dir_cfg.get("simulator", dir_key)
            file_suffix = dir_cfg.get("file_suffix", None)
            transient_simulator = dir_cfg.get("transient_simulator", None)
            transient_keywords = dir_cfg.get("transient_keywords", [])
            include_keywords = dir_cfg.get("include_keywords", [])  # 白名单：只保留含任一关键词的文件

        if not dir_path.exists():
            logger.warning(f"数据目录不存在，跳过: {dir_path}")
            continue

        for h5_file in sorted(dir_path.glob("*.h5")):
            stem = h5_file.stem.lower()

            # 白名单过滤：若配置了 include_keywords，只保留匹配的文件
            if include_keywords and not any(kw.lower() in stem for kw in include_keywords):
                continue

            # 若配置了 transient_keywords，按文件名关键词区分子类型（向后兼容）
            sim_type = simulator
            if transient_simulator and transient_keywords:
                if any(kw in stem for kw in transient_keywords):
                    sim_type = transient_simulator

            found.append((h5_file, sim_type, file_suffix))
            logger.info(f"发现文件: {h5_file.name} → simulator={sim_type}")

    return found


def _scenario_name_from_path(h5_path: Path, file_suffix: str = None) -> str:
    """
    从文件名提取场景名。

    若提供了 file_suffix，去掉该后缀；否则返回原始 stem。
    """
    stem = h5_path.stem
    if file_suffix and stem.endswith(file_suffix):
        stem = stem[: -len(file_suffix)]
    return stem


# ── registry 加载与 domain 解析 ──────────────────────────────────

def _load_registry(registry_path: Path) -> dict:
    """加载 registry.yaml，返回 dict，过滤 None key。"""
    if not registry_path.exists():
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: v for k, v in data.items() if k and v}


def _resolve_domain(
    simulator: str,
    scenario_name: str,
    registry: dict,
) -> dict:
    """
    按优先级解析 domain 元数据（新两层结构）：

    registry 新结构：
      registry[simulator]                          → simulator 级字段（5个通用字段）
      registry[simulator]["scenarios"][scenario]   → 场景描述字符串

    优先级：
      1. registry[simulator]（simulator 级字段）
      2. 代码内置 DOMAIN_REGISTRY（兜底默认值）

    场景描述来源（scenarios dict）：
      1. registry[simulator]["scenarios"][scenario]（新结构）
      2. 内置 DOMAIN_REGISTRY[simulator]["scenarios"][scenario]
      3. 兜底：场景名本身
    """
    sim_entry = registry.get(simulator, {})
    builtin = DOMAIN_REGISTRY.get(simulator, {})

    # 若 simulator 既无 registry 条目也无内置，报错
    if not sim_entry and not builtin:
        raise ValueError(
            f"未找到 simulator '{simulator}' 的元数据。"
            f"请先运行 auto_register.py 生成 registry.yaml。"
        )

    def _get(key, default=None):
        """按优先级取字段：simulator 级 > 内置。"""
        if key in sim_entry:
            return sim_entry[key]
        return builtin.get(key, default)

    # param_info：registry 中是 list，转为 tuple 与内置格式一致
    param_info_raw = _get("param_info", {})
    param_info = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in param_info_raw.items()
    }

    # scenarios dict：从 registry[simulator]["scenarios"] 或内置取
    reg_scenarios = sim_entry.get("scenarios", {})
    builtin_scenarios = builtin.get("scenarios", {})

    # 取当前场景的描述
    scene_desc = reg_scenarios.get(scenario_name) or builtin_scenarios.get(scenario_name) or scenario_name
    # 合并：内置场景描述为基础，registry 覆盖
    scenarios = {**builtin_scenarios, **reg_scenarios}
    if scenario_name not in scenarios:
        scenarios[scenario_name] = scene_desc

    # output_info fallback 链
    default_output_info = [
        {"name": "output", "description": "simulation output", "unit": "-", "slice": [0, None]}
    ]

    return {
        "domain_context":    _get("domain_context", ""),
        "output_description": _get("output_description", "{ch} channels × {ts} timesteps"),
        "scenarios":         scenarios,
        "param_info":        param_info,
        "output_info":       _get("output_info", default_output_info),
        "observation_config": _get("observation_config", {}),
    }

