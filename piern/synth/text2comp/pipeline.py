"""
Stage 2 工具函数：HDF5 文件扫描、场景名解析、registry 加载、domain 解析、配置加载。

这些函数被 generate_templates.py、fill_samples.py、auto_register.py 共用。
"""

import logging
from pathlib import Path

import yaml

from piern.shared.runtime.paths import DATA_ROOT
from piern.shared.storage.hdf5_files import iter_hdf5_files
from piern.synth.text2comp.generator import DOMAIN_REGISTRY

logger = logging.getLogger(__name__)


def _resolve_data_path(value: str | None, base_dir: Path, default: str = "data") -> Path:
    raw = (value or default).strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        return DATA_ROOT.joinpath(*parts[1:])
    return base_dir / path


# ── 配置加载 ──────────────────────────────────────────────────────

def load_config(cfg_path: Path) -> dict:
    """
    加载 default.yaml。

    generation_config 字段（旧格式，指向 generation.yaml）如果存在则自动合并，
    保证向后兼容。新格式直接把 llm/generation/seed 写在 default.yaml 里即可。
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

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

    return cfg


# ── 文件扫描 ──────────────────────────────────────────────────────

def _scan_h5_files(cfg: dict, base_dir: Path) -> list:
    """
    扫描所有 HDF5 文件，返回 [(h5_path, simulator, None)] 列表。

    约定（新格式，data_root）：
      data_root/
        {simulator}/
          {simulator}_{scenario}.h5/.hdf5

    目录名即 simulator 名，文件名去掉 "{simulator}_" 前缀得到场景名。
    跳过非 simulator 目录（templates、text2comp、router 等）。
    """
    # 新格式：data_root
    data_root_str = cfg.get("data_root")
    if data_root_str:
        data_root = _resolve_data_path(data_root_str, base_dir)
        skip = {"templates", "text2comp", "router"}
        found = []
        if data_root.exists():
            for sim_dir in sorted(data_root.iterdir()):
                if not sim_dir.is_dir() or sim_dir.name in skip:
                    continue
                simulator = sim_dir.name
                for h5_file in iter_hdf5_files(sim_dir):
                    found.append((h5_file, simulator, None))
                    logger.info(f"发现文件: {h5_file.name} → simulator={simulator}")
        return found

    # 旧格式兼容：data_dirs dict（已废弃，仅保留向后兼容）
    data_dirs = cfg.get("data_dirs", {})
    found = []
    for dir_key, dir_cfg in data_dirs.items():
        if isinstance(dir_cfg, str):
            dir_path = base_dir / dir_cfg
            simulator = dir_key
            file_suffix = None
        else:
            dir_path = base_dir / dir_cfg["path"]
            simulator = dir_cfg.get("simulator", dir_key)
            file_suffix = dir_cfg.get("file_suffix", None)

        if not dir_path.exists():
            logger.warning(f"数据目录不存在，跳过: {dir_path}")
            continue

        for h5_file in iter_hdf5_files(dir_path):
            found.append((h5_file, simulator, file_suffix))
            logger.info(f"发现文件: {h5_file.name} → simulator={simulator}")

    return found


def duplicate_stage_scenarios(h5_files: list[tuple[Path, str, str | None]]) -> list[str]:
    """Return HDF5 scenario names that would collide in Stage 2/3 outputs."""
    by_scenario: dict[str, list[tuple[str, Path]]] = {}
    for h5_path, simulator, file_suffix in h5_files:
        scenario = _scenario_name_from_path(h5_path, file_suffix)
        by_scenario.setdefault(scenario, []).append((str(simulator), h5_path))

    duplicates: list[str] = []
    for scenario, items in sorted(by_scenario.items()):
        if len(items) <= 1:
            continue
        simulators = ", ".join(sorted({simulator for simulator, _path in items}))
        duplicates.append(f"{scenario} ({simulators})")
    return duplicates


def assert_unique_stage_scenarios(h5_files: list[tuple[Path, str, str | None]]) -> None:
    duplicates = duplicate_stage_scenarios(h5_files)
    if duplicates:
        raise ValueError(
            "同名场景会写入相同的 Stage 2/3 输出文件，无法安全区分：" + "; ".join(duplicates)
        )


def _scenario_name_from_path(h5_path: Path, file_suffix: str = None) -> str:
    """
    从文件名提取场景名。

    新约定：文件名格式为 {simulator}_{scenario}.h5 或 .hdf5，目录名即 simulator。
    去掉 "{simulator}_" 前缀得到场景名。

    旧格式兼容：若提供了 file_suffix，去掉该后缀。
    """
    stem = h5_path.stem
    # 旧格式兼容
    if file_suffix and stem.endswith(file_suffix):
        return stem[: -len(file_suffix)]
    # 新约定：目录名即 simulator，去掉 "{simulator}_" 前缀
    simulator = h5_path.parent.name
    prefix = simulator + "_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
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
