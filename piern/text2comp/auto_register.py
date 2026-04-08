"""
自动注册器：用 LLM 读取 HDF5 基本信息，生成 registry.yaml。

支持分步注册：每次只注册指定字段组，已有字段不覆盖。

字段组（--fields）：
  domain       → domain_context, output_description, param_info
  output_info  → output_info（含 name_zh）
  observation  → observation_config

用法：
  # 注册所有字段（默认，场景级别）
  python -m piern.text2comp.auto_register

  # 只注册指定场景
  python -m piern.text2comp.auto_register --scenarios unified_aquifer ieee14_baseload

  # 只注册某字段组
  python -m piern.text2comp.auto_register --fields domain
  python -m piern.text2comp.auto_register --fields output_info
  python -m piern.text2comp.auto_register --fields observation

  # simulator 级别注册通用字段（output_info/observation_config 各场景相同时推荐）
  python -m piern.text2comp.auto_register --fields output_info observation --simulator-level

  # 再按场景注册特有字段（domain_context/param_info 各场景不同）
  python -m piern.text2comp.auto_register --fields domain

  # 组合：指定场景 + 指定字段组
  python -m piern.text2comp.auto_register --scenarios unified_aquifer --fields observation

  # 强制覆盖已有字段
  python -m piern.text2comp.auto_register --overwrite
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import yaml

from piern.core.llm_client import LLMClient
from piern.core.storage import load_dataset
from piern.text2comp.pipeline import _scenario_name_from_path, _scan_h5_files

logger = logging.getLogger(__name__)

# ── 字段组定义 ────────────────────────────────────────────────

# 每个字段组对应 registry 中的 key 列表
FIELD_GROUPS = {
    "domain":       ["domain_context", "output_description", "param_info"],
    "output_info":  ["output_info"],
    "observation":  ["observation_config"],
}
ALL_FIELDS = list(FIELD_GROUPS.keys())

# ── 系统提示词 ────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a scientific computing expert specializing in physics simulation datasets. "
    "Given basic information about a simulation dataset (parameter names, output shape, file path), "
    "infer the requested metadata fields. "
    "Always respond with valid JSON only, no extra text."
)

# ── prompt 构建（每个字段组独立） ──────────────────────────────

def _dataset_header(
    simulator: str,
    scenario_name: str,
    param_names: list,
    timeseries_shape: tuple,
    params_sample: np.ndarray,
) -> str:
    n_samples, channels, timesteps = timeseries_shape
    sample_rows = params_sample[:3].tolist()
    return f"""A physics simulation dataset has the following properties:

- Inferred simulator type: {simulator}
- Scenario name: {scenario_name}
- Number of samples: {n_samples}
- Output shape: ({channels} channels, {timesteps} timesteps)
- Parameter names: {param_names}
- Example parameter values (first 3 samples):
{json.dumps(sample_rows, indent=2)}
"""


def _build_domain_prompt(
    simulator: str, scenario_name: str, param_names: list,
    timeseries_shape: tuple, params_sample: np.ndarray,
) -> str:
    header = _dataset_header(simulator, scenario_name, param_names, timeseries_shape, params_sample)
    return header + """
Infer the following metadata fields. Respond with a JSON object with exactly these keys:

{
  "domain_context": "One paragraph describing the physical domain, governing equations, numerical method, and what the output represents. Mention the mathematical structure (PDE type, algebraic, DAE, etc.) if inferrable.",
  "output_description": "A short phrase describing the output. MUST contain the literal strings {ch} and {ts} as placeholders, e.g. '{ch} observation wells × {ts} days of hydraulic head (meters above datum)'.",
  "param_info": {
    "<param_name>": ["<physical meaning in English>", "<unit>"],
    ...
  }
}

Rules:
- Include ALL parameter names in param_info, even if uncertain. Use best guess based on name and value range.
- output_description MUST contain the literal strings {ch} and {ts}.
- Respond with JSON only, no markdown fences, no extra text.
"""


def _build_output_info_prompt(
    simulator: str, scenario_name: str, param_names: list,
    timeseries_shape: tuple, params_sample: np.ndarray,
) -> str:
    n_samples, channels, timesteps = timeseries_shape
    header = _dataset_header(simulator, scenario_name, param_names, timeseries_shape, params_sample)
    return header + f"""
Infer the output_info field that describes the channel structure of the output tensor.
This tensor has {channels} channels. You must first decide what each channel represents.

CRITICAL DECISION — choose exactly one of these two cases:

Case A — All channels are the SAME physical quantity measured at DIFFERENT spatial locations
  (e.g. hydraulic head at 5 observation wells, rotor angles of 10 generators)
  → Use ONE output_info entry with slice [0, null]
  → The channel dimension indexes spatial entities, NOT different physical quantities

Case B — Channels are DIFFERENT physical quantities vertically concatenated
  (e.g. bus voltages [0:14] + voltage angles [14:28] + line power flows [28:43])
  → Use ONE entry per physical quantity with the correct row range

How to decide:
- If channels = number of spatial entities (wells, generators, buses) and they all measure
  the same thing → Case A
- If the channel count matches a known concatenation pattern (e.g. n_bus*2 + n_lines for
  power flow) → Case B
- For groundwater (MODFLOW): channels are observation wells → Case A
- For power flow (pandapower): channels are concatenated voltages+angles+flows → Case B
- For transient stability (ANDES): channels are generators → Case A
- For energy-climate (PyPSA/GCAM): channels are different output variables → Case B
- When in doubt with {channels} channels and no clear concatenation pattern → Case A

Respond with a JSON object with exactly this key:

{{
  "output_info": [
    {{
      "name": "<channel_name>",
      "name_zh": "<Chinese name>",
      "description": "<physical meaning in English>",
      "unit": "<unit>",
      "slice": [<start_row>, <end_row_or_null>]
    }},
    ...
  ]
}}

Rules:
- slice end value: use null to mean "to the end of the channel dimension".
- name should be a short snake_case identifier.
- Respond with JSON only, no markdown fences, no extra text.
"""


def _build_observation_prompt(
    simulator: str, scenario_name: str, param_names: list,
    timeseries_shape: tuple, params_sample: np.ndarray,
) -> str:
    header = _dataset_header(simulator, scenario_name, param_names, timeseries_shape, params_sample)
    _, channels, timesteps = timeseries_shape
    return header + f"""
Infer the observation_config field that defines the default downsampling strategy for this dataset.
Respond with a JSON object with exactly this key:

{{
  "observation_config": {{
    "fixed_time_mode": "<mode_name, e.g. monthly / weekly / full / every_10>",
    "fixed_channels": <list_of_int_or_null>,
    "time_modes": [
      {{
        "name": "<same as fixed_time_mode>",
        "indices": "<same as fixed_time_mode>",
        "desc_en": "<English description, e.g. 'monthly, 12 time points'>",
        "desc_zh": "<Chinese description>"
      }}
    ],
    "channel_name_template": "<optional, e.g. 'well {{i}}'>",
    "channel_name_template_zh": "<optional Chinese template, e.g. '第{{i}}号观测井'>"
  }}
}}

Rules:
- fixed_channels: list of 0-based integer indices (e.g. [0,1,2,3,4]), or null to select all channels.
- channel_name_template / channel_name_template_zh: optional, include when channels have meaningful names.
- Only ONE time mode (the fixed one). No weights needed.
- Valid indices: "monthly" (12 pts), "weekly" (52 pts), "full" (all pts), "every_N" (every N steps).
  Choose based on timesteps:
  * 365 timesteps (daily, 1 year): monthly recommended
  * 1000 timesteps (100Hz, 10s): every_100 (1Hz, 10 pts) recommended
  * ≤20 timesteps: full
  * Other: full as fallback
- Respond with JSON only, no markdown fences, no extra text.
"""


# ── 每个字段组的 prompt 和校验 ────────────────────────────────

_FIELD_GROUP_CONFIG = {
    "domain": {
        "prompt_fn": _build_domain_prompt,
        "required_keys": ["domain_context", "output_description", "param_info"],
        "validate": lambda m: _validate_domain(m),
    },
    "output_info": {
        "prompt_fn": _build_output_info_prompt,
        "required_keys": ["output_info"],
        "validate": lambda m: _validate_output_info(m["output_info"]),
    },
    "observation": {
        "prompt_fn": _build_observation_prompt,
        "required_keys": ["observation_config"],
        "validate": lambda m: _validate_observation_config(m["observation_config"]),
    },
}


def _validate_domain(m):
    desc = m.get("output_description", "")
    if "{ch}" not in desc or "{ts}" not in desc:
        raise ValueError("output_description 缺少 {ch} 或 {ts} 占位符")


def _validate_output_info(output_info):
    if not isinstance(output_info, list) or len(output_info) == 0:
        raise ValueError("output_info 必须是非空列表")
    for entry in output_info:
        for k in ("name", "description", "unit", "slice"):
            if k not in entry:
                raise ValueError(f"output_info 条目缺少字段: {k}")


def _validate_observation_config(obs_cfg):
    for k in ("fixed_time_mode", "fixed_channels", "time_modes"):
        if k not in obs_cfg:
            raise ValueError(f"observation_config 缺少字段: {k}")
    fc = obs_cfg.get("fixed_channels")
    if fc is not None:
        if not isinstance(fc, list) or (fc and not all(isinstance(i, int) for i in fc)):
            raise ValueError("fixed_channels 必须是整数列表（0-based）或 null")
    for tm in obs_cfg["time_modes"]:
        idx = tm.get("indices", "")
        valid = {"monthly", "weekly", "full"}
        if idx not in valid and not (idx.startswith("every_") and idx[6:].isdigit()):
            raise ValueError(
                f"time_mode indices '{idx}' 不合法，支持：monthly / weekly / full / every_N"
            )


# ── LLM 调用 ──────────────────────────────────────────────────

def _call_llm_for_fields(
    llm: LLMClient,
    field_group: str,
    simulator: str,
    scenario_name: str,
    param_names: list,
    timeseries_shape: tuple,
    params_sample: np.ndarray,
    max_retries: int = 3,
) -> dict:
    """调用 LLM 推断单个字段组，返回解析后的 dict（只含该组字段）。"""
    cfg = _FIELD_GROUP_CONFIG[field_group]
    prompt = cfg["prompt_fn"](simulator, scenario_name, param_names, timeseries_shape, params_sample)

    for attempt in range(max_retries):
        try:
            response = llm.generate(
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=2000,
            )
            text = response.strip()
            text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
            metadata = json.loads(text.strip())

            for key in cfg["required_keys"]:
                if key not in metadata:
                    raise ValueError(f"LLM 返回缺少字段: {key}")

            cfg["validate"](metadata)
            return metadata

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"字段组 '{field_group}' 第 {attempt+1}/{max_retries} 次解析失败: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"LLM 推断失败（{simulator}/{scenario_name}, fields={field_group}）: {e}"
                )

    raise RuntimeError("unreachable")


# ── 主函数 ────────────────────────────────────────────────────

def run_auto_register(
    cfg_path: str,
    output_path: str,
    scenarios: list = None,
    fields: list = None,
    overwrite: bool = False,
    simulator_level: bool = False,
) -> dict:
    """
    扫描 HDF5 文件，用 LLM 推断元数据，分步写入 registry.yaml。

    Args:
        cfg_path:         default.yaml 路径
        output_path:      registry.yaml 输出路径
        scenarios:        只处理指定场景名列表（None=全部）
        fields:           只注册指定字段组列表（None=全部，可选: domain/output_info/observation）
        overwrite:        True=覆盖已有字段；False=跳过已有字段（默认）
        simulator_level:  True=注册到 simulator 级别 key（如 "modflow"），
                          适合各场景通用的字段（如 output_info/observation_config）；
                          False=注册到 simulator/scenario 级别 key（默认）

    Returns:
        完整的 registry dict

    典型用法：
        # 先用 simulator 级别注册通用字段（output_info + observation）
        run_auto_register(..., fields=["output_info","observation"], simulator_level=True)
        # 再用场景级别注册各场景特有字段（domain）
        run_auto_register(..., fields=["domain"])
    """
    cfg_path = Path(cfg_path)
    output_path = Path(output_path)
    fields = fields or ALL_FIELDS

    # 校验 fields 参数
    invalid = [f for f in fields if f not in FIELD_GROUPS]
    if invalid:
        raise ValueError(f"无效的 fields: {invalid}，合法值: {ALL_FIELDS}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 合并 generation_config
    gen_cfg_path = cfg.get("generation_config")
    if gen_cfg_path:
        gen_cfg_file = Path(gen_cfg_path)
        if not gen_cfg_file.is_absolute():
            gen_cfg_file = cfg_path.parent.parent.parent / gen_cfg_file
            if not gen_cfg_file.exists():
                gen_cfg_file = Path.cwd() / gen_cfg_path
        if gen_cfg_file.exists():
            with open(gen_cfg_file, "r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f) or {}
            for k, v in base_cfg.items():
                if k not in cfg:
                    cfg[k] = v
                elif isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k] = {**v, **cfg[k]}

    llm_cfg = cfg.get("llm", {})
    data_dirs = cfg.get("data_dirs", {})

    base_dir = cfg_path.parent.parent.parent
    if not (base_dir / "data").exists():
        base_dir = Path.cwd()

    # 加载已有 registry
    registry = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}

    # 初始化 LLM
    provider = llm_cfg.get("provider", "siliconflow")
    api_key = llm_cfg.get("api_key") or os.getenv(
        {"siliconflow": "SILICONFLOW_API_KEY",
         "openai": "OPENAI_API_KEY",
         "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "SILICONFLOW_API_KEY")
    )
    llm = LLMClient(
        provider=provider,
        model=llm_cfg.get("model", "Qwen/Qwen2.5-72B-Instruct"),
        api_key=api_key,
        max_retries=llm_cfg.get("max_retries", 3),
        timeout=llm_cfg.get("timeout", 60),
    )

    # 扫描 HDF5
    h5_files = _scan_h5_files(data_dirs, base_dir)
    if not h5_files:
        raise RuntimeError(f"未找到任何 HDF5 文件，检查 data_dirs: {data_dirs}")

    # 按场景过滤
    if scenarios is not None:
        scenarios_set = set(scenarios)
        h5_files = [
            (p, s, sfx) for p, s, sfx in h5_files
            if _scenario_name_from_path(p, sfx) in scenarios_set
        ]
        missing = scenarios_set - {_scenario_name_from_path(p, sfx) for p, _, sfx in h5_files}
        if missing:
            logger.warning(f"以下场景未找到对应 HDF5 文件: {sorted(missing)}")

    level_desc = "simulator 级别" if simulator_level else "场景级别"
    print(f"\n待处理: {len(h5_files)} 个场景，字段组: {fields}，注册级别: {level_desc}，覆盖模式: {overwrite}")

    # simulator 级别时，每个 simulator 只处理一次（取第一个场景作为代表）
    seen_simulators: set = set()

    for h5_path, simulator, file_suffix in h5_files:
        scenario_name = _scenario_name_from_path(h5_path, file_suffix)

        if simulator_level:
            registry_key = simulator
            # 同一 simulator 只推断一次（所有场景共用同一条 registry 条目）
            if simulator in seen_simulators:
                print(f"[跳过] {simulator} 已处理（simulator 级别，只需一次）")
                continue
            seen_simulators.add(simulator)
            print(f"\n  代表场景: {scenario_name}（用于推断 {simulator} 通用字段）")
            # 取 simulator 级条目（新结构：顶层 key = simulator）
            existing_entry = {k: v for k, v in registry.get(simulator, {}).items() if k != "scenarios"}
        else:
            # 场景级注册已不再使用（新结构只在 simulator 级写字段）
            # 兼容旧逻辑：仍允许场景级 key，但推荐使用 simulator 级
            registry_key = simulator
            existing_entry = {k: v for k, v in registry.get(simulator, {}).items() if k != "scenarios"}
            print(f"\n  场景: {scenario_name}（注册到 simulator 级 key: {simulator}）")
        groups_to_run = []
        for fg in fields:
            registry_keys_for_group = FIELD_GROUPS[fg]
            already_have = all(k in existing_entry for k in registry_keys_for_group)
            if already_have and not overwrite:
                print(f"[跳过] {registry_key} / {fg}（已有，使用 --overwrite 强制覆盖）")
            else:
                groups_to_run.append(fg)

        if not groups_to_run:
            continue

        print(f"\n[注册] {registry_key}  →  字段组: {groups_to_run}")
        print(f"  文件: {h5_path.name}")

        try:
            timeseries, params, param_names = load_dataset(str(h5_path))
        except Exception as e:
            logger.error(f"加载 {h5_path} 失败: {e}")
            continue

        print(f"  形状: timeseries={timeseries.shape}, params={params.shape}")

        entry = dict(existing_entry)
        any_success = False

        for fg in groups_to_run:
            print(f"  → 推断 {fg}...", end=" ", flush=True)
            try:
                result = _call_llm_for_fields(
                    llm=llm,
                    field_group=fg,
                    simulator=simulator,
                    scenario_name=scenario_name,
                    param_names=list(param_names),
                    timeseries_shape=timeseries.shape,
                    params_sample=params,
                )
                entry.update(result)
                print("✓")
                any_success = True

                # 事后一致性校验：fixed_channels 必须是整数列表或 null
                if "observation_config" in entry:
                    fc = entry["observation_config"].get("fixed_channels")
                    if isinstance(fc, list) and fc and isinstance(fc[0], str):
                        # 旧版遗留的字符串名称列表，自动转为 null（全选）
                        print(f"  ⚠ 一致性警告: fixed_channels 含字符串元素，已自动重置为 null（全选）")
                        entry["observation_config"]["fixed_channels"] = None
            except RuntimeError as e:
                logger.error(str(e))
                print("✗")

        if any_success:
            # 新结构：写入 registry[simulator]，保留已有 scenarios 子字段
            existing_scenarios = registry.get(simulator, {}).get("scenarios", {})
            if existing_scenarios:
                entry["scenarios"] = existing_scenarios
            registry[simulator] = entry
            # 每处理完一个场景立即落盘，防止中断丢失
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(registry, f, allow_unicode=True, sort_keys=True, indent=2)
            print(f"  已保存 → {output_path}")

    # 打印汇总
    print(f"\n注册完成，共 {len(registry)} 条 → {output_path}")
    for key, entry in sorted(registry.items()):
        have = [fg for fg in ALL_FIELDS if all(k in entry for k in FIELD_GROUPS[fg])]
        missing = [fg for fg in ALL_FIELDS if fg not in have]
        status = "✓ 完整" if not missing else f"缺少: {missing}"
        print(f"  {key}: {status}")

    return registry


# ── CLI 入口 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="自动注册：用 LLM 为 HDF5 数据集生成元数据（支持分步注册）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
字段组（--fields）：
  domain       → domain_context, output_description, param_info
  output_info  → output_info
  observation  → observation_config

示例：
  # 注册所有字段（默认）
  python -m piern.text2comp.auto_register

  # 只注册指定场景
  python -m piern.text2comp.auto_register --scenarios unified_aquifer ieee14_baseload

  # 只注册某字段组
  python -m piern.text2comp.auto_register --fields observation

  # 组合：指定场景 + 指定字段组
  python -m piern.text2comp.auto_register --scenarios unified_aquifer --fields observation

  # 强制覆盖已有字段
  python -m piern.text2comp.auto_register --overwrite
"""
    )
    parser.add_argument("--config", default="configs/text2comp/default.yaml")
    parser.add_argument("--output", default="configs/text2comp/registry.yaml")
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="只处理指定场景名（空格分隔），默认处理全部"
    )
    parser.add_argument(
        "--fields", nargs="+", default=None,
        choices=ALL_FIELDS,
        help=f"只注册指定字段组（{ALL_FIELDS}），默认注册全部"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆盖 registry 中已有的字段（默认跳过已有字段）"
    )
    parser.add_argument(
        "--simulator-level", action="store_true",
        help=(
            "注册到 simulator 级别 key（如 'modflow'），适合各场景通用的字段。"
            "每个 simulator 只推断一次，所有场景共用。"
        )
    )
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    run_auto_register(
        cfg_path=args.config,
        output_path=args.output,
        scenarios=args.scenarios,
        fields=args.fields,
        overwrite=args.overwrite,
        simulator_level=args.simulator_level,
    )


if __name__ == "__main__":
    main()
