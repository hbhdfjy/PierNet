"""
LLM驱动的语言模板生成器（Stage 2）。

支持两种工作模式：

模式 A（解耦，推荐）：
  阶段一 make_template()  → TemplateRecord（只含语言结构，无数值）
  阶段二 fill_sample()    → 5字段训练样本（纯本地，不调 LLM）
  阶段一可批量预生成并缓存，阶段二可离线高速填充。

模式 B（兼容，原有行为）：
  generate_sample() / generate_batch() 一步完成，内部调用 make_template + fill_sample。

每条样本格式：
  input:              LLM生成的自然语言任务描述（含变换后参数值，3-8句话）
  number:             原始参数值列表（未变换，供模型学习真实物理量）
  params_transformed: 变换后参数值列表（与 input 中描述的数值一致）
  target:             自然语言回应前缀 + 真实仿真时序数值
  metadata:           调试用元信息（不参与训练）
"""

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from piern.core.llm_client import LLMClient
from piern.synth.text2comp.template_store import (
    TemplateRecord, PlaceholderSlot, OutputSlot, TransformDesc,
    fill_sample,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 领域知识注册表
# ─────────────────────────────────────────────

DOMAIN_REGISTRY = {
    "modflow": {
        "domain_context": (
            "Groundwater flow simulation (MODFLOW). "
            "Parabolic PDE: S_s·∂h/∂t = ∇(K∇h) + W. "
            "FDM + PCG solver. Outputs hydraulic head (meters above datum)."
        ),
        "scenarios": {
            "unified_aquifer": "Single-layer homogeneous aquifer with uniform hydraulic properties",
            "multilayer_confined": "Multi-layer confined aquifer system with interbedded aquitards",
            "heterogeneous_unconfined": "Spatially heterogeneous unconfined aquifer with variable K field",
            "river_recharge": "Aquifer with river boundary recharge (Cauchy-type boundary condition)",
            "lake_interaction": "Aquifer-lake interaction with dynamic water table exchange",
            "monsoon_seasonal": "Seasonal monsoon recharge pattern with alternating wet/dry cycles",
            "coastal_seawater": "Coastal aquifer subject to saltwater intrusion from the sea boundary",
        },
        "output_description": (
            "{ch} observation wells × {ts} days of hydraulic head (meters above datum)"
        ),
        "output_info": [
            {"name": "hydraulic_head", "name_zh": "水力水头", "description": "hydraulic head at observation wells", "unit": "m", "slice": [0, None]},
        ],
        "param_info": {
            "K_mean": ("mean hydraulic conductivity", "m/d"),
            "K_std": ("hydraulic conductivity standard deviation", "m/d"),
            "K_anisotropy": ("horizontal-to-vertical anisotropy ratio", "dimensionless"),
            "S_storage": ("specific storage coefficient", "1/m"),
            "Q_pumping": ("pumping rate", "m³/d"),
            "H_initial": ("initial hydraulic head", "m"),
            "R_recharge": ("recharge rate", "m/d"),
            "R_variation": ("recharge seasonal variation amplitude", "dimensionless"),
            "BC_strength": ("boundary condition head", "m"),
            "n_layers": ("number of aquifer layers", "dimensionless"),
            "C_concentration": ("contaminant concentration", "mg/L"),
            "D_dispersion": ("hydrodynamic dispersion coefficient", "m²/d"),
            "lambda_decay": ("first-order decay constant", "1/d"),
            "phi_porosity": ("effective porosity", "dimensionless"),
            "rho_density": ("fluid density", "kg/m³"),
            "hk": ("horizontal hydraulic conductivity", "m/d"),
            "sy": ("specific yield", "dimensionless"),
            "pumping": ("well pumping rate", "m³/d"),
            "strt": ("starting hydraulic head", "m"),
            "rch": ("areal recharge rate", "m/d"),
        },
        "observation_config": {
            "fixed_time_mode": "monthly",
            "fixed_channels": [0, 1, 2, 3, 4],
            "time_modes": [
                {
                    "name": "monthly",
                    "indices": "monthly",
                    "desc_en": "monthly (day 15 of each month), 12 time points",
                    "desc_zh": "月度（每月第15天），共12个时间点",
                },
            ],
            "channel_name_template": "well {i}",
            "channel_name_template_zh": "第{i}号观测井",
        },
    },
    "power_flow": {
        "domain_context": (
            "Steady-state AC power flow (pandapower). "
            "Nonlinear algebraic equations solved by Newton-Raphson method. "
            "Outputs: bus voltages (p.u.), voltage angles (rad), line active power flows (MW)."
        ),
        "scenarios": {
            "ieee14_baseload": "IEEE 14-bus test system under base load conditions",
            "ieee14_renewable": "IEEE 14-bus system with high renewable energy penetration",
            "ieee30_contingency": "IEEE 30-bus system with N-1 contingency analysis",
            "ieee118_dispatch": "IEEE 118-bus system with economic dispatch optimization",
            "distribution_33bus": "33-bus radial distribution network with distributed generation",
        },
        "output_description": (
            "{ch} channels (bus voltages + voltage angles + line power flows, vertically concatenated) × {ts} days"
        ),
        "output_info": [
            {"name": "bus_voltages",     "name_zh": "母线电压幅值",   "description": "bus voltage magnitudes",   "unit": "p.u.", "slice": [0, 14]},
            {"name": "voltage_angles",   "name_zh": "母线电压相角",   "description": "bus voltage angles",        "unit": "rad",  "slice": [14, 28]},
            {"name": "line_power_flows", "name_zh": "线路有功功率流", "description": "line active power flows",   "unit": "MW",   "slice": [28, 43]},
        ],
        "param_info": {
            "load_scale": ("load scaling factor relative to nominal", "p.u."),
            "V_base_kv": ("base voltage level", "kV"),
            "renewable_penetration": ("fraction of generation from renewables", "dimensionless"),
            "V_ref": ("reference bus voltage setpoint", "p.u."),
            "line_loading_limit": ("maximum allowed line loading", "p.u."),
            "gen_dispatch_factor": ("generator dispatch scaling factor", "p.u."),
            "shunt_susceptance": ("shunt susceptance for reactive compensation", "p.u."),
            "tap_ratio": ("transformer tap ratio", "dimensionless"),
            "fault_resistance": ("fault resistance at fault location", "Ω"),
            "contingency_line": ("line index for N-1 contingency", "dimensionless"),
            "dg_penetration": ("distributed generation penetration level", "p.u."),
            "feeder_impedance": ("feeder impedance scaling factor", "p.u."),
        },
        "observation_config": {
            "fixed_time_mode": "monthly",
            "fixed_channels": [0, 1, 2],   # 0=bus_voltages, 1=voltage_angles, 2=line_power_flows
            "time_modes": [
                {
                    "name": "monthly",
                    "indices": "monthly",
                    "desc_en": "monthly, 12 time points",
                    "desc_zh": "月度，共12个时间点",
                },
            ],
            "channel_name_template": "channel {i}",
            "channel_name_template_zh": "第{i}个通道",
        },
    },
    "transient": {
        "domain_context": (
            "Transient stability simulation (ANDES). "
            "DAE system: δ̇=ω, M·ω̇=Pm−Pe−D·ω, 0=g(x,y). "
            "Implicit trapezoidal integration (100 Hz). "
            "Outputs: generator rotor angles (radians) over 10-second post-fault window."
        ),
        "scenarios": {
            "ieee39_fault": "IEEE 39-bus New England system with three-phase bus fault",
            "ieee39_trip": "IEEE 39-bus system with sudden generator trip event",
            "ieee14_load_step": "IEEE 14-bus system with large load step disturbance",
        },
        "output_description": (
            "{ch} generators × {ts} time steps of rotor angle (radians), 10 s at 100 Hz"
        ),
        "output_info": [
            {"name": "rotor_angles", "name_zh": "转子角", "description": "generator rotor angles", "unit": "rad", "slice": [0, None]},
        ],
        "param_info": {
            "fault_bus": ("fault location bus index", "dimensionless"),
            "fault_duration": ("fault clearing time", "s"),
            "inertia_H": ("generator inertia constant", "s"),
            "damping_D": ("damping coefficient", "p.u."),
            "Pm_setpoint": ("mechanical power input setpoint", "p.u."),
            "Xd_prime": ("d-axis transient reactance", "p.u."),
            "governor_gain": ("speed governor gain", "p.u."),
            "exciter_gain": ("automatic voltage regulator gain", "p.u."),
            "load_scale": ("pre-fault load scaling factor", "p.u."),
            "trip_gen": ("tripped generator index", "dimensionless"),
            "load_step_size": ("load step magnitude", "p.u."),
            "clearing_time": ("fault clearing time for protection relay", "s"),
        },
        "observation_config": {
            "fixed_time_mode": "1Hz",
            "fixed_channels": None,   # None = 全选所有发电机
            "time_modes": [
                {
                    "name": "1Hz",
                    "indices": "every_100",
                    "desc_en": "downsampled to 1 Hz (every 0.1 s), 10 time points",
                    "desc_zh": "降采样至1Hz（每0.1秒），共10个时间点",
                },
            ],
            "channel_name_template": "generator {i}",
            "channel_name_template_zh": "第{i}号发电机",
        },
    },
    "gcam": {
        "domain_context": (
            "Energy-climate coupled simulation (PyPSA multi-period LP). "
            "Multi-period linear program optimizing energy system costs 2025–2100. "
            "Models technology learning curves, carbon pricing, and climate feedbacks. "
            "Outputs: energy mix shares, CO₂ emissions, energy prices, temperature."
        ),
        "scenarios": {
            "energy_transition": "Accelerated energy technology transition toward net-zero",
            "carbon_pricing": "Carbon pricing policy with rising CO₂ cost trajectory",
            "climate_feedback": "Climate-economy feedback with temperature-dependent damages",
        },
        "output_description": (
            "{ch} variables (coal share, renewable share, CO₂ emissions, energy price, temperature) "
            "× {ts} time steps (2025–2100, 5-year intervals)"
        ),
        "output_info": [
            {"name": "coal_share",      "name_zh": "煤炭占比",     "description": "coal energy share",          "unit": "dimensionless", "slice": [0, 1]},
            {"name": "renewable_share", "name_zh": "可再生能源占比","description": "renewable energy share",     "unit": "dimensionless", "slice": [1, 2]},
            {"name": "co2_emissions",   "name_zh": "CO₂排放量",    "description": "CO₂ emissions",              "unit": "GtCO₂/yr",      "slice": [2, 3]},
            {"name": "energy_price",    "name_zh": "能源价格",     "description": "energy price",               "unit": "$/MWh",         "slice": [3, 4]},
            {"name": "temperature",     "name_zh": "全球平均温度",  "description": "global mean temperature",    "unit": "°C",            "slice": [4, 5]},
        ],
        "param_info": {
            "carbon_price_2030": ("carbon price in 2030", "$/tCO₂"),
            "carbon_price_growth": ("annual carbon price growth rate", "dimensionless"),
            "solar_learning_rate": ("solar PV cost learning rate", "dimensionless"),
            "wind_learning_rate": ("wind power cost learning rate", "dimensionless"),
            "coal_phase_out_year": ("target year for coal phase-out", "year"),
            "nuclear_share_2050": ("nuclear energy share target in 2050", "dimensionless"),
            "energy_demand_growth": ("annual energy demand growth rate", "dimensionless"),
            "climate_sensitivity": ("equilibrium climate sensitivity", "°C per 2×CO₂"),
            "discount_rate": ("social discount rate for cost optimization", "dimensionless"),
            "initial_renewable_share": ("renewable energy share in base year 2025", "dimensionless"),
            "ccs_cost_factor": ("carbon capture and storage cost multiplier", "dimensionless"),
            "efficiency_improvement": ("annual end-use energy efficiency improvement rate", "dimensionless"),
        },
        "observation_config": {
            # 固定策略：全量时间序列，全部 5 个变量
            "fixed_time_mode": "full",
            "fixed_channels": [0, 1, 2, 3, 4],  # 0=coal_share, 1=renewable_share, 2=co2_emissions, 3=energy_price, 4=temperature
            "time_modes": [
                {
                    "name": "full",
                    "indices": "full",
                    "desc_en": "full 2025-2100 trajectory, 16 time points",
                    "desc_zh": "2025-2100年全量，共16个时间点",
                },
            ],
            "channel_name_template": "channel {i}",
            "channel_name_template_zh": "第{i}个通道",
        },
    },
    "simpeg": {
        "domain_context": (
            "Geophysical inversion simulation (SimPEG). "
            "Elliptic PDE: ∇·(σ∇φ) = −Iδ. "
            "1D layered earth model (analytical/transfer-matrix). "
            "Outputs: apparent resistivity (Ω·m), impedance phase (°), "
            "transient EMF (normalized), or apparent chargeability (dimensionless)."
        ),
        "scenarios": {
            "dc_resistivity":   "DC resistivity survey (Wenner array) over a 3-layer earth model",
            "mt_sounding":      "Magnetotelluric (MT) sounding over a 3-layer resistivity structure",
            "tem_decay":        "Time-domain electromagnetic (TEM) central-loop decay measurement",
            "ip_chargeability": "Induced polarization (IP) survey with Cole-Cole polarization model",
        },
        "output_description": (
            "1 channel × {ts} measurement points of geophysical response"
        ),
        "output_info": [
            {
                "name": "geophysical_response",
                "name_zh": "地球物理响应",
                "description": "geophysical measurement curve (apparent resistivity / phase / EMF / chargeability)",
                "unit": "Ω·m / ° / normalized / dimensionless",
                "slice": [0, None],
            },
        ],
        "param_info": {
            "sigma_bg":       ("background electrical conductivity", "S/m"),
            "sigma_anomaly":  ("anomaly body electrical conductivity", "S/m"),
            "depth_top":      ("top depth of anomaly body", "m"),
            "depth_bottom":   ("bottom depth of anomaly body", "m"),
            "width_x":        ("horizontal width of anomaly body", "m"),
            "width_z":        ("vertical thickness of anomaly body", "m"),
            "source_spacing": ("transmitter-receiver spacing / loop radius", "m"),
            "noise_level":    ("relative measurement noise level", "dimensionless"),
            "n_layers":       ("number of earth layers in model", "dimensionless"),
            "survey_length":  ("total survey profile length", "m"),
            "chargeability":  ("intrinsic chargeability (Cole-Cole model)", "dimensionless"),
            "time_constant":  ("Cole-Cole / TEM time constant", "s"),
            "freq_min":       ("minimum MT sounding frequency", "Hz"),
            "freq_max":       ("maximum MT sounding frequency", "Hz"),
            "mu_r":           ("relative magnetic permeability", "dimensionless"),
        },
        "observation_config": {
            "fixed_time_mode": "full",
            "fixed_channels": None,   # None = 全选（单通道）
            "time_modes": [
                {
                    "name": "full",
                    "indices": "full",
                    "desc_en": "full survey curve, 100 measurement points",
                    "desc_zh": "完整测量曲线，共100个测量点",
                },
            ],
            "channel_name_template": "measurement channel {i}",
            "channel_name_template_zh": "第{i}号测量通道",
        },
    },
}

# ─────────────────────────────────────────────
# 系统提示词（固定）
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a scientific computing expert creating training data for physics simulation models. "
    "Write natural language descriptions of physics simulation scenarios. "
    "Be scientifically accurate, specific about parameter values, and vary your writing style. "
    "Do NOT include the actual numerical predictions in your response. "
    "Placeholders such as {value_0} and {output_0} are protected tokens: copy every required "
    "placeholder exactly with single braces, never translate, renumber, omit, or wrap them in code. "
    "Return only the requested template text, with no markdown fences or explanatory notes."
)

# ─────────────────────────────────────────────
# 参数变换配置
# ─────────────────────────────────────────────

TRANSFORM_TYPES = ["multiply", "divide", "add", "subtract"]
TRANSFORM_WEIGHTS = [0.4, 0.3, 0.15, 0.15]
# 乘除法：1-10 连续均匀随机（整数）
MULTIPLY_RANGE = (1, 10)
DIVIDE_RANGE   = (1, 10)
# 加减法：0-100 连续均匀随机
ADD_RANGE      = (0, 100)

LANGUAGES = ["en", "zh"]
STYLES = ["technical", "popular", "concise"]
STYLE_WEIGHTS = [0.4, 0.3, 0.3]

# {output} 占位符，用于 target 模板
OUTPUT_PLACEHOLDER = "{output}"


def _natural_note(t_type: str, factor: float, rng: np.random.Generator) -> tuple[str, str]:
    """
    生成变换的自然语言描述，供 LLM prompt 使用。
    避免机械的"乘以N"/"除以N"，改用领域专家的表达方式。
    同一变换类型有多种措辞，随机选一种增加多样性。
    """
    f = f"{factor:.4g}"

    if t_type == "multiply":
        en_choices = [
            f"scaled up by a factor of {f}",
            f"approximately {f}× the baseline value",
            f"about {f} times the reference",
            f"at {f}× nominal",
        ]
        zh_choices = [
            f"约为基准值的 {f} 倍",
            f"相当于参考值的 {f} 倍",
            f"按 {f} 倍系数放大",
            f"为标称值的 {f} 倍",
        ]
    elif t_type == "divide":
        en_choices = [
            f"scaled down by a factor of {f}",
            f"approximately 1/{f} of the baseline",
            f"reduced to about 1/{f} of the reference",
            f"at 1/{f} of nominal",
        ]
        zh_choices = [
            f"约为基准值的 1/{f}",
            f"缩减至参考值的 1/{f}",
            f"按 1/{f} 比例缩小",
            f"为标称值的 1/{f}",
        ]
    elif t_type == "add":
        en_choices = [
            f"elevated by approximately {f} above baseline",
            f"about {f} units above the reference",
            f"offset upward by {f}",
            f"roughly {f} higher than nominal",
        ]
        zh_choices = [
            f"较基准偏高约 {f}",
            f"高于参考值约 {f}",
            f"在基准之上偏移 {f}",
            f"比标称值高约 {f}",
        ]
    else:  # subtract
        en_choices = [
            f"reduced by approximately {f} from baseline",
            f"about {f} units below the reference",
            f"offset downward by {f}",
            f"roughly {f} lower than nominal",
        ]
        zh_choices = [
            f"较基准偏低约 {f}",
            f"低于参考值约 {f}",
            f"在基准之下偏移 {f}",
            f"比标称值低约 {f}",
        ]

    note_en = en_choices[int(rng.integers(0, len(en_choices)))]
    note_zh = zh_choices[int(rng.integers(0, len(zh_choices)))]
    return note_en, note_zh



# ─────────────────────────────────────────────
# 观测方案数据结构
# ─────────────────────────────────────────────

@dataclass
class ObservationSpec:
    """描述单条样本的观测方案：看哪些通道、哪些时间点。"""
    # 时间维
    time_indices: np.ndarray        # 选取的时间步索引，有序
    time_mode_name: str             # 模式名，如 "monthly"
    time_desc_en: str               # 英文描述
    time_desc_zh: str               # 中文描述

    # 通道维
    # 模式A：output_info 条目级别（power_flow/gcam：选哪些物理量）
    selected_output_info: list      # 筛选后的 output_info 子集
    # 模式B：行级别（modflow/power_transient：选哪些井/发电机）
    channel_indices: Optional[np.ndarray]   # None 表示全选（output_info 级别时为 None）
    channel_desc_en: str            # 英文描述
    channel_desc_zh: str            # 中文描述


# ─────────────────────────────────────────────
# 时间索引生成
# ─────────────────────────────────────────────

def _evenly_spaced_indices(n_timesteps: int, max_points: int) -> np.ndarray:
    """Return up to max_points valid, ordered indices spanning the series."""
    if n_timesteps <= 0:
        return np.array([], dtype=int)
    n_points = min(max_points, n_timesteps)
    return np.round(np.linspace(0, n_timesteps - 1, n_points)).astype(int)


def _get_time_indices(mode_name: str, n_timesteps: int) -> np.ndarray:
    """根据模式名生成时间步索引数组。"""
    if mode_name == "monthly":
        # 均匀分12段，取各段中点（≈每月第15天）。短时序无法表示
        # 12 个“月中”采样点时，退化为全跨度均匀采样，避免负数/重复索引。
        if n_timesteps < 39:
            return _evenly_spaced_indices(n_timesteps, 12)
        return np.round(np.linspace(14, n_timesteps - 14, 12)).astype(int)
    elif mode_name == "weekly":
        if n_timesteps <= 0:
            return np.array([], dtype=int)
        return np.arange(0, n_timesteps, 7)[:52]
    elif mode_name == "full":
        return np.arange(n_timesteps)
    elif mode_name.startswith("every_"):
        # 通用 every_N：每隔 N 步取一个点，N 为任意正整数
        try:
            step = int(mode_name[len("every_"):])
        except ValueError:
            raise ValueError(f"无效的时间模式 '{mode_name}'，every_N 中 N 必须为正整数")
        if step <= 0:
            raise ValueError(f"无效的时间模式 '{mode_name}'，步长必须大于 0")
        return np.arange(0, n_timesteps, step)
    else:
        raise ValueError(
            f"未知时间模式 '{mode_name}'，支持：monthly / weekly / full / every_N（N为正整数）"
        )


# ─────────────────────────────────────────────
# 核心生成器
# ─────────────────────────────────────────────

class LLMTextGenerator:
    """LLM驱动的物理仿真语言描述生成器。"""

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.8,
        max_tokens: int = 600,
        language_mix: float = 0.5,
        transform_prob: float = 0.4,
        styles: Optional[list] = None,
        style_weights: Optional[list] = None,
        max_workers: int = 1,
    ):
        """
        Args:
            llm_client:   已初始化的 LLMClient 实例（作为模板，并发时每线程克隆一份）
            temperature:  LLM 采样温度
            max_tokens:   最大生成 token 数
            language_mix: 生成英文的概率（0=全中文, 1=全英文, 0.5=各半）
            transform_prob: 每个参数被变换的概率
            styles:       写作风格列表
            style_weights: 风格权重（与 styles 等长）
            max_workers:  并发线程数（1=串行，建议 8-32）
        """
        self._llm_template = llm_client   # 保留模板，用于克隆
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.language_mix = language_mix
        self.transform_prob = transform_prob
        self.styles = styles or STYLES
        self.style_weights = style_weights or STYLE_WEIGHTS
        self.max_workers = max_workers

        # 线程本地存储：每个线程持有独立的 LLMClient（requests.Session 非线程安全）
        self._thread_local = threading.local()



    def _get_thread_llm(self) -> LLMClient:
        """获取当前线程的 LLMClient，不存在则克隆一个。"""
        if not hasattr(self._thread_local, "llm"):
            t = self._llm_template
            self._thread_local.llm = LLMClient(
                provider=t.provider,
                model=t.model,
                api_key=t.api_key,
                base_url=t.base_url,
                max_retries=t.max_retries,
                timeout=t.timeout,
                thinking=getattr(t, "thinking", None),
            )
        return self._thread_local.llm

    @property
    def llm(self) -> LLMClient:
        """向后兼容：单线程时直接返回模板客户端。"""
        return self._get_thread_llm()

    # ── 公开接口 ──────────────────────────────

    def generate_sample(
        self,
        simulator: str,
        scenario_name: str,
        params: np.ndarray,
        param_names: list,
        timeseries: np.ndarray,
        sample_idx: int,
        rng: np.random.Generator,
        domain: dict = None,
    ) -> dict:
        """
        生成单条样本。

        返回格式：
            input:              LLM生成的自然语言任务描述（含变换后参数值）
            number:             原始参数值列表（未变换）
            params_transformed: 变换后参数值列表（与 input 中数值一致）
            target:             自然语言前缀 + 真实仿真时序数值
            metadata:           调试用元信息

        Args:
            simulator:     模拟器名称（用于 fallback 查 DOMAIN_REGISTRY）
            scenario_name: 场景名称
            params:        原始参数，shape (n_params,)
            param_names:   参数名称列表
            timeseries:    真实时序 (channels, timesteps)
            sample_idx:    样本索引（用于 metadata）
            rng:           随机数生成器（保证可复现）
            domain:        领域元数据 dict（优先使用）；为 None 时从 DOMAIN_REGISTRY 查找
        """
        if domain is None:
            domain = DOMAIN_REGISTRY.get(simulator)
        if domain is None:
            raise ValueError(
                f"未找到 simulator '{simulator}' 的元数据。"
                f"请先运行 auto_register.py 或在 DOMAIN_REGISTRY 中注册。"
            )

        # 检测无效数值（仿真不收敛时可能产生 NaN/Inf，json.dumps 会失败）
        if not np.isfinite(timeseries).all():
            raise ValueError(
                f"样本 {sample_idx} 的时序数据含 NaN 或 Inf，跳过"
            )

        # ── 阶段一：生成模板（LLM 调用，不接触数值）────────────
        template = self.make_template(
            simulator=simulator,
            scenario_name=scenario_name,
            param_names=param_names,
            timeseries_shape=timeseries.shape,
            rng=rng,
            domain=domain,
            sample_idx=sample_idx,
        )

        # ── 应用降采样（时间 + 通道，与 make_template 中的 obs_spec 一致）────
        ts_time = timeseries[:, np.array(template.time_indices)]
        ts_obs = ts_time[np.array(template.channel_indices), :]

        # ── 阶段二：填充数值（纯本地，不调 LLM）────────────────
        return fill_sample(template, params, ts_obs, sample_idx=sample_idx)

    def generate_batch(
        self,
        simulator: str,
        scenario_name: str,
        params_array: np.ndarray,
        param_names: list,
        timeseries_array: np.ndarray,
        n_samples: int,
        seed: int = 42,
        output_file=None,
        file_lock: Optional[threading.Lock] = None,
        domain: dict = None,
    ) -> list:
        """
        批量生成样本，支持多线程并发。

        Args:
            simulator:        模拟器名称
            scenario_name:    场景名称
            params_array:     (N, n_params)
            param_names:      参数名称列表
            timeseries_array: (N, channels, timesteps)
            n_samples:        实际生成条数（可超过 N，超出部分循环复用）
            seed:             随机种子（每个任务用 seed+i 保证可复现）
            output_file:      若提供，边生成边写入（断点续传友好）
            file_lock:        写文件时的锁（output_file 非 None 时必须提供）

        Returns:
            list of dicts，按原始顺序排列
        """
        n_available = len(params_array)

        def _task(i: int) -> tuple[int, Optional[dict]]:
            idx = i % n_available
            max_retries = getattr(self._llm_template, "max_retries", 3)

            for attempt in range(max_retries):
                sample_rng = np.random.default_rng(seed + i + attempt * 100_000)
                try:
                    sample = self.generate_sample(
                        simulator=simulator,
                        scenario_name=scenario_name,
                        params=params_array[idx],
                        param_names=param_names,
                        timeseries=timeseries_array[idx],
                        sample_idx=idx,
                        rng=sample_rng,
                        domain=domain,
                    )
                    if output_file is not None and file_lock is not None:
                        line = json.dumps(sample, ensure_ascii=False) + "\n"
                        with file_lock:
                            output_file.write(line)
                            output_file.flush()
                    return i, sample
                except ValueError as e:
                    msg = str(e)
                    # NaN/Inf 是数据本身的问题，重试无意义，直接跳过
                    if "NaN" in msg or "Inf" in msg:
                        logger.warning(f"样本 {i} 含无效数值，跳过: {e}")
                        return i, None
                    logger.warning(f"样本 {i} 占位符校验失败（第 {attempt+1}/{max_retries} 次）: {e}")
                except Exception as e:
                    logger.warning(f"样本 {i} 生成失败（跳过）: {e}")
                    return i, None

            logger.error(f"样本 {i} 重试 {max_retries} 次后仍失败，跳过")
            return i, None

        results_map: dict[int, dict] = {}

        if self.max_workers <= 1:
            for i in tqdm(range(n_samples), desc=f"{simulator}/{scenario_name}"):
                _, sample = _task(i)
                if sample is not None:
                    results_map[i] = sample
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_task, i): i for i in range(n_samples)}
                with tqdm(total=n_samples, desc=f"{simulator}/{scenario_name}") as pbar:
                    for future in as_completed(futures):
                        i, sample = future.result()
                        if sample is not None:
                            results_map[i] = sample
                        pbar.update(1)

        return [results_map[i] for i in range(n_samples) if i in results_map]

    # ── 私有方法 ──────────────────────────────

    def _sample_observation(
        self,
        domain: dict,
        timeseries_shape: tuple,
        output_info: list,
    ) -> "ObservationSpec":
        """根据 observation_config 中的固定配置生成观测方案。

        observation_config 必须包含：
          fixed_time_mode: str   — 时间模式名，必须存在于 time_modes 中
          fixed_channels:        — 通道配置：
              None               → 全选所有通道/行
              list[int]          → 行级别：按索引指定（0-based）
              list[str]          → output_info级别：按物理量名称指定

        不支持随机模式。缺少必要字段时抛 ValueError。
        """
        obs_cfg = domain.get("observation_config", {})
        ch, ts = timeseries_shape

        # --- 时间维（必须指定 fixed_time_mode）---
        if "fixed_time_mode" not in obs_cfg:
            raise ValueError(
                "observation_config 缺少 'fixed_time_mode'。"
                "请在 registry.yaml 或 DOMAIN_REGISTRY 中为该 simulator 指定固定时间模式。"
            )
        fixed_time_mode = obs_cfg["fixed_time_mode"]
        time_modes = obs_cfg.get("time_modes", [])
        mode_map = {m["name"]: m for m in time_modes}
        if fixed_time_mode not in mode_map:
            raise ValueError(
                f"fixed_time_mode='{fixed_time_mode}' 不在 time_modes 中: {list(mode_map)}"
            )
        mode = mode_map[fixed_time_mode]
        time_indices = _get_time_indices(mode["indices"], ts)

        if len(time_indices) == 0:
            raise ValueError(
                "observation_config resolved to an empty time selection. Ensure the time series has at least one time step."
            )

        n_actual = len(time_indices)
        desc_en = re.sub(r'\d+ time points?', f'{n_actual} time points', mode["desc_en"])
        desc_zh = re.sub(r'共\d+个时间点', f'共{n_actual}个时间点', mode["desc_zh"])

        # --- 通道维（必须包含 fixed_channels 键）---
        if "fixed_channels" not in obs_cfg:
            raise ValueError(
                "observation_config 缺少 'fixed_channels'。"
                "请设置为 null（全选所有通道）或整数列表（0-based 索引）。"
            )
        fixed_channels = obs_cfg["fixed_channels"]
        # Normalize channel selection. In row mode, fixed_channels are raw row
        # indices. In output_info mode, fixed_channels select output_info entries,
        # and each entry expands to its declared slice in the raw time-series rows.
        raw_channel_level = str(obs_cfg.get("channel_level", "row") or "row").lower()
        channel_level = "output_info" if raw_channel_level in {"output", "output_info"} else "row"
        if fixed_channels == []:
            fixed_channels = None

        def _coerce_int(value, default=None):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _output_info_index(value) -> int:
            if isinstance(value, bool):
                return -1
            if isinstance(value, int):
                return value
            text_value = str(value).strip()
            for j, info in enumerate(output_info):
                if str(info.get("name", "")).strip() == text_value:
                    return j
            return _coerce_int(text_value, -1)

        def _rows_for_output_info(info: dict, fallback_index: int) -> list[int]:
            raw_slice = info.get("slice") if isinstance(info, dict) else None
            if isinstance(raw_slice, (list, tuple)) and len(raw_slice) >= 2:
                start = _coerce_int(raw_slice[0], 0)
                end = ch if raw_slice[1] is None else _coerce_int(raw_slice[1], ch)
            else:
                start = fallback_index
                end = fallback_index + 1
            start = max(0, min(int(start), ch))
            end = max(start, min(int(end), ch))
            return list(range(start, end))

        if channel_level == "output_info":
            if fixed_channels is None:
                selected_indices = list(range(len(output_info)))
            else:
                selected_indices = []
                for value in fixed_channels:
                    idx = _output_info_index(value)
                    if 0 <= idx < len(output_info) and idx not in selected_indices:
                        selected_indices.append(idx)

            channel_rows: list[int] = []
            selected_output_info = []
            compact_offset = 0
            for output_idx in selected_indices:
                info = output_info[output_idx]
                rows = _rows_for_output_info(info, output_idx)
                if not rows:
                    continue
                compact_info = dict(info)
                compact_info["slice"] = [compact_offset, compact_offset + len(rows)]
                selected_output_info.append(compact_info)
                channel_rows.extend(rows)
                compact_offset += len(rows)

            channel_indices = np.array(channel_rows, dtype=int)
            n_sel = len(channel_indices)
            names_en = ", ".join(
                str(info.get("name") or f"output_{i}")
                for i, info in enumerate(selected_output_info)
            )
            names_zh = "、".join(
                str(info.get("name_zh") or info.get("name") or f"输出{i}")
                for i, info in enumerate(selected_output_info)
            )
            channel_desc_en = f"{names_en} ({n_sel} rows of {ch})"
            channel_desc_zh = f"{names_zh}（共{ch}行中的{n_sel}行）"
        else:
            if fixed_channels is None:
                channel_indices = np.arange(ch)
            else:
                def _row_idx(value) -> int:
                    if isinstance(value, bool):
                        return -1
                    if isinstance(value, int):
                        return value
                    return _coerce_int(str(value).strip(), -1)
                indices = [_row_idx(v) for v in fixed_channels]
                indices = [i for i in indices if 0 <= i < ch]
                channel_indices = np.array(sorted(set(indices)), dtype=int)

            selected_output_info = list(output_info)
            n_sel = len(channel_indices)
            tmpl_en = obs_cfg.get("channel_name_template", "channel {i}")
            tmpl_zh = obs_cfg.get("channel_name_template_zh", "通道 {i}")
            names_en = ", ".join(tmpl_en.format(i=int(idx) + 1) for idx in channel_indices)
            names_zh = "、".join(tmpl_zh.format(i=int(idx) + 1) for idx in channel_indices)
            channel_desc_en = f"{names_en} ({n_sel} of {ch})"
            channel_desc_zh = f"{names_zh}（共{ch}行中的{n_sel}行）"

        n_sel = len(channel_indices)
        if n_sel == 0 or not selected_output_info:
            raise ValueError(
                "observation_config resolved to an empty channel selection. "
                "Use fixed_channels=null for all channels or choose valid row/output_info indices."
            )

        return ObservationSpec(
            time_indices=time_indices,
            time_mode_name=mode["name"],
            time_desc_en=desc_en,
            time_desc_zh=desc_zh,
            selected_output_info=selected_output_info,
            channel_indices=channel_indices,
            channel_desc_en=channel_desc_en,
            channel_desc_zh=channel_desc_zh,
        )

    def _apply_transforms(
        self,
        params: np.ndarray,
        param_names: list,
        domain: dict,
        rng: np.random.Generator,
    ) -> tuple:
        """
        对每个参数独立以 transform_prob 概率施加变换。

        Returns:
            params_transformed: 变换后的参数数组（shape 同输入）
            transform_notes:    list of (name, original, transformed, note_en, note_zh)
                                note 为空字符串表示未变换
        """
        params_transformed = params.copy().astype(float)
        transform_notes = []
        for i, (name, orig_val) in enumerate(zip(param_names, params)):
            # 元数据参数跳过变换
            if name in ("scenario_type", "output_type", "complexity"):
                transform_notes.append((name, orig_val, orig_val, "", ""))
                continue

            if rng.random() >= self.transform_prob:
                transform_notes.append((name, orig_val, orig_val, "", ""))
                continue

            t_type = rng.choice(TRANSFORM_TYPES, p=TRANSFORM_WEIGHTS)
            new_val, note_en, note_zh = self._do_transform(orig_val, t_type, rng)
            params_transformed[i] = new_val
            transform_notes.append((name, orig_val, new_val, note_en, note_zh))

        return params_transformed, transform_notes

    def _do_transform(
        self, value: float, t_type: str, rng: np.random.Generator
    ) -> tuple:
        """返回 (new_value, note_en, note_zh)"""
        if t_type == "multiply":
            factor = float(rng.integers(MULTIPLY_RANGE[0], MULTIPLY_RANGE[1] + 1))
            new_val = value * factor
            note_en, note_zh = _natural_note(t_type, factor, rng)
            return new_val, note_en, note_zh

        elif t_type == "divide":
            factor = float(rng.integers(DIVIDE_RANGE[0], DIVIDE_RANGE[1] + 1))
            if abs(value) < 1e-12:
                return value, "", ""
            new_val = value / factor
            note_en, note_zh = _natural_note(t_type, factor, rng)
            return new_val, note_en, note_zh

        elif t_type == "add":
            offset = float(rng.uniform(ADD_RANGE[0], ADD_RANGE[1]))
            new_val = value + offset
            note_en, note_zh = _natural_note(t_type, offset, rng)
            return new_val, note_en, note_zh

        elif t_type == "subtract":
            offset = float(rng.uniform(ADD_RANGE[0], ADD_RANGE[1]))
            new_val = value - offset
            note_en, note_zh = _natural_note(t_type, offset, rng)
            return new_val, note_en, note_zh

        return value, "", ""

    # ── 解耦辅助：变换描述 ────────────────────────────────────

    def _compute_transform_descs(
        self,
        param_names: list,
        domain: dict,
        rng: np.random.Generator,
    ) -> list[TransformDesc]:
        """
        阶段一专用：只决定"对哪个参数做什么变换"，不接触实际数值。

        Returns:
            list[TransformDesc]，每个参数一条（未变换的 transform_type=None）
        """
        descs: list[TransformDesc] = []
        for i, name in enumerate(param_names):
            if name in ("scenario_type", "output_type", "complexity"):
                descs.append(TransformDesc(
                    param_name=name, param_index=i,
                    transform_type=None, factor=None,
                    note_en="", note_zh="",
                ))
                continue

            if rng.random() >= self.transform_prob:
                descs.append(TransformDesc(
                    param_name=name, param_index=i,
                    transform_type=None, factor=None,
                    note_en="", note_zh="",
                ))
                continue

            t_type = rng.choice(TRANSFORM_TYPES, p=TRANSFORM_WEIGHTS)
            # 只生成因子，不应用到值
            if t_type in ("multiply", "divide"):
                factor = float(rng.integers(MULTIPLY_RANGE[0], MULTIPLY_RANGE[1] + 1))
            else:  # add / subtract
                factor = float(rng.uniform(ADD_RANGE[0], ADD_RANGE[1]))

            note_en, note_zh = _natural_note(t_type, factor, rng)

            descs.append(TransformDesc(
                param_name=name, param_index=i,
                transform_type=t_type, factor=factor,
                note_en=note_en, note_zh=note_zh,
            ))
        return descs

    @staticmethod
    def _apply_transform_descs(
        params: np.ndarray,
        descs: list[TransformDesc],
    ) -> np.ndarray:
        """
        阶段一/二通用：将 TransformDesc 列表应用到参数数组，返回变换后数组。
        """
        result = params.copy().astype(float)
        for td in descs:
            result[td.param_index] = td.apply(float(params[td.param_index]))
        return result

    @staticmethod
    def _descs_to_transform_notes(
        descs: list[TransformDesc],
        params: np.ndarray,
        params_transformed: np.ndarray,
    ) -> list[tuple]:
        """将 TransformDesc 列表转换为旧格式 transform_notes（向后兼容）。"""
        notes = []
        for td in descs:
            orig = float(params[td.param_index])
            trans = float(params_transformed[td.param_index])
            notes.append((td.param_name, orig, trans, td.note_en, td.note_zh))
        return notes

    def _build_placeholder_schema(
        self,
        param_names: list,
        domain: dict,
        descs: list[TransformDesc],
        language: str,
    ) -> tuple[list[PlaceholderSlot], list[tuple]]:
        """
        从参数名和变换描述构建 placeholder_schema（无数值）。
        同时返回 placeholder_index_meta（prompt 构建用，含 note 描述但无数值）。

        Returns:
            (placeholder_schema, placeholder_index_meta)
            placeholder_index_meta: list of (ph_key, param_index, note_str)
        """
        schema: list[PlaceholderSlot] = []
        meta: list[tuple] = []  # (ph_key, param_index, note_str)
        slot = 0

        for td in descs:
            name = td.param_name
            if name in ("scenario_type", "output_type", "complexity"):
                continue
            # 跳过零值且不在 param_info 中的参数（与 _build_user_prompt 保持一致）
            # 注意：这里无法判断值是否为零（无数值），改为只跳过不在 param_info 中的元数据参数
            # 实际零值过滤在 _build_user_prompt 中做（有数值时才能判断）
            # 此处保守策略：所有非元数据参数都生成占位符
            ph_key = "{value_" + str(slot) + "}"
            note = td.note_en if language == "en" else td.note_zh
            schema.append(PlaceholderSlot(
                index=slot,
                param_name=name,
                param_index=td.param_index,
                use_transformed=(td.transform_type is not None),
                fmt="scalar",  # 参数值均为标量
            ))
            meta.append((ph_key, td.param_index, note))
            slot += 1

        return schema, meta

    @staticmethod
    def _build_output_schema(
        obs_spec: "ObservationSpec",
    ) -> list[OutputSlot]:
        """从 ObservationSpec 构建 output_schema（无数值）。
        通道已由 channel_indices 选定，每个 OutputSlot 仅记录名称。
        """
        return [
            OutputSlot(index=i, name=oi["name"])
            for i, oi in enumerate(obs_spec.selected_output_info)
        ]

    def _build_user_prompt(
        self,
        scenario_name: str,
        domain: dict,
        transform_notes: list,
        ts_shape: tuple,
        language: str,
        style: str,
        obs_spec: "ObservationSpec",
    ) -> tuple[str, list[tuple]]:
        """
        构建发送给 LLM 的用户提示词。

        参数值用占位符 {{value_N}} 表示，LLM 只负责生成语言结构。
        调用方用 _fill_placeholders() 把真实数值替换进去。

        Returns:
            (prompt_str, placeholder_index)
            placeholder_index: list of (placeholder, value, note)
        """
        ch, ts = ts_shape
        scenario_desc = domain["scenarios"].get(scenario_name, scenario_name)
        param_info = domain.get("param_info", {})

        # 用实际观测通道数描述输出
        output_desc = domain["output_description"].format(ch=ch, ts=ts)

        # 构建参数列表文本
        param_lines = []
        placeholder_index = []
        slot = 0

        for name, orig, trans, note_en, note_zh in transform_notes:
            if name in ("scenario_type", "output_type", "complexity"):
                continue
            meaning, unit = param_info.get(name, (name, "-"))
            ph_key = "{value_" + str(slot) + "}"
            ph_display = ph_key
            note = note_en if language == "en" else note_zh
            placeholder_index.append((ph_key, trans, note))
            slot += 1

            if language == "en":
                if note_en:
                    param_lines.append(
                        f"  - {name} ({meaning}): {ph_display} {unit}  ← converted ({note_en})"
                    )
                else:
                    param_lines.append(
                        f"  - {name} ({meaning}): {ph_display} {unit}"
                    )
            else:
                if note_zh:
                    param_lines.append(
                        f"  - {name}（{meaning}）：{ph_display} {unit}  ← 已换算（{note_zh}）"
                    )
                else:
                    param_lines.append(
                        f"  - {name}（{meaning}）：{ph_display} {unit}"
                    )

        param_text = "\n".join(param_lines) if param_lines else "  (no parameters)"

        n_params = len(placeholder_index)
        all_ph_en = ", ".join(ph for ph, _, _ in placeholder_index)
        all_ph_zh = "、".join(ph for ph, _, _ in placeholder_index)

        if language == "en":
            style_map = {
                "technical": "technical and precise, use domain-specific terminology",
                "popular": "accessible to a general audience, use analogies where helpful",
                "concise": "concise and direct, minimal background explanation",
            }
            length_map = {
                "technical": "4-6 sentences",
                "popular":   "4-7 sentences",
                "concise":   "2-3 sentences",
            }
            style_desc  = style_map.get(style, "technical")
            length_desc = length_map.get(style, "3-5 sentences")
            obs_section = (
                f"[Observation setup — weave naturally into text where appropriate]\n"
                f"- Time sampling: {obs_spec.time_desc_en}\n"
                f"- Channels observed: {obs_spec.channel_desc_en}"
            )
            ending_examples = (
                "e.g. \"Please predict the resulting time series.\", "
                "\"Forecast the system response over this period.\", "
                "\"What does the model output look like?\", "
                "\"Estimate the evolution of these quantities.\""
            )
            prompt = f"""Write a natural language instruction for the following physics simulation prediction task.

[Domain]
{domain["domain_context"]}

[Scenario]
{scenario_desc}

[Parameters — {n_params} placeholders: {all_ph_en}]
{param_text}

[Output to predict]
{output_desc}

{obs_section}

[Requirements]
- Output only the final instruction text. Do not use markdown, bullets, code blocks, or explanations.
- Language: English
- Style: {style_desc}
- Length: {length_desc}
- PLACEHOLDER RULES (critical): The final text must contain these exact {n_params} protected tokens once each: {all_ph_en}. Use single braces exactly as shown. Do not replace, skip, rename, duplicate, or invent placeholders.
- For converted parameters (marked ← converted): describe the conversion naturally as a domain expert would — e.g. "scaled by a factor of 3", "at roughly 3× the baseline", "adjusted upward by ~58 units". Avoid mechanical phrasing like "multiplied by 3". Do not mention the original raw value.
- Observation setup: mention time sampling and observed channels naturally if it fits the style; omit if it would feel forced (especially for concise style).
- Ending: close with a prediction request in your own words — vary the phrasing across samples. ({ending_examples})
"""
        else:
            style_map = {
                "technical": "专业技术风格，使用领域专业术语",
                "popular": "科普风格，适合普通读者，适当使用类比",
                "concise": "简洁直接风格，减少背景铺垫",
            }
            length_map = {
                "technical": "4-6句话",
                "popular":   "4-7句话",
                "concise":   "2-3句话",
            }
            style_desc  = style_map.get(style, "专业技术风格")
            length_desc = length_map.get(style, "3-5句话")
            obs_section = (
                f"【观测设置——根据风格自然融入，不必逐字照搬】\n"
                f"- 时间采样：{obs_spec.time_desc_zh}\n"
                f"- 观测通道：{obs_spec.channel_desc_zh}"
            )
            ending_examples = (
                "例如「请预测该系统的时序输出」、「请给出模型的预测结果」、「该系统的演变趋势如何」、「请估算上述量的时间序列」"
            )
            prompt = f"""请为以下物理仿真预测任务撰写一段自然语言指令。

【领域背景】
{domain["domain_context"]}

【仿真场景】
{scenario_desc}

【参数列表——共 {n_params} 个占位符：{all_ph_zh}】
{param_text}

【预测目标】
{output_desc}

{obs_section}

【写作要求】
- 只输出最终指令文本。不要使用 Markdown、列表、代码块或解释说明。
- 语言：中文
- 风格：{style_desc}
- 长度：{length_desc}
- 占位符规则（关键）：最终文本必须且只需包含这些 {n_params} 个受保护 token 各一次：{all_ph_zh}。必须使用这里展示的单层花括号原样复制，不得替换为数字、不得遗漏、不得改名、不得重复、不得凭空增加。
- 对已换算的参数（标注了"已换算"的）：用领域专家的自然表达融入文中，如"约为参考值的3倍"、"较基准偏高约58"、"换算后约为标准值的3倍"，避免机械说"乘以3"或"减去58"，不要提及原始值。
- 观测设置：根据风格自然提及时间采样和观测通道；简洁风格可省略，不必强制出现。
- 结尾：以预测请求收尾，措辞自由发挥，不同样本应有所变化。（{ending_examples}）
"""
        return prompt, placeholder_index

    @staticmethod
    def _fill_placeholders(template: str, placeholder_index: list) -> str:
        """
        将语言模板中的 {value_N} 占位符替换为真实数值，并做完整性校验。

        value 可以是 float 或 list：
          - float → 格式化为 "{value:.5g}"
          - list/ndarray → json.dumps 序列化为列表字符串

        校验规则：
          1. 模板中出现的每个 {value_N}，N 必须在 placeholder_index 范围内
          2. placeholder_index 中每个占位符必须在模板中至少出现一次
        违反则抛 ValueError，由 generate_batch 捕获后重试。
        """
        n_expected = len(placeholder_index)

        found_indices = set(int(m) for m in re.findall(r'\{value_(\d+)\}', template))
        invalid = [n for n in found_indices if n >= n_expected]
        if invalid:
            raise ValueError(
                f"LLM 生成了不存在的占位符: "
                f"{['{value_' + str(n) + '}' for n in sorted(invalid)]}，"
                f"共 {n_expected} 个参数"
            )

        missing = [ph for ph, _, _ in placeholder_index if ph not in template]
        if missing:
            raise ValueError(
                f"LLM 丢弃了 {len(missing)} 个占位符: {missing}"
            )

        result = template
        for ph, value, _ in placeholder_index:
            if isinstance(value, (list, np.ndarray)):
                value_str = json.dumps(
                    value.tolist() if isinstance(value, np.ndarray) else value
                )
            else:
                value_str = f"{float(value):.5g}"
            result = result.replace(ph, value_str)
        return result

    @staticmethod
    def _build_target_prompt(
        output_info: list,
        language: str,
        style: str,
        scenario_desc: str = "",
    ) -> str:
        """
        构建生成 target_template 的 LLM prompt。

        格式规则（Token Router 强制要求）：
          ┌─ 单输出（n=1）─────────────────────────────────────────┐
          │  引导语紧接 {output_0}，之间无空格，之后无任何文字       │
          │  例："本次仿真输出了水力水头{output_0}"                 │
          └────────────────────────────────────────────────────────┘
          ┌─ 多输出（n>1）─────────────────────────────────────────┐
          │  引导语中列举所有物理量名称，所有占位符紧挨着放在末尾    │
          │  占位符之间无任何文字，最后一个占位符后无任何文字        │
          │  例（3输出）："本次仿真输出母线电压幅值、相角及线路      │
          │               有功功率流，数值依次为{output_0}{output_1}{output_2}" │
          └────────────────────────────────────────────────────────┘

        Token Router 触发点 = 引导语末尾（紧接第一个占位符之前的最后几个字）。
        触发后专家模型一次性输出所有占位符对应的数值矩阵。
        """
        n_outputs = len(output_info)
        if n_outputs <= 0:
            raise ValueError(
                "target_template 无法生成：output_info 为空。"
                "请检查 registry 的 observation_config.channel_level/fixed_channels。"
            )
        ph_seq = "".join(f"{{output_{i}}}" for i in range(n_outputs))  # 占位符序列（紧挨）

        if language == "zh":
            style_map = {
                "technical": "专业技术风格，使用领域专业术语",
                "popular": "科普风格，适合普通读者",
                "concise": "简洁直接风格",
            }
            style_desc = style_map.get(style, "专业技术风格")

            output_lines = "\n".join(
                f"  {{output_{i}}}: {o.get('name_zh', o.get('name', '仿真输出'))}（{o.get('unit', '-')}）"
                for i, o in enumerate(output_info)
            )

            # 所有物理量名称，用于引导语中列举
            names_zh = "、".join(o.get("name_zh", o.get("name", f"输出{i}")) for i, o in enumerate(output_info))

            scenario_line = f"\n【仿真场景】{scenario_desc}\n" if scenario_desc else ""

            if n_outputs == 1:
                name0 = output_info[0].get("name_zh", output_info[0].get("name", "仿真输出"))
                examples = (
                    f"  \"本次仿真输出了{name0}{{output_0}}\"\n"
                    f"  \"模型计算得到的{name0}为{{output_0}}\"\n"
                    f"  \"仿真结果中{name0}数据为{{output_0}}\"\n"
                    f"  \"以下为{name0}的预测时序{{output_0}}\""
                )
                rule_multi = ""
            else:
                examples = (
                    f"  \"本次仿真输出{names_zh}，数值依次为{ph_seq}\"\n"
                    f"  \"模型计算得到的{names_zh}结果为{ph_seq}\"\n"
                    f"  \"仿真输出了{names_zh}，各量时序依次为{ph_seq}\""
                )
                rule_multi = (
                    f"7. 【多输出专项规则】所有占位符必须【紧挨着】集中放在引导语末尾，"
                    f"占位符之间不得插入任何文字或标点。\n"
                    f"   正确：...数值依次为{ph_seq}\n"
                    f"   错误：...电压为{{output_0}}，相角为{{output_1}}，功率为{{output_2}}\n"
                )

            return (
                f"请为物理仿真任务的输出结果写一段【引导语】，将以下占位符自然地融入其中。"
                f"{scenario_line}\n"
                f"【最终字符硬约束】你的完整回答必须以这个精确后缀结尾：{ph_seq}\n"
                f"不得在这个后缀中插入空格、标点或任何文字。\n\n"
                f"【必须使用的占位符（共 {n_outputs} 个）】\n"
                f"{ph_seq}\n\n"
                f"【各占位符含义】\n{output_lines}\n\n"
                f"【格式硬性规则——违反任意一条则输出无效】\n"
                f"1. 每个占位符恰好出现一次，不得重复，不得遗漏。\n"
                f"2. 引导语最后一个字符后面直接是第一个占位符 {{output_0}}，中间不能有空格或任何字符。\n"
                f"3. 最后一个占位符之后【不得有任何文字】，整段输出必须以占位符结尾。\n"
                f"4. 占位符原样保留，不得替换为数字或文字描述。\n"
                f"5. 不得凭空创造规定范围之外的占位符。\n"
                f"6. 只输出引导语本身，不要加引号、不要加解释、不要使用 Markdown。\n"
                f"{rule_multi}\n"
                f"【风格：{style_desc}】\n\n"
                f"【示例（不要照抄，仅供格式参考）】\n"
                f"{examples}"
            )
        else:
            style_map = {
                "technical": "technical and precise, use domain terminology",
                "popular": "accessible to a general audience",
                "concise": "concise and direct",
            }
            style_desc = style_map.get(style, "technical")

            output_lines = "\n".join(
                f"  {{output_{i}}}: {o.get('name', 'output')} ({o.get('unit', '-')})"
                for i, o in enumerate(output_info)
            )

            names_en = ", ".join(o.get("name", f"output{i}") for i, o in enumerate(output_info))
            scenario_line = f"\n[Scenario] {scenario_desc}\n" if scenario_desc else ""

            if n_outputs == 1:
                name0 = output_info[0].get("name", "simulation output")
                examples = (
                    f"  \"The simulation result for {name0} is{{output_0}}\"\n"
                    f"  \"The predicted {name0} time series:{{output_0}}\"\n"
                    f"  \"Model output for {name0}:{{output_0}}\"\n"
                    f"  \"The computed {name0} values are{{output_0}}\""
                )
                rule_multi = ""
            else:
                examples = (
                    f"  \"The simulation produces {names_en}, values in order:{ph_seq}\"\n"
                    f"  \"Model outputs for {names_en} are as follows:{ph_seq}\"\n"
                    f"  \"The computed {names_en} time series are:{ph_seq}\""
                )
                rule_multi = (
                    f"7. [Multi-output rule] ALL placeholders must be placed TOGETHER at the end, "
                    f"with NO text or punctuation between them.\n"
                    f"   Correct:  ...values in order:{ph_seq}\n"
                    f"   Wrong:    ...voltage is{{output_0}}, angle is{{output_1}}, power is{{output_2}}\n"
                )

            return (
                f"Write a natural language lead-in phrase for the outputs of a physics simulation, "
                f"incorporating all placeholders below."
                f"{scenario_line}\n"
                f"[Required final suffix] Your complete answer must end with this exact suffix: {ph_seq}\n"
                f"Do not insert spaces, punctuation, or text inside this suffix.\n\n"
                f"[Placeholders — ALL {n_outputs}, placed together at the end]\n"
                f"{ph_seq}\n\n"
                f"[What each placeholder represents]\n{output_lines}\n\n"
                f"[Hard format rules — violating any rule makes your output unusable]\n"
                f"1. Use EVERY placeholder exactly once. Never repeat or omit one.\n"
                f"2. The last character of the lead-in text must be immediately followed by {{output_0}} "
                f"— no space or any character in between.\n"
                f"3. There must be NO text after the last placeholder — the output ends with a placeholder.\n"
                f"4. Keep placeholders verbatim — do not replace with numbers or descriptions.\n"
                f"5. Do NOT invent placeholders beyond the listed ones.\n"
                f"6. Output only the lead-in phrase itself, no quotes, no explanation, no markdown.\n"
                f"{rule_multi}\n"
                f"[Style: {style_desc}]\n\n"
                f"[Examples — for format reference only, do not copy]\n"
                f"{examples}"
            )

    # ════════════════════════════════════════════════════════════
    # 阶段一：make_template()
    # ════════════════════════════════════════════════════════════

    def make_template(
        self,
        simulator: str,
        scenario_name: str,
        param_names: list,
        timeseries_shape: tuple,        # (ch_orig, ts_orig)
        rng: np.random.Generator,
        domain: dict,
        sample_idx: int = 0,
    ) -> TemplateRecord:
        """
        阶段一：纯语言生成，不接触任何实际数值。

        决定：语言、风格、变换类型（不含数值）、观测方案
        调用 LLM 生成：input_template、target_template
        返回：TemplateRecord（可序列化，供阶段二复用）

        Args:
            simulator:        模拟器名称
            scenario_name:    场景名称
            param_names:      参数名列表
            timeseries_shape: 原始时序形状 (ch_orig, ts_orig)
            rng:              随机数生成器
            domain:           领域元数据 dict
            sample_idx:       样本索引（记录用）
        """
        ch_orig, ts_orig = timeseries_shape

        # ── 选择语言和风格 ────────────────────────────────────
        language = "en" if rng.random() < self.language_mix else "zh"
        style = rng.choice(self.styles, p=self.style_weights)

        # ── 决定变换方案（只决定类型和因子，不应用到数值）────────
        descs = self._compute_transform_descs(param_names, domain, rng)

        # ── 生成观测方案（只依赖形状，不依赖数值）────────────────
        output_info = domain.get("output_info", [
            {"name": "output", "description": "simulation output", "unit": "-", "slice": [0, None]}
        ])
        obs_spec = self._sample_observation(
            domain, timeseries_shape, output_info,
        )

        # ── 计算降采样后形状 ──────────────────────────────────
        n_time = len(obs_spec.time_indices)
        n_ch_obs = len(obs_spec.channel_indices)
        ts_obs_shape = (n_ch_obs, n_time)

        output_schema = self._build_output_schema(obs_spec)

        # ── 构建 input prompt（用 descs 代替 transform_notes）────
        # orig/trans 填 1.0 而非 0.0，避免触发 _build_user_prompt 的零值过滤
        # （过滤条件：abs(orig) < 1e-12 and name not in param_info），
        # 确保 placeholder_index 与 placeholder_schema 的条目数完全一致。
        dummy_transform_notes = [
            (td.param_name, 1.0, 1.0, td.note_en, td.note_zh)
            for td in descs
        ]
        input_prompt, placeholder_index = self._build_user_prompt(
            scenario_name, domain,
            dummy_transform_notes, ts_obs_shape, language, style, obs_spec
        )

        # 用 placeholder_index 的参数名顺序重建 placeholder_schema：
        # dummy_transform_notes 用 orig=1.0 避免了零值过滤，因此
        # placeholder_index 与 valid_descs 的顺序严格一一对应。
        # 通过 td.param_index 绑定正确的参数位置，彻底避免顺序错位。
        _META = ("scenario_type", "output_type", "complexity")
        valid_descs = [td for td in descs if td.param_name not in _META]
        rebuilt_schema: list[PlaceholderSlot] = [
            PlaceholderSlot(
                index=slot_idx,
                param_name=valid_descs[slot_idx].param_name,
                param_index=valid_descs[slot_idx].param_index,
                use_transformed=(valid_descs[slot_idx].transform_type is not None),
                fmt="scalar",
            )
            for slot_idx in range(len(placeholder_index))
            if slot_idx < len(valid_descs)
        ]

        # ── LLM 调用：生成 input_template ────────────────────
        input_template = self.llm.generate(
            prompt=input_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).strip()

        # ── LLM 调用：生成 target_template（每次独立调用，保证多样性）────
        # 格式要求：引导语紧接{output_N}，无空格，无后缀（Token Router 友好）
        _scenario_desc = domain["scenarios"].get(scenario_name, "")
        target_prompt = self._build_target_prompt(
            obs_spec.selected_output_info, language, style, _scenario_desc
        )
        target_template = self.llm.generate(
            prompt=target_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=min(float(self.temperature), 0.3),
            max_tokens=200,
        ).strip()

        # ── 校验占位符完整性 ──────────────────────────────────
        # 复用现有校验逻辑（只校验结构，不填值）
        n_expected = len(placeholder_index)
        found_indices = set(int(m) for m in re.findall(r'\{value_(\d+)\}', input_template))
        invalid = [n for n in found_indices if n >= n_expected]
        if invalid:
            raise ValueError(
                f"LLM 生成了不存在的占位符: "
                f"{['{value_' + str(n) + '}' for n in sorted(invalid)]}，"
                f"共 {n_expected} 个参数"
            )
        missing = [ph for ph, _, _ in placeholder_index if ph not in input_template]
        if missing:
            raise ValueError(f"LLM 丢弃了 {len(missing)} 个占位符: {missing}")

        n_out = len(obs_spec.selected_output_info)
        last_ph = f"{{output_{n_out - 1}}}"
        # 校验每个占位符恰好出现一次
        for i in range(n_out):
            ph = f"{{output_{i}}}"
            count = target_template.count(ph)
            if count == 0:
                raise ValueError(f"target_template 缺少输出占位符: {{output_{i}}}")
            if count > 1:
                raise ValueError(f"target_template 中输出占位符 {{output_{i}}} 出现了 {count} 次")
        # 格式校验1：最后一个占位符后面不能有文字（Token Router 要求）
        pos = target_template.rfind(last_ph)
        if pos == -1:
            raise ValueError(f"target_template 缺少末尾占位符: {last_ph}")
        trailing = target_template[pos + len(last_ph):].strip()
        if trailing:
            raise ValueError(
                f"target_template 末尾有多余文字（Token Router 要求以占位符结尾）: '{trailing}'"
            )
        # 格式校验2（多输出）：所有占位符必须紧挨着集中在末尾，之间不能有任何文字
        if n_out > 1:
            ph_seq = "".join(f"{{output_{i}}}" for i in range(n_out))
            if ph_seq not in target_template:
                raise ValueError(
                    f"target_template 多输出占位符未紧挨集中（Token Router 要求）: "
                    f"期望末尾包含 '{ph_seq}'，实际: '{target_template}'"
                )

        # ── 组装 TemplateRecord ───────────────────────────────
        return TemplateRecord(
            input_template=input_template,
            target_template=target_template,
            placeholder_schema=rebuilt_schema,
            output_schema=output_schema,
            transform_descs=descs,
            simulator=simulator,
            scenario=scenario_name,
            language=language,
            style=style,
            time_mode=obs_spec.time_mode_name,
            n_time_points=len(obs_spec.time_indices),
            time_indices=obs_spec.time_indices.tolist(),
            channel_indices=obs_spec.channel_indices.tolist(),
            selected_output_names=[o["name"] for o in obs_spec.selected_output_info],
            timeseries_shape_orig=[ch_orig, ts_orig],
            timeseries_shape_obs=list(ts_obs_shape),
            param_names=param_names,
        )

    def make_template_batch(
        self,
        simulator: str,
        scenario_name: str,
        param_names: list,
        timeseries_shape: tuple,
        n_templates: int,
        domain: dict,
        seed: int = 42,
        output_file=None,
        file_lock: Optional[threading.Lock] = None,
        progress_callback=None,
    ) -> list[TemplateRecord]:
        """
        批量生成 n_templates 条模板（阶段一批量接口）。

        不需要 HDF5 数据，只需要元数据和时序形状。
        支持并发生成（max_workers）。

        Args:
            simulator:        模拟器名称
            scenario_name:    场景名称
            param_names:      参数名列表
            timeseries_shape: (ch_orig, ts_orig)
            n_templates:      生成模板数
            domain:           领域元数据
            seed:             随机种子
            output_file:      若提供，边生成边写入 JSONL（断点续传友好）
            file_lock:        写文件锁

        Returns:
            list[TemplateRecord]，按生成顺序排列
        """
        def _task(i: int) -> tuple[int, Optional[TemplateRecord]]:
            max_retries = max(1, int(getattr(self._llm_template, "max_retries", 3) or 1))
            for attempt in range(max_retries):
                sample_rng = np.random.default_rng(seed + i + attempt * 100_000)
                try:
                    tmpl = self.make_template(
                        simulator=simulator,
                        scenario_name=scenario_name,
                        param_names=param_names,
                        timeseries_shape=timeseries_shape,
                        rng=sample_rng,
                        domain=domain,
                        sample_idx=i,
                    )
                    return i, tmpl
                except InterruptedError:
                    raise
                except ValueError as e:
                    logger.warning(f"Template {i} failed validation (attempt {attempt+1}/{max_retries}): {e}")
                except Exception as e:
                    logger.warning(f"Template {i} failed with exception and will be replaced: {e}")
                    return i, None
            logger.error(f"Template {i} still failed after {max_retries} retries; replacing it")
            return i, None

        def _store_template(i: int, tmpl: TemplateRecord) -> None:
            results_map[i] = tmpl
            if output_file is not None and file_lock is not None:
                line = tmpl.to_json_line() + "\n"
                with file_lock:
                    output_file.write(line)
                    output_file.flush()

        results_map: dict[int, TemplateRecord] = {}
        written_count = 0
        max_retries = max(1, int(getattr(self._llm_template, "max_retries", 3) or 1))
        max_candidates = max(n_templates, n_templates * max(3, max_retries))

        if self.max_workers <= 1:
            with tqdm(total=n_templates, desc=f"Generate {simulator}/{scenario_name}") as pbar:
                for candidate_idx in range(max_candidates):
                    if written_count >= n_templates:
                        break
                    _, tmpl = _task(candidate_idx)
                    if tmpl is not None:
                        _store_template(candidate_idx, tmpl)
                        written_count += 1
                        pbar.update(1)
                    if progress_callback is not None:
                        progress_callback(written_count)
        else:
            worker_count = max(1, int(self.max_workers))
            submitted_count = 0
            futures = {}

            def _submit_next(executor) -> bool:
                nonlocal submitted_count
                if submitted_count >= max_candidates:
                    return False
                futures[executor.submit(_task, submitted_count)] = submitted_count
                submitted_count += 1
                return True

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for _ in range(min(worker_count, max_candidates)):
                    _submit_next(executor)

                with tqdm(total=n_templates, desc=f"Generate {simulator}/{scenario_name}") as pbar:
                    while futures and written_count < n_templates:
                        for future in as_completed(list(futures)):
                            futures.pop(future, None)
                            try:
                                i, tmpl = future.result()
                            except InterruptedError:
                                for pending in futures:
                                    pending.cancel()
                                raise
                            if tmpl is not None and written_count < n_templates:
                                _store_template(i, tmpl)
                                written_count += 1
                                pbar.update(1)
                            if progress_callback is not None:
                                progress_callback(written_count)
                            if written_count < n_templates:
                                _submit_next(executor)
                            break

                for pending in futures:
                    pending.cancel()

        if written_count < n_templates:
            raise RuntimeError(
                f"{simulator}/{scenario_name} only generated {written_count}/{n_templates} valid templates; "
                f"tried {max_candidates} candidates. Lower concurrency or inspect LLM API errors/rate limits."
            )

        return [results_map[i] for i in sorted(results_map)[:n_templates]]
