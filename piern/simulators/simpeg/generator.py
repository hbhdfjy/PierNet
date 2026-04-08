"""
SimPEG 地球物理正演生成器（真实 SimPEG 调用版）。

四个场景，全部基于 SimPEG 开源库（版本 0.25.x）：
  dc_resistivity  — 直流电阻率 Wenner 装置，视电阻率曲线 (1, 100)
  mt_sounding     — 大地电磁测深，视电阻率曲线 (1, 100)
  tem_decay       — 时域电磁衰减，归一化 dB/dt 曲线 (1, 100)
  ip_chargeability— 激发极化，视充电率曲线 (1, 100)

椭圆型PDE：∇·(σ∇φ) = -Iδ，σ=电导率，φ=电位
"""

import logging
import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np

# 抑制 SimPEG 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 场景名称常量
# ─────────────────────────────────────────────────────────────────
SCENARIO_DC   = "dc_resistivity"
SCENARIO_MT   = "mt_sounding"
SCENARIO_TEM  = "tem_decay"
SCENARIO_IP   = "ip_chargeability"

VALID_SCENARIOS = {SCENARIO_DC, SCENARIO_MT, SCENARIO_TEM, SCENARIO_IP}

# ─────────────────────────────────────────────────────────────────
# 预构建 survey 对象（在 generate_batch 中复用，避免每样本重建）
# ─────────────────────────────────────────────────────────────────

def _build_dc_survey(a_spacings: np.ndarray):
    """构建 DC Wenner 装置 survey（只依赖极距，与地层参数无关）。"""
    from simpeg.electromagnetics.static import resistivity as dc

    source_list = []
    for a in a_spacings:
        rx = dc.receivers.Dipole(
            locations_m=np.array([[-0.5 * a, 0.0, 0.0]]),
            locations_n=np.array([[0.5 * a, 0.0, 0.0]]),
        )
        src = dc.sources.Dipole(
            [rx],
            location_a=np.array([-1.5 * a, 0.0, 0.0]),
            location_b=np.array([1.5 * a, 0.0, 0.0]),
        )
        source_list.append(src)
    return dc.Survey(source_list)


def _build_mt_survey(frequencies: np.ndarray):
    """构建 MT 1D survey（只依赖频率）。"""
    from simpeg.electromagnetics.natural_source import (
        simulation_1d as mt_1d,
        sources,
        survey as mt_survey,
    )

    loc = np.array([[0.0, 0.0, 0.0]])
    rx = mt_1d.Impedance(
        locations_e=loc,
        orientation="xy",
        component="apparent_resistivity",
    )
    src_list = [sources.PlanewaveXYPrimary([rx], frequency=f) for f in frequencies]
    return mt_survey.Survey(src_list)


def _build_tem_survey(times: np.ndarray, loop_radius: float):
    """构建 TEM 1D 中心回线 survey（只依赖时间道和回线半径）。"""
    import simpeg.electromagnetics.time_domain as tdem

    rx = tdem.receivers.PointMagneticFluxTimeDerivative(
        locations=np.array([[0.0, 0.0, 0.0]]),
        times=times,
        orientation="z",
    )
    src = tdem.sources.CircularLoop(
        receiver_list=[rx],
        location=np.array([0.0, 0.0, 0.0]),
        radius=loop_radius,
        current=1.0,
        waveform=tdem.sources.StepOffWaveform(),
    )
    return tdem.Survey([src])


# ─────────────────────────────────────────────────────────────────
# DC 电阻率正演
# ─────────────────────────────────────────────────────────────────

def _run_dc(
    params: Dict[str, float],
    cfg: Dict[str, Any],
    survey=None,
    a_spacings: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    直流电阻率正演：SimPEG Simulation1DLayers，Wenner 装置。

    Returns:
        视电阻率曲线 [Ω·m]，形状 (1, 100)；失败返回 None
    """
    try:
        from simpeg.electromagnetics.static.resistivity import simulation_1d as dc_1d
        from simpeg import maps

        # 构建3层模型
        rho_bg = 1.0 / max(params.get("sigma_bg", 0.01), 1e-10)
        rho_anomaly = 1.0 / max(params.get("sigma_anomaly", 0.1), 1e-10)
        depth_top = params.get("depth_top", 10.0)
        depth_bottom = params.get("depth_bottom", 50.0)

        h1 = max(depth_top, 0.1)
        h2 = max(depth_bottom - depth_top, 0.1)
        rho = np.array([rho_bg, rho_anomaly, rho_bg])
        thicknesses = np.array([h1, h2])

        # 极距（若未预构建则临时构建）
        if a_spacings is None:
            n_points = cfg.get("n_points", 100)
            survey_length = params.get("survey_length", 1000.0)
            a_spacings = np.logspace(0, np.log10(survey_length), n_points)

        if survey is None:
            survey = _build_dc_survey(a_spacings)

        sim = dc_1d.Simulation1DLayers(
            survey=survey,
            thicknesses=thicknesses,
            rhoMap=maps.IdentityMap(nP=len(rho)),
        )
        rho_a = sim.dpred(rho)

        # 添加噪声
        noise_level = params.get("noise_level", 0.02)
        if noise_level > 0:
            rng_noise = np.random.default_rng()
            noise = rng_noise.normal(0, noise_level * np.abs(rho_a))
            rho_a = rho_a + noise

        rho_a = np.clip(np.abs(rho_a), 0.1, 1e6)
        return rho_a.reshape(1, -1).astype(np.float32)

    except Exception as e:
        logger.debug(f"DC正演失败: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# MT 大地电磁正演
# ─────────────────────────────────────────────────────────────────

def _run_mt(
    params: Dict[str, float],
    cfg: Dict[str, Any],
    survey=None,
    frequencies: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    大地电磁测深正演：SimPEG Simulation1DRecursive，视电阻率曲线。

    Returns:
        视电阻率曲线 [Ω·m]，形状 (1, 100)；失败返回 None
    """
    try:
        from simpeg.electromagnetics.natural_source import simulation_1d as mt_1d
        from simpeg import maps

        # 构建3层模型
        sigma_bg = max(params.get("sigma_bg", 0.01), 1e-10)
        sigma_anomaly = max(params.get("sigma_anomaly", 0.1), 1e-10)
        depth_top = params.get("depth_top", 500.0)
        depth_bottom = params.get("depth_bottom", 2000.0)

        h1 = max(depth_top, 1.0)
        h2 = max(depth_bottom - depth_top, 1.0)
        sigma = np.array([sigma_bg, sigma_anomaly, sigma_bg])
        thicknesses = np.array([h1, h2])

        # 频率范围
        if frequencies is None:
            n_points = cfg.get("n_points", 100)
            freq_min = params.get("freq_min", 1e-3)
            freq_max = params.get("freq_max", 1e3)
            frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)

        if survey is None:
            survey = _build_mt_survey(frequencies)

        sim = mt_1d.Simulation1DRecursive(
            survey=survey,
            thicknesses=thicknesses,
            sigmaMap=maps.IdentityMap(nP=len(sigma)),
        )
        app_res = sim.dpred(sigma)

        # 添加噪声
        noise_level = params.get("noise_level", 0.02)
        if noise_level > 0:
            rng_noise = np.random.default_rng()
            noise = rng_noise.normal(0, noise_level * np.abs(app_res))
            app_res = app_res + noise

        app_res = np.clip(np.abs(app_res), 0.1, 1e6)
        return app_res.reshape(1, -1).astype(np.float32)

    except Exception as e:
        logger.debug(f"MT正演失败: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# TEM 时域电磁正演
# ─────────────────────────────────────────────────────────────────

def _run_tem(
    params: Dict[str, float],
    cfg: Dict[str, Any],
    survey=None,
    times: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    时域电磁衰减正演：SimPEG Simulation1DLayered，中心回线 dB/dt。

    Returns:
        归一化 dB/dt（log10 变换后线性化到 [0,1]），形状 (1, 100)；失败返回 None
    """
    try:
        import simpeg.electromagnetics.time_domain as tdem
        from simpeg import maps

        # 构建3层模型
        sigma_bg = max(params.get("sigma_bg", 0.01), 1e-10)
        sigma_anomaly = max(params.get("sigma_anomaly", 0.1), 1e-10)
        depth_top = params.get("depth_top", 10.0)
        depth_bottom = params.get("depth_bottom", 50.0)
        loop_radius = max(params.get("source_spacing", 50.0), 1.0)

        h1 = max(depth_top, 0.1)
        h2 = max(depth_bottom - depth_top, 0.1)
        sigma = np.array([sigma_bg, sigma_anomaly, sigma_bg])
        thicknesses = np.array([h1, h2])

        # 时间道
        if times is None:
            n_points = cfg.get("n_points", 100)
            times = np.logspace(-6, -2, n_points)

        if survey is None:
            survey = _build_tem_survey(times, loop_radius)

        sim = tdem.Simulation1DLayered(
            survey=survey,
            thicknesses=thicknesses,
            sigmaMap=maps.IdentityMap(nP=len(sigma)),
        )
        dbdt = sim.dpred(sigma)  # shape (n_times,)，负值（dB/dt < 0）

        # 归一化：emf = -dbdt（取正），相对于第一个时间道
        emf = -dbdt
        emf_abs = np.abs(emf)

        # 防止全零
        if emf_abs[0] < 1e-30:
            return None

        emf_norm = emf_abs / emf_abs[0]
        emf_norm = np.clip(emf_norm, 1e-10, 1.0)

        # log10 变换后线性化到 [0, 1]
        log_emf = np.log10(emf_norm)  # 范围 (-inf, 0]
        log_min = np.log10(1e-10)     # = -10
        log_max = 0.0
        emf_out = (log_emf - log_min) / (log_max - log_min)
        emf_out = np.clip(emf_out, 0.0, 1.0)

        # 添加噪声
        noise_level = params.get("noise_level", 0.02)
        if noise_level > 0:
            rng_noise = np.random.default_rng()
            noise = rng_noise.normal(0, noise_level * 0.05)
            emf_out = emf_out + noise
            emf_out = np.clip(emf_out, 0.0, 1.0)

        return emf_out.reshape(1, -1).astype(np.float32)

    except Exception as e:
        logger.debug(f"TEM正演失败: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# IP 激发极化正演（DC 1D × 2 + Cole-Cole）
# ─────────────────────────────────────────────────────────────────

def _cole_cole_sigma(
    sigma_dc: float,
    chargeability: float,
    tau: float,
    c: float = 0.5,
    f: float = 0.3,
) -> float:
    """
    Cole-Cole 模型：计算频率 f 处的等效实部电导率。

    sigma_ip = sigma_dc * (1 - eta*(1 - 1/(1+(i*w*tau)^c)))
    取实部作为极化后的等效电导率。
    """
    omega = 2.0 * np.pi * f
    z = (1j * omega * tau) ** c
    cc = 1.0 - chargeability * (1.0 - 1.0 / (1.0 + z))
    return float(np.real(sigma_dc * cc))


def _run_ip(
    params: Dict[str, float],
    cfg: Dict[str, Any],
    survey=None,
    a_spacings: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    激发极化正演：SimPEG DC 1D × 2（极化前/后）+ Cole-Cole 充电率。

    视充电率 = (rho_ip - rho_dc) / rho_dc

    Returns:
        视充电率曲线 [无量纲]，形状 (1, 100)；失败返回 None
    """
    try:
        from simpeg.electromagnetics.static.resistivity import simulation_1d as dc_1d
        from simpeg import maps

        # 模型参数
        sigma_bg = max(params.get("sigma_bg", 0.01), 1e-10)
        sigma_anomaly = max(params.get("sigma_anomaly", 0.1), 1e-10)
        depth_top = params.get("depth_top", 10.0)
        depth_bottom = params.get("depth_bottom", 50.0)
        chargeability = params.get("chargeability", 0.1)
        tau = max(params.get("time_constant", 0.1), 1e-6)
        chargeability_max = params.get("chargeability_max",
                                       cfg.get("params", {}).get("chargeability_max", 0.5))

        h1 = max(depth_top, 0.1)
        h2 = max(depth_bottom - depth_top, 0.1)
        thicknesses = np.array([h1, h2])

        # 极距
        if a_spacings is None:
            n_points = cfg.get("n_points", 100)
            survey_length = params.get("survey_length", 1000.0)
            a_spacings = np.logspace(0, np.log10(survey_length), n_points)

        if survey is None:
            survey = _build_dc_survey(a_spacings)

        # ── 1. 纯 DC（无极化）──
        rho_dc_arr = np.array([
            1.0 / sigma_bg,
            1.0 / sigma_anomaly,
            1.0 / sigma_bg,
        ])
        sim_dc = dc_1d.Simulation1DLayers(
            survey=survey,
            thicknesses=thicknesses,
            rhoMap=maps.IdentityMap(nP=3),
        )
        rho_a_dc = sim_dc.dpred(rho_dc_arr)

        # ── 2. Cole-Cole 修正后的 sigma（极化后）──
        sigma_bg_ip = _cole_cole_sigma(sigma_bg, chargeability * 0.3, tau)
        sigma_anom_ip = _cole_cole_sigma(sigma_anomaly, chargeability, tau)
        # 保证为正
        sigma_bg_ip = max(sigma_bg_ip, 1e-10)
        sigma_anom_ip = max(sigma_anom_ip, 1e-10)

        rho_ip_arr = np.array([
            1.0 / sigma_bg_ip,
            1.0 / sigma_anom_ip,
            1.0 / sigma_bg_ip,
        ])
        sim_ip = dc_1d.Simulation1DLayers(
            survey=survey,
            thicknesses=thicknesses,
            rhoMap=maps.IdentityMap(nP=3),
        )
        rho_a_ip = sim_ip.dpred(rho_ip_arr)

        # ── 3. 视充电率 ──
        denom = np.abs(rho_a_dc) + 1e-30
        app_chargeability = (rho_a_ip - rho_a_dc) / denom

        # 添加噪声
        noise_level = params.get("noise_level", 0.02)
        if noise_level > 0:
            rng_noise = np.random.default_rng()
            noise = rng_noise.normal(0, noise_level * chargeability * 0.05)
            app_chargeability = app_chargeability + noise

        app_chargeability = np.clip(app_chargeability, 0.0, chargeability_max)
        return app_chargeability.reshape(1, -1).astype(np.float32)

    except Exception as e:
        logger.debug(f"IP正演失败: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# 参数采样
# ─────────────────────────────────────────────────────────────────

def _sample_params(cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, float]:
    """从配置范围中均匀采样一组参数。"""
    p = cfg["params"]
    params = {}
    for key in p:
        if key.endswith("_min"):
            base = key[:-4]
            min_val = p[f"{base}_min"]
            max_val = p[f"{base}_max"]
            params[base] = float(rng.uniform(min_val, max_val))
    return params


def _validate_params(params: Dict[str, float]) -> bool:
    """检查参数物理合理性。"""
    sigma_bg = params.get("sigma_bg", 0.01)
    sigma_anomaly = params.get("sigma_anomaly", 0.1)
    depth_top = params.get("depth_top", 10.0)
    depth_bottom = params.get("depth_bottom", 50.0)

    if sigma_bg <= 0 or sigma_anomaly <= 0:
        return False
    if depth_top <= 0 or depth_bottom <= depth_top:
        return False

    chargeability = params.get("chargeability", None)
    if chargeability is not None and not (0 < chargeability < 1):
        return False

    return True


# ─────────────────────────────────────────────────────────────────
# 单样本生成
# ─────────────────────────────────────────────────────────────────

def generate_sample(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    survey=None,
    measurement_array: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """
    生成单个样本：采样参数 → 正演 → 返回时序。

    Args:
        cfg:               配置字典（来自 YAML）
        rng:               随机数生成器
        survey:            预构建的 SimPEG survey 对象（可复用）
        measurement_array: 极距/频率/时间道数组（与 survey 对应）

    Returns:
        (timeseries [1, n_points], params_dict) 或 (None, None) 若失败
    """
    scenario = cfg.get("scenario", SCENARIO_DC)
    params = _sample_params(cfg, rng)

    if not _validate_params(params):
        return None, None

    if scenario == SCENARIO_DC:
        ts = _run_dc(params, cfg, survey=survey, a_spacings=measurement_array)
    elif scenario == SCENARIO_MT:
        ts = _run_mt(params, cfg, survey=survey, frequencies=measurement_array)
    elif scenario == SCENARIO_TEM:
        ts = _run_tem(params, cfg, survey=survey, times=measurement_array)
    elif scenario == SCENARIO_IP:
        ts = _run_ip(params, cfg, survey=survey, a_spacings=measurement_array)
    else:
        raise ValueError(f"未知场景: {scenario}，有效值: {VALID_SCENARIOS}")

    if ts is None:
        return None, None

    return ts, params


# ─────────────────────────────────────────────────────────────────
# 批量生成
# ─────────────────────────────────────────────────────────────────

def generate_batch(
    cfg: Dict[str, Any],
    n_samples: int,
    seed: int = 42,
    progress_callback=None,
    progress_total: int = 0,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    批量生成 n_samples 个样本。

    在 generate_batch 中预构建 survey 对象（极距/频率/时间固定），
    避免每个样本重建，显著提升性能。

    Args:
        cfg:       配置字典（来自 YAML）
        n_samples: 目标样本数
        seed:      随机种子

    Returns:
        timeseries:   [N, 1, n_points]
        params_array: [N, n_params]
        param_names:  参数名称列表
    """
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    scenario = cfg.get("scenario", SCENARIO_DC)
    n_points = cfg.get("n_points", 100)

    # 从配置中提取参数名
    p = cfg["params"]
    param_names = sorted(set(k[:-4] for k in p if k.endswith("_min")))

    # ── 预构建 survey 对象（与地层参数无关，可安全复用）──
    survey = None
    measurement_array = None

    try:
        if scenario == SCENARIO_DC:
            # 使用配置中 survey_length 范围的中间值构建极距（仅用于构建 survey 结构）
            # 注意：survey 结构（源/接收器位置）与实际极距绑定，每次 _run_dc 会重建
            # 这里预构建一个"默认"极距的 survey，但实际上每次都需要重建（极距随参数变化）
            # 因此 DC/IP 场景的 survey 不能预构建（极距依赖 survey_length 参数）
            # MT/TEM 的频率/时间固定，可以预构建
            survey = None
            measurement_array = None

        elif scenario == SCENARIO_MT:
            # 若 freq_min/freq_max 不随样本变化（配置中无 freq_min/freq_max 参数），
            # 则可以预构建 survey；否则逐样本构建。
            has_variable_freq = "freq_min_min" in p or "freq_max_min" in p
            if not has_variable_freq:
                freq_min = cfg.get("freq_min", 1e-3)
                freq_max = cfg.get("freq_max", 1e3)
                measurement_array = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)
                survey = _build_mt_survey(measurement_array)
                logger.info(f"预构建 MT survey：{n_points} 个频率点，{freq_min:.3g}~{freq_max:.3g} Hz")
            else:
                # freq_min/freq_max 随样本变化，无法预构建
                survey = None
                measurement_array = None

        elif scenario == SCENARIO_TEM:
            # 时间道固定
            measurement_array = np.logspace(-6, -2, n_points)
            # TEM survey 依赖 loop_radius，不同样本 loop_radius 不同，无法统一预构建
            # 改为：预构建一个"中间值"的 survey，但每次仍需重建（loop_radius 随参数变化）
            survey = None
            measurement_array = measurement_array  # 时间道可以共享

        elif scenario == SCENARIO_IP:
            survey = None
            measurement_array = None

    except Exception as e:
        logger.warning(f"预构建 survey 失败，将逐样本构建: {e}")
        survey = None
        measurement_array = None

    ts_list = []
    params_list = []
    attempts = 0
    max_attempts = n_samples * 5

    with tqdm(total=n_samples, desc=f"生成 SimPEG [{scenario}] 样本") as pbar:
        while len(ts_list) < n_samples and attempts < max_attempts:
            ts, params = generate_sample(
                cfg, rng,
                survey=survey,
                measurement_array=measurement_array,
            )
            attempts += 1
            if ts is None:
                continue
            row = [params.get(k, float('nan')) for k in param_names]
            if any(np.isnan(v) for v in row):
                continue
            ts_list.append(ts)
            params_list.append(row)
            pbar.update(1)
            if len(ts_list) % 50 == 0 or len(ts_list) == n_samples:
                logger.info(f"生成进度：{len(ts_list)}/{n_samples}")
                if progress_callback:
                    progress_callback(len(ts_list), progress_total or n_samples)

    if len(ts_list) == 0:
        raise RuntimeError(f"所有 SimPEG 正演均失败，场景: {scenario}")

    timeseries = np.stack(ts_list, axis=0)      # [N, 1, n_points]
    params_array = np.array(params_list, dtype=np.float32)  # [N, n_params]

    logger.info(
        f"成功生成 {len(ts_list)}/{n_samples} 个样本"
        f"（尝试 {attempts} 次，成功率 {len(ts_list)/attempts*100:.1f}%）"
    )
    return timeseries, params_array, param_names
