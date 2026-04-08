"""
电力系统数据生成器。

支持两类仿真：
1. 稳态潮流（pandapower.runpp）：非线性代数方程组，Newton-Raphson求解
   输出：各节点24小时电压幅值，形状 (n_bus, 24)
2. 暂态稳定（ANDES）：完整 DAE 系统，含励磁系统和调速器
   输出：发电机转子角时序，形状 (n_gen, 1000)

权威开源工具：
  - pandapower (https://www.pandapower.org)：稳态潮流
  - ANDES (https://github.com/cuihantao/andes)：暂态稳定
"""

import numpy as np
import logging
import os
import warnings
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# 抑制 ANDES 内部日志
logging.getLogger('andes').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# ANDES 案例文件路径
# ─────────────────────────────────────────────────────────────────────────────

def _get_andes_case_path(case_name: str) -> str:
    """获取 ANDES 内置案例文件路径。"""
    try:
        import andes
        andes_path = os.path.dirname(andes.__file__)
        cases_dir = os.path.join(andes_path, 'cases')
    except ImportError:
        raise ImportError("请安装 ANDES：pip install andes")

    # 案例名称到文件路径的映射
    case_map = {
        'ieee14_fault':  os.path.join(cases_dir, 'ieee14', 'ieee14_fault.xlsx'),
        'ieee14_gentrip': os.path.join(cases_dir, 'ieee14', 'ieee14_gentrip.xlsx'),
        'ieee39_full':   os.path.join(cases_dir, 'ieee39', 'ieee39_full.xlsx'),
        'ieee14':        os.path.join(cases_dir, 'ieee14', 'ieee14_fault.xlsx'),
        'ieee39':        os.path.join(cases_dir, 'ieee39', 'ieee39_full.xlsx'),
    }

    path = case_map.get(case_name)
    if path is None or not os.path.exists(path):
        # 默认回退
        fallback = os.path.join(cases_dir, 'ieee14', 'ieee14_fault.xlsx')
        logger.warning(f"案例 {case_name} 未找到，使用默认 ieee14_fault")
        return fallback
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 稳态潮流（pandapower）
# ─────────────────────────────────────────────────────────────────────────────

def _create_pandapower_network(scenario: str, params: Dict[str, float]) -> Any:
    """
    根据场景名称创建 pandapower 网络对象并应用参数扰动。
    """
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError:
        raise ImportError("请安装 pandapower：pip install pandapower")

    load_scale = params.get('load_scale', 1.0)
    voltage_pu = params.get('grid_voltage_pu', 1.0)
    renewable_ratio = params.get('renewable_ratio', 0.0)

    if 'ieee14' in scenario:
        net = pn.case14()
    elif 'ieee30' in scenario:
        net = pn.case30()
    elif 'ieee118' in scenario:
        net = pn.case118()
    elif 'distribution_33bus' in scenario:
        net = pn.case33bw()
    else:
        net = pn.case14()

    if len(net.load) > 0:
        net.load['p_mw'] *= load_scale
        net.load['q_mvar'] *= load_scale

    if len(net.ext_grid) > 0:
        net.ext_grid['vm_pu'] = voltage_pu

    if renewable_ratio > 0 and len(net.gen) > 0:
        n_renewable = max(1, int(len(net.gen) * renewable_ratio))
        for i in range(min(n_renewable, len(net.gen))):
            net.gen.at[net.gen.index[i], 'p_mw'] *= (0.7 + 0.3 * renewable_ratio)

    return net


def _generate_annual_load_profile(params: Dict[str, float], rng: np.random.Generator, n_days: int = 365) -> np.ndarray:
    """
    生成365天年度负荷曲线。

    由三部分叠加：
    - 季节性趋势（年周期正弦）
    - 周变化（工作日 vs 周末）
    - 随机噪声

    Returns:
        负荷因子数组 (365,)，值域约 [0.4, 1.2]
    """
    days = np.arange(n_days)

    # 季节性：冬夏负荷高，春秋低（以冬季为峰值）
    seasonal_amp = params.get('seasonal_amp', 0.15)
    seasonal = 1.0 + seasonal_amp * np.cos(2 * np.pi * days / 365)

    # 周变化：周末负荷下降
    weekly_drop = params.get('weekly_drop', 0.15)
    weekly = np.where(days % 7 < 5, 1.0, 1.0 - weekly_drop)

    # 随机噪声
    noise_std = params.get('noise_std', 0.03)
    noise = rng.normal(0, noise_std, n_days)

    load_scale = params.get('load_scale', 1.0)
    profile = load_scale * seasonal * weekly * (1.0 + noise)
    return np.clip(profile, 0.3, 1.5).astype(np.float32)


def _run_powerflow_365d(
    scenario: str,
    params: Dict[str, float],
    rng: np.random.Generator,
    n_days: int = 365,
) -> Optional[Dict[str, np.ndarray]]:
    """
    运行365天年度潮流仿真（pandapower）。

    每天求解一次稳态潮流，输出三个时序：
      V_bus:     (n_bus, 365)  节点电压幅值 (pu)
      theta_bus: (n_bus, 365)  节点电压相角 (degree)
      P_line:    (n_line, 365) 线路有功功率 (MW)

    Returns:
        {'V_bus': ..., 'theta_bus': ..., 'P_line': ...} 或 None
    """
    try:
        import pandapower as pp
    except ImportError:
        raise ImportError("请安装 pandapower：pip install pandapower")

    annual_profile = _generate_annual_load_profile(params, rng, n_days)

    try:
        net = _create_pandapower_network(scenario, params)
        n_bus = len(net.bus)
        n_line = len(net.line)

        V_bus     = np.zeros((n_bus,  n_days), dtype=np.float32)
        theta_bus = np.zeros((n_bus,  n_days), dtype=np.float32)
        P_line    = np.zeros((n_line, n_days), dtype=np.float32)

        base_p = net.load['p_mw'].values.copy()   if len(net.load) > 0 else np.array([])
        base_q = net.load['q_mvar'].values.copy() if len(net.load) > 0 else np.array([])

        for day in range(n_days):
            lf = float(annual_profile[day])
            if len(net.load) > 0:
                net.load['p_mw']   = base_p * lf
                net.load['q_mvar'] = base_q * lf
            # 第一天冷启动，后续热启动（复用上一天收敛结果，减少迭代次数）
            init = 'auto' if day == 0 else 'results'
            try:
                pp.runpp(net, algorithm='nr', numba=False,
                         max_iteration=50, tolerance_mva=1e-6, init=init)
            except Exception:
                try:
                    pp.runpp(net, algorithm='nr', numba=False,
                             max_iteration=100, tolerance_mva=1e-4, init='auto')
                except Exception:
                    return None
            if not net.converged:
                return None

            V_bus[:, day]     = net.res_bus['vm_pu'].values.astype(np.float32)
            theta_bus[:, day] = net.res_bus['va_degree'].values.astype(np.float32)
            P_line[:, day]    = net.res_line['p_from_mw'].values.astype(np.float32)

        # 质量检查
        if np.any(V_bus < 0.7) or np.any(V_bus > 1.3):
            return None
        if np.any(np.isnan(V_bus)) or np.any(np.isnan(P_line)):
            return None

        return {'V_bus': V_bus, 'theta_bus': theta_bus, 'P_line': P_line}

    except Exception as e:
        logger.debug(f"潮流仿真失败: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 暂态稳定（ANDES）
# ─────────────────────────────────────────────────────────────────────────────

def _run_transient_stability_andes(
    scenario: str,
    params: Dict[str, float],
    rng: np.random.Generator,
    n_timesteps: int = 1000,
    tf: float = 10.0,
) -> Optional[np.ndarray]:
    """
    使用 ANDES 运行暂态稳定仿真。

    ANDES 内置完整的 DAE 求解器，包含：
    - GENROU 发电机模型（4阶/6阶）
    - TGOV1N 调速器
    - IEEEX1 励磁系统
    - 故障/跳闸事件（Toggle/Toggler）

    Args:
        scenario: 场景名称（决定使用哪个 ANDES 内置案例）
        params: 参数字典
        rng: 随机数生成器（用于参数扰动）
        n_timesteps: 输出时间步数（通过重采样得到）
        tf: 仿真时长（秒）

    Returns:
        转子角时序 (n_gen, n_timesteps)；失败返回 None
    """
    try:
        import andes
        from scipy.interpolate import interp1d
    except ImportError:
        raise ImportError("请安装 ANDES：pip install andes\n请安装 scipy：pip install scipy")

    # 选择案例文件
    if 'ieee39' in scenario:
        case_path = _get_andes_case_path('ieee39_full')
    elif 'load_step' in scenario:
        # 负荷阶跃：使用 ieee14_fault（故障持续时间极短）
        case_path = _get_andes_case_path('ieee14_fault')
    elif 'trip' in scenario:
        # 发电机跳闸
        case_path = _get_andes_case_path('ieee14_gentrip')
    else:
        # 默认：ieee14 三相故障
        case_path = _get_andes_case_path('ieee14_fault')

    try:
        # 加载系统（每次仿真重新加载以避免状态污染）
        ss = andes.load(case_path, setup=False, no_output=True)

        # 参数扰动：负荷缩放（PQ.p0.v 是 list，需转为 numpy 再运算）
        load_scale = params.get('load_scale', 1.0)
        if load_scale != 1.0 and hasattr(ss, 'PQ') and ss.PQ.n > 0:
            ss.PQ.p0.v = list(np.array(ss.PQ.p0.v) * load_scale)
            ss.PQ.q0.v = list(np.array(ss.PQ.q0.v) * load_scale)

        # 参数扰动：惯性系数缩放（M.v 是 list）
        inertia_scale = np.clip(params.get('inertia_mean', 5.0) / 5.0, 0.5, 2.0)
        if hasattr(ss, 'GENROU') and ss.GENROU.n > 0 and hasattr(ss.GENROU, 'M'):
            ss.GENROU.M.v = list(np.array(ss.GENROU.M.v) * inertia_scale)

        # 参数扰动：故障持续时间
        # ANDES 用 Fault 模型：tf=故障投入时刻，tc=故障切除时刻
        fault_duration = params.get('fault_duration', 0.1)
        t_fault_on = 1.0
        if hasattr(ss, 'Fault') and ss.Fault.n > 0:
            ss.Fault.tf.v[0] = t_fault_on
            ss.Fault.tc.v[0] = t_fault_on + fault_duration

        ss.setup()

        # 运行潮流
        ss.PFlow.run()
        if not ss.PFlow.converged:
            logger.debug("ANDES 潮流不收敛")
            return None

        # 运行暂态仿真
        ss.TDS.config.tf = tf
        ss.TDS.config.tstep = 0.01
        ss.TDS.init()
        if not ss.TDS.initialized:
            logger.debug("ANDES TDS 初始化失败（可能是调速器限幅问题），跳过此样本")
            return None
        ss.TDS.run()

        # 提取转子角时序
        if not hasattr(ss, 'GENROU') or ss.GENROU.n == 0:
            logger.debug("无 GENROU 发电机模型")
            return None

        delta_ts = ss.dae.ts.get_data(ss.GENROU.delta)  # (n_steps, n_gen)
        t_sim = np.array(ss.dae.ts.t)
        n_gen = ss.GENROU.n

        if len(t_sim) < 10:
            return None

        # 重采样到 n_timesteps 个等间距点
        t_new = np.linspace(0.0, tf, n_timesteps)
        delta_resampled = np.zeros((n_gen, n_timesteps), dtype=np.float32)
        for i in range(n_gen):
            f = interp1d(t_sim, delta_ts[:, i], kind='linear', fill_value='extrapolate')
            delta_resampled[i] = f(t_new).astype(np.float32)

        # 验证：转子角不应发散（>10π 认为不稳定）
        if np.any(np.abs(delta_resampled) > 10 * np.pi):
            logger.debug(f"转子角发散：max={np.abs(delta_resampled).max():.2f} rad")
            return None
        if np.any(np.isnan(delta_resampled)):
            return None

        return delta_resampled

    except Exception as e:
        logger.debug(f"ANDES 暂态仿真失败: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 参数采样
# ─────────────────────────────────────────────────────────────────────────────

def _sample_params(cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, float]:
    """从配置范围中均匀采样电力系统参数。"""
    p = cfg['params']
    params = {}
    for key in p:
        if key.endswith('_min'):
            base = key[:-4]
            params[base] = float(rng.uniform(p[f'{base}_min'], p[f'{base}_max']))
    return params


def _get_param_names_from_config(cfg: Dict[str, Any]) -> list:
    """从配置中提取参数名称列表。"""
    p = cfg['params']
    return sorted([key[:-4] for key in p if key.endswith('_min')])


# ─────────────────────────────────────────────────────────────────────────────
# 单样本生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """
    生成单个电力系统样本。

    稳态潮流使用 pandapower，暂态稳定使用 ANDES。

    Returns:
        (timeseries, params_dict) 或 (None, None) 若失败
    """
    # 支持增强阶段直接注入参数（跳过采样）
    if '_override_params' in cfg:
        params = dict(cfg['_override_params'])
    else:
        params = _sample_params(cfg, rng)
    scenario = cfg.get('scenario_name', 'ieee14_baseload')
    sim_type = cfg.get('simulation_type', 'powerflow')

    if sim_type == 'powerflow':
        result = _run_powerflow_365d(scenario, params, rng)
        if result is None:
            return None, None
        # 将三个数组拼接为 (n_bus*2 + n_line, 365) 的统一张量
        # 但保留分开存储的信息，通过 params 传递形状信息
        # 返回 stacked: (3, max_dim, 365)，用 padding 对齐
        V     = result['V_bus']      # (n_bus, 365)
        theta = result['theta_bus']  # (n_bus, 365)
        P     = result['P_line']     # (n_line, 365)
        # 直接拼接 V 和 theta（同维度），P 单独存
        # 最终返回 (n_bus, 365) 作为主时序（V_bus），其余存 metadata
        # 简单起见：返回 V_bus 作为主时序，theta/P 通过 params 附带形状
        params['_n_bus']  = float(V.shape[0])
        params['_n_line'] = float(P.shape[0])
        # 将三个数组垂直拼接：(n_bus + n_bus + n_line, 365)
        ts = np.concatenate([V, theta, P], axis=0)
    elif sim_type == 'transient':
        n_timesteps = cfg.get('n_timesteps', 1000)
        ts = _run_transient_stability_andes(scenario, params, rng, n_timesteps=n_timesteps)
    else:
        logger.warning(f"未知仿真类型: {sim_type}")
        return None, None

    if ts is None:
        return None, None
    return ts, params


# ─────────────────────────────────────────────────────────────────────────────
# 批量生成
# ─────────────────────────────────────────────────────────────────────────────

def _generate_single_worker(args):
    """并行生成的工作函数（顶层函数，可被 pickle）。"""
    cfg, seed_offset = args
    rng = np.random.default_rng(seed_offset)
    try:
        return generate_sample(cfg, rng)
    except Exception as e:
        return None, None


def generate_batch(
    cfg: Dict[str, Any],
    n_samples: int,
    seed: int = 42,
    parallel: bool = False,
    max_workers: int = 4,
    progress_callback=None,
    progress_total: int = 0,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    批量生成电力系统样本。

    parallel=True 时使用多进程并行，加速约 max_workers 倍。

    Returns:
        timeseries: [N, n_channels, n_timesteps]
        params_array: [N, n_params]
        param_names: 参数名称列表
    """
    from tqdm import tqdm
    import time

    param_names = _get_param_names_from_config(cfg)
    sim_type = cfg.get('simulation_type', 'powerflow')
    tool = 'ANDES' if sim_type == 'transient' else 'pandapower'
    scenario = cfg.get('scenario_name', 'unknown')
    logger.info(f"参数列表 ({len(param_names)}维): {param_names}")

    if parallel:
        return _generate_batch_parallel(cfg, n_samples, seed, max_workers, param_names, tool, scenario,
                                        progress_callback=progress_callback,
                                        progress_total=progress_total or n_samples)

    # 串行生成
    rng = np.random.default_rng(seed)
    ts_list = []
    params_list = []
    attempts = 0
    max_attempts = n_samples * 5
    start_time = time.time()

    with tqdm(total=n_samples, desc=f"生成电力系统样本 [{tool}] ({scenario})") as pbar:
        while len(ts_list) < n_samples and attempts < max_attempts:
            ts, params = generate_sample(cfg, rng)
            attempts += 1
            if ts is None:
                continue
            row = [params.get(k, float('nan')) for k in param_names]
            if any(np.isnan(v) for v in row):
                continue
            ts_list.append(ts)
            params_list.append(row)
            rate = len(ts_list) / (time.time() - start_time) if time.time() > start_time else 0
            pbar.set_postfix({'成功率': f'{len(ts_list)/attempts*100:.1f}%', '速度': f'{rate:.2f}样本/s', '工具': tool})
            pbar.update(1)
            if progress_callback:
                progress_callback(len(ts_list), progress_total or n_samples)

    if len(ts_list) == 0:
        raise RuntimeError(f"所有样本生成失败，请检查配置：{scenario}")

    timeseries = np.stack(ts_list, axis=0)
    params_array = np.array(params_list, dtype=np.float32)
    logger.info(f"生成完成：{len(ts_list)}/{n_samples} 个样本，形状 {timeseries.shape}")
    return timeseries, params_array, param_names


def _generate_batch_parallel(
    cfg: Dict[str, Any],
    n_samples: int,
    seed: int,
    max_workers: int,
    param_names: list,
    tool: str,
    scenario: str,
    progress_callback=None,
    progress_total: int = 0,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """多线程并行批量生成。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import time

    logger.info(f"并行模式：{max_workers} 个线程")
    max_attempts = n_samples * 5

    ts_list = []
    params_list = []
    start_time = time.time()

    # 按批次提交，每批 max_workers*2 个任务，收够样本立即停
    # 避免预先提交全部任务导致 executor.__exit__ 等待所有任务完成
    batch_size = max_workers * 2
    seed_offset = seed
    with tqdm(total=n_samples, desc=f"生成电力系统样本 [{tool}] ({scenario}) [{max_workers}核]") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            attempts = 0
            while len(ts_list) < n_samples and attempts < max_attempts:
                this_batch = min(batch_size, max_attempts - attempts)
                batch_tasks = [(cfg, seed_offset + i) for i in range(this_batch)]
                seed_offset += this_batch
                attempts += this_batch

                futures = [executor.submit(_generate_single_worker, t) for t in batch_tasks]
                for future in as_completed(futures):
                    if len(ts_list) >= n_samples:
                        break
                    try:
                        ts, params = future.result()
                    except Exception:
                        continue
                    if ts is None or params is None:
                        continue
                    row = [params.get(k, float('nan')) for k in param_names]
                    if any(np.isnan(v) for v in row):
                        continue
                    ts_list.append(ts)
                    params_list.append(row)
                    rate = len(ts_list) / max(time.time() - start_time, 1e-6)
                    pbar.set_postfix({'速度': f'{rate:.2f}样本/s', '工具': tool})
                    pbar.update(1)
                    logger.info(f"生成进度：{len(ts_list)}/{n_samples}")
                    if progress_callback:
                        progress_callback(len(ts_list), progress_total or n_samples)
                if len(ts_list) >= n_samples:
                    break

    if len(ts_list) == 0:
        raise RuntimeError(f"所有样本生成失败：{scenario}")

    ts_list = ts_list[:n_samples]
    params_list = params_list[:n_samples]
    timeseries = np.stack(ts_list, axis=0)
    params_array = np.array(params_list, dtype=np.float32)
    logger.info(f"并行生成完成：{len(ts_list)}/{n_samples} 个样本，形状 {timeseries.shape}")
    return timeseries, params_array, param_names
