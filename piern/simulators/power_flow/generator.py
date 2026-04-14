"""
稳态潮流数据生成器（pandapower）。

非线性代数方程组，Newton-Raphson 求解。
输出：节点电压幅值、相角、线路有功功率，形状 (n_bus*2 + n_line, 365)。
"""

import numpy as np
import logging
import warnings
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 网络创建
# ─────────────────────────────────────────────────────────────────────────────

def _create_pandapower_network(scenario: str, params: Dict[str, float]) -> Any:
    """
    根据场景名称创建 pandapower 网络对象并应用静态参数扰动。

    注意：不在此处应用 load_scale，因为年度负荷曲线（_generate_annual_load_profile）
    已经包含了 load_scale。此处只设置基准网络结构相关的参数。
    """
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError:
        raise ImportError("请安装 pandapower：pip install pandapower")

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

    if len(net.ext_grid) > 0:
        net.ext_grid['vm_pu'] = voltage_pu

    if renewable_ratio > 0 and len(net.gen) > 0:
        n_renewable = max(1, int(len(net.gen) * renewable_ratio))
        for i in range(min(n_renewable, len(net.gen))):
            net.gen.at[net.gen.index[i], 'p_mw'] *= (0.7 + 0.3 * renewable_ratio)

    return net


def _generate_annual_load_profile(
    params: Dict[str, float],
    rng: np.random.Generator,
    n_days: int = 365,
) -> np.ndarray:
    """
    生成365天年度负荷曲线。

    由三部分叠加：季节性趋势、周变化（工作日/周末）、随机噪声。

    Returns:
        负荷因子数组 (365,)，值域约 [0.4, 1.2]
    """
    days = np.arange(n_days)
    seasonal_amp = params.get('seasonal_amp', 0.15)
    seasonal = 1.0 + seasonal_amp * np.cos(2 * np.pi * days / 365)
    weekly_drop = params.get('weekly_drop', 0.15)
    weekly = np.where(days % 7 < 5, 1.0, 1.0 - weekly_drop)
    noise_std = params.get('noise_std', 0.03)
    noise = rng.normal(0, noise_std, n_days)
    load_scale = params.get('load_scale', 1.0)
    profile = load_scale * seasonal * weekly * (1.0 + noise)
    return np.clip(profile, 0.3, 1.5).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 仿真核心
# ─────────────────────────────────────────────────────────────────────────────

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

        if np.any(V_bus < 0.7) or np.any(V_bus > 1.3):
            return None
        if np.any(np.isnan(V_bus)) or np.any(np.isnan(P_line)):
            return None

        return {'V_bus': V_bus, 'theta_bus': theta_bus, 'P_line': P_line}

    except Exception as e:
        logger.debug(f"潮流仿真失败: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 参数采样
# ─────────────────────────────────────────────────────────────────────────────

def _sample_params(cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, float]:
    """从配置范围中均匀采样潮流参数。"""
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
# 单样本 / 批量生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """
    生成单个潮流样本。

    Returns:
        (timeseries, params_dict) 或 (None, None) 若失败
    """
    if '_override_params' in cfg:
        params = dict(cfg['_override_params'])
    else:
        params = _sample_params(cfg, rng)
    scenario = cfg.get('scenario_name', 'ieee14_baseload')

    result = _run_powerflow_365d(scenario, params, rng)
    if result is None:
        return None, None

    V     = result['V_bus']
    theta = result['theta_bus']
    P     = result['P_line']
    params['_n_bus']  = float(V.shape[0])
    params['_n_line'] = float(P.shape[0])
    ts = np.concatenate([V, theta, P], axis=0)
    return ts, params


def _generate_single_worker(args):
    """并行生成的工作函数（顶层函数，可被 pickle）。"""
    cfg, seed_offset = args
    rng = np.random.default_rng(seed_offset)
    try:
        return generate_sample(cfg, rng)
    except Exception:
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
    批量生成潮流样本。

    Returns:
        timeseries: [N, n_channels, n_timesteps]
        params_array: [N, n_params]
        param_names: 参数名称列表
    """
    from tqdm import tqdm
    import time

    param_names = _get_param_names_from_config(cfg)
    scenario = cfg.get('scenario_name', 'unknown')
    logger.info(f"参数列表 ({len(param_names)}维): {param_names}")

    if parallel:
        return _generate_batch_parallel(
            cfg, n_samples, seed, max_workers, param_names, scenario,
            progress_callback=progress_callback,
            progress_total=progress_total or n_samples,
        )

    rng = np.random.default_rng(seed)
    ts_list = []
    params_list = []
    attempts = 0
    max_attempts = n_samples * 5
    start_time = time.time()

    with tqdm(total=n_samples, desc=f"生成潮流样本 [pandapower] ({scenario})") as pbar:
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
            rate = len(ts_list) / max(time.time() - start_time, 1e-6)
            pbar.set_postfix({'成功率': f'{len(ts_list)/attempts*100:.1f}%', '速度': f'{rate:.2f}样本/s'})
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
    scenario: str,
    progress_callback=None,
    progress_total: int = 0,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """多线程并行批量生成潮流样本。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import time

    logger.info(f"并行模式：{max_workers} 个线程")
    max_attempts = n_samples * 5
    ts_list = []
    params_list = []
    start_time = time.time()
    batch_size = max_workers * 2
    seed_offset = seed

    with tqdm(total=n_samples, desc=f"生成潮流样本 [pandapower] ({scenario}) [{max_workers}核]") as pbar:
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
                    pbar.set_postfix({'速度': f'{rate:.2f}样本/s'})
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
