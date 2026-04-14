"""
暂态稳定数据生成器（ANDES）。

完整 DAE 系统，含励磁系统和调速器，隐式梯形法求解。
输出：发电机转子角时序，形状 (n_gen, 1000)。
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
# ANDES 案例文件
# ─────────────────────────────────────────────────────────────────────────────

def _get_andes_case_path(case_name: str) -> str:
    """获取 ANDES 内置案例文件路径。"""
    try:
        import andes
        andes_path = os.path.dirname(andes.__file__)
        cases_dir = os.path.join(andes_path, 'cases')
    except ImportError:
        raise ImportError("请安装 ANDES：pip install andes")

    case_map = {
        'ieee14_fault':   os.path.join(cases_dir, 'ieee14', 'ieee14_fault.xlsx'),
        'ieee14_gentrip': os.path.join(cases_dir, 'ieee14', 'ieee14_gentrip.xlsx'),
        'ieee39_full':    os.path.join(cases_dir, 'ieee39', 'ieee39_full.xlsx'),
        'ieee14':         os.path.join(cases_dir, 'ieee14', 'ieee14_fault.xlsx'),
        'ieee39':         os.path.join(cases_dir, 'ieee39', 'ieee39_full.xlsx'),
    }

    path = case_map.get(case_name)
    if path is None or not os.path.exists(path):
        fallback = os.path.join(cases_dir, 'ieee14', 'ieee14_fault.xlsx')
        logger.warning(f"案例 {case_name} 未找到，使用默认 ieee14_fault")
        return fallback
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 仿真核心
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
    - 故障/跳闸事件（Fault/Toggler）

    Args:
        scenario: 场景名称
        params: 参数字典
        rng: 随机数生成器
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

    if 'ieee39' in scenario:
        case_path = _get_andes_case_path('ieee39_full')
    elif 'load_step' in scenario:
        case_path = _get_andes_case_path('ieee14_fault')
    elif 'trip' in scenario:
        case_path = _get_andes_case_path('ieee14_gentrip')
    else:
        case_path = _get_andes_case_path('ieee14_fault')

    try:
        ss = andes.load(case_path, setup=False, no_output=True)

        load_scale = params.get('load_scale', 1.0)
        if load_scale != 1.0 and hasattr(ss, 'PQ') and ss.PQ.n > 0:
            ss.PQ.p0.v = list(np.array(ss.PQ.p0.v) * load_scale)
            ss.PQ.q0.v = list(np.array(ss.PQ.q0.v) * load_scale)

        inertia_scale = np.clip(params.get('inertia_mean', 5.0) / 5.0, 0.5, 2.0)
        if hasattr(ss, 'GENROU') and ss.GENROU.n > 0 and hasattr(ss.GENROU, 'M'):
            ss.GENROU.M.v = list(np.array(ss.GENROU.M.v) * inertia_scale)

        fault_duration = params.get('fault_duration', 0.1)
        t_fault_on = 1.0
        if hasattr(ss, 'Fault') and ss.Fault.n > 0:
            ss.Fault.tf.v[0] = t_fault_on
            ss.Fault.tc.v[0] = t_fault_on + fault_duration

        ss.setup()

        ss.PFlow.run()
        if not ss.PFlow.converged:
            logger.debug("ANDES 潮流不收敛")
            return None

        ss.TDS.config.tf = tf
        ss.TDS.config.tstep = 0.01
        ss.TDS.init()
        if not ss.TDS.initialized:
            logger.debug("ANDES TDS 初始化失败，跳过此样本")
            return None
        ss.TDS.run()

        if not hasattr(ss, 'GENROU') or ss.GENROU.n == 0:
            logger.debug("无 GENROU 发电机模型")
            return None

        delta_ts = ss.dae.ts.get_data(ss.GENROU.delta)  # (n_steps, n_gen)
        t_sim = np.array(ss.dae.ts.t)
        n_gen = ss.GENROU.n

        if len(t_sim) < 10:
            return None

        t_new = np.linspace(0.0, tf, n_timesteps)
        delta_resampled = np.zeros((n_gen, n_timesteps), dtype=np.float32)
        for i in range(n_gen):
            f = interp1d(t_sim, delta_ts[:, i], kind='linear', fill_value='extrapolate')
            delta_resampled[i] = f(t_new).astype(np.float32)

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
    """从配置范围中均匀采样暂态参数。"""
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
    生成单个暂态稳定样本。

    Returns:
        (timeseries, params_dict) 或 (None, None) 若失败
    """
    if '_override_params' in cfg:
        params = dict(cfg['_override_params'])
    else:
        params = _sample_params(cfg, rng)
    scenario = cfg.get('scenario_name', 'ieee14_fault')
    n_timesteps = cfg.get('n_timesteps', 1000)

    ts = _run_transient_stability_andes(scenario, params, rng, n_timesteps=n_timesteps)
    if ts is None:
        return None, None
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
    批量生成暂态稳定样本。

    Returns:
        timeseries: [N, n_gen, n_timesteps]
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

    with tqdm(total=n_samples, desc=f"生成暂态样本 [ANDES] ({scenario})") as pbar:
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
    """多线程并行批量生成暂态样本。"""
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

    with tqdm(total=n_samples, desc=f"生成暂态样本 [ANDES] ({scenario}) [{max_workers}核]") as pbar:
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
