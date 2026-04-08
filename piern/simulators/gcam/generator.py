"""
GCAM 领域数据生成器（基于 PyPSA）。

使用 PyPSA（https://github.com/PyPSA/PyPSA）进行多期能源系统规划，
生成 2025-2100 年能源转型时序数据。

输入：18维参数（能源-气候参数）
输出：(5, 16) 时序数组
  - 5个变量：煤炭份额、可再生能源份额、CO2排放、能源价格、温度异常
  - 16个时间步：2025-2100年，5年间隔
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

from piern.simulators.gcam.pypsa_energy import run_pypsa_energy_transition, validate_output

logger = logging.getLogger(__name__)


def _sample_params(cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, float]:
    """从配置范围中均匀采样 GCAM 参数。"""
    p = cfg['params']
    params = {}
    for key in p:
        if key.endswith('_min'):
            base = key[:-4]
            params[base] = float(rng.uniform(p[f'{base}_min'], p[f'{base}_max']))
    return params


def _get_param_names_from_config(cfg: Dict[str, Any]) -> List[str]:
    """从配置中提取参数名称列表。"""
    p = cfg['params']
    return sorted([key[:-4] for key in p if key.endswith('_min')])


def generate_sample(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """
    生成单个 PyPSA 能源转型样本。

    Returns:
        (timeseries [5, 16], params_dict) 或 (None, None) 若失败
    """
    params = _sample_params(cfg, rng)
    scenario = cfg.get('scenario_name', 'energy_transition')

    output = run_pypsa_energy_transition(params, scenario=scenario)

    if output is None or not validate_output(output):
        return None, None

    return output, params


def generate_batch(
    cfg: Dict[str, Any],
    n_samples: int,
    seed: int = 42,
    parallel: bool = False,
    max_workers: int = 4,
    progress_callback=None,
    progress_total: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    批量生成 PyPSA 能源转型样本。

    Returns:
        timeseries: [N, 5, 16]
        params_array: [N, n_params]
        param_names: 参数名称列表
    """
    from tqdm import tqdm
    import time

    rng = np.random.default_rng(seed)
    param_names = _get_param_names_from_config(cfg)
    logger.info(f"PyPSA GCAM 参数列表 ({len(param_names)}维): {param_names}")

    ts_list = []
    params_list = []
    attempts = 0
    max_attempts = n_samples * 5
    start_time = time.time()

    with tqdm(total=n_samples, desc=f"生成 PyPSA 能源样本 ({cfg.get('scenario_name', 'unknown')})") as pbar:
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
            pbar.set_postfix({
                '成功率': f'{len(ts_list)/attempts*100:.1f}%',
                '速度': f'{rate:.2f}样本/s',
                '工具': 'PyPSA',
            })
            pbar.update(1)
            if len(ts_list) % 50 == 0 or len(ts_list) == n_samples:
                logger.info(f"生成进度：{len(ts_list)}/{n_samples}")
                if progress_callback:
                    progress_callback(len(ts_list), progress_total or n_samples)

    if len(ts_list) == 0:
        raise RuntimeError(f"所有 PyPSA 样本生成失败：{cfg.get('scenario_name')}")

    timeseries = np.stack(ts_list, axis=0)
    params_array = np.array(params_list, dtype=np.float32)
    logger.info(f"PyPSA 生成完成：{len(ts_list)}/{n_samples} 个样本，形状 {timeseries.shape}")
    return timeseries, params_array, param_names
