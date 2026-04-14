"""
从指定参数数组生成电力系统样本（用于参数空间采样增强）。

与 generator.py 的区别：接收已有参数数组，而非从配置采样。
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple

from piern.simulators.power_flow.generator import _run_powerflow_365d
from piern.simulators.transient.generator import _run_transient_stability_andes

logger = logging.getLogger(__name__)


def generate_batch_from_params(
    params_array: np.ndarray,
    param_names: List[str],
    cfg: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    从给定参数数组批量生成样本。

    Args:
        params_array: 参数矩阵 [N, n_params]
        param_names: 参数名称列表
        cfg: 仿真配置

    Returns:
        (ts_list, params_list) 成功样本的时序和参数列表
    """
    ts_list = []
    params_list = []

    # 使用随机 seed 避免多次调用时生成相同的增强样本
    rng = np.random.default_rng()
    scenario = cfg.get('scenario_name', 'ieee14_baseload')
    sim_type = cfg.get('simulation_type', 'powerflow')

    for i, param_row in enumerate(params_array):
        params_dict = {name: float(param_row[j]) for j, name in enumerate(param_names)}

        try:
            if sim_type == 'powerflow':
                result = _run_powerflow_365d(scenario, params_dict, rng)
                if result is None:
                    continue
                V     = result['V_bus']
                theta = result['theta_bus']
                P     = result['P_line']
                ts = np.concatenate([V, theta, P], axis=0)
            elif sim_type == 'transient':
                n_timesteps = cfg.get('n_timesteps', 1000)
                ts = _run_transient_stability_andes(scenario, params_dict, rng, n_timesteps=n_timesteps)
            else:
                continue

            if ts is not None:
                ts_list.append(ts)
                params_list.append(param_row)
        except Exception as e:
            logger.debug(f"样本生成失败: {e}")
            continue

    return ts_list, params_list
