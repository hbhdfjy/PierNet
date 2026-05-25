"""
从指定参数数组生成 PyPSA 能源转型样本（用于参数空间采样增强）。
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple

from PierNet.simulators.gcam.pypsa_energy import run_pypsa_energy_transition, validate_output

logger = logging.getLogger(__name__)


def generate_batch_from_params(
    params_array: np.ndarray,
    param_names: List[str],
    cfg: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    从给定参数数组批量生成 PyPSA 样本。

    Args:
        params_array: 参数矩阵 [N, n_params]
        param_names: 参数名称列表
        cfg: 仿真配置

    Returns:
        (ts_list, params_list)
    """
    scenario = cfg.get('scenario_name', 'energy_transition')
    ts_list = []
    params_list = []
    total = len(params_array)

    for i, param_row in enumerate(params_array):
        params_dict = {name: float(param_row[j]) for j, name in enumerate(param_names)}
        output = run_pypsa_energy_transition(params_dict, scenario=scenario)

        if output is not None and validate_output(output):
            ts_list.append(output)
            params_list.append(param_row)

        # 每 20 个或最后一个报告一次进度
        if (i + 1) % 20 == 0 or (i + 1) == total:
            logger.info(f"增强批次进度：{i+1}/{total}（已成功 {len(ts_list)} 个）")

    return ts_list, params_list
