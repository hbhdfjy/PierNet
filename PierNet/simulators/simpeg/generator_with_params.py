"""
SimPEG 从指定参数批量生成（用于参数空间采样增强）。
"""

import logging
from typing import List, Dict, Any, Tuple

import numpy as np

from PierNet.simulators.simpeg.generator import generate_sample

logger = logging.getLogger(__name__)


def generate_batch_from_params(
    params_array: np.ndarray,
    param_names: List[str],
    cfg: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    从给定参数数组批量生成样本（用于增强阶段）。

    Args:
        params_array: [N, n_params] 参数数组
        param_names:  参数名称列表
        cfg:          配置字典

    Returns:
        (ts_list, params_list)
        ts_list:     成功的时序列表，每个 [1, n_points]
        params_list: 对应的参数数组列表，每个 [n_params]
    """
    ts_list = []
    params_list = []
    total = len(params_array)

    for i in range(total):
        # 将参数数组转换为字典
        params_dict = {name: float(params_array[i, j]) for j, name in enumerate(param_names)}

        # 构建临时配置（覆盖参数范围，使 _sample_params 能采样到指定值）
        tmp_cfg = dict(cfg)
        tmp_cfg["params"] = {}
        for name, val in params_dict.items():
            tmp_cfg["params"][f"{name}_min"] = val
            tmp_cfg["params"][f"{name}_max"] = val

        rng = np.random.default_rng(42 + i)
        ts, _ = generate_sample(tmp_cfg, rng)

        if ts is not None:
            ts_list.append(ts)
            params_list.append(params_array[i])

        if (i + 1) % 50 == 0 or (i + 1) == total:
            logger.info(f"增强批次进度：{i+1}/{total}（已成功 {len(ts_list)} 个）")

    return ts_list, params_list
