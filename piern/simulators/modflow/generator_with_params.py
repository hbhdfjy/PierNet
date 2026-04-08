"""
MODFLOW 生成器扩展：支持从指定参数生成样本。

为参数空间采样增强提供支持。
"""

import numpy as np
import tempfile
import logging
from typing import Dict, Any
from .generator import _run_modflow, _get_well_positions

logger = logging.getLogger(__name__)


def generate_sample_from_params(
    params_dict: Dict[str, float],
    cfg: Dict[str, Any],
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """
    从指定参数生成样本（不采样参数）。

    Args:
        params_dict: 指定的参数字典，例如 {"hk": 15.0, "sy": 0.12, ...}
        cfg: MODFLOW 配置
        rng: 随机数生成器（用于非均质场等需要随机性的场景）

    Returns:
        timeseries [n_wells, n_timesteps] 或 None（如果失败）
    """
    # 如果未传入rng，创建一个默认的
    if rng is None:
        rng = np.random.default_rng(42)

    with tempfile.TemporaryDirectory() as work_dir:
        ts = _run_modflow(params_dict, cfg, work_dir, rng)

    return ts


def generate_batch_from_params(
    params_array: np.ndarray,
    param_names: list[str],
    cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    从指定参数数组批量生成样本。

    Args:
        params_array: 参数数组，形状 [N, n_params]
        param_names: 参数名称列表，例如 ["hk", "sy", "pumping", "strt", "rch"]
        cfg: MODFLOW 配置

    Returns:
        timeseries: [N_success, n_wells, n_timesteps]
        params_success: [N_success, n_params]（仅包含成功生成的样本）
    """
    from tqdm import tqdm

    N = params_array.shape[0]
    n_wells = cfg["n_wells"]
    n_timesteps = cfg["n_timesteps"]

    ts_list = []
    params_success_list = []

    with tqdm(total=N, desc="从指定参数生成样本") as pbar:
        for i in range(N):
            # 将参数数组转换为字典
            params_dict = {name: float(params_array[i, j])
                          for j, name in enumerate(param_names)}

            # 运行 MODFLOW
            ts = generate_sample_from_params(params_dict, cfg)

            if ts is not None:
                ts_list.append(ts)
                params_success_list.append(params_array[i])

            pbar.update(1)

    if len(ts_list) == 0:
        logger.warning("所有样本生成失败")
        return np.array([]), np.array([])

    timeseries = np.stack(ts_list, axis=0)
    params_success = np.array(params_success_list, dtype=np.float32)

    success_rate = len(ts_list) / N * 100
    logger.info(f"成功生成 {len(ts_list)}/{N} 个样本（成功率 {success_rate:.1f}%）")

    return timeseries, params_success
