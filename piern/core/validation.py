"""
数据质量过滤器：过滤低质量的合成时序样本。

过滤条件：
1. NaN/Inf 比例超过阈值
2. 时序方差过小（常数序列，模型可能未收敛）
3. 水头值超出物理合理范围（模型发散）
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _check_nan_ratio(ts: np.ndarray, max_nan_ratio: float) -> bool:
    """返回 True 表示通过（NaN/Inf 比例未超标）。"""
    nan_ratio = (~np.isfinite(ts)).mean()
    return float(nan_ratio) <= max_nan_ratio


def _check_variance(ts: np.ndarray, min_variance: float) -> bool:
    """返回 True 表示通过（方差足够大，非常数序列）。"""
    return float(np.nanvar(ts)) >= min_variance


def _check_head_range(
    ts: np.ndarray,
    min_head: float,
    max_head: float,
) -> bool:
    """返回 True 表示通过（水头值未发散）。

    只过滤数值明显发散的样本（如 >1000m 或 <-100m），
    不过滤物理上合理的大水头值——水头范围应由参数范围控制。
    """
    valid = np.isfinite(ts)
    if not valid.any():
        return False
    # 使用宽松的发散检测阈值（±1000m），不依赖配置文件中的场景特定阈值
    diverge_max = max(max_head * 10, 1000.0)
    diverge_min = min(min_head * 10, -100.0)
    return bool(ts[valid].min() >= diverge_min and ts[valid].max() <= diverge_max)


def filter_sample(
    timeseries: np.ndarray,
    val_cfg: Dict[str, Any],
) -> bool:
    """
    对单个样本执行全部质量检查。

    Args:
        timeseries: [n_wells, n_timesteps]
        val_cfg: 验证配置（来自 modflow.yaml 的 validation 节）

    Returns:
        True 表示样本通过质量检查，可保留
    """
    if not _check_nan_ratio(timeseries, val_cfg.get("max_nan_ratio", 0.05)):
        return False
    if not _check_variance(timeseries, val_cfg.get("min_variance", 1e-8)):
        return False
    if not _check_head_range(
        timeseries,
        val_cfg.get("min_head_value", -1e6),
        val_cfg.get("max_head_value", 1e6),
    ):
        return False
    return True


def filter_dataset(
    timeseries: np.ndarray,
    params: np.ndarray,
    val_cfg: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对整个数据集批量过滤，返回通过质量检查的子集。

    Args:
        timeseries: [N, n_wells, n_timesteps]
        params: [N, n_params]
        val_cfg: 验证配置

    Returns:
        filtered_timeseries: [M, n_wells, n_timesteps]，M <= N
        filtered_params: [M, n_params]
        keep_mask: [N] 布尔掩码，True 表示保留
    """
    N = timeseries.shape[0]
    keep_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        keep_mask[i] = filter_sample(timeseries[i], val_cfg)

    n_kept = keep_mask.sum()
    n_dropped = N - n_kept
    logger.info(f"质量过滤：保留 {n_kept}/{N} 个样本，丢弃 {n_dropped} 个")

    return timeseries[keep_mask], params[keep_mask], keep_mask
