"""HDF5 存储工具：将合成的时序数据集写入 HDF5 文件。"""

import numpy as np
import h5py
import os
from typing import Dict, Any


def save_dataset(
    output_path: str,
    timeseries: np.ndarray,
    params: np.ndarray,
    param_names: list[str],
    metadata: Dict[str, Any] | None = None,
    compression_level: int = 4,
) -> None:
    """
    将合成数据集写入 HDF5 文件。

    Args:
        output_path: 输出文件路径（.h5）
        timeseries: 时序数据，形状 [N, n_wells, n_timesteps]
        params: 参数矩阵，形状 [N, n_params]
        param_names: 参数名称列表，长度 n_params
        metadata: 附加元数据字典（可选）
        compression_level: gzip 压缩级别（1-9，默认4）（优化建议4）
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # 主数据集（支持自定义压缩级别）
        f.create_dataset(
            "timeseries",
            data=timeseries.astype(np.float32),
            compression="gzip",
            compression_opts=compression_level,
            shuffle=True,  # 启用 shuffle 过滤器提高压缩率
        )
        f.create_dataset(
            "params",
            data=params.astype(np.float32),
            compression="gzip",
            compression_opts=compression_level,
            shuffle=True,
        )

        # 参数名称（存为字节字符串）
        f.create_dataset(
            "param_names",
            data=np.array([n.encode("utf-8") for n in param_names]),
        )

        # 元数据
        if metadata:
            meta_grp = f.create_group("metadata")
            for k, v in metadata.items():
                if isinstance(v, (int, float, str, bool)):
                    meta_grp.attrs[k] = v
                elif isinstance(v, list):
                    # 处理字符串列表：转为字节字符串
                    if v and isinstance(v[0], str):
                        meta_grp.create_dataset(k, data=np.array([s.encode("utf-8") for s in v]))
                    else:
                        meta_grp.create_dataset(k, data=np.array(v))
                elif isinstance(v, np.ndarray):
                    meta_grp.create_dataset(k, data=v)

        # 数据集基本信息写入根属性
        # 注意：shape[1] 对不同模拟器含义不同（MODFLOW=观测井数，电力=n_bus*2+n_line，SimPEG=1，GCAM=5）
        # 统一用 n_channels 表示时序的第二维，避免 n_wells 的语义混淆
        f.attrs["n_samples"] = timeseries.shape[0]
        f.attrs["n_channels"] = timeseries.shape[1]
        f.attrs["n_wells"] = timeseries.shape[1]   # 向后兼容旧读取代码，与 n_channels 相同
        f.attrs["n_timesteps"] = timeseries.shape[2]
        f.attrs["n_params"] = params.shape[1]


def load_dataset(
    input_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    从 HDF5 文件加载数据集。

    Returns:
        timeseries: [N, n_wells, n_timesteps]
        params: [N, n_params]
        param_names: 参数名称列表
    """
    with h5py.File(input_path, "r") as f:
        timeseries = f["timeseries"][:]
        params = f["params"][:]
        param_names = [n.decode("utf-8") for n in f["param_names"][:]]
    return timeseries, params, param_names
