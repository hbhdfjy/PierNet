"""
数据格式转换器：5字段 → 2字段

架构说明：
- Text2Comp模型：文本 → 数值预测（生成专家模型输入参数）
- 专家模型：接收Text2Comp输出 → 计算最终物理结果

原始平台 fill_samples 输出格式（5字段）：
{
    "input": "自然语言描述",
    "number": [原始参数],
    "params_transformed": [变换后参数],
    "target": "引导语+物理时序矩阵",   # 专家模型输出真值，不是 Text2Comp 标签
    "metadata": {...}
}

训练模块期望格式（2字段）：
{
    "prompt": "自然语言描述",
    "label": [预测数值]                # 专家模型输入参数
}

关键概念：
- label 维度 = 专家模型输入维度
- label 必须来自 expert_input / params_transformed / number
- target 是专家模型的物理输出，只能用于验证或端到端评测
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_numbers_from_target(target: str) -> list[float] | None:
    """
    从target文本提取数值矩阵，展平为1D列表

    target格式示例：
    - "模型结果为[[[1.2, 3.4], [5.6, 7.8]]]"
    - "仿真输出温度数据[[1.0, 2.0, 3.0]]"

    Args:
        target: 包含数值矩阵的文本

    Returns:
        展平后的数值列表，或None（解析失败）
    """
    # 匹配JSON数组（支持嵌套）
    # 找到最后一个完整的JSON数组结构
    pattern = r'\[[\s\d\.\,\-\+eE]+\]$'
    match = re.search(pattern, target)

    if match:
        try:
            matrix_str = match.group()
            matrix = json.loads(matrix_str)
            return flatten_matrix(matrix)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, target片段: {target[-100:]}")

    # 回退方案：尝试提取所有数值
    numbers = re.findall(r'[\d\.\-\+eE]+', target)
    if numbers:
        try:
            return [float(n) for n in numbers]
        except ValueError:
            pass

    logger.error(f"无法从target提取数值: {target[:200]}...")
    return None


def flatten_matrix(matrix) -> list[float]:
    """
    展平多维数组为1D列表

    Args:
        matrix: 嵌套列表或数值

    Returns:
        展平后的float列表
    """
    result = []

    if isinstance(matrix, list):
        for item in matrix:
            if isinstance(item, list):
                result.extend(flatten_matrix(item))
            else:
                try:
                    result.append(float(item))
                except (TypeError, ValueError):
                    logger.warning(f"跳过非数值元素: {item}")
    else:
        try:
            result.append(float(matrix))
        except (TypeError, ValueError):
            logger.warning(f"跳过非数值元素: {matrix}")

    return result


def convert_sample(sample: dict) -> dict | None:
    """
    转换单条样本：5字段 → 2字段

    Args:
        sample: 5字段原始样本

    Returns:
        2字段训练样本，或None（转换失败）
    """
    prompt = sample.get("input", "")

    if not prompt:
        logger.warning("样本缺少input字段")
        return None

    # Text2Comp 负责生成专家模型输入参数。Stage 3 的 target 是物理输出，
    # 不能再作为标签。变换后的参数与 input 中出现的数值一致，优先级最高。
    raw_label = sample.get("expert_input")
    label_source = "expert_input"
    if raw_label is None:
        raw_label = sample.get("params_transformed")
        label_source = "params_transformed"
    if raw_label is None:
        raw_label = sample.get("number")
        label_source = "number"

    label = _coerce_expert_input(raw_label)
    if label is None:
        logger.warning("样本缺少有效的专家输入参数，跳过样本")
        return None

    return {
        "prompt": prompt,
        "label": label,
        "expert_input": label,
        "metadata": {
            "schema_name": "piernet.text2comp",
            "schema_version": 1,
            "label_semantics": "expert_input",
            "label_source": label_source,
        },
    }


def _coerce_expert_input(value) -> list[float] | None:
    """Return a finite one-dimensional expert input vector."""
    if value is None or isinstance(value, (str, bytes, bytearray, dict)):
        return None
    flattened = flatten_matrix(value)
    if not flattened or not all(math.isfinite(item) for item in flattened):
        return None
    return flattened


def convert_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    skip_invalid: bool = True,
    expected_dim: int | None = None,
) -> dict:
    """
    批量转换JSONL文件

    Args:
        input_path: 5字段JSONL输入路径
        output_path: 2字段JSONL输出路径
        skip_invalid: 是否跳过无效样本
        expected_dim: 期望的label维度（用于验证）

    Returns:
        统计信息：{converted, skipped, invalid, output_dim}
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    invalid = 0
    output_dims = set()

    with open(input_path, "r", encoding="utf-8") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            for line_no, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    if skip_invalid:
                        invalid += 1
                        continue
                    raise ValueError(f"JSON解析错误 (行{line_no}): {e}")

                result = convert_sample(sample)

                if result is None:
                    if skip_invalid:
                        skipped += 1
                        continue
                    raise ValueError(f"转换失败 (行{line_no})")

                # 验证维度一致性
                label_dim = len(result["label"])
                output_dims.add(label_dim)

                if expected_dim is not None and label_dim != expected_dim:
                    if skip_invalid:
                        logger.warning(f"维度不一致 (行{line_no}): {label_dim} vs {expected_dim}")
                        skipped += 1
                        continue
                    raise ValueError(f"维度不一致 (行{line_no}): {label_dim} vs {expected_dim}")

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                converted += 1

    # 检查维度一致性
    if len(output_dims) > 1:
        logger.warning(f"输出维度不一致: {sorted(output_dims)}")

    final_dim = list(output_dims)[0] if output_dims else 0

    stats = {
        "converted": converted,
        "skipped": skipped,
        "invalid": invalid,
        "output_dim": final_dim,
        "input_file": str(input_path),
        "output_file": str(output_path),
    }

    logger.info(f"转换完成: {converted}条成功, {skipped}条跳过, output_dim={final_dim}")
    return stats


def compute_output_dim_from_h5(
    h5_path: str | Path,
    observation_config: dict | None = None,
) -> int:
    """
    从HDF5数据自动计算输出维度

    公式: output_dim = selected_channels × selected_time_points

    Args:
        h5_path: HDF5文件路径
        observation_config: 观测配置（可选，默认monthly采样）

    Returns:
        输出维度
    """
    import h5py

    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        # 检测格式
        keys = list(f.keys())
        is_pdebench = all(k.isdigit() or (k.startswith("0") and len(k) == 4) for k in keys[:10])

        if is_pdebench:
            # PDEBench格式
            first_key = sorted(keys)[0]
            data = f[first_key]["data"]
            data_shape = data.shape  # (T, S, C)
            n_channels = data_shape[2] if len(data_shape) > 2 else 1
            n_time_steps = data_shape[0]
        else:
            # PiERN格式
            timeseries = f["timeseries"]
            ts_shape = timeseries.shape  # (N, C, T) 或 (N, S, T)
            if len(ts_shape) == 3:
                n_channels = ts_shape[1]
                n_time_steps = ts_shape[2]
            else:
                # (N, T) 格式
                n_channels = 1
                n_time_steps = ts_shape[1]

    # 应用观测配置
    if observation_config:
        fixed_channels = observation_config.get("fixed_channels")
        if fixed_channels is not None:
            n_channels = len(fixed_channels)

        time_mode = observation_config.get("fixed_time_mode", "monthly")
        n_time_points = get_time_points(time_mode, n_time_steps)
    else:
        # 默认：全选通道，月度采样
        n_time_points = get_time_points("monthly", n_time_steps)

    return n_channels * n_time_points


def get_time_points(mode: str, n_time_steps: int) -> int:
    """
    根据时间模式计算采样点数

    Args:
        mode: 时间模式（monthly, weekly, full, every_N）
        n_time_steps: 总时间步数

    Returns:
        采样时间点数
    """
    if mode == "monthly":
        # 12个月
        return min(12, n_time_steps)
    elif mode == "weekly":
        # 52周
        return min(52, n_time_steps)
    elif mode == "full":
        return n_time_steps
    elif mode.startswith("every_"):
        try:
            step = int(mode.split("_")[1])
            return n_time_steps // step
        except ValueError:
            return n_time_steps
    else:
        return n_time_steps
