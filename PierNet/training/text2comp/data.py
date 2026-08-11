"""
文生计算训练数据模块

功能：
1. PromptNumbersDataset: 从JSONL加载 {prompt, label} 格式数据
2. HDF5DataGenerator: 从HDF5物理数据生成训练数据
3. 自动还原输入：从embedding角度处理数据
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import ExpertModelInfo


class PromptNumbersDataset(Dataset):
    """
    文生计算训练数据集

    数据格式：{prompt: str, label: list[float]}
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        expected_len: int | None = None,
        skip_invalid: bool = True,
        system_prompt: str = "",
        response_prefix: str = "好的，科学计算预测结果为：",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expected_len = expected_len
        self.skip_invalid = skip_invalid
        self.system_prompt = system_prompt
        self.response_prefix = response_prefix
        self.samples = []
        self.group_ids: list[str] = []

        # 加载JSONL数据
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    if skip_invalid:
                        continue
                    raise ValueError(f"JSON decode error at line {line_no}: {e}")

                prompt = (item.get("prompt") or "").strip()
                label = item.get("label", [])

                if not prompt:
                    if skip_invalid:
                        continue
                    raise ValueError(f"Empty prompt at line {line_no}")

                # 转换label为float列表
                try:
                    nums = [float(x) for x in label]
                except (TypeError, ValueError) as e:
                    if skip_invalid:
                        continue
                    raise ValueError(f"Invalid label at line {line_no}: {e}")

                # 检查维度
                if expected_len is not None and len(nums) != expected_len:
                    if skip_invalid:
                        continue
                    raise ValueError(
                        f"Label dimension mismatch at line {line_no}: "
                        f"got {len(nums)}, expected {expected_len}"
                    )

                # 包装prompt为chat格式
                wrapped_prompt = self._wrap_prompt(prompt)

                label_tensor = torch.tensor(nums, dtype=torch.float32)
                self.samples.append((wrapped_prompt, label_tensor))
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                sample_index = metadata.get("sample_index")
                if sample_index is None:
                    sample_index = metadata.get("sample_idx")
                if sample_index is None:
                    group_id = f"row:{line_no}"
                else:
                    group_id = ":".join(
                        [
                            str(metadata.get("workflow_id") or ""),
                            str(metadata.get("dataset_id") or ""),
                            str(metadata.get("scenario") or ""),
                            str(sample_index),
                        ]
                    )
                self.group_ids.append(group_id)

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples in {file_path}")

        self.label_dim = expected_len or len(self.samples[0][1])
        print(f"Loaded {len(self.samples)} samples, label_dim={self.label_dim}")

    # 服务器实际的完整system prompt（从de_token_*.py复制）
    DEFAULT_SYSTEM_PROMPT = '''你是一名PDEBench求解助手，专门负责根据用户提供的任务输入进行预测。如果用户输入的是PDEBench求解任务数据（例如1d diff-sorp数据），你将输出下一时刻的预测结果；如果是普通对话任务，则按正常对话回答。

======================================================================
【PDEBench求解任务模式】
触发条件：
- 用户输入包含PDEBench任务数据，例如时间步的数据或场分布。

输出要求：
- 根据输入数据，输出简洁的科学计算预测结果。
示例：
用户输入: "这是1d_diff-reaction任务，请根据以下经过处理的过去1帧32个网格点数据，预测下一帧的状态。调整基数复位：值被增加了 0.05，请还原。数据如下：\n[0.79756, 0.79756, ...]"
强制输出: "好的，科学计算预测结果为：[预测结果]。"

======================================================================
【普通对话模式】
触发条件：用户输入的内容不符合PDEBench求解任务的格式。

输出要求：
- 按普通聊天模式回答一段自然语言。
示例：
用户输入: "你是谁？"
输出: "你好，我是PDEBench求解助手，很高兴为你提供帮助！"

======================================================================
通用要求：
- 回答必须自然流畅。
- 如果是PDEBench任务数据，输出简洁的预测结果；其他情况下按普通对话模式回答。'''

    def _wrap_prompt(self, prompt: str) -> str:
        """将prompt包装为chat格式（使用服务器实际格式）"""
        return self.wrap_prompt_text(
            prompt,
            system_prompt=self.system_prompt,
            response_prefix=self.response_prefix,
        )

    @classmethod
    def wrap_prompt_text(
        cls,
        prompt: str,
        *,
        system_prompt: str = "",
        response_prefix: str = "好的，科学计算预测结果为：",
    ) -> str:
        """Build the exact Text2Comp tokenization input shared by training and inference."""
        effective_system_prompt = system_prompt or cls.DEFAULT_SYSTEM_PROMPT
        return f'''<|im_start|>system
{effective_system_prompt}
<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{response_prefix}'''

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt, label = self.samples[idx]
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label,
        }


class HDF5DataGenerator:
    """
    从HDF5物理数据生成文生计算训练数据

    输入：HDF5文件 (timeseries, params)
    输出：JSONL文件 ({prompt, label}格式)

    流程：
    1. 从HDF5读取时序数据和参数
    2. 使用专家模型信息生成prompt模板
    3. 提取目标数值作为label
    4. 输出JSONL训练数据
    """

    def __init__(
        self,
        expert_info: ExpertModelInfo,
        tokenizer_path: str = "",
        language: str = "zh",
    ):
        self.expert_info = expert_info
        self.language = language

        # 默认prompt模板
        self._setup_prompt_templates()

    def _setup_prompt_templates(self):
        """设置prompt模板"""
        if self.language == "zh":
            self.task_prompt_template = (
                "这是{domain}任务，请根据以下经过处理的{input_desc}，"
                "预测{output_desc}。\n"
                "数据如下：\n{data_str}"
            )
            self.default_system_prompt = (
                "你是一名物理求解助手，专门负责根据用户提供的物理参数进行预测。\n"
                "根据输入数据，输出简洁的科学计算预测结果。"
            )
        else:
            self.task_prompt_template = (
                "This is a {domain} task. Based on the processed {input_desc}, "
                "predict the {output_desc}.\n"
                "Data: {data_str}"
            )
            self.default_system_prompt = (
                "You are a physics solver assistant. "
                "Output concise scientific prediction results based on input data."
            )

    def generate_from_hdf5(
        self,
        h5_path: str,
        output_path: str,
        n_samples: int | None = None,
        input_frames: int = 2,
        output_frames: int = 1,
        precision: int = 4,
        spatial_downsample: int = 1,  # 空间降采样因子
    ) -> dict[str, int]:
        """
        从HDF5生成训练数据

        支持两种HDF5格式:
        1. PiERN格式: timeseries[N,C,T], params[N,P]
        2. PDEBench格式: 每个样本独立group(0000,0001...), data[T,S,C]

        Args:
            h5_path: HDF5文件路径
            output_path: 输出JSONL路径
            n_samples: 生成样本数（None=全部）
            input_frames: 输入帧数
            output_frames: 输出帧数（预测目标）
            precision: 数值精度
            spatial_downsample: 空间降采样因子（如16=从1024点降到64点）

        Returns:
            统计信息：生成样本数、跳过数等
        """
        path = Path(h5_path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        # 读取HDF5，自动检测格式
        with h5py.File(path, "r") as f:
            keys = list(f.keys())

            # 检测格式：PDEBench格式有数字命名的group
            is_pdebench = all(k.isdigit() or (k.startswith("0") and len(k) == 4) for k in keys[:10])

            if is_pdebench:
                # PDEBench格式: 每个样本独立group
                sample_keys = sorted(keys)
                n_total = len(sample_keys)

                # 检查第一个样本结构
                first_sample = f[sample_keys[0]]
                if "data" in first_sample:
                    data_shape = first_sample["data"].shape  # (T, S, C)
                    n_spatial = data_shape[1]
                    n_channels = data_shape[2] if len(data_shape) > 2 else 1
                else:
                    raise ValueError(f"Unknown PDEBench format, keys: {list(first_sample.keys())}")

                # 计算降采样后的空间点数
                n_spatial_downsampled = n_spatial // spatial_downsample

                # 设置output_dim
                if self.expert_info.output_dim == 0:
                    self.expert_info.output_dim = n_spatial_downsampled * n_channels

                self.expert_info.spatial_points = n_spatial_downsampled

            else:
                # PiERN格式: 统一的timeseries数组
                timeseries = f["timeseries"][:]  # [N, C, T]
                n_total = timeseries.shape[0]
                n_channels = timeseries.shape[1]

                if self.expert_info.output_dim == 0:
                    self.expert_info.output_dim = n_channels * output_frames

        if n_samples is None:
            n_samples = n_total

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generated = 0
        skipped = 0

        with h5py.File(path, "r") as f:
            with open(output_path, "w", encoding="utf-8") as fout:
                for i in range(min(n_samples, n_total)):
                    if is_pdebench:
                        # PDEBench格式读取
                        sample_key = sorted(f.keys())[i]
                        sample_data = f[sample_key]["data"][:]  # (T, S, C)
                        ts = sample_data.transpose(2, 1, 0)  # 转换为 (C, S, T)
                    else:
                        # PiERN格式读取
                        ts = f["timeseries"][i]  # (C, T) 或 (C, S, T)

                    # 处理不同维度
                    if ts.ndim == 2:
                        # (C, T) 格式，添加空间维度
                        ts = ts[:, None, :]  # (C, 1, T)
                    elif ts.ndim == 3:
                        # 已经是 (C, S, T) 格式
                        pass
                    else:
                        skipped += 1
                        continue

                    # 空间降采样
                    if spatial_downsample > 1 and ts.shape[1] > 1:
                        ts = ts[:, ::spatial_downsample, :]  # (C, S/downsample, T)

                    n_channels = ts.shape[0]
                    n_spatial = ts.shape[1]
                    n_time = ts.shape[2]

                    # 提取输入数据（前input_frames帧）
                    if input_frames >= n_time:
                        skipped += 1
                        continue

                    # 取所有通道、所有空间点、前input_frames帧
                    input_data = ts[:, :, :input_frames].flatten()

                    # 提取目标数据（预测下一帧）
                    target_frame = input_frames
                    label = ts[:, :, target_frame].flatten().tolist()

                    # 检查NaN/Inf
                    if not np.isfinite(input_data).all() or not np.isfinite(label).all():
                        skipped += 1
                        continue

                    # 生成prompt
                    data_str = self._format_data(input_data, precision)
                    prompt = self._generate_prompt(
                        data_str,
                        input_frames=input_frames,
                        spatial_points=n_spatial * n_channels
                    )

                    # 写入JSONL
                    item = {"prompt": prompt, "label": label}
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    generated += 1

        stats = {
            "total_samples": n_total,
            "generated": generated,
            "skipped": skipped,
            "output_dim": self.expert_info.output_dim,
            "format": "pdebench" if is_pdebench else "piern",
            "spatial_points": self.expert_info.spatial_points,
        }
        print(f"Generated {generated} samples to {output_path}")
        return stats

    def _format_data(self, data: np.ndarray, precision: int) -> str:
        """格式化数值数据为字符串"""
        fmt = f"{{:.{precision}f}}"
        values = [fmt.format(v) for v in data]
        return "[" + ", ".join(values) + "]"

    def _generate_prompt(self, data_str: str, input_frames: int = 2, spatial_points: int = 0) -> str:
        """生成prompt文本"""
        template = self.expert_info.prompt_template or self.task_prompt_template

        # 自动推断spatial_points
        if spatial_points == 0:
            spatial_points = self.expert_info.spatial_points or self.expert_info.output_dim

        # 构建模板参数
        params = {
            "domain": self.expert_info.domain or self.expert_info.name,
            "input_desc": self.expert_info.input_description or f"过去{input_frames}帧{spatial_points}个网格点数据",
            "output_desc": self.expert_info.output_description or "下一帧的状态",
            "data_str": data_str,
            "time_steps": input_frames,
            "spatial_points": spatial_points,
        }

        prompt = template.format(**params)
        return prompt


def generate_text2comp_data(
    h5_path: str,
    output_path: str,
    expert_info: ExpertModelInfo,
    **kwargs,
) -> dict[str, int]:
    """便捷函数：生成文生计算训练数据"""
    generator = HDF5DataGenerator(expert_info)
    return generator.generate_from_hdf5(h5_path, output_path, **kwargs)
