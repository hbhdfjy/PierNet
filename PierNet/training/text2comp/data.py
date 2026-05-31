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

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples in {file_path}")

        self.label_dim = expected_len or len(self.samples[0][1])
        print(f"Loaded {len(self.samples)} samples, label_dim={self.label_dim}")

    DEFAULT_SYSTEM_PROMPT = '''你是一名PDEBench求解助手，专门负责根据用户提供的任务输入进行预测。'''

    def _wrap_prompt(self, prompt: str) -> str:
        """将prompt包装为chat格式"""
        system_prompt = self.system_prompt or self.DEFAULT_SYSTEM_PROMPT
        return f'<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{self.response_prefix}'

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
    """从HDF5物理数据生成文生计算训练数据"""

    def __init__(self, expert_info: ExpertModelInfo, language: str = "zh"):
        self.expert_info = expert_info
        self.language = language

    def generate_from_hdf5(
        self,
        h5_path: str,
        output_path: str,
        n_samples: int | None = None,
        input_frames: int = 2,
        precision: int = 4,
        spatial_downsample: int = 1,
    ) -> dict[str, int]:
        """从HDF5生成训练数据"""
        path = Path(h5_path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generated = 0
        skipped = 0

        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            is_pdebench = all(k.isdigit() or (k.startswith("0") and len(k) == 4) for k in keys[:10])

            if is_pdebench:
                sample_keys = sorted(keys)
                n_total = len(sample_keys)
            else:
                timeseries = f["timeseries"][:]
                n_total = timeseries.shape[0]

            if n_samples is None:
                n_samples = n_total

            with open(output_path, "w", encoding="utf-8") as fout:
                for i in range(min(n_samples, n_total)):
                    try:
                        if is_pdebench:
                            sample_key = sorted(f.keys())[i]
                            sample_data = f[sample_key]["data"][:]
                            ts = sample_data.transpose(2, 1, 0)
                        else:
                            ts = f["timeseries"][i]

                        if ts.ndim == 2:
                            ts = ts[:, None, :]
                        elif ts.ndim != 3:
                            skipped += 1
                            continue

                        if spatial_downsample > 1 and ts.shape[1] > 1:
                            ts = ts[:, ::spatial_downsample, :]

                        n_time = ts.shape[2]

                        if input_frames >= n_time:
                            skipped += 1
                            continue

                        input_data = ts[:, :, :input_frames].flatten()
                        target_frame = input_frames
                        label = ts[:, :, target_frame].flatten().tolist()

                        if not np.isfinite(input_data).all() or not np.isfinite(label).all():
                            skipped += 1
                            continue

                        data_str = "[" + ", ".join([f"{v:.{precision}f}" for v in input_data]) + "]"
                        prompt = f"这是{self.expert_info.domain or self.expert_info.name}任务，数据如下：\n{data_str}"

                        item = {"prompt": prompt, "label": label}
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                        generated += 1
                    except Exception:
                        skipped += 1

        stats = {
            "total_samples": n_total,
            "generated": generated,
            "skipped": skipped,
            "output_dim": self.expert_info.output_dim,
        }
        print(f"Generated {generated} samples to {output_path}")
        return stats


def generate_text2comp_data(
    h5_path: str,
    output_path: str,
    expert_info: ExpertModelInfo,
    **kwargs,
) -> dict[str, int]:
    """便捷函数：生成文生计算训练数据"""
    generator = HDF5DataGenerator(expert_info)
    return generator.generate_from_hdf5(h5_path, output_path, **kwargs)