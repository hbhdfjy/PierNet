"""
模板库数据结构：阶段一（模板生成）与阶段二（数值填充）的解耦接口。

阶段一输出 / 阶段二输入：TemplateRecord
  - 只含语言结构（{value_N} / {output_N} 占位符）
  - 不含任何实际数值或时序数据
  - 可序列化存储，供批量复用

阶段二输入（额外）：
  - params: np.ndarray          原始参数（18维）
  - params_transformed: np.ndarray  变换后参数
  - timeseries_obs: np.ndarray  已降采样的时序

文件格式：templates.jsonl（每行一个 TemplateRecord 的 JSON）
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 子结构：描述占位符的结构信息（无数值）
# ─────────────────────────────────────────────────────────────────

@dataclass
class TransformDesc:
    """
    描述单个参数的变换，不含原始值或变换后值。
    用于 prompt 构建和阶段二的变换应用。
    """
    param_name: str
    param_index: int            # 在 param_names 列表中的位置
    transform_type: Optional[str]  # "multiply"|"divide"|"add"|"subtract"|None
    factor: Optional[float]     # 变换因子（乘法/除法的倍数，或加减的偏移量）
    note_en: str                # "multiplied by 3"（空字符串=未变换）
    note_zh: str                # "乘以 3"

    def apply(self, value: float) -> float:
        """将变换应用到原始值，返回变换后值。"""
        if self.transform_type is None or self.factor is None:
            return value
        if self.transform_type == "multiply":
            return value * self.factor
        elif self.transform_type == "divide":
            return value / self.factor if abs(self.factor) > 1e-12 else value
        elif self.transform_type == "add":
            return value + self.factor
        elif self.transform_type == "subtract":
            return value - self.factor
        return value


@dataclass
class PlaceholderSlot:
    """
    描述 {value_N} 占位符的结构，不含实际数值。
    阶段二用此结构从 params_transformed 中取值并格式化。
    """
    index: int          # N，对应 {value_N}
    param_name: str     # 参数名，如 "K_mean"
    param_index: int    # 在 param_names 中的位置
    use_transformed: bool  # True=用 params_transformed，False=用 params 原始值
    fmt: str            # "scalar"（格式化为 .5g）| "list"（json.dumps）


@dataclass
class OutputSlot:
    """
    描述 {output_N} 占位符。阶段二将整个 timeseries_obs 序列化后填入。
    通道选择已在阶段一通过 channel_indices 完成，此处无需再切片。
    """
    index: int   # N，对应 {output_N}
    name: str    # 物理量名，如 "hydraulic_head"


# ─────────────────────────────────────────────────────────────────
# 主结构：TemplateRecord
# ─────────────────────────────────────────────────────────────────

@dataclass
class TemplateRecord:
    """
    阶段一产出的语言模板，不含任何实际数值。

    可序列化为 JSON 行存入 templates.jsonl，
    阶段二读取后与实际数值结合生成最终训练样本。
    """
    # ── 模板内容（LLM 输出）──────────────────────────────────────
    input_template: str
    # 含 {value_N} 占位符的语言结构，例：
    # "该含水层水力传导系数为 {value_0} m/day（已乘以10），月度采样，请预测水头时序。"

    target_template: str
    # 含 {output_N} 占位符的引导语，例：
    # "模型结果为 {output_0}。"

    # ── 占位符结构（阶段二填充所需）────────────────────────────
    placeholder_schema: list[PlaceholderSlot]
    # 每个 {value_N} 对应哪个参数、是否用变换值、如何格式化

    output_schema: list[OutputSlot]
    # 每个 {output_N} 对应哪段时序、如何切片

    transform_descs: list[TransformDesc]
    # 每个参数的变换描述（含因子，无原始值）
    # 阶段二用此从 params 计算 params_transformed

    # ── 生成元信息（调试 / 复用判断）────────────────────────────
    simulator: str
    scenario: str
    language: str           # "en" | "zh"
    style: str              # "technical" | "popular" | "concise"

    # 观测方案（已应用，记录供 metadata 写入）
    time_mode: str
    n_time_points: int
    time_indices: list[int]    # 实际选取的时间步索引（0-based）
    channel_indices: list[int] # 实际选取的通道索引（0-based）
    selected_output_names: list[str]

    # 原始时序形状（用于 metadata）
    timeseries_shape_orig: list[int]    # [ch_orig, ts_orig]
    timeseries_shape_obs: list[int]     # [ch_obs, ts_obs]（降采样后）

    # 参数名列表（阶段二需要知道顺序）
    param_names: list[str]

    # ── 序列化 ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """转换为可 JSON 序列化的字典。"""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TemplateRecord":
        """从字典（JSON 反序列化后）重建 TemplateRecord。兼容旧版模板文件中的多余字段。"""
        import dataclasses
        d = dict(d)
        d["placeholder_schema"] = [PlaceholderSlot(**s) for s in d["placeholder_schema"]]
        d["output_schema"] = [OutputSlot(index=s["index"], name=s["name"]) for s in d["output_schema"]]
        d["transform_descs"] = [TransformDesc(**t) for t in d["transform_descs"]]
        # 过滤掉旧版模板文件中 TemplateRecord 不认识的字段，保持向后兼容
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> "TemplateRecord":
        return cls.from_dict(json.loads(line))


def _compact_output_info_for_metadata(template: TemplateRecord, output_info: list = None) -> list:
    if output_info is None:
        return [s.name for s in template.output_schema]
    if not isinstance(output_info, list) or not template.selected_output_names:
        return output_info

    by_name = {
        str(item.get("name")): item
        for item in output_info
        if isinstance(item, dict) and item.get("name") is not None
    }
    if not by_name:
        return output_info

    try:
        original_channels = int(template.timeseries_shape_orig[0])
    except (TypeError, ValueError, IndexError):
        original_channels = max([int(i) for i in template.channel_indices], default=-1) + 1

    def _as_int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _rows_for_info(info: dict, fallback_index: int) -> list[int]:
        raw_slice = info.get("slice") if isinstance(info, dict) else None
        if isinstance(raw_slice, (list, tuple)) and len(raw_slice) >= 2:
            start = _as_int(raw_slice[0], 0)
            end = original_channels if raw_slice[1] is None else _as_int(raw_slice[1], original_channels)
        else:
            start = fallback_index
            end = fallback_index + 1
        start = max(0, min(start, original_channels))
        end = max(start, min(end, original_channels))
        return list(range(start, end))

    selected = []
    expanded_rows = []
    for name in template.selected_output_names:
        info = by_name.get(str(name))
        if info is None:
            continue
        rows = _rows_for_info(info, len(selected))
        if not rows:
            continue
        selected.append((info, rows))
        expanded_rows.extend(rows)

    channel_indices = [int(i) for i in template.channel_indices]
    if not selected or channel_indices != expanded_rows:
        return output_info

    compact = []
    offset = 0
    for info, rows in selected:
        item = dict(info)
        item["slice"] = [offset, offset + len(rows)]
        compact.append(item)
        offset += len(rows)
    return compact


# ─────────────────────────────────────────────────────────────────
# 阶段二：数值填充（纯本地，不调 LLM）
# ─────────────────────────────────────────────────────────────────

def fill_sample(
    template: TemplateRecord,
    params: np.ndarray,
    timeseries_obs: np.ndarray,
    sample_idx: int,
    output_info: list = None,
    precision: int = 4,
) -> dict:
    """
    阶段二：将实际数值填入模板，生成最终训练样本。

    不调用任何 LLM。

    Args:
        template:        阶段一产出的 TemplateRecord
        params:          原始参数（18维），shape (n_params,)
        timeseries_obs:  已降采样的时序，shape (ch_obs, ts_obs)
        sample_idx:      样本索引（写入 metadata）

    Returns:
        5字段训练样本 dict（input/number/params_transformed/target/metadata）
    """
    # ── 参数维度校验 ──────────────────────────────────────────
    if template.transform_descs:
        max_idx = max(td.param_index for td in template.transform_descs)
        if max_idx >= len(params):
            raise ValueError(
                f"参数维度不匹配：模板期望至少 {max_idx + 1} 维，"
                f"实际 params 长度为 {len(params)}。"
                f"HDF5 文件可能由旧版代码生成，请重新运行 Stage 1。"
            )

    # ── 计算 params_transformed ───────────────────────────────
    params_transformed = params.copy().astype(float)
    for td in template.transform_descs:
        params_transformed[td.param_index] = td.apply(float(params[td.param_index]))

    # ── 填充 input：替换 {value_N} ───────────────────────────
    input_text = template.input_template
    for slot in template.placeholder_schema:
        ph = f"{{value_{slot.index}}}"
        val = params_transformed[slot.param_index] if slot.use_transformed else params[slot.param_index]
        if slot.fmt == "list":
            val_str = json.dumps(
                val.tolist() if isinstance(val, np.ndarray) else list(val),
                ensure_ascii=False,
            )
        else:
            val_str = f"{float(val):.{precision}g}"
        input_text = input_text.replace(ph, val_str)

    # ── 填充 target：替换 {output_N} ─────────────────────────
    # timeseries_obs 已在阶段一按 channel_indices 和 time_indices 切好，直接填入
    # 用与 input 相同的有效数字精度（:.{precision}g）格式化每个数值
    target_text = template.target_template
    if template.output_schema:
        # 预先将整个矩阵按有效数字截断，避免在循环内重复计算
        ts_list = timeseries_obs.tolist()
        ts_rounded = [
            [float(f"{x:.{precision}g}") for x in row]
            for row in ts_list
        ]
        ts_json = json.dumps(ts_rounded, ensure_ascii=False)
        for slot in template.output_schema:
            ph = f"{{output_{slot.index}}}"
            target_text = target_text.replace(ph, ts_json)

    # ── 组装最终样本 ──────────────────────────────────────────
    return {
        "input": input_text,
        "number": params.tolist(),
        "params_transformed": params_transformed.tolist(),
        "target": target_text,
        "metadata": {
            "simulator": template.simulator,
            "scenario": template.scenario,
            "param_names": template.param_names,
            "timeseries_shape": template.timeseries_shape_orig,
            "timeseries_shape_obs": template.timeseries_shape_obs,
            "observation": {
                "time_mode": template.time_mode,
                "n_time_points": template.n_time_points,
                "channel_indices": template.channel_indices,
                "selected_output_names": template.selected_output_names,
            },
            "sample_idx": int(sample_idx),
            "language": template.language,
            "style": template.style,
            "input_template": template.input_template,
            "target_template": template.target_template,
            "output_info": _compact_output_info_for_metadata(template, output_info),
        },
    }


# ─────────────────────────────────────────────────────────────────
# 模板库 I/O
# ─────────────────────────────────────────────────────────────────

def save_templates(templates: list[TemplateRecord], path: Path | str) -> None:
    """将模板列表写入 JSONL 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in templates:
            f.write(t.to_json_line() + "\n")
    logger.info(f"已保存 {len(templates)} 条模板 → {path}")


def load_templates(path: Path | str) -> list[TemplateRecord]:
    """从 JSONL 文件加载模板列表。"""
    path = Path(path)
    templates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                templates.append(TemplateRecord.from_json_line(line))
    logger.info(f"已加载 {len(templates)} 条模板 ← {path}")
    return templates


def fill_dataset(
    templates_path: Path | str,
    h5_path: Path | str,
    output_path: Path | str,
    seed: int = 42,
    n_samples: Optional[int] = None,
) -> int:
    """
    批量阶段二：读取模板库 + HDF5，输出 JSONL 训练样本。

    Args:
        templates_path: templates.jsonl 路径
        h5_path:        HDF5 数据集路径
        output_path:    输出 JSONL 路径
        seed:           随机种子（用于样本-模板匹配）
        n_samples:      生成样本数（None=与模板数相同）

    Returns:
        实际写入的样本数
    """
    from piern.core.storage import load_dataset

    templates = load_templates(templates_path)
    timeseries, params, param_names = load_dataset(str(h5_path))

    n = n_samples if n_samples is not None else len(templates)
    n_avail_t = len(templates)
    n_avail_d = len(params)

    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_order = rng.permutation(n_avail_t)
    d_order = rng.permutation(n_avail_d)

    written = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for i in range(n):
            t_idx = int(t_order[i % n_avail_t])
            d_idx = int(d_order[i % n_avail_d])
            template = templates[t_idx]

            ts = timeseries[d_idx]   # (ch_orig, ts_orig)
            p = params[d_idx]        # (n_params,)

            # 按模板记录的 time_indices 和 channel_indices 切片（防越界）
            time_idx = np.array(template.time_indices)
            valid_time_idx = time_idx[time_idx < ts.shape[1]]
            if len(valid_time_idx) == 0:
                continue
            ts_time = ts[:, valid_time_idx]

            if template.channel_indices is not None:
                ch_arr = np.array(template.channel_indices)
                valid_ch = ch_arr[ch_arr < ts_time.shape[0]]
                if len(valid_ch) == 0:
                    continue
                ts_obs = ts_time[valid_ch, :]
            else:
                ts_obs = ts_time

            # NaN/Inf 检查
            if not np.isfinite(ts_obs).all():
                logger.warning(f"样本 {i} 含 NaN/Inf，跳过")
                continue

            sample = fill_sample(template, p, ts_obs, sample_idx=d_idx)
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    logger.info(f"已填充 {written}/{n} 条样本 → {output_path}")
    return written
