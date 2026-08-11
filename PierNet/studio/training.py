from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from PierNet.studio.expert import execute_expert

DEFAULT_BASE_MODEL = os.getenv(
    "PIERN_STUDIO_BASE_MODEL",
    "/root/data/PierNet/models/Qwen/Qwen2.5-0.5B-Instruct",
)
MAX_TRAINING_SAMPLES = int(os.getenv("PIERN_STUDIO_MAX_TRAINING_SAMPLES", "256"))
MAX_PROMPTS = int(os.getenv("PIERN_STUDIO_MAX_PROMPTS", "512"))

ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]


class RouterHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings).squeeze(-1)


class Text2CompHead(nn.Module):
    def __init__(self, embedding_dim: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, embeddings: torch.Tensor, parsed_inputs: torch.Tensor) -> torch.Tensor:
        correction = self.network(torch.cat([embeddings, parsed_inputs], dim=-1))
        return parsed_inputs + 0.1 * correction


def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    candidates: list[tuple[int, int]] = []
    for index in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(index)
            candidates.append((int(free), index))
        except Exception:
            candidates.append((0, index))
    return torch.device(f"cuda:{max(candidates)[1]}")


def _embed_texts(
    texts: list[str],
    *,
    base_model: str,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> np.ndarray:
    if not Path(base_model).exists():
        raise FileNotFoundError(f"基础语言模型不存在: {base_model}")
    device = _select_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModel.from_pretrained(base_model, local_files_only=True, dtype=dtype)
    model.to(device)
    model.eval()
    embeddings: list[np.ndarray] = []
    batch_size = 16 if device.type == "cuda" else 4
    try:
        for start in range(0, len(texts), batch_size):
            if cancel and cancel():
                raise InterruptedError("用户取消了训练")
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=160,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model(**encoded)
                hidden = output.last_hidden_state.float()
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
            if progress:
                progress((start + len(batch)) / max(1, len(texts)), "正在理解训练语句")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


def _format_value(value: float) -> str:
    return f"{float(value):.8g}"


def _prompt(goal: str, names: list[str], values: np.ndarray, variant: int) -> str:
    pairs = "；".join(f"{name}={_format_value(value)}" for name, value in zip(names, values, strict=True))
    templates = (
        "请完成{goal}。已知参数：{pairs}。请调用计算模型给出结果。",
        "我需要{goal}，输入条件为：{pairs}。请进行计算。",
        "根据以下参数执行科学计算并返回数值：{pairs}。任务：{goal}。",
        "帮我求解这个问题：{goal}。参数是 {pairs}。",
    )
    return templates[variant % len(templates)].format(goal=goal, pairs=pairs)


NEGATIVE_PROMPTS = (
    "请解释什么是科学计算。",
    "今天天气怎么样？",
    "请把这段文字翻译成英文。",
    "给我写一段项目介绍。",
    "什么是神经网络？",
    "请总结这份资料。",
    "你好，很高兴认识你。",
    "请介绍常见的数据格式。",
)


def prepare_training_data(
    canonical_path: Path,
    output_dir: Path,
    *,
    goal: str,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(canonical_path, allow_pickle=False) as payload:
        inputs = np.asarray(payload["inputs"], dtype=np.float32)
        outputs = np.asarray(payload["outputs"], dtype=np.float32)
        input_names = [str(item) for item in payload["input_names"].tolist()]
        output_names = [str(item) for item in payload["output_names"].tolist()]
    flat_inputs = inputs.reshape(inputs.shape[0], -1)
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        flat_inputs.shape[0],
        size=min(flat_inputs.shape[0], MAX_TRAINING_SAMPLES),
        replace=False,
    )
    prompt_rows: list[dict[str, Any]] = []
    variants = max(1, min(4, math.ceil(128 / max(1, len(selected)))))
    for position, index in enumerate(selected):
        for variant in range(variants):
            prompt_rows.append(
                {
                    "prompt": _prompt(goal, input_names, flat_inputs[index], position + variant),
                    "inputs": flat_inputs[index].astype(float).tolist(),
                    "sample_index": int(index),
                }
            )
            if len(prompt_rows) >= MAX_PROMPTS:
                break
        if len(prompt_rows) >= MAX_PROMPTS:
            break
    recommended_prompt = prompt_rows[0]["prompt"]
    jsonl_path = output_dir / "training_prompts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as output:
        for row in prompt_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(
        output_dir / "training_data.npz",
        prompts=np.asarray([row["prompt"] for row in prompt_rows]),
        inputs=np.asarray([row["inputs"] for row in prompt_rows], dtype=np.float32),
    )
    metadata = {
        "prompt_count": len(prompt_rows),
        "source_samples": int(inputs.shape[0]),
        "input_shape": list(inputs.shape[1:]),
        "output_shape": list(outputs.shape[1:]),
        "input_names": input_names,
        "output_names": output_names,
        "recommended_prompt": recommended_prompt,
        "training_data_path": str(output_dir / "training_data.npz"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _split_indices(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(size)
    split = max(1, min(size - 1, int(size * 0.8)))
    return indices[:split], indices[split:]


def _train_router(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    seed: int,
    cancel: CancelCallback | None,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train_indices, test_indices = _split_indices(len(labels), seed)
    x = torch.from_numpy(embeddings)
    y = torch.from_numpy(labels.astype(np.float32))
    model = RouterHead(embeddings.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(80):
        if cancel and cancel():
            raise InterruptedError("用户取消了训练")
        optimizer.zero_grad()
        loss = loss_fn(model(x[train_indices]), y[train_indices])
        loss.backward()
        optimizer.step()
        if progress and epoch % 8 == 0:
            progress((epoch + 1) / 80, "正在学习识别用户任务")
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x[test_indices]))
        predictions = (probabilities >= 0.5).float()
        accuracy = float((predictions == y[test_indices]).float().mean())
        test_loss = float(loss_fn(model(x[test_indices]), y[test_indices]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "embedding_dim": embeddings.shape[1]}, output_path)
    return {
        "accuracy": accuracy,
        "loss": test_loss,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
    }


def _train_text2comp(
    embeddings: np.ndarray,
    inputs: np.ndarray,
    output_path: Path,
    *,
    seed: int,
    cancel: CancelCallback | None,
    progress: ProgressCallback | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    mean = inputs.mean(axis=0).astype(np.float32)
    std = inputs.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized = ((inputs - mean) / std).astype(np.float32)
    rng = np.random.default_rng(seed)
    noisy = normalized + rng.normal(0, 0.02, size=normalized.shape).astype(np.float32)
    train_indices, test_indices = _split_indices(len(inputs), seed)
    x_embeddings = torch.from_numpy(embeddings)
    x_numeric = torch.from_numpy(noisy)
    targets = torch.from_numpy(normalized)
    hidden_dim = max(32, min(256, inputs.shape[1] * 8))
    model = Text2CompHead(embeddings.shape[1], inputs.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(120):
        if cancel and cancel():
            raise InterruptedError("用户取消了训练")
        optimizer.zero_grad()
        prediction = model(x_embeddings[train_indices], x_numeric[train_indices])
        loss = loss_fn(prediction, targets[train_indices])
        loss.backward()
        optimizer.step()
        if progress and epoch % 12 == 0:
            progress((epoch + 1) / 120, "正在学习生成计算参数")
    model.eval()
    with torch.no_grad():
        prediction = model(x_embeddings[test_indices], x_numeric[test_indices])
        test_mse = float(loss_fn(prediction, targets[test_indices]))
        original = prediction.numpy() * std + mean
        original_mse = float(np.mean((original - inputs[test_indices]) ** 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": embeddings.shape[1],
            "input_dim": inputs.shape[1],
            "hidden_dim": hidden_dim,
            "mean": mean,
            "std": std,
        },
        output_path,
    )
    return (
        {
            "normalized_mse": test_mse,
            "input_mse": original_mse,
            "train_samples": len(train_indices),
            "test_samples": len(test_indices),
        },
        mean,
        std,
    )


def train_project_models(
    training_data_path: Path,
    router_path: Path,
    text2comp_path: Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    seed: int = 42,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    with np.load(training_data_path, allow_pickle=False) as payload:
        positive_prompts = [str(item) for item in payload["prompts"].tolist()]
        positive_inputs = np.asarray(payload["inputs"], dtype=np.float32)
    negative_prompts = [NEGATIVE_PROMPTS[index % len(NEGATIVE_PROMPTS)] for index in range(len(positive_prompts))]
    all_prompts = [*positive_prompts, *negative_prompts]
    embeddings = _embed_texts(
        all_prompts,
        base_model=base_model,
        progress=((lambda value, message: progress(value * 0.45, message)) if progress else None),
        cancel=cancel,
    )
    positive_embeddings = embeddings[: len(positive_prompts)]
    labels = np.concatenate(
        [
            np.ones(len(positive_prompts), dtype=np.float32),
            np.zeros(len(negative_prompts), dtype=np.float32),
        ]
    )
    router_metrics = _train_router(
        embeddings,
        labels,
        router_path,
        seed=seed,
        cancel=cancel,
        progress=((lambda value, message: progress(0.45 + value * 0.2, message)) if progress else None),
    )
    text2comp_metrics, mean, std = _train_text2comp(
        positive_embeddings,
        positive_inputs,
        text2comp_path,
        seed=seed,
        cancel=cancel,
        progress=((lambda value, message: progress(0.65 + value * 0.35, message)) if progress else None),
    )
    return {
        "base_model": base_model,
        "embedding_dim": int(embeddings.shape[1]),
        "input_dim": int(positive_inputs.shape[1]),
        "input_mean": mean.astype(float).tolist(),
        "input_std": std.astype(float).tolist(),
        "router": router_metrics,
        "text2comp": text2comp_metrics,
    }


def extract_named_values(message: str, names: list[str]) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    missing: list[str] = []
    for name in names:
        pattern = rf"(?<![\w]){re.escape(name)}\s*[:=：]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            values.append(float(match.group(1)))
        else:
            values.append(0.0)
            missing.append(name)
    if len(missing) == len(names):
        found = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", message)
        if len(found) >= len(names):
            values = [float(item) for item in found[-len(names) :]]
            missing = []
    return np.asarray(values, dtype=np.float32), missing


def _load_router(path: Path) -> RouterHead:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = RouterHead(int(payload["embedding_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _load_text2comp(path: Path) -> tuple[Text2CompHead, np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = Text2CompHead(
        int(payload["embedding_dim"]),
        int(payload["input_dim"]),
        int(payload["hidden_dim"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return (
        model,
        np.asarray(payload["mean"], dtype=np.float32),
        np.asarray(payload["std"], dtype=np.float32),
    )


def merge_text2comp_inputs(
    parsed: np.ndarray,
    generated: np.ndarray,
    names: list[str],
    missing: list[str],
) -> np.ndarray:
    """Preserve explicit user values and use Text2Comp only for missing fields."""
    missing_names = set(missing)
    return np.asarray(
        [
            float(generated[index]) if name in missing_names else float(parsed[index])
            for index, name in enumerate(names)
        ],
        dtype=np.float32,
    )


def _chart_for_output(output: np.ndarray, names: list[str]) -> dict[str, Any]:
    values = output.astype(float)
    if values.ndim == 0 or values.size == 1:
        return {"kind": "metric", "value": float(values.reshape(-1)[0]), "label": names[0] if names else "结果"}
    if values.ndim == 1:
        return {
            "kind": "line",
            "x": list(range(1, values.size + 1)),
            "series": [{"name": names[0] if names else "计算结果", "values": values.tolist()}],
        }
    if values.ndim == 2:
        return {
            "kind": "heatmap",
            "rows": int(values.shape[0]),
            "columns": int(values.shape[1]),
            "values": values.tolist(),
        }
    flat = values.reshape(-1)
    return {
        "kind": "line",
        "x": list(range(1, flat.size + 1)),
        "series": [{"name": "计算结果", "values": flat.tolist()}],
    }


def run_project_inference(
    *,
    message: str,
    manifest: dict[str, Any],
    expert: dict[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    base_model = str(manifest["base_model"])
    embedding = _embed_texts([message], base_model=base_model)
    router = _load_router(Path(manifest["router_path"]))
    with torch.no_grad():
        confidence = float(torch.sigmoid(router(torch.from_numpy(embedding)))[0])
    input_names = [str(item) for item in manifest["input_names"]]
    parsed, missing = extract_named_values(message, input_names)
    text2comp, mean, std = _load_text2comp(Path(manifest["text2comp_path"]))
    if missing:
        missing_set = set(missing)
        parsed = np.asarray(
            [
                float(mean[index]) if name in missing_set else float(parsed[index])
                for index, name in enumerate(input_names)
            ],
            dtype=np.float32,
        )
    parsed_normalized = (parsed - mean) / std
    with torch.no_grad():
        predicted_normalized = text2comp(
            torch.from_numpy(embedding),
            torch.from_numpy(parsed_normalized.reshape(1, -1)),
        )[0].numpy()
    generated_inputs = predicted_normalized * std + mean
    resolved_inputs = merge_text2comp_inputs(
        parsed,
        generated_inputs,
        input_names,
        missing,
    )
    expert_input = resolved_inputs.reshape((1, *manifest["input_shape"]))
    expert_output = execute_expert(expert, expert_input, work_dir=work_dir)[0]
    flat = expert_output.reshape(-1).astype(float)
    trend = ""
    if flat.size > 1:
        trend = "整体上升" if flat[-1] > flat[0] else "整体下降" if flat[-1] < flat[0] else "整体稳定"
    preview = "，".join(f"{value:.6g}" for value in flat[: min(12, flat.size)])
    missing_copy = f" 未识别到参数：{', '.join(missing)}，已使用训练数据中心值补全。" if missing else ""
    answer = (
        f"计算已完成，共得到 {flat.size} 个数值。"
        f"范围为 {flat.min():.6g} 至 {flat.max():.6g}"
        f"{f'，{trend}' if trend else ''}。前 {min(12, flat.size)} 个数值为：{preview}。"
        f"{missing_copy}"
    )
    return {
        "message": message,
        "answer": answer,
        "routed": confidence >= 0.5,
        "confidence": confidence,
        "inputs": resolved_inputs.astype(float).tolist(),
        "output": expert_output.astype(float).tolist(),
        "chart": _chart_for_output(expert_output, [str(item) for item in manifest.get("output_names", [])]),
        "latency_ms": (time.perf_counter() - started) * 1000,
    }
