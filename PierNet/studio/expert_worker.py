from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_callable(root: Path, entrypoint: str, callable_name: str):
    module_path = (root / entrypoint).resolve()
    if root.resolve() not in module_path.parents:
        raise ValueError("entrypoint 超出专家模型目录")
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("studio_uploaded_expert", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入 {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    spec.loader.exec_module(module)
    target = getattr(module, callable_name, None)
    if target is None or not callable(target):
        raise AttributeError(f"{entrypoint} 中不存在可调用的 {callable_name}")
    return target


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, dict):
        if "outputs" in value:
            value = value["outputs"]
        elif "output" in value:
            value = value["output"]
        else:
            raise ValueError("字典返回值需要包含 outputs 或 output")
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError("专家模型输出必须是数值")
    result = result.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("专家模型输出包含 NaN 或 Inf")
    return result


def _call_per_sample(predict, inputs: np.ndarray) -> np.ndarray:
    rows = [_to_numpy(predict(sample)) for sample in inputs]
    try:
        return np.stack(rows, axis=0)
    except ValueError as exc:
        raise ValueError("逐样本调用返回了不一致的输出形状") from exc


def _call_predict(
    predict,
    inputs: np.ndarray,
    *,
    force_per_sample: bool,
) -> tuple[np.ndarray, str]:
    if force_per_sample:
        return _call_per_sample(predict, inputs), "per_sample"
    try:
        batched = _to_numpy(predict(inputs))
    except Exception as batch_error:
        try:
            return _call_per_sample(predict, inputs), "per_sample"
        except Exception:
            raise batch_error
    if batched.ndim == 0 or batched.shape[0] != inputs.shape[0]:
        try:
            return _call_per_sample(predict, inputs), "per_sample"
        except Exception:
            pass
    return batched, "batch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-sample", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    predict = _load_callable(
        root,
        str(manifest.get("entrypoint") or "model.py"),
        str(manifest.get("callable") or "predict"),
    )
    with np.load(args.input, allow_pickle=False) as payload:
        inputs = np.asarray(payload["inputs"], dtype=np.float32)
    outputs, execution_mode = _call_predict(
        predict,
        inputs,
        force_per_sample=args.per_sample,
    )
    np.savez_compressed(
        args.output,
        outputs=outputs,
        execution_mode=np.asarray(execution_mode),
    )


if __name__ == "__main__":
    main()
