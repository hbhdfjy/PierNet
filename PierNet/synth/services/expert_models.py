"""Expert-model uploads and Stage 1 HDF5 generation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import yaml

from PierNet.core.storage import save_dataset
from PierNet.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT
from PierNet.synth.services.hdf5_data import validate_hdf5_file, validate_name


EXPERT_SIMULATOR = "expert_model"
EXPERT_INTERFACE = "def predict(inputs: list[float]) -> float | list[float]"
EXPERT_INTERFACE_VERSION = 2
EXPERT_MODEL_ROOT = DATA_ROOT / "expert_models"
EXPERT_MODEL_FILES = EXPERT_MODEL_ROOT / "files"
EXPERT_METADATA_PATH = EXPERT_MODEL_ROOT / "models.json"
EXPERT_CONFIG_ROOT = PROJECT_ROOT / "configs" / EXPERT_SIMULATOR / "variants"
MAX_MODEL_BYTES = 20 * 1024 * 1024
MAX_INPUT_DIM = 32
MAX_INPUT_POINTS = 1_000_000
DEFAULT_EXAMPLE_INPUT = [0.0]
EXPERT_MODEL_CONSTRAINTS = {
    "file": [
        "上传文件必须是单个 Python .py 文件，大小不超过 20 MB。",
        "模块在上传校验和生成数据时会被导入执行；请不要在模块顶层启动长任务、读写大文件或访问外部网络。",
    ],
    "interface": [
        "必须定义可调用函数 predict(inputs)。",
        f"inputs 必须按 list[float] 处理，长度为 1 到 {MAX_INPUT_DIM}，所有数值必须是有限 float。",
        "如果模型需要多维输入，可选定义 EXAMPLE_INPUT = [0.0, 0.0, ...] 作为上传时的最小校验输入。",
    ],
    "output": [
        "predict 必须返回一个有限 float，或一维 finite float 列表/元组/numpy.ndarray。",
        "同一次数据生成中，每个输入点返回的输出维度必须一致。",
    ],
    "dataset": [
        "生成结果会写入 Stage 1 HDF5：params 形状为 [N, input_dim]，timeseries 形状为 [N, output_dim, 1]。",
        f"一次输入规划最多生成 {MAX_INPUT_POINTS} 个点。",
    ],
}
EXPERT_MODEL_EXAMPLE_SOURCE = """# expert_model.py
EXAMPLE_INPUT = [0.0]

def predict(inputs):
    x = float(inputs[0])
    return [x, x * x]
"""

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


class ExpertModelError(ValueError):
    """Raised when uploaded expert model content or prompts are invalid."""


def describe_constraints() -> dict[str, Any]:
    return {
        "interface": EXPERT_INTERFACE,
        "interface_version": EXPERT_INTERFACE_VERSION,
        "constraints": EXPERT_MODEL_CONSTRAINTS,
        "example_source": EXPERT_MODEL_EXAMPLE_SOURCE,
        "max_model_bytes": MAX_MODEL_BYTES,
        "max_input_dim": MAX_INPUT_DIM,
        "max_input_points": MAX_INPUT_POINTS,
    }


def _now() -> float:
    return time.time()


def _safe_stem(value: str) -> str:
    raw = Path(str(value or "expert_model")).stem.strip() or "expert_model"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_-") or "expert_model"
    if not cleaned[0].isalnum():
        cleaned = f"model_{cleaned}"
    return cleaned[:64]


def _read_models() -> list[dict[str, Any]]:
    if not EXPERT_METADATA_PATH.exists():
        return []
    try:
        payload = json.loads(EXPERT_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    models = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(models, list):
        return []
    return [item for item in models if isinstance(item, dict)]


def _write_models(models: list[dict[str, Any]]) -> None:
    EXPERT_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = EXPERT_METADATA_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {"version": 1, **describe_constraints(), "models": models},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tmp.replace(EXPERT_METADATA_PATH)


def list_models() -> list[dict[str, Any]]:
    models = []
    for item in _read_models():
        path = Path(str(item.get("path") or ""))
        copy = dict(item)
        copy["exists"] = bool(path.exists())
        if path.exists():
            copy["file_size_bytes"] = path.stat().st_size
        models.append(copy)
    return sorted(models, key=lambda item: float(item.get("created_at") or 0), reverse=True)


def get_model(model_id: str) -> dict[str, Any]:
    model_id = validate_name("model_id", model_id)
    for item in _read_models():
        if item.get("model_id") == model_id:
            path = Path(str(item.get("path") or ""))
            if not path.exists():
                raise FileNotFoundError(f"专家模型文件不存在: {model_id}")
            return item
    raise KeyError(model_id)


def _load_module(path: Path, model_id: str) -> ModuleType:
    module_name = "PierNet_user_expert_" + hashlib.sha256(f"{model_id}:{path}".encode("utf-8")).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ExpertModelError("无法加载专家模型 Python 文件")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_predict(model: dict[str, Any]) -> Callable[[list[float]], Any]:
    model_id = str(model.get("model_id") or "")
    path = Path(str(model.get("path") or ""))
    module = _load_module(path, model_id)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise ExpertModelError(f"专家模型必须实现接口: {EXPERT_INTERFACE}")
    return predict


def _finite_vector(value: Any, label: str) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ExpertModelError(f"{label} 必须是非空 list[float]")
    if len(value) > MAX_INPUT_DIM:
        raise ExpertModelError(f"{label} 长度不能超过 {MAX_INPUT_DIM}")
    return [_finite_float(item, f"{label}[{idx}]") for idx, item in enumerate(value)]


def validate_model_contract(model: dict[str, Any]) -> dict[str, Any]:
    model_id = str(model.get("model_id") or "")
    path = Path(str(model.get("path") or ""))
    module = _load_module(path, model_id)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise ExpertModelError(f"专家模型必须实现接口: {EXPERT_INTERFACE}")
    example_input = _finite_vector(getattr(module, "EXAMPLE_INPUT", DEFAULT_EXAMPLE_INPUT), "EXAMPLE_INPUT")
    try:
        smoke_output = _normalise_output(predict(list(example_input)))
    except Exception as exc:
        raise ExpertModelError(f"专家模型上传校验失败：predict(EXAMPLE_INPUT) 无法返回合法 float 输出（{exc}）") from exc
    return {
        "example_input": example_input,
        "example_input_dim": len(example_input),
        "smoke_output_dim": len(smoke_output),
    }


def upload_model(name: str, content: bytes) -> dict[str, Any]:
    if not content:
        raise ExpertModelError("上传文件为空")
    if len(content) > MAX_MODEL_BYTES:
        raise ExpertModelError(f"专家模型文件不能超过 {MAX_MODEL_BYTES // (1024 * 1024)} MB")

    stem = _safe_stem(name)
    digest = hashlib.sha256(content + str(time.time_ns()).encode("ascii")).hexdigest()[:12]
    model_id = validate_name("model_id", f"{stem}-{digest}")
    target = EXPERT_MODEL_FILES / f"{model_id}.py"
    EXPERT_MODEL_FILES.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)

    model = {
        "model_id": model_id,
        "name": stem,
        "file_name": target.name,
        "path": str(target),
        "created_at": _now(),
        "file_size_bytes": target.stat().st_size,
        "interface": EXPERT_INTERFACE,
        "interface_version": EXPERT_INTERFACE_VERSION,
        "constraints_version": EXPERT_INTERFACE_VERSION,
    }
    try:
        model.update(validate_model_contract(model))
    except Exception:
        target.unlink(missing_ok=True)
        raise

    models = [item for item in _read_models() if item.get("model_id") != model_id]
    models.append(model)
    _write_models(models)
    return model


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExpertModelError(f"{label} 必须是 float") from exc
    if not math.isfinite(number):
        raise ExpertModelError(f"{label} 必须是有限 float")
    return number


def _finite_int(value: Any, label: str, *, min_value: int, max_value: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ExpertModelError(f"{label} 必须是整数") from exc
    if number < min_value or number > max_value:
        raise ExpertModelError(f"{label} 必须在 {min_value} 到 {max_value} 之间")
    return number


def _extract_first(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _matrix_from_values(values: Any) -> np.ndarray:
    if not isinstance(values, list) or not values:
        raise ExpertModelError("values 必须是非空数组")
    rows: list[list[float]] = []
    for idx, row in enumerate(values):
        if isinstance(row, (int, float)):
            rows.append([_finite_float(row, f"values[{idx}]")])
            continue
        if not isinstance(row, list) or not row:
            raise ExpertModelError(f"values[{idx}] 必须是 float 或 float 数组")
        rows.append([_finite_float(value, f"values[{idx}][{j}]") for j, value in enumerate(row)])
    dim = len(rows[0])
    if dim < 1 or dim > MAX_INPUT_DIM:
        raise ExpertModelError(f"输入维度必须在 1 到 {MAX_INPUT_DIM} 之间")
    if len(rows) > MAX_INPUT_POINTS:
        raise ExpertModelError(f"输入点数不能超过 {MAX_INPUT_POINTS}")
    if any(len(row) != dim for row in rows):
        raise ExpertModelError("values 中每一行的输入维度必须一致")
    return np.asarray(rows, dtype=np.float32)


def _matrix_from_linear(count: int, start: float, step: float, input_dim: int) -> np.ndarray:
    base = start + np.arange(count, dtype=np.float32) * np.float32(step)
    if input_dim == 1:
        return base.reshape(count, 1).astype(np.float32)
    return np.repeat(base.reshape(count, 1), input_dim, axis=1).astype(np.float32)


def build_input_plan(prompt: str, input_dim: int | None = None) -> dict[str, Any]:
    text = str(prompt or "").strip()
    if not text:
        raise ExpertModelError("请输入专家模型输入设定")

    warnings: list[str] = []
    source = "rule_agent"
    parsed: dict[str, Any] | None = None
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExpertModelError(f"输入设定 JSON 无法解析: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ExpertModelError("输入设定 JSON 必须是对象")
        source = "json"

    if parsed and "values" in parsed:
        matrix = _matrix_from_values(parsed["values"])
        plan = {
            "kind": "explicit_values",
            "source": source,
            "count": int(matrix.shape[0]),
            "input_dim": int(matrix.shape[1]),
            "start": float(matrix[0, 0]),
            "step": None,
            "param_names": [f"input_{i}" for i in range(int(matrix.shape[1]))],
            "summary": f"显式输入 {matrix.shape[0]} 个点，输入维度 {matrix.shape[1]}",
            "warnings": warnings,
        }
        return {"plan": plan, "matrix": matrix, "preview": matrix[:10].astype(float).tolist()}

    if parsed:
        count = _finite_int(parsed.get("count", 100), "count", min_value=1, max_value=MAX_INPUT_POINTS)
        start = _finite_float(parsed.get("start", 0.0), "start")
        step = _finite_float(parsed.get("step", 1.0), "step")
        dim_value = parsed.get("input_dim", input_dim or 1)
        dim = _finite_int(dim_value, "input_dim", min_value=1, max_value=MAX_INPUT_DIM)
    else:
        count_raw = _extract_first(
            [
                r"有\s*(\d+)\s*个(?:点|样本|输入)?",
                r"(\d+)\s*个(?:点|样本|输入)",
                r"count\s*[:=]\s*(\d+)",
                r"(\d+)\s*points?",
            ],
            text,
        )
        start_raw = _extract_first(
            [
                rf"从\s*({_NUMBER})\s*开始",
                rf"start\s*[:=]\s*({_NUMBER})",
                rf"from\s*({_NUMBER})",
            ],
            text,
        )
        step_raw = _extract_first(
            [
                rf"依次\s*(?:加|增加|递增)\s*({_NUMBER})",
                rf"每(?:个|次)[^，,。;.]*?(?:加|增加|递增)\s*({_NUMBER})",
                rf"步长\s*[:=]?\s*({_NUMBER})",
                rf"step\s*[:=]?\s*({_NUMBER})",
            ],
            text,
        )
        dim_raw = _extract_first(
            [
                r"输入维度\s*[:=]?\s*(\d+)",
                r"input[_ -]?dim\s*[:=]?\s*(\d+)",
            ],
            text,
        )

        if count_raw is None:
            warnings.append("未识别点数，默认使用 100 个点")
        if start_raw is None:
            warnings.append("未识别起点，默认从 0.0 开始")
        if step_raw is None:
            warnings.append("未识别步长，默认每次加 1.0")
        count = _finite_int(count_raw or 100, "count", min_value=1, max_value=MAX_INPUT_POINTS)
        start = _finite_float(start_raw or 0.0, "start")
        step = _finite_float(step_raw or 1.0, "step")
        dim = _finite_int(dim_raw or input_dim or 1, "input_dim", min_value=1, max_value=MAX_INPUT_DIM)

    if dim > 1:
        warnings.append("当前自然语言线性规划会把同一个标量复制到每个输入维度；多维差异输入请使用 JSON values")

    matrix = _matrix_from_linear(count, start, step, dim)
    plan = {
        "kind": "linear_sweep",
        "source": source,
        "count": count,
        "input_dim": dim,
        "start": start,
        "step": step,
        "param_names": [f"input_{i}" for i in range(dim)],
        "summary": f"生成 {count} 个输入点：从 {start:g} 开始，每次加 {step:g}，输入维度 {dim}",
        "warnings": warnings,
    }
    return {"plan": plan, "matrix": matrix, "preview": matrix[:10].astype(float).tolist()}


def _normalise_output(value: Any) -> list[float]:
    if isinstance(value, (int, float, np.number)):
        return [_finite_float(value, "专家模型输出")]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ExpertModelError("专家模型输出必须是 float 或非空 float 数组")
    output: list[float] = []
    for idx, item in enumerate(value):
        if isinstance(item, (list, tuple, dict)):
            raise ExpertModelError("专家模型输出必须是一维 float 数组")
        output.append(_finite_float(item, f"专家模型输出[{idx}]"))
    return output


def _write_expert_config(model: dict[str, Any], scenario: str, output_file: str, plan: dict[str, Any], prompt: str) -> None:
    EXPERT_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "simulator": EXPERT_SIMULATOR,
        "scenario": scenario,
        "output_dir": f"data/{EXPERT_SIMULATOR}",
        "output_file": output_file,
        "expert_model_id": model.get("model_id"),
        "expert_model_name": model.get("name"),
        "expert_interface": EXPERT_INTERFACE,
        "input_prompt": prompt,
        "input_plan": {key: value for key, value in plan.items() if key != "warnings"},
    }
    path = EXPERT_CONFIG_ROOT / f"{scenario}.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def generate_dataset(
    *,
    model_id: str,
    scenario: str,
    prompt: str,
    input_dim: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    scenario = validate_name("scenario", scenario)
    model = get_model(model_id)
    plan_result = build_input_plan(prompt, input_dim=input_dim)
    plan = plan_result["plan"]
    params = plan_result["matrix"].astype(np.float32)

    output_path = DATA_ROOT / EXPERT_SIMULATOR / f"{EXPERT_SIMULATOR}_{scenario}.h5"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在: {output_path}")

    predict = _load_predict(model)
    outputs: list[list[float]] = []
    output_dim: int | None = None
    for row_idx, row in enumerate(params):
        result = predict([float(value) for value in row.tolist()])
        values = _normalise_output(result)
        if output_dim is None:
            output_dim = len(values)
        elif len(values) != output_dim:
            raise ExpertModelError(f"专家模型第 {row_idx} 个输出维度变化：期望 {output_dim}，实际 {len(values)}")
        outputs.append(values)

    output_matrix = np.asarray(outputs, dtype=np.float32)
    timeseries = output_matrix.reshape(output_matrix.shape[0], output_matrix.shape[1], 1)
    metadata = {
        "simulator": EXPERT_SIMULATOR,
        "scenario": scenario,
        "expert_model_id": str(model.get("model_id") or ""),
        "expert_model_name": str(model.get("name") or ""),
        "expert_interface_version": EXPERT_INTERFACE_VERSION,
        "input_prompt": prompt,
        "input_plan_summary": str(plan.get("summary") or ""),
        "output_dim": int(output_matrix.shape[1]),
    }
    save_dataset(str(output_path), timeseries, params, list(plan["param_names"]), metadata=metadata)
    _write_expert_config(model, scenario, output_path.name, plan, prompt)
    validation = validate_hdf5_file(output_path)
    if not validation.get("valid"):
        errors = "; ".join(str(item) for item in validation.get("errors") or ["未知错误"])
        raise ExpertModelError(f"专家模型生成的 HDF5 未通过校验: {errors}")
    return {
        "ok": True,
        "model": model,
        "simulator": EXPERT_SIMULATOR,
        "scenario": scenario,
        "saved_path": str(output_path),
        "input_plan": {**plan, "preview": plan_result["preview"]},
        "validation": validation,
    }
