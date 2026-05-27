"""Expert-model uploads and Stage 1 HDF5 generation."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import stat
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable

import numpy as np
import yaml

from PierNet.core.storage import save_dataset
from PierNet.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT
from PierNet.synth.services.hdf5_data import validate_hdf5_file, validate_name


EXPERT_SIMULATOR = "expert_model"
EXPERT_INTERFACE = "def predict(inputs: list[float]) -> float | list[float]"
EXPERT_INTERFACE_VERSION = 3
EXPERT_MODEL_ROOT = DATA_ROOT / "expert_models"
EXPERT_MODEL_FILES = EXPERT_MODEL_ROOT / "files"
EXPERT_METADATA_PATH = EXPERT_MODEL_ROOT / "models.json"
EXPERT_CONFIG_ROOT = PROJECT_ROOT / "configs" / EXPERT_SIMULATOR / "variants"
MANIFEST_NAME = "piernet_expert_model.json"
MAX_MODEL_BYTES = 1 * 1024 * 1024 * 1024
MAX_MODEL_BYTES_LABEL = "1 GB"
MAX_INPUT_DIM = 32
MAX_INPUT_POINTS = 1_000_000
MAX_BUNDLE_FILES = 512
DEFAULT_EXAMPLE_INPUT = [0.0]
SUPPORTED_UPLOAD_SUFFIXES = (".py", ".zip", ".tar.gz", ".tgz")
_MODULE_CONTEXT_LOCK = threading.RLock()
EXPERT_MODEL_CONSTRAINTS = {
    "file": [
        "可直接上传单个 Python .py 文件，或上传 .zip/.tar.gz/.tgz 专家模型包。",
        f"上传体大小不超过 {MAX_MODEL_BYTES_LABEL}；压缩包最多 {MAX_BUNDLE_FILES} 个文件，禁止绝对路径、.. 路径穿越和软链接。",
        "压缩包内可以包含权重、配置、JSON、CSV 等模型资产，但必须提供 piernet_expert_model.json 和 Python 入口适配器。",
        "第一版假定服务器已有模型需要的运行环境；不会自动安装 requirements。",
    ],
    "manifest": [
        "压缩包必须包含 piernet_expert_model.json。",
        "manifest 必须声明 runtime=python、entrypoint、callable 和 example_input。",
        "entrypoint 必须指向包内 Python 文件；callable 通常为 predict。",
    ],
    "interface": [
        "入口 callable 必须接收 inputs，并按 list[float] 处理。",
        f"inputs 长度必须为 1 到 {MAX_INPUT_DIM}，所有数值必须是有限 float。",
        "单 .py 文件可选定义 EXAMPLE_INPUT = [0.0, ...]；压缩包从 manifest.example_input 读取上传校验输入。",
    ],
    "output": [
        "callable 必须返回一个有限 float，或一维 finite float 列表/元组/numpy.ndarray。",
        "同一次数据生成中，每个输入点返回的输出维度必须一致。",
    ],
    "dataset": [
        "生成结果会写入 Stage 1 HDF5：params 形状为 [N, input_dim]，timeseries 形状为 [N, output_dim, 1]。",
        f"一次输入规划最多生成 {MAX_INPUT_POINTS} 个点。",
    ],
}
EXPERT_MODEL_EXAMPLE_SOURCE = """# expert_model.zip
# ├── piernet_expert_model.json
# ├── adapter.py
# └── model/config/weights 等任意资产

# piernet_expert_model.json
{
  "schema_version": 1,
  "runtime": "python",
  "entrypoint": "adapter.py",
  "callable": "predict",
  "example_input": [0.0]
}

# adapter.py
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
        "max_bundle_files": MAX_BUNDLE_FILES,
        "supported_upload_suffixes": list(SUPPORTED_UPLOAD_SUFFIXES),
        "manifest_name": MANIFEST_NAME,
    }


def _now() -> float:
    return time.time()


def _strip_supported_suffix(value: str) -> str:
    name = Path(str(value or "expert_model")).name
    lower = name.lower()
    for suffix in (".tar.gz", ".tgz", ".zip", ".py"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _safe_stem(value: str) -> str:
    raw = _strip_supported_suffix(value).strip() or "expert_model"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_-") or "expert_model"
    if not cleaned[0].isalnum():
        cleaned = f"model_{cleaned}"
    return cleaned[:64]


def _safe_file_name(value: str, fallback: str) -> str:
    name = Path(str(value or fallback)).name or fallback
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-") or fallback
    if not cleaned.lower().endswith(".py") and fallback.lower().endswith(".py"):
        cleaned = f"{cleaned}.py"
    return cleaned[:160]


def _upload_kind(name: str) -> str:
    lower = str(name or "").lower()
    if lower.endswith(".tar.gz"):
        return "tar.gz"
    if lower.endswith(".tgz"):
        return "tgz"
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".py"):
        return "python_file"
    raise ExpertModelError("上传文件必须是 .py、.zip、.tar.gz 或 .tgz；普通模型资产请放入压缩包并提供适配器")


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
            if path.is_file():
                copy["file_size_bytes"] = path.stat().st_size
            elif path.is_dir():
                copy["file_size_bytes"] = int(copy.get("file_size_bytes") or _directory_size(path))
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


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _assert_inside(root: Path, candidate: Path) -> None:
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ExpertModelError(f"压缩包包含非法路径: {candidate}") from exc


def _safe_archive_member(raw: str) -> Path | None:
    if not raw or "\x00" in raw:
        raise ExpertModelError("压缩包包含空路径或非法路径")
    normalized = raw.replace("\\", "/")
    raw_path = PurePosixPath(normalized)
    if raw_path.is_absolute():
        raise ExpertModelError(f"压缩包包含绝对路径: {raw}")
    normalized = normalized.strip("/")
    if not normalized:
        return None
    posix_path = PurePosixPath(normalized)
    parts = posix_path.parts
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise ExpertModelError(f"压缩包包含非法路径: {raw}")
    return Path(*parts)


def _check_archive_limits(total_bytes: int, file_count: int) -> None:
    if total_bytes > MAX_MODEL_BYTES:
        raise ExpertModelError(f"压缩包解压后文件总大小不能超过 {MAX_MODEL_BYTES_LABEL}")
    if file_count > MAX_BUNDLE_FILES:
        raise ExpertModelError(f"压缩包文件数量不能超过 {MAX_BUNDLE_FILES}")


def _extract_zip(content: bytes, dest: Path) -> tuple[int, int]:
    total = 0
    count = 0
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for info in archive.infolist():
            member_path = _safe_archive_member(info.filename)
            if member_path is None:
                continue
            target = dest / member_path
            _assert_inside(dest, target)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ExpertModelError(f"压缩包不允许包含软链接: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            count += 1
            total += int(info.file_size)
            _check_archive_limits(total, count)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
    return total, count


def _extract_tar(content: bytes, dest: Path) -> tuple[int, int]:
    total = 0
    count = 0
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as archive:
        for member in archive.getmembers():
            member_path = _safe_archive_member(member.name)
            if member_path is None:
                continue
            target = dest / member_path
            _assert_inside(dest, target)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ExpertModelError(f"压缩包只允许普通文件和目录: {member.name}")
            count += 1
            total += int(member.size)
            _check_archive_limits(total, count)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ExpertModelError(f"无法读取压缩包文件: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with extracted, target.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
    return total, count


def _safe_relative_file(value: str, label: str) -> Path:
    path = _safe_archive_member(str(value or ""))
    if path is None:
        raise ExpertModelError(f"{label} 不能为空")
    return path


def _load_bundle_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise ExpertModelError(f"专家模型包必须包含 {MANIFEST_NAME}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExpertModelError(f"{MANIFEST_NAME} 不是有效 JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ExpertModelError(f"{MANIFEST_NAME} 必须是 JSON object")
    runtime = str(manifest.get("runtime") or "").strip().lower()
    if runtime != "python":
        raise ExpertModelError("专家模型包第一版仅支持 runtime=python")
    entrypoint = _safe_relative_file(str(manifest.get("entrypoint") or ""), "entrypoint")
    callable_name = str(manifest.get("callable") or "predict").strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", callable_name):
        raise ExpertModelError("callable 必须是 Python 标识符")
    if "example_input" not in manifest:
        raise ExpertModelError("专家模型包 manifest 必须提供 example_input")
    example_input = _finite_vector(manifest.get("example_input"), "example_input")
    entrypoint_path = root / entrypoint
    _assert_inside(root, entrypoint_path)
    if not entrypoint_path.exists() or not entrypoint_path.is_file():
        raise ExpertModelError(f"entrypoint 文件不存在: {entrypoint}")
    if entrypoint_path.suffix.lower() != ".py":
        raise ExpertModelError("entrypoint 必须是 Python .py 文件")
    return {
        "schema_version": int(manifest.get("schema_version") or 1),
        "runtime": "python",
        "entrypoint": str(entrypoint.as_posix()),
        "callable": callable_name,
        "example_input": example_input,
    }


def _prepare_python_file(model_id: str, name: str, content: bytes) -> dict[str, Any]:
    model_dir = EXPERT_MODEL_FILES / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    file_name = _safe_file_name(name, f"{model_id}.py")
    target = model_dir / file_name
    target.write_bytes(content)
    return {
        "package_type": "python_file",
        "file_name": file_name,
        "path": str(target),
        "entrypoint": file_name,
        "callable": "predict",
        "asset_count": 1,
        "asset_size_bytes": len(content),
    }


def _prepare_bundle(model_id: str, name: str, content: bytes, kind: str) -> dict[str, Any]:
    model_dir = EXPERT_MODEL_FILES / model_id
    source_dir = model_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_name = _safe_file_name(name, f"{model_id}.{kind}")
    archive_path = model_dir / archive_name
    archive_path.write_bytes(content)
    if kind == "zip":
        total, count = _extract_zip(content, source_dir)
    else:
        total, count = _extract_tar(content, source_dir)
    manifest = _load_bundle_manifest(source_dir)
    return {
        "package_type": kind,
        "file_name": archive_name,
        "path": str(source_dir),
        "archive_path": str(archive_path),
        "entrypoint": manifest["entrypoint"],
        "callable": manifest["callable"],
        "example_input": manifest["example_input"],
        "manifest": manifest,
        "asset_count": count,
        "asset_size_bytes": total,
    }


@contextlib.contextmanager
def _module_context(root: Path, entrypoint: Path):
    with _MODULE_CONTEXT_LOCK:
        previous_cwd = os.getcwd()
        added: list[str] = []
        for candidate in (str(root), str(entrypoint.parent)):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
                added.append(candidate)
        try:
            os.chdir(root)
            yield
        finally:
            os.chdir(previous_cwd)
            for candidate in added:
                try:
                    sys.path.remove(candidate)
                except ValueError:
                    pass


def _model_entrypoint(model: dict[str, Any]) -> tuple[Path, Path, str]:
    path = Path(str(model.get("path") or ""))
    callable_name = str(model.get("callable") or "predict")
    if path.is_file():
        return path.parent, path, callable_name
    root = path
    entrypoint = _safe_relative_file(str(model.get("entrypoint") or ""), "entrypoint")
    entrypoint_path = root / entrypoint
    _assert_inside(root, entrypoint_path)
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"专家模型入口文件不存在: {entrypoint}")
    return root, entrypoint_path, callable_name


def _load_module(path: Path, model_id: str, root: Path | None = None) -> ModuleType:
    module_name = "PierNet_user_expert_" + hashlib.sha256(f"{model_id}:{path}".encode("utf-8")).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ExpertModelError("无法加载专家模型 Python 文件")
    module = importlib.util.module_from_spec(spec)
    with _module_context(root or path.parent, path):
        spec.loader.exec_module(module)
    return module


def _load_predict(model: dict[str, Any]) -> Callable[[list[float]], Any]:
    model_id = str(model.get("model_id") or "")
    root, entrypoint, callable_name = _model_entrypoint(model)
    module = _load_module(entrypoint, model_id, root=root)
    predict = getattr(module, callable_name, None)
    if not callable(predict):
        raise ExpertModelError(f"专家模型必须实现可调用入口: {callable_name}(inputs)")

    def wrapped(inputs: list[float]) -> Any:
        with _module_context(root, entrypoint):
            return predict(inputs)

    return wrapped


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExpertModelError(f"{label} 必须是 float") from exc
    if not math.isfinite(number):
        raise ExpertModelError(f"{label} 必须是有限 float")
    return number


def _finite_vector(value: Any, label: str) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ExpertModelError(f"{label} 必须是非空 list[float]")
    if len(value) > MAX_INPUT_DIM:
        raise ExpertModelError(f"{label} 长度不能超过 {MAX_INPUT_DIM}")
    return [_finite_float(item, f"{label}[{idx}]") for idx, item in enumerate(value)]


def validate_model_contract(model: dict[str, Any]) -> dict[str, Any]:
    root, entrypoint, callable_name = _model_entrypoint(model)
    model_id = str(model.get("model_id") or "")
    module = _load_module(entrypoint, model_id, root=root)
    predict = getattr(module, callable_name, None)
    if not callable(predict):
        raise ExpertModelError(f"专家模型必须实现可调用入口: {callable_name}(inputs)")
    example_input = _finite_vector(model.get("example_input", getattr(module, "EXAMPLE_INPUT", DEFAULT_EXAMPLE_INPUT)), "EXAMPLE_INPUT")
    try:
        with _module_context(root, entrypoint):
            smoke_output = _normalise_output(predict(list(example_input)))
    except Exception as exc:
        raise ExpertModelError(f"专家模型上传校验失败：{callable_name}(EXAMPLE_INPUT) 无法返回合法 float 输出（{exc}）") from exc
    return {
        "example_input": example_input,
        "example_input_dim": len(example_input),
        "smoke_output_dim": len(smoke_output),
    }


def upload_model(name: str, content: bytes) -> dict[str, Any]:
    if not content:
        raise ExpertModelError("上传文件为空")
    if len(content) > MAX_MODEL_BYTES:
        raise ExpertModelError(f"专家模型文件不能超过 {MAX_MODEL_BYTES_LABEL}")

    kind = _upload_kind(name)
    stem = _safe_stem(name)
    digest = hashlib.sha256(content + str(time.time_ns()).encode("ascii")).hexdigest()[:12]
    model_id = validate_name("model_id", f"{stem}-{digest}")
    model_dir = EXPERT_MODEL_FILES / model_id

    try:
        if kind == "python_file":
            prepared = _prepare_python_file(model_id, name, content)
        else:
            prepared = _prepare_bundle(model_id, name, content, kind)
        model = {
            "model_id": model_id,
            "name": stem,
            "created_at": _now(),
            "file_size_bytes": len(content),
            "interface": EXPERT_INTERFACE,
            "interface_version": EXPERT_INTERFACE_VERSION,
            "constraints_version": EXPERT_INTERFACE_VERSION,
            **prepared,
        }
        model.update(validate_model_contract(model))
    except Exception:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise

    models = [item for item in _read_models() if item.get("model_id") != model_id]
    models.append(model)
    _write_models(models)
    return model


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
        "expert_package_type": model.get("package_type"),
        "expert_entrypoint": model.get("entrypoint"),
        "expert_callable": model.get("callable"),
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
        "expert_package_type": str(model.get("package_type") or ""),
        "expert_entrypoint": str(model.get("entrypoint") or ""),
        "expert_callable": str(model.get("callable") or ""),
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
