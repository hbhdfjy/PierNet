from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

MAX_EXPERT_FILES = 512
MANIFEST_NAME = "piernet_expert_model.json"


class ExpertValidationError(ValueError):
    pass


def _safe_path(name: str) -> Path:
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ExpertValidationError(f"专家模型压缩包包含不安全路径: {name}")
    return Path(*pure.parts)


def _extract_archive(source: Path, destination: Path) -> None:
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            files = [info for info in archive.infolist() if not info.is_dir()]
            if len(files) > MAX_EXPERT_FILES:
                raise ExpertValidationError(f"专家模型文件数量超过 {MAX_EXPERT_FILES}")
            for info in files:
                target = destination / _safe_path(info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        return
    if tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            files = [member for member in archive.getmembers() if member.isfile()]
            if len(files) > MAX_EXPERT_FILES:
                raise ExpertValidationError(f"专家模型文件数量超过 {MAX_EXPERT_FILES}")
            for member in files:
                if member.issym() or member.islnk():
                    raise ExpertValidationError("专家模型压缩包不允许包含符号链接")
                target = destination / _safe_path(member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                opened = archive.extractfile(member)
                if opened is None:
                    continue
                with opened, target.open("wb") as dst:
                    shutil.copyfileobj(opened, dst)
        return
    raise ExpertValidationError("无法识别专家模型压缩包")


def _defines_predict(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "predict"
        for node in tree.body
    )


def prepare_expert_package(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(source.suffixes).lower()
    if source.suffix.lower() == ".py":
        shutil.copy2(source, destination / "model.py")
    elif suffixes.endswith((".zip", ".tar.gz", ".tgz")):
        _extract_archive(source, destination)
    else:
        raise ExpertValidationError("专家模型需要是 .py、.zip、.tar.gz 或 .tgz")
    manifests = list(destination.rglob(MANIFEST_NAME))
    if len(manifests) > 1:
        raise ExpertValidationError(f"专家包中只能包含一个 {MANIFEST_NAME}")
    if manifests:
        manifest_path = manifests[0]
        root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        python_files = [
            path
            for path in destination.rglob("*.py")
            if "__pycache__" not in path.parts and path.name != "__init__.py"
        ]
        candidates = (
            python_files
            if len(python_files) == 1
            else [path for path in python_files if _defines_predict(path)]
        )
        if len(candidates) != 1:
            raise ExpertValidationError(
                f"未提供 {MANIFEST_NAME} 时，专家包需要且只能有一个"
                "定义 predict(inputs) 的 Python 入口文件"
            )
        entrypoint_path = candidates[0]
        root = entrypoint_path.parent
        entrypoint = entrypoint_path.name
        manifest = {
            "runtime": "python",
            "entrypoint": entrypoint,
            "callable": "predict",
            "batch_mode": "auto",
        }
        manifest_path = root / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if str(manifest.get("runtime") or "python").lower() != "python":
        raise ExpertValidationError("第一版 Studio 只支持 Python 运行时")
    entrypoint = str(manifest.get("entrypoint") or "model.py")
    callable_name = str(manifest.get("callable") or "predict")
    batch_mode = str(manifest.get("batch_mode") or "auto").lower()
    if batch_mode not in {"auto", "batch", "per_sample"}:
        raise ExpertValidationError(
            "batch_mode 仅支持 auto、batch 或 per_sample"
        )
    if not (root / entrypoint).exists():
        raise ExpertValidationError(f"专家入口文件不存在: {entrypoint}")
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "runtime": "python",
        "entrypoint": entrypoint,
        "callable": callable_name,
        "batch_mode": batch_mode,
        "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
    }


def execute_expert(
    expert: dict[str, Any],
    inputs: np.ndarray,
    *,
    work_dir: Path,
    timeout_seconds: int = 120,
) -> np.ndarray:
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="expert-", dir=work_dir) as temp_name:
        temp = Path(temp_name)
        input_path = temp / "input.npz"
        output_path = temp / "output.npz"
        np.savez_compressed(input_path, inputs=np.asarray(inputs, dtype=np.float32))
        env = dict(os.environ)
        project_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = (
            project_root
            if not env.get("PYTHONPATH")
            else f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
        )
        command = [
            sys.executable,
            "-m",
            "PierNet.studio.expert_worker",
            "--root",
            str(expert["root"]),
            "--manifest",
            str(expert["manifest_path"]),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        if expert.get("batch_mode") == "per_sample":
            command.append("--per-sample")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else ""
            raise ExpertValidationError(f"计算模型执行失败: {detail or '未知错误'}")
        if not output_path.exists():
            raise ExpertValidationError("计算模型没有产生输出")
        with np.load(output_path, allow_pickle=False) as payload:
            outputs = np.asarray(payload["outputs"], dtype=np.float32)
            if "execution_mode" in payload:
                expert["batch_mode"] = str(payload["execution_mode"].item())
        if not np.isfinite(outputs).all():
            raise ExpertValidationError("计算模型输出包含 NaN 或 Inf")
        return outputs


def check_compatibility(
    canonical_path: Path,
    expert: dict[str, Any],
    *,
    work_dir: Path,
) -> dict[str, Any]:
    with np.load(canonical_path, allow_pickle=False) as payload:
        inputs = np.asarray(payload["inputs"], dtype=np.float32)
        expected = np.asarray(payload["outputs"], dtype=np.float32)
    sample_count = min(3, inputs.shape[0])
    actual = execute_expert(expert, inputs[:sample_count], work_dir=work_dir)
    expected_sample = expected[:sample_count]
    shape_match = actual.shape == expected_sample.shape
    report: dict[str, Any] = {
        "compatible": shape_match,
        "sample_count": sample_count,
        "input_shape": list(inputs.shape[1:]),
        "expected_output_shape": list(expected_sample.shape[1:]),
        "actual_output_shape": list(actual.shape[1:]) if actual.ndim > 0 else [],
        "finite": bool(np.isfinite(actual).all()),
    }
    if shape_match:
        diff = actual.astype(np.float64) - expected_sample.astype(np.float64)
        mse = float(np.mean(diff**2))
        scale = float(np.sqrt(np.mean(expected_sample.astype(np.float64) ** 2)))
        report.update(
            {
                "sample_mse": mse,
                "relative_rmse": float(np.sqrt(mse) / max(scale, 1e-12)),
                "preview": actual[0].reshape(-1).astype(float).tolist()[:24],
            }
        )
    else:
        report["message"] = "计算模型输出形状与数据输出形状不一致"
    return report
