from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
import numpy as np
import pandas as pd

SUPPORTED_DATA_SUFFIXES = {".h5", ".hdf5", ".npz", ".csv", ".parquet"}
MAX_ARCHIVE_FILES = 512


class DataInspectionError(ValueError):
    pass


def _safe_member_path(name: str) -> Path:
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise DataInspectionError(f"压缩包包含不安全路径: {name}")
    return Path(*pure.parts)


def extract_data_archive(archive_path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if len(infos) > MAX_ARCHIVE_FILES:
                raise DataInspectionError(f"压缩包文件数量超过 {MAX_ARCHIVE_FILES}")
            for info in infos:
                relative = _safe_member_path(info.filename)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append(target)
        return extracted
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            if len(members) > MAX_ARCHIVE_FILES:
                raise DataInspectionError(f"压缩包文件数量超过 {MAX_ARCHIVE_FILES}")
            for member in members:
                if member.issym() or member.islnk():
                    raise DataInspectionError("压缩包不允许包含符号链接")
                relative = _safe_member_path(member.name)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append(target)
        return extracted
    raise DataInspectionError("无法识别压缩包格式")


def discover_data_file(source_path: Path, extraction_dir: Path) -> Path:
    suffixes = "".join(source_path.suffixes).lower()
    if source_path.suffix.lower() in SUPPORTED_DATA_SUFFIXES:
        return source_path
    if suffixes.endswith(".zip") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        files = extract_data_archive(source_path, extraction_dir)
        candidates = [path for path in files if path.suffix.lower() in SUPPORTED_DATA_SUFFIXES]
        if len(candidates) != 1:
            raise DataInspectionError(
                "数据压缩包必须包含且只包含一个可识别的数据文件；"
                f"当前发现 {len(candidates)} 个"
            )
        return candidates[0]
    raise DataInspectionError(
        "暂不支持此数据格式，请上传科学数据文件、表格，"
        "或包含单个数据文件的压缩包"
    )


def _as_numeric(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise DataInspectionError(f"{name} 不是数值数组")
    if array.ndim == 0:
        raise DataInspectionError(f"{name} 缺少样本维度")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise DataInspectionError(f"{name} 包含 NaN 或 Inf")
    return array


def _default_names(prefix: str, size: int) -> list[str]:
    return [f"{prefix}_{index + 1}" for index in range(size)]


def _decode_names(value: Any, expected: int, prefix: str) -> list[str]:
    if value is None:
        return _default_names(prefix, expected)
    names = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(value).reshape(-1)
    ]
    return names if len(names) == expected else _default_names(prefix, expected)


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        input_key = "inputs" if "inputs" in data else "params" if "params" in data else None
        output_key = "outputs" if "outputs" in data else "timeseries" if "timeseries" in data else None
        if input_key is None or output_key is None:
            numeric_keys = [
                key
                for key in keys
                if np.issubdtype(np.asarray(data[key]).dtype, np.number)
                and np.asarray(data[key]).ndim > 0
            ]
            if len(numeric_keys) < 2:
                raise DataInspectionError("NPZ 中需要 inputs/outputs，或至少两个数值数组")
            input_key, output_key = numeric_keys[:2]
        inputs = _as_numeric(input_key, data[input_key])
        outputs = _as_numeric(output_key, data[output_key])
        input_dim = int(np.prod(inputs.shape[1:])) if inputs.ndim > 1 else 1
        output_dim = int(np.prod(outputs.shape[1:])) if outputs.ndim > 1 else 1
        input_names = _decode_names(
            data["input_names"] if "input_names" in data else None, input_dim, "input"
        )
        output_names = _decode_names(
            data["output_names"] if "output_names" in data else None, output_dim, "output"
        )
    return inputs, outputs, input_names, output_names


def _h5_first(dataset: h5py.File, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in dataset:
            return dataset[name][...]
    return None


def _load_hdf5(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    with h5py.File(path, "r") as data:
        raw_inputs = _h5_first(data, ("inputs", "params"))
        raw_outputs = _h5_first(data, ("outputs", "timeseries"))
        if raw_inputs is None or raw_outputs is None:
            raise DataInspectionError("HDF5 需要 inputs/outputs 或 params/timeseries 数据集")
        inputs = _as_numeric("inputs", raw_inputs)
        outputs = _as_numeric("outputs", raw_outputs)
        input_dim = int(np.prod(inputs.shape[1:])) if inputs.ndim > 1 else 1
        output_dim = int(np.prod(outputs.shape[1:])) if outputs.ndim > 1 else 1
        raw_input_names = _h5_first(data, ("input_names", "param_names"))
        raw_output_names = _h5_first(data, ("output_names", "channel_names"))
        input_names = _decode_names(raw_input_names, input_dim, "input")
        output_names = _decode_names(raw_output_names, output_dim, "output")
    return inputs, outputs, input_names, output_names


def inspect_tabular(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, nrows=100) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    suggested_inputs = [column for column in numeric if str(column).lower().startswith("input")]
    suggested_outputs = [column for column in numeric if str(column).lower().startswith("output")]
    return {
        "kind": "tabular",
        "columns": [
            {
                "name": str(column),
                "dtype": str(frame[column].dtype),
                "numeric": column in numeric,
                "sample": frame[column].head(3).tolist(),
            }
            for column in frame.columns
        ],
        "suggested_input_fields": suggested_inputs,
        "suggested_output_fields": suggested_outputs,
        "needs_mapping": not (suggested_inputs and suggested_outputs),
    }


def _load_tabular(
    path: Path,
    input_fields: list[str] | None,
    output_fields: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    inputs = input_fields or [column for column in numeric if column.lower().startswith("input")]
    outputs = output_fields or [column for column in numeric if column.lower().startswith("output")]
    if not inputs or not outputs:
        raise DataInspectionError("请为表格数据选择输入列和输出列")
    missing = [column for column in [*inputs, *outputs] if column not in frame.columns]
    if missing:
        raise DataInspectionError(f"数据列不存在: {', '.join(missing)}")
    return (
        _as_numeric("inputs", frame[inputs].to_numpy()),
        _as_numeric("outputs", frame[outputs].to_numpy()),
        list(inputs),
        list(outputs),
    )


def _stats(array: np.ndarray) -> dict[str, float]:
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def canonicalize_data(
    data_path: Path,
    canonical_path: Path,
    *,
    input_fields: list[str] | None = None,
    output_fields: list[str] | None = None,
) -> dict[str, Any]:
    suffix = data_path.suffix.lower()
    if suffix == ".npz":
        inputs, outputs, input_names, output_names = _load_npz(data_path)
    elif suffix in {".h5", ".hdf5"}:
        inputs, outputs, input_names, output_names = _load_hdf5(data_path)
    elif suffix in {".csv", ".parquet"}:
        inputs, outputs, input_names, output_names = _load_tabular(
            data_path, input_fields, output_fields
        )
    else:
        raise DataInspectionError(f"不支持的数据格式: {suffix}")
    if inputs.shape[0] != outputs.shape[0]:
        raise DataInspectionError(
            f"输入与输出样本数不一致: {inputs.shape[0]} != {outputs.shape[0]}"
        )
    if inputs.shape[0] < 4:
        raise DataInspectionError("至少需要 4 条输入输出样本")
    input_shape = list(inputs.shape[1:] or (1,))
    output_shape = list(outputs.shape[1:] or (1,))
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        canonical_path,
        inputs=inputs,
        outputs=outputs,
        input_names=np.asarray(input_names),
        output_names=np.asarray(output_names),
    )
    metadata = {
        "canonical_path": str(canonical_path),
        "samples": int(inputs.shape[0]),
        "input_shape": input_shape,
        "input_dim": int(np.prod(input_shape)),
        "output_shape": output_shape,
        "output_dim": int(np.prod(output_shape)),
        "input_names": input_names,
        "output_names": output_names,
        "input_stats": _stats(inputs),
        "output_stats": _stats(outputs),
        "preview": [
            {
                "inputs": inputs[index].reshape(-1).astype(float).tolist(),
                "outputs": outputs[index].reshape(-1).astype(float).tolist()[:24],
            }
            for index in range(min(3, inputs.shape[0]))
        ],
    }
    (canonical_path.parent / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def inspect_source_data(data_path: Path) -> dict[str, Any]:
    suffix = data_path.suffix.lower()
    if suffix in {".csv", ".parquet"}:
        return inspect_tabular(data_path)
    return {"kind": suffix.removeprefix("."), "needs_mapping": False}
