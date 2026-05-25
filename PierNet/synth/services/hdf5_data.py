"""HDF5 data-file validation and discovery for Stage 1 uploads/registration."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from PierNet.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT
from PierNet.shared.storage.hdf5_files import hdf5_scenario_from_path as _hdf5_scenario_from_path
from PierNet.shared.storage.hdf5_files import iter_hdf5_files


BUILTIN_SIMULATORS = ("modflow", "simpeg", "power_flow", "transient", "gcam")
SKIP_DATA_DIRS = {"templates", "text2comp", "router", ".manifests", ".indexes"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_REQUIRED_DATASETS = ("timeseries", "params", "param_names")
_REQUIRED_ATTRS = ("n_samples", "n_channels", "n_timesteps", "n_params")


def validate_name(kind: str, value: str, *, allowed: tuple[str, ...] | None = None) -> str:
    cleaned = (value or "").strip()
    if not _NAME_RE.match(cleaned):
        raise ValueError(f"{kind} 只能使用字母、数字、下划线或短横线，且必须以字母/数字开头")
    if allowed is not None and cleaned not in allowed:
        raise ValueError(f"{kind} 必须是以下值之一: {', '.join(allowed)}")
    return cleaned


def canonical_hdf5_path(simulator: str, scenario: str) -> Path:
    simulator = validate_name("simulator", simulator)
    scenario = validate_name("scenario", scenario)
    return DATA_ROOT / simulator / f"{simulator}_{scenario}.h5"


def scenario_from_hdf5_path(path: Path, simulator: str | None = None) -> str:
    return _hdf5_scenario_from_path(path, simulator or path.parent.name)


def _display_path(path: Path) -> str:
    try:
        if path.is_absolute():
            return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        pass
    return str(path)


def _resolve_data_path(value: str | None, *, default: str = "data") -> Path:
    raw = (value or default).strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        return DATA_ROOT.joinpath(*parts[1:])
    return PROJECT_ROOT / path


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
    return value


def _decode_param_names(dataset: h5py.Dataset) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    names: list[str] = []
    try:
        raw_values = dataset[:]
    except Exception as exc:
        return [], [f"param_names 无法读取: {exc}"]

    for idx, raw in enumerate(raw_values):
        if isinstance(raw, (bytes, bytearray, np.bytes_)):
            try:
                value = bytes(raw).decode("utf-8")
            except UnicodeDecodeError:
                errors.append(f"param_names[{idx}] 不是 UTF-8 字符串")
                value = bytes(raw).decode("utf-8", errors="replace")
        else:
            value = str(raw)
        value = value.strip()
        if not value:
            errors.append(f"param_names[{idx}] 为空")
        names.append(value)
    return names, errors


def _dataset_is_numeric(dataset: h5py.Dataset) -> bool:
    return dataset.dtype.kind in {"f", "i", "u"}


def _is_finite_dataset(dataset: h5py.Dataset, *, block_rows: int = 64) -> bool:
    if dataset.size == 0:
        return True
    if dataset.ndim == 0:
        return bool(np.isfinite(dataset[()]).all())
    rows = int(dataset.shape[0])
    if rows == 0:
        return True
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        if not np.isfinite(dataset[start:stop]).all():
            return False
    return True


def validate_hdf5_file(path: Path) -> dict[str, Any]:
    """Validate the canonical PierNet Stage 1 HDF5 contract.

    Required root datasets:
    - timeseries: numeric [N, C, T]
    - params: numeric [N, P]
    - param_names: string-like [P]

    Required root attrs must match dataset shapes:
    - n_samples, n_channels, n_timesteps, n_params
    """

    result: dict[str, Any] = {
        "valid": False,
        "path": str(path),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "sample_count": 0,
        "output_shape": None,
        "params_shape": None,
        "n_params": 0,
        "param_names_preview": [],
        "attrs": {},
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]

    if not path.exists():
        errors.append("文件不存在")
        return result
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        errors.append("文件扩展名必须是 .h5 或 .hdf5")
        return result
    if path.stat().st_size == 0:
        errors.append("文件为空")
        return result

    try:
        with h5py.File(path, "r") as hf:
            missing = [name for name in _REQUIRED_DATASETS if name not in hf]
            if missing:
                errors.append(f"缺少必需 dataset: {', '.join(missing)}")
                return result

            timeseries = hf["timeseries"]
            params = hf["params"]
            param_names = hf["param_names"]

            if not isinstance(timeseries, h5py.Dataset):
                errors.append("timeseries 必须是 dataset")
            if not isinstance(params, h5py.Dataset):
                errors.append("params 必须是 dataset")
            if not isinstance(param_names, h5py.Dataset):
                errors.append("param_names 必须是 dataset")
            if errors:
                return result

            if timeseries.ndim != 3:
                errors.append(f"timeseries 必须是 3 维 [N, C, T]，当前 shape={timeseries.shape}")
            if params.ndim != 2:
                errors.append(f"params 必须是 2 维 [N, P]，当前 shape={params.shape}")
            if param_names.ndim != 1:
                errors.append(f"param_names 必须是 1 维 [P]，当前 shape={param_names.shape}")

            if timeseries.ndim == 3:
                n_samples, n_channels, n_timesteps = map(int, timeseries.shape)
                result["sample_count"] = n_samples
                result["output_shape"] = [n_channels, n_timesteps]
                if n_samples <= 0:
                    errors.append("timeseries 样本数 N 必须大于 0")
                if n_channels <= 0:
                    errors.append("timeseries 通道数 C 必须大于 0")
                if n_timesteps <= 0:
                    errors.append("timeseries 时间步 T 必须大于 0")
            else:
                n_samples = n_channels = n_timesteps = None

            if params.ndim == 2:
                p_samples, n_params = map(int, params.shape)
                result["params_shape"] = [p_samples, n_params]
                result["n_params"] = n_params
                if n_params <= 0:
                    errors.append("params 参数数 P 必须大于 0")
                if n_samples is not None and p_samples != n_samples:
                    errors.append(f"params 样本数 {p_samples} 必须等于 timeseries 样本数 {n_samples}")
            else:
                p_samples = n_params = None

            if param_names.ndim == 1:
                names, name_errors = _decode_param_names(param_names)
                errors.extend(name_errors)
                result["param_names_preview"] = names[:12]
                if n_params is not None and len(names) != n_params:
                    errors.append(f"param_names 长度 {len(names)} 必须等于 params 参数数 {n_params}")
                if len(set(names)) != len(names):
                    warnings.append("param_names 存在重复名称，建议修正以避免模板填充歧义")

            if not _dataset_is_numeric(timeseries):
                errors.append(f"timeseries 必须是数值类型，当前 dtype={timeseries.dtype}")
            if not _dataset_is_numeric(params):
                errors.append(f"params 必须是数值类型，当前 dtype={params.dtype}")

            attrs = {key: _json_scalar(value) for key, value in hf.attrs.items()}
            result["attrs"] = attrs
            for key in _REQUIRED_ATTRS:
                if key not in hf.attrs:
                    errors.append(f"缺少必需根属性: {key}")

            expected_attrs = {
                "n_samples": n_samples,
                "n_channels": n_channels,
                "n_timesteps": n_timesteps,
                "n_params": n_params,
            }
            for key, expected in expected_attrs.items():
                if expected is None or key not in hf.attrs:
                    continue
                try:
                    actual = int(hf.attrs[key])
                except Exception:
                    errors.append(f"根属性 {key} 必须是整数")
                    continue
                if actual != expected:
                    errors.append(f"根属性 {key}={actual} 与数据 shape 推导值 {expected} 不一致")

            if "n_wells" in hf.attrs and n_channels is not None:
                try:
                    n_wells = int(hf.attrs["n_wells"])
                    if n_wells != n_channels:
                        warnings.append(f"兼容属性 n_wells={n_wells} 与 n_channels={n_channels} 不一致")
                except Exception:
                    warnings.append("兼容属性 n_wells 不是整数")

            if not errors:
                if not _is_finite_dataset(timeseries):
                    errors.append("timeseries 包含 NaN 或 Inf")
                if not _is_finite_dataset(params):
                    errors.append("params 包含 NaN 或 Inf")
    except OSError as exc:
        errors.append(f"无法打开 HDF5 文件: {exc}")
    except Exception as exc:
        errors.append(f"HDF5 校验失败: {exc}")

    result["valid"] = len(errors) == 0
    return result


def list_hdf5_data_files() -> list[dict[str, Any]]:
    data_root = DATA_ROOT
    items: list[dict[str, Any]] = []
    if not data_root.exists():
        return items

    for sim_dir in sorted(data_root.iterdir()):
        if not sim_dir.is_dir() or sim_dir.name in SKIP_DATA_DIRS:
            continue
        simulator = sim_dir.name
        for path in iter_hdf5_files(sim_dir):
            validation = validate_hdf5_file(path)
            items.append({
                **validation,
                "simulator": simulator,
                "scenario": scenario_from_hdf5_path(path, simulator),
                "path": _display_path(path),
                "mtime": path.stat().st_mtime,
            })
    return items


def _load_text2comp_scenarios(config_path: str) -> tuple[Path, list[tuple[str, str]]]:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise ValueError(f"配置文件不存在: {config_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_root = _resolve_data_path(cfg.get("data_root", "data"))
    scenarios_cfg = cfg.get("scenarios", [])
    pairs: list[tuple[str, str]] = []

    if isinstance(scenarios_cfg, dict):
        for simulator, scenario_list in scenarios_cfg.items():
            if isinstance(scenario_list, list):
                for scenario in scenario_list:
                    pairs.append((str(simulator), str(scenario)))
    elif isinstance(scenarios_cfg, list):
        for item in scenarios_cfg:
            if isinstance(item, str):
                simulator, scenario = _infer_simulator_for_scenario(data_root, item)
                pairs.append((simulator, scenario))
            elif isinstance(item, dict):
                simulator = str(item.get("simulator") or "").strip()
                scenario = str(item.get("scenario") or item.get("name") or "").strip()
                if simulator and scenario:
                    pairs.append((simulator, scenario))
    return data_root, pairs


def _resolve_hdf5_path_for_scenario(data_root: Path, simulator: str, scenario: str) -> Path:
    sim_dir = data_root / simulator
    for h5_file in iter_hdf5_files(sim_dir):
        if scenario_from_hdf5_path(h5_file, simulator) == scenario:
            return h5_file
    return sim_dir / f"{simulator}_{scenario}.h5"


def _infer_simulator_for_scenario(data_root: Path, scenario: str) -> tuple[str, str]:
    for simulator in BUILTIN_SIMULATORS:
        if _resolve_hdf5_path_for_scenario(data_root, simulator, scenario).exists():
            return simulator, scenario
    for sim_dir in sorted(data_root.iterdir()) if data_root.exists() else []:
        if not sim_dir.is_dir() or sim_dir.name in SKIP_DATA_DIRS:
            continue
        prefix = f"{sim_dir.name}_"
        for h5_file in iter_hdf5_files(sim_dir):
            derived = h5_file.stem[len(prefix):] if h5_file.stem.startswith(prefix) else h5_file.stem
            if derived == scenario:
                return sim_dir.name, scenario
    return "", scenario


def _scan_hdf5_pairs(data_root: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not data_root.exists():
        return pairs
    for sim_dir in sorted(data_root.iterdir()):
        if not sim_dir.is_dir() or sim_dir.name in SKIP_DATA_DIRS:
            continue
        simulator = sim_dir.name
        for h5_file in iter_hdf5_files(sim_dir):
            pairs.append((simulator, scenario_from_hdf5_path(h5_file, simulator)))
    return pairs


def collect_registration_hdf5_validations(
    config_path: str,
    scenarios: list[str] | None = None,
) -> list[dict[str, Any]]:
    data_root, pairs = _load_text2comp_scenarios(config_path)
    selected = set(scenarios or [])
    if selected and pairs:
        matched = [(sim, sc) for sim, sc in pairs if sc in selected]
        matched_names = {sc for _, sc in matched}
        missing = sorted(selected - matched_names)
        pairs = matched + [_infer_simulator_for_scenario(data_root, scenario) for scenario in missing]
    elif selected:
        pairs = [_infer_simulator_for_scenario(data_root, scenario) for scenario in sorted(selected)]
    elif not pairs:
        pairs = _scan_hdf5_pairs(data_root)

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for simulator, scenario in pairs:
        if not scenario:
            continue
        if not simulator:
            results.append({
                "valid": False,
                "simulator": "",
                "scenario": scenario,
                "path": str(data_root),
                "file_size_bytes": 0,
                "sample_count": 0,
                "output_shape": None,
                "params_shape": None,
                "n_params": 0,
                "param_names_preview": [],
                "attrs": {},
                "errors": [f"无法根据场景名 {scenario} 推断对应 simulator/HDF5 文件"],
                "warnings": [],
            })
            continue
        key = (simulator, scenario)
        if key in seen:
            continue
        seen.add(key)
        path = _resolve_hdf5_path_for_scenario(data_root, simulator, scenario)
        validation = validate_hdf5_file(path)
        results.append({
            **validation,
            "simulator": simulator,
            "scenario": scenario,
            "path": _display_path(path),
        })
    return results
