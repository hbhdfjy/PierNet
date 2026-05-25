
"""Unified file catalog and safe file operations for PierNet data assets."""

from __future__ import annotations

import base64
import json
import os
import shutil
import time

import h5py

from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from PierNet.shared.runtime.paths import DATA_DIR, DATA_ROOT, PROJECT_ROOT, TEMPLATES_DIR
from PierNet.shared.storage import portable
from PierNet.shared.storage.hdf5_files import iter_hdf5_files
from PierNet.synth.api.routers.config import invalidate_text2comp_scenarios_cache
from PierNet.synth.services import file_manager, hdf5_data, job_manager, jsonl_filter_index, jsonl_index, manifest_store
from PierNet.training.services import training_manager

ROUTER_DIR = DATA_ROOT / "router"
ROUTER_SCENARIO_DIR = ROUTER_DIR / "by_scenario"
MANIFEST_DIR = DATA_ROOT / ".manifests"
INDEX_DIR = DATA_ROOT / ".indexes"

_PLATFORM_LABELS = {
    "synth": "数据合成",
    "training": "训练产物",
    "system": "系统",
}



def _assert_no_active_jobs(job_types: set[str], message: str) -> None:
    active = job_manager.running_jobs(job_types)
    if active:
        raise RuntimeError(f"{message}: {', '.join(job.job_id for job in active)}")


def _assert_no_active_training_jobs(message: str) -> None:
    active = [
        str(job["job_id"])
        for job in training_manager.list_jobs(refresh=True)
        if job.get("status") in training_manager.TRAINING_ACTIVE_STATUSES
    ]
    if active:
        raise RuntimeError(f"{message}: {', '.join(active)}")

_STAGE_LABELS = {
    "stage1": "阶段 1 HDF5",
    "stage2": "阶段 2 模板",
    "stage3": "阶段 3 样本",
    "stage4": "阶段 4 路由",
    "training": "训练产物",
    "system": "清单 / 索引",
}

_KIND_LABELS = {
    "hdf5": "HDF5",
    "template": "模板 JSONL",
    "sample": "样本 JSONL（兼容）",
    "sample_merged": "合并样本 JSONL",
    "router_scenario": "路由场景 JSONL（兼容）",
    "sample_parquet": "样本 Parquet",
    "router_parquet": "路由 Parquet",
    "router_cache": "路由 JSONL 缓存",
    "router_train": "路由训练 JSONL",
    "training_job": "训练任务",
    "training_prepared": "训练 prepared 缓存",
    "training_checkpoint": "训练权重",
    "manifest": "清单",
    "index": "索引",
    "catalog_db": "目录数据库",
}

_DELETABLE_KINDS = {"sample", "router_scenario", "sample_parquet", "router_parquet", "router_cache", "training_job", "training_prepared", "training_checkpoint"}
_PROTECTED_KINDS = {"hdf5", "template", "sample_merged", "router_train", "manifest", "index"}


def encode_asset_id(*parts: str) -> str:
    payload = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def decode_asset_id(asset_id: str) -> list[str]:
    try:
        padded = asset_id + "=" * (-len(asset_id) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        parts = json.loads(raw)
    except Exception as exc:  # pragma: no cover - exercised through route error path
        raise ValueError("invalid asset id") from exc
    if not isinstance(parts, list) or not parts or not all(isinstance(part, str) for part in parts):
        raise ValueError("invalid asset id")
    return parts


def _safe_name_component(value: str, label: str) -> str:
    component = str(value or "")
    if not component or component in {".", ".."} or "\x00" in component:
        raise ValueError(f"{label} must be a non-empty file name component")
    if Path(component).name != component or "\\" in component:
        raise ValueError(f"{label} must be a file name component")
    return component

def _safe_identity_component(value: str, label: str) -> str:
    component = str(value or "")
    if not component or "\x00" in component or "\\" in component:
        raise ValueError(f"{label} must be a non-empty portable identity")
    path = Path(component)
    if path.is_absolute():
        raise ValueError(f"{label} must be a relative portable identity")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must not contain traversal segments")
    return component


def _router_cache_file_candidates(sim_dir: Path, scenario: str) -> list[Path]:
    names = [f"{portable.safe_partition_value(scenario)}.jsonl", f"{scenario}.jsonl"]
    candidates = [sim_dir / name for name in dict.fromkeys(names)]
    return candidates


def _relative_path(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _safe_stat(path: Path) -> tuple[int, float]:
    try:
        stat = path.stat()
        return int(stat.st_size), float(stat.st_mtime)
    except OSError:
        return 0, 0.0


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _delete_index_files(source_path: Path, profiles: tuple[str, ...] = ()) -> int:
    index_paths = [jsonl_index.get_index_path(source_path)]
    index_paths.extend(jsonl_filter_index.get_filter_index_path(source_path, profile) for profile in profiles)
    deleted = 0
    for index_path in index_paths:
        try:
            index_path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return deleted


def _clear_index_files_for_dirs(source_dirs: tuple[Path, ...]) -> int:
    deleted = 0
    index_dirs = {
        jsonl_index.get_index_path(source_dir / "__PierNet_index_scope__.jsonl").parent
        for source_dir in source_dirs
    }
    for index_dir in sorted(index_dirs):
        if not index_dir.exists():
            continue
        for path in sorted(index_dir.glob("*.idx.json")):
            try:
                path.unlink()
                deleted += 1
            except OSError:
                continue
    return deleted


def _delete_router_indexes(path: Path) -> int:
    return _delete_index_files(path, ("router_label",))


def _delete_sample_parquet_partitions(scenario: str, simulator: str | None = None) -> int:
    deleted = 0
    for part in portable.discover_partitions("text2comp"):
        if part.scenario != scenario:
            continue
        if simulator is not None and part.simulator != simulator:
            continue
        if portable.delete_partition("text2comp", part.scenario, simulator=part.simulator):
            deleted += 1
    return deleted


def _router_jsonl_cache_root() -> Path:
    return Path(os.getenv("PierNet_ROUTER_JSONL_CACHE_DIR", str(ROUTER_DIR / ".parquet_jsonl_cache")))


def _delete_router_jsonl_cache(simulator: str | None = None, scenario: str | None = None) -> int:
    root = _router_jsonl_cache_root()
    if not root.exists():
        return 0
    deleted = 0
    simulator_dirs = [root / simulator] if simulator else [path for path in root.iterdir() if path.is_dir()]
    for sim_dir in simulator_dirs:
        if not sim_dir.exists() or not sim_dir.is_dir():
            continue
        paths = _router_cache_file_candidates(sim_dir, scenario) if scenario else sorted(sim_dir.rglob("*.jsonl"))
        for path in paths:
            meta_path = path.with_suffix(".meta.json")
            if path.exists():
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    continue
            if meta_path.exists():
                try:
                    meta_path.unlink()
                    deleted += 1
                except OSError:
                    continue
            try:
                if path.parent != sim_dir and path.parent.exists() and not any(path.parent.iterdir()):
                    path.parent.rmdir()
            except OSError:
                pass
        try:
            if not any(sim_dir.iterdir()):
                sim_dir.rmdir()
        except OSError:
            pass
    try:
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
    except OSError:
        pass
    return deleted


def _delete_router_parquet_partitions(scenario: str, simulator: str | None = None) -> int:
    deleted = 0
    for part in portable.discover_partitions("router"):
        if part.scenario != scenario:
            continue
        if simulator is not None and part.simulator != simulator:
            continue
        if portable.delete_partition("router", part.scenario, simulator=part.simulator):
            deleted += 1
    return deleted


def _dir_size(path: Path | None, *, max_files: int = 5000) -> int:
    if path is None or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    seen = 0
    for child in path.rglob("*"):
        if child.is_file():
            seen += 1
            if seen > max_files:
                break
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _delete_parquet_partitions(kind: str) -> int:
    deleted = 0
    for part in portable.discover_partitions(kind):
        if portable.delete_partition(kind, part.scenario, simulator=part.simulator):
            deleted += 1
    return deleted


def _asset(
    *,
    platform: str,
    stage: str,
    kind: str,
    title: str,
    path: Path | str | None,
    id_parts: tuple[str, ...],
    simulator: str | None = None,
    scenario: str | None = None,
    job_id: str | None = None,
    count: int | None = None,
    count_label: str | None = None,
    file_size_bytes: int | None = None,
    mtime: float | None = None,
    valid: bool | None = None,
    status: str = "ok",
    protected: bool | None = None,
    deletable: bool | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path_obj = Path(path) if path else None
    if path_obj is not None and (file_size_bytes is None or mtime is None):
        size, ts = _safe_stat(path_obj)
        file_size_bytes = size if file_size_bytes is None else file_size_bytes
        mtime = ts if mtime is None else mtime

    protected_value = kind in _PROTECTED_KINDS if protected is None else protected
    deletable_value = (kind in _DELETABLE_KINDS and not protected_value) if deletable is None else deletable

    return {
        "id": encode_asset_id(*id_parts),
        "platform": platform,
        "platform_label": _PLATFORM_LABELS.get(platform, platform),
        "stage": stage,
        "stage_label": _STAGE_LABELS.get(stage, stage),
        "kind": kind,
        "kind_label": _KIND_LABELS.get(kind, kind),
        "title": title,
        "simulator": simulator,
        "scenario": scenario,
        "job_id": job_id,
        "path": _relative_path(path_obj),
        "count": count,
        "count_label": count_label,
        "file_size_bytes": int(file_size_bytes or 0),
        "mtime": float(mtime or 0),
        "valid": valid,
        "status": status,
        "protected": protected_value,
        "deletable": deletable_value,
        "warnings": warnings or [],
        "errors": errors or [],
        "details": details or {},
    }


def _hdf5_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    data_root = DATA_ROOT
    if not data_root.exists():
        return assets

    for sim_dir in sorted(data_root.iterdir()):
        if not sim_dir.is_dir() or sim_dir.name in hdf5_data.SKIP_DATA_DIRS:
            continue
        simulator = sim_dir.name
        for path in iter_hdf5_files(sim_dir):
            stat = path.stat()
            scenario = hdf5_data.scenario_from_hdf5_path(path, simulator)
            info = _inspect_hdf5_header(path)
            assets.append(_asset(
                platform="synth",
                stage="stage1",
                kind="hdf5",
                title=f"{simulator}/{scenario}",
                simulator=simulator,
                scenario=scenario,
                path=path,
                id_parts=("hdf5", simulator, scenario),
                count=info["sample_count"],
                count_label="样本",
                file_size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                valid=info["valid"],
                status="ok" if info["valid"] else "invalid",
                warnings=info["warnings"],
                errors=info["errors"],
                details={
                    "output_shape": info["output_shape"],
                    "params_shape": info["params_shape"],
                    "n_params": info["n_params"],
                    "param_names_preview": info["param_names_preview"],
                    "attrs": info["attrs"],
                    "validation_mode": "header_only",
                },
            ))
    return assets


def _inspect_hdf5_header(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "sample_count": 0,
        "output_shape": None,
        "params_shape": None,
        "n_params": 0,
        "param_names_preview": [],
        "attrs": {},
        "errors": [],
        "warnings": ["当前只扫描文件头；上传和注册仍会执行完整 HDF5 校验。"],
    }
    try:
        with h5py.File(path, "r") as hf:
            missing = [name for name in ("timeseries", "params", "param_names") if name not in hf]
            if missing:
                result["errors"].append(f"missing datasets: {', '.join(missing)}")
                return result
            timeseries = hf["timeseries"]
            params = hf["params"]
            param_names = hf["param_names"]
            if getattr(timeseries, "ndim", None) != 3:
                result["errors"].append(f"timeseries must be 3D, got {getattr(timeseries, 'shape', None)}")
            else:
                n_samples, n_channels, n_timesteps = map(int, timeseries.shape)
                result["sample_count"] = n_samples
                result["output_shape"] = [n_channels, n_timesteps]
            if getattr(params, "ndim", None) != 2:
                result["errors"].append(f"params must be 2D, got {getattr(params, 'shape', None)}")
            else:
                p_samples, n_params = map(int, params.shape)
                result["params_shape"] = [p_samples, n_params]
                result["n_params"] = n_params
                if result["sample_count"] and p_samples != result["sample_count"]:
                    result["errors"].append("params sample count does not match timeseries")
            if getattr(param_names, "ndim", None) != 1:
                result["errors"].append(f"param_names must be 1D, got {getattr(param_names, 'shape', None)}")
            else:
                preview = []
                for raw in param_names[: min(len(param_names), 12)]:
                    if isinstance(raw, (bytes, bytearray)):
                        preview.append(bytes(raw).decode("utf-8", errors="replace"))
                    else:
                        preview.append(str(raw))
                result["param_names_preview"] = preview
                if result["n_params"] and len(param_names) != result["n_params"]:
                    result["errors"].append("param_names length does not match params width")
            result["attrs"] = {key: _json_safe(value) for key, value in hf.attrs.items()}
            for key in ("n_samples", "n_channels", "n_timesteps", "n_params"):
                if key not in hf.attrs:
                    result["errors"].append(f"missing root attr: {key}")
            result["valid"] = len(result["errors"]) == 0
    except Exception as exc:
        result["errors"].append(str(exc))
    return result


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return value


def _template_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    manifest = manifest_store.ensure_template_manifest()
    for item in manifest.get("items", []):
        scenario = str(item.get("scenario") or "")
        simulator = str(item.get("simulator") or "").strip() or None
        assets.append(_asset(
            platform="synth",
            stage="stage2",
            kind="template",
            title=f"{simulator}/{scenario}" if simulator else scenario,
            simulator=simulator,
            scenario=scenario,
            path=item.get("path"),
            id_parts=("template", scenario),
            count=int(item.get("template_count") or 0),
            count_label="行",
            file_size_bytes=int(item.get("file_size_bytes") or 0),
            mtime=float(item.get("mtime") or 0),
            valid=True,
            details={
                "by_language": item.get("by_language", {}),
                "by_style": item.get("by_style", {}),
            },
        ))
    return assets


def _sample_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    manifest = manifest_store.ensure_sample_manifest()
    for item in manifest.get("items", []):
        scenario = str(item.get("scenario") or "")
        simulator = str(item.get("simulator") or "unknown")
        assets.append(_asset(
            platform="synth",
            stage="stage3",
            kind="sample",
            title=f"{simulator}/{scenario}",
            simulator=simulator,
            scenario=scenario,
            path=item.get("path"),
            id_parts=("sample", simulator, scenario),
            count=int(item.get("sample_count") or 0),
            count_label="行",
            file_size_bytes=int(item.get("file_size_bytes") or 0),
            mtime=float(item.get("mtime") or 0),
            valid=True,
            details={
                "by_language": item.get("by_language", {}),
                "by_style": item.get("by_style", {}),
                "by_time_mode": item.get("by_time_mode", {}),
                "timeseries_shape_obs": item.get("timeseries_shape_obs"),
            },
        ))

    merged_path = DATA_DIR / "all_training_data.jsonl"
    if merged_path.exists():
        merged_count = int(manifest.get("summary", {}).get("total_samples", 0))
        merged_size, merged_mtime = _safe_stat(merged_path)
        is_empty_merged = merged_count == 0 and merged_size == 0
        assets.append(_asset(
            platform="synth",
            stage="stage3",
            kind="sample_merged",
            title="all_training_data.jsonl",
            path=merged_path,
            id_parts=("sample_merged", "all_training_data"),
            count=merged_count,
            count_label="行",
            file_size_bytes=merged_size,
            mtime=merged_mtime,
            valid=True,
            status="empty" if is_empty_merged else "ok",
            protected=True,
            deletable=False,
            warnings=["合并样本文件受保护；请删除源场景样本后由平台重建。"] if is_empty_merged else [],
            details={"note": "由各场景样本文件合并生成，删除样本后会重建。"},
        ))
    return assets


def _router_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    manifest = manifest_store.ensure_router_manifest()
    for item in manifest.get("scenarios", []):
        scenario = str(item.get("scenario") or "")
        simulator = str(item.get("simulator") or "unknown")
        assets.append(_asset(
            platform="synth",
            stage="stage4",
            kind="router_scenario",
            title=f"{simulator}/{scenario}",
            simulator=simulator,
            scenario=scenario,
            path=item.get("path"),
            id_parts=("router_scenario", simulator, scenario),
            count=int(item.get("router_count") or 0),
            count_label="路由样本",
            file_size_bytes=int(item.get("file_size_bytes") or 0),
            mtime=float(item.get("mtime") or 0),
            valid=True,
        ))

    split = manifest.get("splits", {}).get("train") or {}
    train_path = ROUTER_DIR / "train.jsonl"
    if split.get("exists") or train_path.exists():
        assets.append(_asset(
            platform="synth",
            stage="stage4",
            kind="router_train",
            title="train.jsonl",
            path=train_path,
            id_parts=("router_train", "train"),
            count=int(split.get("count") or _count_jsonl(train_path)),
            count_label="路由样本",
            file_size_bytes=int(split.get("file_size_bytes") or 0),
            mtime=float(split.get("mtime") or 0),
            valid=True,
            protected=True,
            deletable=False,
            details={"label_counts": manifest.get("label_counts", {})},
        ))
    return assets


def _parquet_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for kind, stage, asset_kind, count_label in (
        ("text2comp", "stage3", "sample_parquet", "行"),
        ("router", "stage4", "router_parquet", "路由样本"),
    ):
        try:
            partitions = portable.discover_partitions(kind)
        except Exception as exc:
            assets.append(_asset(
                platform="system",
                stage="system",
                kind="manifest",
                title=f"{kind} parquet scan failed",
                path=None,
                id_parts=("parquet_error", kind),
                valid=False,
                status="invalid",
                protected=True,
                deletable=False,
                errors=[str(exc)],
            ))
            continue
        for part in partitions:
            assets.append(_asset(
                platform="synth",
                stage=stage,
                kind=asset_kind,
                title=f"{part.simulator}/{part.scenario}",
                simulator=part.simulator,
                scenario=part.scenario,
                path=part.path,
                id_parts=(asset_kind, part.simulator, part.scenario),
                count=part.row_count,
                count_label=count_label,
                file_size_bytes=part.file_size_bytes,
                mtime=part.mtime,
                valid=True,
                details={"storage": "parquet", "manifest": part.metadata},
            ))
    if portable.CATALOG_DB_PATH.exists():
        assets.append(_asset(
            platform="system",
            stage="system",
            kind="catalog_db",
            title="catalog.sqlite",
            path=portable.CATALOG_DB_PATH,
            id_parts=("catalog_db", "catalog.sqlite"),
            valid=True,
            protected=True,
            deletable=False,
        ))
    return assets


def _router_cache_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    root = _router_jsonl_cache_root()
    if not root.exists():
        return assets
    for sim_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(sim_dir.rglob("*.jsonl")):
            relative_stem = str(path.relative_to(sim_dir).with_suffix(""))
            meta_path = path.with_suffix(".meta.json")
            details: dict[str, Any] = {"cache": True}
            if meta_path.exists():
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        details.update(payload)
                except Exception:
                    details["meta_error"] = "failed to read cache metadata"
            simulator = str(details.get("simulator") or unquote(sim_dir.name))
            scenario = str(details.get("scenario") or unquote(relative_stem))
            assets.append(_asset(
                platform="synth",
                stage="stage4",
                kind="router_cache",
                title=f"{simulator}/{scenario}",
                simulator=simulator,
                scenario=scenario,
                path=path,
                id_parts=("router_cache", simulator, scenario),
                count=_count_jsonl(path),
                count_label="缓存行",
                valid=True,
                details=details,
                warnings=["由 Router Parquet 临时导出的训练准备缓存，可安全删除。"],
            ))
    return assets


def _manifest_index_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    if MANIFEST_DIR.exists():
        for path in sorted(MANIFEST_DIR.glob("*.json")):
            assets.append(_asset(
                platform="system",
                stage="system",
                kind="manifest",
                title=path.name,
                path=path,
                id_parts=("manifest", path.name),
                valid=True,
                protected=True,
                deletable=False,
            ))
    if INDEX_DIR.exists():
        for path in sorted(INDEX_DIR.rglob("*.json")):
            relative = path.relative_to(INDEX_DIR)
            if relative.parts and relative.parts[0] == ".tmp":
                continue
            assets.append(_asset(
                platform="system",
                stage="system",
                kind="index",
                title=str(relative),
                path=path,
                id_parts=("index", str(relative)),
                valid=True,
                protected=True,
                deletable=False,
            ))
    return assets


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_nonnegative_int(value: Any) -> tuple[int, bool]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0, False
    return max(0, parsed), parsed >= 0


def _active_prepared_refs(jobs: list[dict[str, Any]]) -> tuple[set[tuple[str, str]], set[str]]:
    active_statuses = getattr(training_manager, "TRAINING_ACTIVE_STATUSES", set())
    refs: set[tuple[str, str]] = set()
    unknown_simulators: set[str] = set()
    for job in jobs:
        status = str(job.get("status") or "")
        prepared_name = str(job.get("prepared_name") or "").strip()
        simulator = str(job.get("simulator") or "").strip()
        if status not in active_statuses or not simulator:
            continue
        if prepared_name:
            refs.add((simulator, prepared_name))
        else:
            unknown_simulators.add(simulator)
    return refs, unknown_simulators


def _prepared_cache_path(simulator: str, prepared_name: str) -> Path:
    return Path(training_manager.ARTIFACTS_ROOT) / simulator / "prepared" / prepared_name


def _training_prepared_assets(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root = Path(training_manager.ARTIFACTS_ROOT)
    if not root.exists():
        return []
    active_refs, unknown_active_simulators = _active_prepared_refs(jobs)
    assets: list[dict[str, Any]] = []
    for prepared_root in sorted(root.glob("*/prepared")):
        if not prepared_root.is_dir():
            continue
        simulator = prepared_root.parent.name
        for prepared_dir in sorted(path for path in prepared_root.iterdir() if path.is_dir()):
            prepared_name = prepared_dir.name
            meta_path = prepared_dir / "meta.json"
            summary = _read_json_object(meta_path)
            is_active = (simulator, prepared_name) in active_refs or simulator in unknown_active_simulators
            scenarios = summary.get("scenarios") if isinstance(summary.get("scenarios"), list) else []
            train_samples, train_samples_ok = _coerce_nonnegative_int(summary.get("train_samples"))
            test_samples, test_samples_ok = _coerce_nonnegative_int(summary.get("test_samples"))
            sample_count_valid = train_samples_ok and test_samples_ok
            meta_valid = bool(summary) and sample_count_valid
            errors = []
            if not summary:
                errors.append("缺少或无法读取 meta.json")
            if summary and not sample_count_valid:
                errors.append("meta.json 中的样本计数字段无效")
            assets.append(_asset(
                platform="training",
                stage="training",
                kind="training_prepared",
                title=f"{simulator}/{prepared_name}",
                simulator=simulator,
                scenario=", ".join(str(item) for item in scenarios),
                path=prepared_dir,
                id_parts=("training_prepared", simulator, prepared_name),
                count=train_samples + test_samples if summary else None,
                count_label="样本",
                file_size_bytes=_dir_size(prepared_dir, max_files=20000),
                mtime=_safe_stat(meta_path if meta_path.exists() else prepared_dir)[1],
                valid=meta_valid,
                status="ok" if meta_valid else "invalid",
                protected=is_active,
                deletable=not is_active,
                warnings=["活跃训练任务正在使用，不能删除。"] if is_active else [],
                errors=errors,
                details={
                    "prepared_name": prepared_name,
                    "prepared_format": summary.get("prepared_format"),
                    "input_representation": summary.get("input_representation"),
                    "train_samples": train_samples,
                    "test_samples": test_samples,
                    "source_fingerprint": summary.get("source_fingerprint"),
                },
            ))
    return assets


def _assert_prepared_cache_inactive(simulator: str, prepared_name: str) -> None:
    active = []
    active_statuses = getattr(training_manager, "TRAINING_ACTIVE_STATUSES", set())
    for job in training_manager.list_jobs(refresh=True):
        if str(job.get("status") or "") not in active_statuses:
            continue
        if str(job.get("simulator") or "") != simulator:
            continue
        job_prepared_name = str(job.get("prepared_name") or "").strip()
        if not job_prepared_name or job_prepared_name == prepared_name:
            active.append(str(job.get("job_id") or ""))
    if active:
        raise RuntimeError(f"训练 prepared 缓存正在被任务使用: {', '.join(active)}")


def _delete_training_prepared_cache(simulator: str, prepared_name: str) -> bool:
    root = Path(training_manager.ARTIFACTS_ROOT).resolve(strict=False)
    target = _prepared_cache_path(simulator, prepared_name).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"refusing to delete path outside training artifact root: {target}") from exc
    if not target.exists() or not target.is_dir():
        return False
    shutil.rmtree(target)
    return True


def _training_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    try:
        jobs = training_manager.list_jobs(refresh=True)
    except Exception as exc:
        return [_asset(
            platform="training",
            stage="training",
            kind="training_job",
            title="读取训练任务失败",
            path=None,
            id_parts=("training_error", "jobs"),
            valid=False,
            status="invalid",
            protected=True,
            deletable=False,
            errors=[str(exc)],
        )]

    active_statuses = getattr(training_manager, "TRAINING_ACTIVE_STATUSES", set())
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        run_dir = Path(str(job.get("run_dir") or "")) if job.get("run_dir") else None
        checkpoints = job.get("checkpoints") or []
        log_path = Path(str(job.get("log_path") or "")) if job.get("log_path") else None
        size = _dir_size(run_dir)
        if log_path and log_path.exists():
            try:
                size += log_path.stat().st_size
            except OSError:
                pass
        ended_at = job.get("ended_at")
        mtime = float(ended_at or job.get("started_at") or job.get("created_at") or 0)
        status = str(job.get("status") or "unknown")
        is_active = status in active_statuses
        assets.append(_asset(
            platform="training",
            stage="training",
            kind="training_job",
            title=str(job.get("name") or job_id),
            simulator=str(job.get("simulator") or ""),
            scenario=", ".join(job.get("scenarios") or []),
            job_id=job_id,
            path=run_dir,
            id_parts=("training_job", job_id),
            count=len(checkpoints),
            count_label="权重",
            file_size_bytes=size,
            mtime=mtime,
            valid=status not in {"error", "external_terminated"},
            status=status,
            protected=is_active,
            deletable=not is_active,
            warnings=["训练任务运行中，不能删除。"] if is_active else [],
            errors=[str(job.get("error_message"))] if job.get("error_message") else [],
            details={
                "gpu_id": job.get("gpu_id"),
                "latest_epoch": job.get("latest_epoch"),
                "latest_step": job.get("latest_step"),
                "latest_metrics": job.get("latest_metrics"),
                "checkpoints": checkpoints,
                "log_path": str(log_path) if log_path else "",
                "artifact_root": job.get("artifact_root"),
            },
        ))

        for checkpoint in checkpoints:
            checkpoint_path = Path(str(checkpoint.get("path") or ""))
            checkpoint_name = str(checkpoint.get("name") or checkpoint_path.name)
            is_epoch_checkpoint = checkpoint_name.startswith("router_epoch_") and checkpoint_name.endswith(".pt")
            is_protected_checkpoint = is_active or not is_epoch_checkpoint
            assets.append(_asset(
                platform="training",
                stage="training",
                kind="training_checkpoint",
                title=f"{job.get('name') or job_id}/{checkpoint_name}",
                simulator=str(job.get("simulator") or ""),
                scenario=", ".join(job.get("scenarios") or []),
                job_id=job_id,
                path=checkpoint_path,
                id_parts=("training_checkpoint", job_id, checkpoint_name),
                count=checkpoint.get("epoch"),
                count_label="轮次",
                file_size_bytes=int(checkpoint.get("size_bytes") or 0),
                mtime=float(checkpoint.get("mtime") or 0),
                valid=True,
                status=status,
                protected=is_protected_checkpoint,
                deletable=not is_protected_checkpoint,
                warnings=["运行中任务的权重不能删除。"] if is_active else (["latest/final/interrupted 权重受保护。"] if not is_epoch_checkpoint else []),
                details={
                    "job_id": job_id,
                    "job_name": job.get("name"),
                    "epoch": checkpoint.get("epoch"),
                    "checkpoint_name": checkpoint_name,
                },
            ))
    assets.extend(_training_prepared_assets(jobs))
    return assets


def list_file_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for collector in (_hdf5_assets, _template_assets, _sample_assets, _router_assets, _parquet_assets, _router_cache_assets, _training_assets, _manifest_index_assets):
        try:
            assets.extend(collector())
        except Exception as exc:
            assets.append(_asset(
                platform="system",
                stage="system",
                kind="manifest",
                title=f"{collector.__name__} failed",
                path=None,
                id_parts=("collector_error", collector.__name__),
                valid=False,
                status="invalid",
                protected=True,
                deletable=False,
                errors=[str(exc)],
            ))
    return sorted(assets, key=lambda item: (item["platform"], item["stage"], item["kind"], item["title"]))


def get_file_catalog() -> dict[str, Any]:
    assets = list_file_assets()
    return {
        "generated_at": time.time(),
        "assets": assets,
        "summary": _summary(assets),
    }


def _summary(assets: list[dict[str, Any]]) -> dict[str, Any]:
    by_platform = Counter(asset["platform"] for asset in assets)
    by_stage = Counter(asset["stage"] for asset in assets)
    by_kind = Counter(asset["kind"] for asset in assets)
    return {
        "total_assets": len(assets),
        "total_size_bytes": sum(int(asset.get("file_size_bytes") or 0) for asset in assets),
        "invalid_count": sum(1 for asset in assets if asset.get("valid") is False or asset.get("status") == "invalid"),
        "deletable_count": sum(1 for asset in assets if asset.get("deletable")),
        "protected_count": sum(1 for asset in assets if asset.get("protected")),
        "by_platform": dict(by_platform),
        "by_stage": dict(by_stage),
        "by_kind": dict(by_kind),
    }


def delete_asset(asset_id: str) -> dict[str, Any]:
    parts = decode_asset_id(asset_id)
    kind = parts[0]
    if kind == "hdf5" and len(parts) == 3:
        _safe_name_component(parts[1], "simulator")
        _safe_name_component(parts[2], "scenario")
        raise ValueError("Stage 1 HDF5 files are protected and must be preserved")

    if kind == "template" and len(parts) == 2:
        _safe_name_component(parts[1], "scenario")
        raise ValueError("Stage 2 template files are protected and must be preserved")

    if kind == "sample" and len(parts) in {2, 3}:
        _assert_no_active_jobs({"fill_samples", "router"}, "样本正在被填充或路由构建任务读取")
        simulator = _safe_name_component(parts[1], "simulator") if len(parts) == 3 else None
        scenario = (
            _safe_name_component(parts[2], "scenario")
            if len(parts) == 3
            else _safe_name_component(parts[1], "scenario")
        )
        deleted_jsonl = 1 if file_manager.delete_sample_file(scenario, simulator=simulator) else 0
        deleted_parquet = _delete_sample_parquet_partitions(scenario, simulator=simulator)
        deleted = deleted_jsonl + deleted_parquet
        if deleted_parquet:
            invalidate_text2comp_scenarios_cache()
        if deleted == 0:
            target = f"{simulator}/{scenario}" if simulator else scenario
            raise FileNotFoundError(f"file does not exist: {target}")
        return {"ok": True, "kind": kind, "deleted": deleted}

    if kind == "sample_merged" and len(parts) == 2:
        _safe_name_component(parts[1], "sample_merged")
        raise ValueError("merged sample file is protected; delete source sample files instead")

    if kind == "router_scenario" and len(parts) in {2, 3}:
        _assert_no_active_jobs({"router"}, "路由数据正在构建")
        _assert_no_active_training_jobs("路由数据正在被训练任务使用")
        simulator = _safe_name_component(parts[1], "simulator") if len(parts) == 3 else None
        scenario = (
            _safe_name_component(parts[2], "scenario")
            if len(parts) == 3
            else _safe_name_component(parts[1], "scenario")
        )
        return _delete_router_scenario(scenario, simulator=simulator)

    if kind == "sample_parquet" and len(parts) == 3:
        _assert_no_active_jobs({"fill_samples", "router"}, "样本分区正在被填充或路由构建任务使用")
        simulator = _safe_name_component(parts[1], "simulator")
        scenario = _safe_identity_component(parts[2], "scenario")
        deleted = portable.delete_partition("text2comp", scenario, simulator=simulator)
        if deleted:
            invalidate_text2comp_scenarios_cache()
        if not deleted:
            raise FileNotFoundError(f"Parquet partition does not exist: {simulator}/{scenario}")
        return {"ok": True, "kind": kind, "deleted": 1}

    if kind == "router_parquet" and len(parts) == 3:
        _assert_no_active_jobs({"router"}, "路由分区正在构建")
        _assert_no_active_training_jobs("路由分区正在被训练任务使用")
        simulator = _safe_name_component(parts[1], "simulator")
        scenario = _safe_identity_component(parts[2], "scenario")
        if not portable.delete_partition("router", scenario, simulator=simulator):
            raise FileNotFoundError(f"Parquet partition does not exist: {simulator}/{scenario}")
        deleted = 1 + _delete_router_jsonl_cache(simulator=simulator, scenario=scenario)
        return {"ok": True, "kind": kind, "deleted": deleted}

    if kind == "router_cache" and len(parts) == 3:
        _assert_no_active_training_jobs("路由缓存正在被训练任务使用")
        simulator = _safe_name_component(parts[1], "simulator")
        scenario = _safe_identity_component(parts[2], "scenario")
        deleted = _delete_router_jsonl_cache(simulator=simulator, scenario=scenario)
        if not deleted:
            raise FileNotFoundError(f"Router JSONL cache does not exist: {simulator}/{scenario}")
        return {"ok": True, "kind": kind, "deleted": deleted}

    if kind == "training_job" and len(parts) == 2:
        job_id = _safe_name_component(parts[1], "job_id")
        training_manager.delete_job(job_id)
        return {"ok": True, "kind": kind, "deleted": 1}

    if kind == "training_prepared" and len(parts) == 3:
        simulator = _safe_name_component(parts[1], "simulator")
        prepared_name = _safe_name_component(parts[2], "prepared_name")
        _assert_prepared_cache_inactive(simulator, prepared_name)
        if not _delete_training_prepared_cache(simulator, prepared_name):
            raise FileNotFoundError(f"Training prepared cache does not exist: {simulator}/{prepared_name}")
        return {"ok": True, "kind": kind, "deleted": 1}

    if kind == "training_checkpoint" and len(parts) == 3:
        job_id = _safe_name_component(parts[1], "job_id")
        checkpoint_name = _safe_name_component(parts[2], "checkpoint_name")
        training_manager.delete_checkpoint(job_id, checkpoint_name)
        return {"ok": True, "kind": kind, "deleted": 1}

    raise ValueError(f"unsupported asset deletion kind: {kind}")


def clear_group(kind: str) -> dict[str, Any]:
    if kind == "templates":
        raise ValueError("Stage 2 template files are protected and must be preserved")
    if kind == "samples":
        _assert_no_active_jobs({"fill_samples", "router"}, "样本正在被填充或路由构建任务读取")
        deleted_jsonl = file_manager.clear_all_samples()
        deleted_parquet = _delete_parquet_partitions("text2comp")
        if deleted_parquet:
            invalidate_text2comp_scenarios_cache()
        deleted = deleted_jsonl + deleted_parquet
        return {"ok": True, "kind": kind, "deleted": deleted}
    if kind == "router":
        _assert_no_active_jobs({"router"}, "路由数据正在构建")
        _assert_no_active_training_jobs("路由数据正在被训练任务使用")
        return _delete_all_router_data()
    raise ValueError(f"unsupported clear group kind: {kind}")



def _router_record_matches_identity(record: dict, fallback_scenario: str, scenario: str, simulator: str | None) -> bool:
    metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    record_scenario = str(metadata.get("scenario") or fallback_scenario).strip()
    record_simulator = str(metadata.get("simulator") or "").strip() or None
    return record_scenario == scenario and (simulator is None or record_simulator in {None, simulator})


def _router_jsonl_contains_identity(path: Path, scenario: str, simulator: str | None = None) -> bool:
    saw_record = False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                saw_record = True
                if _router_record_matches_identity(record, path.stem, scenario, simulator):
                    return True
    except OSError:
        return False
    return not saw_record and path.stem == scenario


def _rewrite_router_scenario_file(path: Path, scenario: str, simulator: str | None = None) -> tuple[bool, int]:
    kept: list[str] = []
    removed = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    kept.append(line if line.endswith("\n") else line + "\n")
                    continue
                if isinstance(record, dict) and _router_record_matches_identity(record, path.stem, scenario, simulator):
                    removed += 1
                    continue
                kept.append(line if line.endswith("\n") else line + "\n")
    except OSError:
        return False, 0

    if removed == 0:
        return False, len(kept)
    _delete_router_indexes(path)
    if kept:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("".join(kept), encoding="utf-8")
        tmp_path.replace(path)
    else:
        path.unlink(missing_ok=True)
    return True, len(kept)


def _update_router_meta_count(meta_path: Path, remaining: int) -> None:
    if not meta_path.exists():
        return
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    payload["output_count"] = remaining
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_router_scenario_file(scenario: str, simulator: str | None = None) -> Path | None:
    direct = ROUTER_SCENARIO_DIR / f"{scenario}.jsonl"
    matches: list[Path] = []
    if direct.exists() and _router_jsonl_contains_identity(direct, scenario, simulator=simulator):
        matches.append(direct)

    if ROUTER_SCENARIO_DIR.exists():
        for path in sorted(ROUTER_SCENARIO_DIR.glob("*.jsonl")):
            if path == direct:
                continue
            if _router_jsonl_contains_identity(path, scenario, simulator=simulator):
                matches.append(path)

    if len(matches) > 1:
        target = f"{simulator}/{scenario}" if simulator else scenario
        raise ValueError(f"ambiguous router scenario files for {target}")
    return matches[0] if matches else None


def _delete_router_scenario(scenario: str, simulator: str | None = None) -> dict[str, Any]:
    path = _resolve_router_scenario_file(scenario, simulator=simulator)
    meta_path = path.with_suffix(".meta.json") if path is not None else None
    deleted = _delete_router_parquet_partitions(scenario, simulator=simulator)
    if path is None or not path.exists():
        if deleted:
            deleted += _delete_router_jsonl_cache(simulator=simulator, scenario=scenario)
            try:
                manifest_store.rebuild_router_manifest()
            except Exception:
                pass
            return {"ok": True, "kind": "router_scenario", "deleted": deleted, "train_count": 0}
        raise FileNotFoundError(f"router scenario file does not exist: {scenario}")
    removed, remaining = _rewrite_router_scenario_file(path, scenario, simulator=simulator)
    if not removed:
        if deleted:
            try:
                manifest_store.rebuild_router_manifest()
            except Exception:
                pass
            return {"ok": True, "kind": "router_scenario", "deleted": deleted, "train_count": 0}
        raise FileNotFoundError(f"router scenario file does not contain: {scenario}")
    deleted += 1
    deleted += _delete_router_jsonl_cache(simulator=simulator, scenario=scenario)
    if remaining == 0 and meta_path is not None and meta_path.exists():
        meta_path.unlink()
        deleted += 1
    elif meta_path is not None:
        _update_router_meta_count(meta_path, remaining)
    total = _rewrite_router_train_from_scenarios()
    try:
        manifest_store.rebuild_router_manifest()
    except Exception:
        pass
    return {"ok": True, "kind": "router_scenario", "deleted": deleted, "train_count": total}


def _delete_all_router_data() -> dict[str, Any]:
    deleted = 0
    if ROUTER_SCENARIO_DIR.exists():
        for path in ROUTER_SCENARIO_DIR.glob("*.jsonl"):
            deleted += _delete_router_indexes(path)
            path.unlink()
            deleted += 1
        for meta_path in ROUTER_SCENARIO_DIR.glob("*.meta.json"):
            meta_path.unlink()
            deleted += 1
    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        deleted += _delete_router_indexes(train_path)
        train_path.unlink()
        deleted += 1
    deleted += _delete_parquet_partitions("router")
    deleted += _delete_router_jsonl_cache()
    try:
        manifest_store.rebuild_router_manifest()
    except Exception:
        pass
    return {"ok": True, "kind": "router", "deleted": deleted}


def _rewrite_router_train_from_scenarios() -> int:
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    ROUTER_SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for path in sorted(ROUTER_SCENARIO_DIR.glob("*.jsonl")):
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(sample, dict):
                    samples.append(sample)
    out_path = ROUTER_DIR / "train.jsonl"
    _delete_router_indexes(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples)


def rebuild_indexes(scope: str = "all") -> dict[str, Any]:
    """Rebuild manifests and JSONL indexes used by file browsers."""

    rebuilt: list[str] = []
    errors: list[str] = []
    deleted_indexes = 0

    def run(label: str, func) -> None:
        try:
            func()
            rebuilt.append(label)
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    if scope in {"all", "templates"}:
        deleted_indexes += _clear_index_files_for_dirs((TEMPLATES_DIR,))
        run("manifest:templates", manifest_store.rebuild_template_manifest)
        for path in sorted(TEMPLATES_DIR.glob("*_templates.jsonl")) if TEMPLATES_DIR.exists() else []:
            run(f"index:{_relative_path(path)}", lambda p=path: jsonl_index.rebuild_index(p))
            run(f"filter:{_relative_path(path)}:template_language_style", lambda p=path: jsonl_filter_index.rebuild_filter_index(p, "template_language_style"))

    if scope in {"all", "samples"}:
        deleted_indexes += _clear_index_files_for_dirs((DATA_DIR,))
        run("manifest:samples", manifest_store.rebuild_sample_manifest)
        for path in sorted(DATA_DIR.glob("*.jsonl")) if DATA_DIR.exists() else []:
            run(f"index:{_relative_path(path)}", lambda p=path: jsonl_index.rebuild_index(p))
            if path.name != "all_training_data.jsonl":
                run(f"filter:{_relative_path(path)}:sample_language_style", lambda p=path: jsonl_filter_index.rebuild_filter_index(p, "sample_language_style"))

    if scope in {"all", "router"}:
        deleted_indexes += _clear_index_files_for_dirs((ROUTER_DIR, ROUTER_SCENARIO_DIR))
        run("manifest:router", manifest_store.rebuild_router_manifest)
        router_files: list[Path] = []
        if ROUTER_DIR.exists():
            router_files.append(ROUTER_DIR / "train.jsonl")
        if ROUTER_SCENARIO_DIR.exists():
            router_files.extend(sorted(ROUTER_SCENARIO_DIR.glob("*.jsonl")))
        for path in router_files:
            if not path.exists():
                continue
            run(f"index:{_relative_path(path)}", lambda p=path: jsonl_index.rebuild_index(p))
            run(f"filter:{_relative_path(path)}:router_label", lambda p=path: jsonl_filter_index.rebuild_filter_index(p, "router_label"))

    return {"ok": len(errors) == 0, "rebuilt": rebuilt, "errors": errors, "deleted_indexes": deleted_indexes}
