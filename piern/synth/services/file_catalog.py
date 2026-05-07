
"""Unified file catalog and safe file operations for PiERN data assets."""

from __future__ import annotations

import base64
import json
import time

import h5py

from collections import Counter
from pathlib import Path
from typing import Any

from piern.shared.runtime.paths import DATA_DIR, PROJECT_ROOT, TEMPLATES_DIR
from piern.shared.storage import portable
from piern.synth.api.routers.config import invalidate_text2comp_scenarios_cache
from piern.synth.services import file_manager, hdf5_data, jsonl_filter_index, jsonl_index, manifest_store
from piern.training.services import training_manager

ROUTER_DIR = PROJECT_ROOT / "data" / "router"
ROUTER_SCENARIO_DIR = ROUTER_DIR / "by_scenario"
MANIFEST_DIR = PROJECT_ROOT / "data" / ".manifests"
INDEX_DIR = PROJECT_ROOT / "data" / ".indexes"

_PLATFORM_LABELS = {
    "synth": "数据合成",
    "training": "训练产物",
    "system": "系统",
}

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
    "sample": "样本 JSONL",
    "sample_merged": "合并样本",
    "router_scenario": "路由场景数据",
    "sample_parquet": "样本 Parquet",
    "router_parquet": "路由 Parquet",
    "router_train": "路由训练数据",
    "training_job": "训练任务",
    "training_checkpoint": "训练权重",
    "manifest": "清单",
    "index": "索引",
    "catalog_db": "目录数据库",
}

_DELETABLE_KINDS = {"hdf5", "template", "sample", "router_scenario", "sample_parquet", "router_parquet", "training_job", "training_checkpoint"}
_PROTECTED_KINDS = {"sample_merged", "router_train", "manifest", "index"}


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
    data_root = PROJECT_ROOT / "data"
    if not data_root.exists():
        return assets

    for sim_dir in sorted(data_root.iterdir()):
        if not sim_dir.is_dir() or sim_dir.name in hdf5_data.SKIP_DATA_DIRS:
            continue
        simulator = sim_dir.name
        for path in sorted([*sim_dir.glob("*.h5"), *sim_dir.glob("*.hdf5")]):
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
        assets.append(_asset(
            platform="synth",
            stage="stage2",
            kind="template",
            title=scenario,
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
            title=scenario,
            simulator=simulator,
            scenario=scenario,
            path=item.get("path"),
            id_parts=("sample", scenario),
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
        assets.append(_asset(
            platform="synth",
            stage="stage3",
            kind="sample_merged",
            title="all_training_data.jsonl",
            path=merged_path,
            id_parts=("sample_merged", "all_training_data"),
            count=int(manifest.get("summary", {}).get("total_samples", 0)),
            count_label="行",
            valid=True,
            protected=True,
            deletable=False,
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
            title=scenario,
            simulator=simulator,
            scenario=scenario,
            path=item.get("path"),
            id_parts=("router_scenario", scenario),
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
            assets.append(_asset(
                platform="system",
                stage="system",
                kind="index",
                title=str(path.relative_to(INDEX_DIR)),
                path=path,
                id_parts=("index", str(path.relative_to(INDEX_DIR))),
                valid=True,
                protected=True,
                deletable=False,
            ))
    return assets


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
    return assets


def list_file_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for collector in (_hdf5_assets, _template_assets, _sample_assets, _router_assets, _parquet_assets, _training_assets, _manifest_index_assets):
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
        simulator, scenario = parts[1], parts[2]
        path = hdf5_data.canonical_hdf5_path(simulator, scenario)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 file does not exist: {_relative_path(path)}")
        path.unlink()
        invalidate_text2comp_scenarios_cache()
        return {"ok": True, "kind": kind, "deleted": 1}

    if kind == "template" and len(parts) == 2:
        if not file_manager.delete_template_file(parts[1]):
            raise FileNotFoundError(f"file does not exist: {parts[1]}")
        return {"ok": True, "kind": kind, "deleted": 1}

    if kind == "sample" and len(parts) == 2:
        if not file_manager.delete_sample_file(parts[1]):
            raise FileNotFoundError(f"file does not exist: {parts[1]}")
        return {"ok": True, "kind": kind, "deleted": 1}

    if kind == "router_scenario" and len(parts) == 2:
        return _delete_router_scenario(parts[1])

    if kind == "sample_parquet" and len(parts) == 3:
        if not portable.delete_partition("text2comp", parts[2], simulator=parts[1]):
            raise FileNotFoundError(f"Parquet partition does not exist: {parts[1]}/{parts[2]}")
        return {"ok": True, "kind": kind, "deleted": 1}

    if kind == "router_parquet" and len(parts) == 3:
        if not portable.delete_partition("router", parts[2], simulator=parts[1]):
            raise FileNotFoundError(f"Parquet partition does not exist: {parts[1]}/{parts[2]}")
        return {"ok": True, "kind": kind, "deleted": 1}

    if kind == "training_job" and len(parts) == 2:
        training_manager.delete_job(parts[1])
        return {"ok": True, "kind": kind, "deleted": 1}

    if kind == "training_checkpoint" and len(parts) == 3:
        training_manager.delete_checkpoint(parts[1], parts[2])
        return {"ok": True, "kind": kind, "deleted": 1}

    raise ValueError(f"unsupported asset deletion kind: {kind}")


def clear_group(kind: str) -> dict[str, Any]:
    if kind == "templates":
        return {"ok": True, "kind": kind, "deleted": file_manager.clear_all_templates()}
    if kind == "samples":
        return {"ok": True, "kind": kind, "deleted": file_manager.clear_all_samples()}
    if kind == "router":
        return _delete_all_router_data()
    raise ValueError(f"unsupported clear group kind: {kind}")


def _delete_router_scenario(scenario: str) -> dict[str, Any]:
    path = ROUTER_SCENARIO_DIR / f"{scenario}.jsonl"
    meta_path = path.with_suffix(".meta.json")
    if not path.exists():
        raise FileNotFoundError(f"router scenario file does not exist: {scenario}")
    path.unlink()
    meta_path.unlink(missing_ok=True)
    total = _rewrite_router_train_from_scenarios()
    try:
        manifest_store.rebuild_router_manifest()
    except Exception:
        pass
    return {"ok": True, "kind": "router_scenario", "deleted": 1, "train_count": total}


def _delete_all_router_data() -> dict[str, Any]:
    deleted = 0
    if ROUTER_SCENARIO_DIR.exists():
        for path in ROUTER_SCENARIO_DIR.glob("*.jsonl"):
            path.unlink()
            deleted += 1
        for meta_path in ROUTER_SCENARIO_DIR.glob("*.meta.json"):
            meta_path.unlink()
            deleted += 1
    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        train_path.unlink()
        deleted += 1
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
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        except Exception:
            continue
    out_path = ROUTER_DIR / "train.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples)


def rebuild_indexes(scope: str = "all") -> dict[str, Any]:
    """Rebuild manifests and JSONL indexes used by file browsers."""

    rebuilt: list[str] = []
    errors: list[str] = []

    def run(label: str, func) -> None:
        try:
            func()
            rebuilt.append(label)
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    if scope in {"all", "templates"}:
        run("manifest:templates", manifest_store.rebuild_template_manifest)
        for path in sorted(TEMPLATES_DIR.glob("*_templates.jsonl")) if TEMPLATES_DIR.exists() else []:
            run(f"index:{_relative_path(path)}", lambda p=path: jsonl_index.rebuild_index(p))
            run(f"filter:{_relative_path(path)}:template_language_style", lambda p=path: jsonl_filter_index.rebuild_filter_index(p, "template_language_style"))

    if scope in {"all", "samples"}:
        run("manifest:samples", manifest_store.rebuild_sample_manifest)
        for path in sorted(DATA_DIR.glob("*.jsonl")) if DATA_DIR.exists() else []:
            run(f"index:{_relative_path(path)}", lambda p=path: jsonl_index.rebuild_index(p))
            if path.name != "all_training_data.jsonl":
                run(f"filter:{_relative_path(path)}:sample_language_style", lambda p=path: jsonl_filter_index.rebuild_filter_index(p, "sample_language_style"))

    if scope in {"all", "router"}:
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

    return {"ok": len(errors) == 0, "rebuilt": rebuilt, "errors": errors}
