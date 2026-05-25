"""Operational health endpoints for local and remote deployments."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import APIRouter

from PierNet.shared.runtime.config import validate_runtime_config
from PierNet.shared.runtime.paths import ARTIFACT_ROOT, DATA_ROOT, PROJECT_ROOT, RUNLOG_ROOT
from PierNet.training.services import training_manager

router = APIRouter()


def _path_status(path: Path, *, writable: bool = False) -> dict:
    exists = path.exists()
    info = {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir() if exists else False,
        "writable": False,
        "writable_checked": writable,
    }
    if writable:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f".health-{time.time_ns()}.tmp"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            info["writable"] = True
        except OSError as exc:
            info["error"] = str(exc)
    return info


@router.get("/health/live")
def live() -> dict:
    return {"status": "ok", "ts": time.time()}


@router.get("/health/ready")
def ready() -> dict:
    validation = validate_runtime_config()
    checks = {
        "project_root": _path_status(PROJECT_ROOT),
        "data_root": _path_status(DATA_ROOT, writable=True),
        "artifact_root": _path_status(ARTIFACT_ROOT, writable=True),
        "runlog_root": _path_status(RUNLOG_ROOT, writable=True),
        "runtime_config": {
            "ok": validation.ok,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
            "summary": validation.config.safe_summary(),
        },
    }
    path_checks = [value for key, value in checks.items() if key != "runtime_config"]
    writable_checks = [item for item in path_checks if item.get("writable_checked")]
    ok = (
        all(item.get("exists") for item in path_checks)
        and all(item.get("writable") for item in writable_checks)
        and validation.ok
    )
    return {"status": "ok" if ok else "degraded", "checks": checks}


@router.get("/health/storage")
def storage() -> dict:
    usage = shutil.disk_usage(DATA_ROOT if DATA_ROOT.exists() else PROJECT_ROOT)
    return {
        "status": "ok",
        "data_root": str(DATA_ROOT),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_ratio": usage.free / usage.total if usage.total else 0.0,
    }


@router.get("/health/gpu")
def gpu() -> dict:
    try:
        gpus = training_manager.get_gpu_inventory()
    except Exception as exc:  # pragma: no cover - defensive runtime endpoint
        return {
            "status": "unavailable",
            "gpus": [],
            "error": str(exc),
        }
    return {
        "status": "ok" if gpus else "unavailable",
        "gpus": gpus,
    }
