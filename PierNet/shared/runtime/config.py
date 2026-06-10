"""Runtime configuration loading, validation, and sanitized reporting."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PierNet.shared.runtime.env import load_env_file

logger = logging.getLogger(__name__)

PROJECT_DEFAULT = Path(__file__).resolve().parents[3]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_path(name: str, default: Path) -> Path:
    raw = _env(name)
    value = Path(os.path.expandvars(raw)).expanduser() if raw else default
    return value.resolve()


def _default_conda_env(project_root: Path) -> Path:
    repo_env = project_root / ".conda" / "env"
    if (repo_env / "bin" / "python").exists():
        return repo_env
    return Path.home() / ".conda" / "envs" / "PierNet"


def _default_node_bin(project_root: Path) -> Path | None:
    current = project_root / ".node" / "current" / "bin" / "node"
    if current.exists():
        return current.resolve()
    node_root = project_root / ".node"
    if node_root.exists():
        for candidate_dir in sorted(node_root.glob("node-v*/bin")):
            candidate = candidate_dir / "node"
            if candidate.exists():
                return candidate.resolve()
    return None


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class RuntimeConfig:
    env_file: Optional[Path]
    project_root: Path
    data_root: Path
    artifact_root: Path
    runlog_root: Path
    job_store_path: Path
    training_job_store_path: Path
    router_jsonl_cache_dir: Path
    lock_store_path: Path
    audit_store_path: Path
    worker_store_path: Path
    host: str
    backend_port: int
    frontend_port: int
    cors_origins: Tuple[str, ...]
    service_run_dir: Path
    conda_env: Path
    python: Path
    training_python: Path
    npm: str
    node_bin: Optional[Path]
    qwen_embedding_model: Path
    qwen_embedding_tokenizer: Path
    modflow_exe: Optional[Path]
    max_workers: int
    default_train_workers: int
    default_prepare_workers: int
    gpu_free_memory_threshold_mib: int
    gpu_available_util_threshold: float
    cache_cleanup_enabled: bool
    cache_cleanup_dry_run: bool
    cache_cleanup_interval_hours: float
    router_jsonl_cache_ttl_days: float
    training_prepared_cache_ttl_days: float
    cache_cleanup_max_delete_gb: float
    log_level: str

    def safe_summary(self) -> Dict[str, Any]:
        return {
            "env_file": str(self.env_file) if self.env_file else None,
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "artifact_root": str(self.artifact_root),
            "runlog_root": str(self.runlog_root),
            "job_store_path": str(self.job_store_path),
            "training_job_store_path": str(self.training_job_store_path),
            "router_jsonl_cache_dir": str(self.router_jsonl_cache_dir),
            "lock_store_path": str(self.lock_store_path),
            "audit_store_path": str(self.audit_store_path),
            "worker_store_path": str(self.worker_store_path),
            "host": self.host,
            "backend_port": self.backend_port,
            "frontend_port": self.frontend_port,
            "cors_origins": list(self.cors_origins),
            "service_run_dir": str(self.service_run_dir),
            "conda_env": str(self.conda_env),
            "python": str(self.python),
            "training_python": str(self.training_python),
            "npm": self.npm,
            "node_bin": str(self.node_bin) if self.node_bin else None,
            "qwen_embedding_model": str(self.qwen_embedding_model),
            "qwen_embedding_tokenizer": str(self.qwen_embedding_tokenizer),
            "modflow_exe": str(self.modflow_exe) if self.modflow_exe else None,
            "max_workers": self.max_workers,
            "default_train_workers": self.default_train_workers,
            "default_prepare_workers": self.default_prepare_workers,
            "gpu_free_memory_threshold_mib": self.gpu_free_memory_threshold_mib,
            "gpu_available_util_threshold": self.gpu_available_util_threshold,
            "cache_cleanup_enabled": self.cache_cleanup_enabled,
            "cache_cleanup_dry_run": self.cache_cleanup_dry_run,
            "cache_cleanup_interval_hours": self.cache_cleanup_interval_hours,
            "router_jsonl_cache_ttl_days": self.router_jsonl_cache_ttl_days,
            "training_prepared_cache_ttl_days": self.training_prepared_cache_ttl_days,
            "cache_cleanup_max_delete_gb": self.cache_cleanup_max_delete_gb,
            "log_level": self.log_level,
        }


@dataclass(frozen=True)
class ConfigValidation:
    config: RuntimeConfig
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "summary": self.config.safe_summary(),
        }


def load_runtime_config() -> RuntimeConfig:
    env_file = load_env_file()
    project_root = _env_path("PierNet_ROOT", PROJECT_DEFAULT)
    data_root = _env_path("PierNet_DATA_ROOT", project_root / "data")
    artifact_root = _env_path("PierNet_ARTIFACT_ROOT", project_root / "artifacts")
    runlog_root = _env_path("PierNet_RUNLOG_ROOT", project_root / ".runlogs")
    backend_port = _env_int("PierNet_BACKEND_PORT", 8000)
    frontend_port = _env_int("PierNet_FRONTEND_PORT", 3000)
    cors = _split_csv(_env("PierNet_CORS_ORIGINS"))
    if not cors:
        cors = (
            f"http://localhost:{frontend_port}",
            "http://localhost:4173",
            f"http://127.0.0.1:{frontend_port}",
        )
    conda_env = _env_path("PierNet_CONDA_ENV", _default_conda_env(project_root))
    python = _env_path("PierNet_PYTHON", conda_env / "bin" / "python")
    training_python = _env_path("PierNet_TRAINING_PYTHON", python)
    node_raw = _env("PierNet_NODE_BIN") or _env("PierNet_NODE")
    default_node = _default_node_bin(project_root)
    qwen_default = Path.home() / "Qwen" / "Qwen2.5-0.5B-Instruct"
    model = _env_path("PierNet_QWEN_EMBEDDING_MODEL", qwen_default)
    return RuntimeConfig(
        env_file=env_file,
        project_root=project_root,
        data_root=data_root,
        artifact_root=artifact_root,
        runlog_root=runlog_root,
        job_store_path=_env_path("PierNet_JOB_STORE_PATH", runlog_root / "jobs.sqlite"),
        training_job_store_path=_env_path("PierNet_TRAINING_JOB_STORE_PATH", runlog_root / "training_jobs.sqlite"),
        router_jsonl_cache_dir=_env_path(
            "PierNet_ROUTER_JSONL_CACHE_DIR", data_root / "router" / ".parquet_jsonl_cache"
        ),
        lock_store_path=_env_path("PierNet_LOCK_STORE_PATH", runlog_root / "job_locks.sqlite"),
        audit_store_path=_env_path("PierNet_AUDIT_STORE_PATH", runlog_root / "audit_events.sqlite"),
        worker_store_path=_env_path("PierNet_WORKER_STORE_PATH", runlog_root / "worker_heartbeats.sqlite"),
        host=_env("PierNet_HOST", _env("PierNet_SERVICE_HOST", "0.0.0.0")) or "0.0.0.0",
        backend_port=backend_port,
        frontend_port=frontend_port,
        cors_origins=cors,
        service_run_dir=_env_path("PierNet_SERVICE_RUN_DIR", runlog_root / "services"),
        conda_env=conda_env,
        python=python,
        training_python=training_python,
        npm=_env("PierNet_NPM", "npm") or "npm",
        node_bin=Path(os.path.expandvars(node_raw)).expanduser().resolve() if node_raw else default_node,
        qwen_embedding_model=model,
        qwen_embedding_tokenizer=_env_path("PierNet_QWEN_EMBEDDING_TOKENIZER", model),
        modflow_exe=_env_path("PierNet_MODFLOW_EXE", Path("/__missing_modflow__")) if _env("PierNet_MODFLOW_EXE") else None,
        max_workers=_env_int("PierNet_MAX_WORKERS", 8),
        default_train_workers=_env_int("PierNet_DEFAULT_TRAIN_WORKERS", 8),
        default_prepare_workers=_env_int("PierNet_DEFAULT_PREPARE_WORKERS", 8),
        gpu_free_memory_threshold_mib=_env_int("PierNet_GPU_FREE_MEMORY_THRESHOLD_MIB", 2048),
        gpu_available_util_threshold=_env_float("PierNet_GPU_AVAILABLE_UTIL_THRESHOLD", 20.0),
        cache_cleanup_enabled=_env_bool("PierNet_CACHE_CLEANUP_ENABLED", False),
        cache_cleanup_dry_run=_env_bool("PierNet_CACHE_CLEANUP_DRY_RUN", False),
        cache_cleanup_interval_hours=_env_float("PierNet_CACHE_CLEANUP_INTERVAL_HOURS", 24.0),
        router_jsonl_cache_ttl_days=_env_float("PierNet_ROUTER_JSONL_CACHE_TTL_DAYS", 7.0),
        training_prepared_cache_ttl_days=_env_float("PierNet_TRAINING_PREPARED_CACHE_TTL_DAYS", 7.0),
        cache_cleanup_max_delete_gb=_env_float("PierNet_CACHE_CLEANUP_MAX_DELETE_GB", 1024.0),
        log_level=_env("PierNet_LOG_LEVEL", "INFO") or "INFO",
    )


def _check_existing_dir(errors: List[str], path: Path, label: str) -> None:
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")
    elif not path.is_dir():
        errors.append(f"{label} is not a directory: {path}")


def _check_writable_dir(errors: List[str], path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".config-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        errors.append(f"{label} is not writable: {path} ({exc})")


def _check_parent_writable(errors: List[str], path: Path, label: str) -> None:
    _check_writable_dir(errors, path.parent, f"{label} parent")


def _check_optional_file(warnings: List[str], path: Optional[Path], label: str) -> None:
    if path is not None and not path.exists():
        warnings.append(f"{label} not found: {path}")


def _check_port(errors: List[str], port: int, label: str) -> None:
    if port < 1 or port > 65535:
        errors.append(f"{label} must be in 1..65535, got {port}")


def validate_runtime_config(config: Optional[RuntimeConfig] = None) -> ConfigValidation:
    cfg = config or load_runtime_config()
    errors: List[str] = []
    warnings: List[str] = []

    _check_existing_dir(errors, cfg.project_root, "PierNet_ROOT")
    _check_existing_dir(errors, cfg.data_root, "PierNet_DATA_ROOT")
    _check_writable_dir(errors, cfg.artifact_root, "PierNet_ARTIFACT_ROOT")
    _check_writable_dir(errors, cfg.runlog_root, "PierNet_RUNLOG_ROOT")
    _check_parent_writable(errors, cfg.job_store_path, "PierNet_JOB_STORE_PATH")
    _check_parent_writable(errors, cfg.training_job_store_path, "PierNet_TRAINING_JOB_STORE_PATH")
    _check_parent_writable(errors, cfg.lock_store_path, "PierNet_LOCK_STORE_PATH")
    _check_parent_writable(errors, cfg.audit_store_path, "PierNet_AUDIT_STORE_PATH")
    _check_parent_writable(errors, cfg.worker_store_path, "PierNet_WORKER_STORE_PATH")
    _check_writable_dir(errors, cfg.router_jsonl_cache_dir, "PierNet_ROUTER_JSONL_CACHE_DIR")
    _check_writable_dir(errors, cfg.service_run_dir, "PierNet_SERVICE_RUN_DIR")
    _check_port(errors, cfg.backend_port, "PierNet_BACKEND_PORT")
    _check_port(errors, cfg.frontend_port, "PierNet_FRONTEND_PORT")

    _check_optional_file(warnings, cfg.python, "PierNet_PYTHON")
    _check_optional_file(warnings, cfg.training_python, "PierNet_TRAINING_PYTHON")
    _check_optional_file(warnings, cfg.node_bin, "PierNet_NODE_BIN")
    if not cfg.conda_env.exists():
        warnings.append(f"PierNet_CONDA_ENV not found: {cfg.conda_env}")
    if not cfg.qwen_embedding_model.exists():
        warnings.append(f"PierNet_QWEN_EMBEDDING_MODEL not found: {cfg.qwen_embedding_model}")
    if not cfg.qwen_embedding_tokenizer.exists():
        warnings.append(f"PierNet_QWEN_EMBEDDING_TOKENIZER not found: {cfg.qwen_embedding_tokenizer}")
    _check_optional_file(warnings, cfg.modflow_exe, "PierNet_MODFLOW_EXE")

    return ConfigValidation(config=cfg, errors=tuple(errors), warnings=tuple(warnings))


def log_runtime_config() -> ConfigValidation:
    validation = validate_runtime_config()
    summary = json.dumps(validation.config.safe_summary(), ensure_ascii=False, sort_keys=True)
    logger.info("PierNet runtime configuration: %s", summary)
    for warning in validation.warnings:
        logger.warning("PierNet runtime configuration warning: %s", warning)
    if validation.errors:
        for error in validation.errors:
            logger.error("PierNet runtime configuration error: %s", error)
        raise RuntimeError("PierNet runtime configuration is invalid: " + "; ".join(validation.errors))
    return validation


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate PierNet runtime configuration.")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    validation = validate_runtime_config()
    if args.json:
        print(json.dumps(validation.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("PierNet runtime configuration")
        for key, value in validation.config.safe_summary().items():
            print(f"  {key}: {value}")
        for warning in validation.warnings:
            print(f"WARN: {warning}", file=sys.stderr)
        for error in validation.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
