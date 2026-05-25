from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from threading import RLock, Thread
from typing import Any

from PierNet.training.router.data import inspect_router_input_representation
from PierNet.shared.runtime.paths import ARTIFACT_ROOT, DATA_ROOT, PROJECT_ROOT, RUNLOG_ROOT
from PierNet.shared.tasks import locks as task_locks
from PierNet.training.services import job_store as training_job_store
from PierNet.training.services.checkpoint_store import (
    checkpoint_entries as _checkpoint_entries,
    clear_checkpoint_metadata_cache as _clear_checkpoint_metadata_cache,
    validate_resume_checkpoint as _validate_resume_checkpoint,
)
from PierNet.training.services.process_control import (
    pid_alive as _pid_alive,
    safe_kill_process_group as _safe_kill_process_group,
)
from PierNet.training.services.training_cleanup import (
    delete_job_artifacts_in_background as _delete_job_artifacts_in_background,
    remove_job_log as _remove_job_log,
    remove_job_stop_file as _remove_job_stop_file,
    restore_staged_job_artifacts as _restore_staged_job_artifacts,
    stage_job_artifacts_for_delete as _stage_job_artifacts_for_delete,
)
from PierNet.training.services import gpu_inventory as training_gpu
from PierNet.training.services import training_datasets
from PierNet.training.services import training_progress
from PierNet.training.services.training_runtime import (
    TRAINING_ACTIVE_STATUSES,
    TRAINING_TERMINAL_STATUSES,
    coerce_float as _coerce_float,
    coerce_int as _coerce_int,
    coerce_status as _coerce_status,
    normalize_training_config as _normalize_training_config,
    tail_lines as _tail_lines,
)

PYTHON_BIN = Path(os.getenv("PierNet_TRAINING_PYTHON", sys.executable))
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "router" / "train_token_router.py"
ARTIFACTS_ROOT = ARTIFACT_ROOT / "token_router"
RUNLOGS_ROOT = RUNLOG_ROOT
CONTROL_ROOT = RUNLOGS_ROOT / "training-controls"
DEFAULT_ROUTER_MANIFEST_PATH = DATA_ROOT / ".manifests" / "router.json"
ROUTER_MANIFEST_PATH = DEFAULT_ROUTER_MANIFEST_PATH
ROUTER_DATA_DIR = DATA_ROOT / "router"
GPU_FREE_MEMORY_THRESHOLD_MIB = 2048
GPU_AVAILABLE_UTIL_THRESHOLD = 20
GPU_LOCK_TTL_SECONDS = float(os.getenv("PierNet_GPU_LOCK_TTL_SECONDS", str(7 * 24 * 3600)))
TRAINING_QUEUE_DEFAULT_PRIORITY = int(os.getenv("PierNet_TRAINING_QUEUE_DEFAULT_PRIORITY", "100"))
TRAINING_STOP_GRACE_SECONDS = 45.0
PLATFORM_STOP_PENDING_MESSAGE = "Platform stop requested; waiting for checkpoint save."
PLATFORM_STOP_PENDING_DISPLAY = "已发送停止请求，正在等待当前 checkpoint 安全保存。"
PLATFORM_STOP_TERMINAL_MESSAGES = {
    PLATFORM_STOP_PENDING_MESSAGE,
    PLATFORM_STOP_PENDING_DISPLAY,
    "Stopped by platform request.",
}
PLATFORM_STOP_EXIT_REASONS = {"platform_stop", "platform_stop_requested"}

_REGISTRY_LOCK = RLock()
LOGGER = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNLOGS_ROOT.mkdir(parents=True, exist_ok=True)
    CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        CONTROL_ROOT.chmod(0o700)
    except OSError:
        LOGGER.debug("Failed to chmod training control directory %s", CONTROL_ROOT, exc_info=True)


def _infer_simulator_from_run_dir(run_dir: str | None) -> str | None:
    if not run_dir:
        return None
    path = Path(run_dir)
    if path.parent.name == "runs" and path.parent.parent.name:
        return path.parent.parent.name
    return None


def _infer_artifact_root(run_dir: str | None, simulator: str) -> str:
    if run_dir:
        path = Path(run_dir)
        if path.parent.name == "runs":
            return str(path.parent.parent)
        if path.parent != path:
            return str(path.parent)
    return str(ARTIFACTS_ROOT / simulator)


def _normalize_job_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    job_id = str(entry.get("job_id") or "").strip()
    if not job_id:
        LOGGER.warning("Skipping training job snapshot without job_id: %s", entry)
        return None

    normalized = dict(entry)
    config = normalized.get("config") if isinstance(normalized.get("config"), dict) else {}
    config = dict(config)
    config = _normalize_training_config(config)

    run_dir = str(normalized.get("run_dir") or ARTIFACTS_ROOT / "unknown" / "runs" / job_id)
    log_path = str(normalized.get("log_path") or RUNLOGS_ROOT / f"{job_id}.log")
    simulator = str(
        normalized.get("simulator") or config.get("simulator") or _infer_simulator_from_run_dir(run_dir) or "unknown"
    )
    scenarios = normalized.get("scenarios")
    if not isinstance(scenarios, list):
        scenarios = config.get("scenarios") if isinstance(config.get("scenarios"), list) else []

    normalized.update(
        {
            "job_id": job_id,
            "name": _normalize_job_name(normalized.get("name"), fallback=job_id),
            "status": _coerce_status(normalized.get("status")),
            "simulator": simulator,
            "scenarios": [str(item) for item in scenarios],
            "gpu_id": _coerce_int(normalized.get("gpu_id"), -1),
            "created_at": _coerce_float(normalized.get("created_at"), 0.0) or 0.0,
            "started_at": _coerce_float(normalized.get("started_at")),
            "ended_at": _coerce_float(normalized.get("ended_at")),
            "pid": _coerce_int(normalized.get("pid")),
            "artifact_root": str(normalized.get("artifact_root") or _infer_artifact_root(run_dir, simulator)),
            "run_dir": run_dir,
            "log_path": log_path,
            "config": config,
            "command": normalized.get("command") if isinstance(normalized.get("command"), list) else [],
            "checkpoints": normalized.get("checkpoints") if isinstance(normalized.get("checkpoints"), list) else [],
        }
    )
    return normalized


def _normalize_job_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            LOGGER.warning("Skipping non-dict training job snapshot: %s", entry)
            continue
        item = _normalize_job_entry(entry)
        if item is not None:
            normalized.append(item)
    return normalized


def _load_registry() -> list[dict[str, Any]]:
    _ensure_dirs()
    try:
        return _normalize_job_entries(training_job_store.list_job_snapshots())
    except Exception:
        LOGGER.exception("Failed to load training jobs from SQLite")
        return []


def _save_registry(entries: list[dict[str, Any]]) -> None:
    _ensure_dirs()
    try:
        training_job_store.save_jobs(_normalize_job_entries(entries))
    except Exception:
        LOGGER.exception("Failed to save training jobs to SQLite")
        raise


def _append_launch_log(path: Path, *lines: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _with_registry_lock(func):
    def wrapper(*args, **kwargs):
        with _REGISTRY_LOCK:
            return func(*args, **kwargs)

    return wrapper


def _find_job(entries: list[dict[str, Any]], job_id: str) -> dict[str, Any]:
    for entry in entries:
        if entry["job_id"] == job_id:
            return entry
    raise KeyError(job_id)


def _make_job_id() -> str:
    return f"train-{uuid.uuid4().hex[:8]}"


def _stop_file_for_job(job_id: str) -> Path:
    return CONTROL_ROOT / f"{job_id}.stop.json"


def _normalize_job_name(value: Any, *, fallback: str) -> str:
    if value is None:
        return fallback
    name = str(value).strip()
    return name[:80] if name else fallback


def _hash_prepared_name(
    simulator: str,
    scenarios: list[str],
    test_ratio: float,
    *,
    input_representation: str,
    embedding_model: str = "",
    embedding_tokenizer: str = "",
) -> str:
    payload = json.dumps(
        {
            "simulator": simulator,
            "scenarios": sorted(scenarios),
            "test_ratio": test_ratio,
            "input_representation": input_representation,
            "embedding_model": embedding_model,
            "embedding_tokenizer": embedding_tokenizer,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=6).hexdigest()
    return f"{simulator}-{digest}"


def _write_stop_request(entry: dict[str, Any]) -> None:
    stop_file_value = entry.get("stop_file")
    if not stop_file_value:
        raise ValueError("training job has no stop file; cannot request stop")

    stop_file = Path(stop_file_value)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": entry["job_id"],
        "requested_at": time.time(),
        "reason": "platform_stop",
    }
    tmp_path = stop_file.with_suffix(stop_file.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp_path.chmod(0o600)
    except OSError:
        LOGGER.debug("Failed to chmod temporary stop request %s", tmp_path, exc_info=True)
    tmp_path.replace(stop_file)
    try:
        stop_file.chmod(0o600)
    except OSError:
        LOGGER.debug("Failed to chmod stop request %s", stop_file, exc_info=True)


def _sync_platform_stop_message(entry: dict[str, Any], *, alive: bool) -> None:
    exit_reason = entry.get("exit_reason")
    is_platform_stop = exit_reason in PLATFORM_STOP_EXIT_REASONS or bool(entry.get("stop_requested"))
    if not is_platform_stop:
        return

    if alive and entry.get("status") == "stopping":
        if entry.get("error_message") in PLATFORM_STOP_TERMINAL_MESSAGES or not entry.get("error_message"):
            entry["error_message"] = PLATFORM_STOP_PENDING_DISPLAY
        return

    if entry.get("status") == "terminated" and exit_reason in PLATFORM_STOP_EXIT_REASONS:
        if entry.get("error_message") in PLATFORM_STOP_TERMINAL_MESSAGES:
            entry["error_message"] = None


def _refresh_entry(entry: dict[str, Any]) -> dict[str, Any]:
    entry["name"] = _normalize_job_name(entry.get("name"), fallback=entry["job_id"])
    run_dir_str = entry.get("run_dir")
    log_path_str = entry.get("log_path")
    if not run_dir_str or not log_path_str:
        return entry
    run_dir = Path(run_dir_str)
    log_path = Path(log_path_str)
    if entry.get("status") == "queued":
        return entry
    pid = entry.get("pid")
    alive = _pid_alive(pid)
    latest_point = training_progress.latest_training_point(run_dir)
    if latest_point:
        entry["latest_epoch"] = _coerce_int(latest_point.get("epoch")) or None
        entry["latest_step"] = _coerce_int(latest_point.get("step")) or None
        entry["steps_per_epoch"] = _coerce_int(latest_point.get("steps_per_epoch")) or None
        entry["global_step"] = _coerce_int(latest_point.get("global_step")) or None
        entry["avg_loss"] = _coerce_float(latest_point.get("avg_loss"))
        entry["steps_per_sec"] = _coerce_float(latest_point.get("steps_per_sec"))
        entry["eta_seconds"] = _coerce_float(latest_point.get("eta_seconds"))

    latest_metrics = training_progress.latest_test_metrics(run_dir)
    if latest_metrics:
        entry["latest_test_epoch"] = _coerce_int(latest_metrics.get("epoch")) or None
        overall = latest_metrics.get("overall", {}) if isinstance(latest_metrics.get("overall"), dict) else {}
        entry["latest_metrics"] = {
            "accuracy": _coerce_float(overall.get("accuracy")),
            "precision": _coerce_float(overall.get("precision")),
            "recall": _coerce_float(overall.get("recall")),
            "f1": _coerce_float(overall.get("f1")),
            "pr_auc": _coerce_float(overall.get("pr_auc")),
        }

    stop_requested = bool(entry.get("stop_requested"))
    if alive:
        last_lines = _tail_lines(log_path, 2)
        if stop_requested:
            entry["status"] = "stopping"
        elif last_lines and last_lines[-1].startswith("[test]"):
            entry["status"] = "evaluating"
        elif latest_point:
            entry["status"] = "running"
        else:
            entry["status"] = "starting"
    else:
        if entry.get("status") not in TRAINING_TERMINAL_STATUSES:
            last_lines = _tail_lines(log_path, 40)
            if stop_requested or entry.get("terminated") or any(line.startswith("[stop]") for line in last_lines):
                entry["status"] = "terminated"
                entry["terminated"] = True
                entry["exit_reason"] = entry.get("exit_reason") or "platform_stop"
                entry["error_message"] = entry.get("error_message") or "Stopped by platform request."
            elif (run_dir / "router_final.pt").exists():
                entry["status"] = "done"
                entry["exit_reason"] = "completed"
            else:
                if any(line.startswith("[done]") for line in last_lines):
                    entry["status"] = "done"
                    entry["exit_reason"] = "completed"
                else:
                    entry["status"] = "external_terminated"
                    entry["exit_reason"] = "external_termination"
                    if last_lines:
                        entry["error_message"] = (
                            "Training process exited without a platform stop request or completion marker. "
                            f"Last log: {last_lines[-1]}"
                        )
            entry["ended_at"] = entry.get("ended_at") or time.time()

    if entry.get("status") in TRAINING_TERMINAL_STATUSES:
        task_locks.release_lock(f"gpu:{entry.get('gpu_id')}", str(entry.get("job_id")))
    _sync_platform_stop_message(entry, alive=alive)
    entry["checkpoints"] = _checkpoint_entries(run_dir)
    return entry


@_with_registry_lock
def list_jobs(refresh: bool = True) -> list[dict[str, Any]]:
    entries = _load_registry()
    if refresh:
        entries = [_refresh_entry(entry) for entry in entries]
        _save_registry(entries)
    return sorted(entries, key=lambda item: item["created_at"], reverse=True)


@_with_registry_lock
def get_job(job_id: str, refresh: bool = True) -> dict[str, Any]:
    entries = _load_registry()
    entry = _find_job(entries, job_id)
    if refresh:
        entry = _refresh_entry(entry)
        _save_registry(entries)
    return entry


@_with_registry_lock
def delete_job(job_id: str) -> dict[str, Any]:
    entries = _load_registry()
    entry = _find_job(entries, job_id)
    entry = _refresh_entry(entry)

    if entry.get("status") in TRAINING_ACTIVE_STATUSES or _pid_alive(entry.get("pid")):
        raise ValueError(f"training job is still active: {job_id}")

    remaining = [item for item in entries if item["job_id"] != job_id]
    staged = _stage_job_artifacts_for_delete(entry)
    try:
        _save_registry(remaining)
        training_job_store.mark_deleted(job_id)
    except Exception:
        _restore_staged_job_artifacts(staged)
        raise

    _remove_job_log(entry)
    task_locks.release_lock(f"gpu:{entry.get('gpu_id')}", job_id)
    _remove_job_stop_file(entry)
    _delete_job_artifacts_in_background([target for _, target in staged])
    return entry


def list_datasets() -> list[dict[str, Any]]:
    return training_datasets.list_datasets(
        router_manifest_path=ROUTER_MANIFEST_PATH,
        default_router_manifest_path=DEFAULT_ROUTER_MANIFEST_PATH,
    )


def get_gpu_inventory() -> list[dict[str, Any]]:
    return training_gpu.build_gpu_inventory(
        jobs=list_jobs(refresh=True),
        lock_rows=task_locks.list_locks(prefix="gpu:"),
        active_statuses=TRAINING_ACTIVE_STATUSES,
        free_memory_threshold_mib=GPU_FREE_MEMORY_THRESHOLD_MIB,
        utilization_threshold=GPU_AVAILABLE_UTIL_THRESHOLD,
    )


def get_overview() -> dict[str, Any]:
    jobs = list_jobs(refresh=True)
    running_job_count = sum(1 for job in jobs if job["status"] in TRAINING_ACTIVE_STATUSES)
    completed_job_count = sum(1 for job in jobs if job["status"] == "done")
    return {
        "datasets": list_datasets(),
        "gpus": get_gpu_inventory(),
        "jobs": jobs[:12],
        "running_job_count": running_job_count,
        "completed_job_count": completed_job_count,
    }


def _validate_scenarios(simulator: str, scenarios: list[str]) -> list[str]:
    datasets = {item["simulator"]: item for item in list_datasets()}
    if simulator not in datasets:
        raise ValueError(f"Unsupported simulator: {simulator}")
    available = {scenario["scenario"] for scenario in datasets[simulator]["scenarios"]}
    requested = sorted({str(item).strip() for item in scenarios if str(item).strip()})
    if not requested:
        return sorted(available)
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"Unknown scenarios for {simulator}: {missing}")
    return requested


def _training_queue_enabled() -> bool:
    return os.getenv("PierNet_WORKER_QUEUE_TRAINING", "1").strip().lower() not in {"0", "false", "no", "off"}


def _queued_training_entry(payload: dict[str, Any]) -> dict[str, Any]:
    gpu_id = int(payload["gpu_id"])
    gpu_map = {gpu["index"]: gpu for gpu in get_gpu_inventory()}
    if gpu_id not in gpu_map:
        raise ValueError(f"GPU {gpu_id} not found")

    simulator = str(payload["simulator"])
    scenarios = _validate_scenarios(simulator, list(payload.get("scenarios") or []))
    requested_input_representation = "embedding"
    resolved_input_representation, embedding_metadata = inspect_router_input_representation(
        simulator=simulator,
        router_dir=ROUTER_DATA_DIR,
        scenarios=scenarios,
        input_representation=requested_input_representation,
    )
    job_id = _make_job_id()
    job_name = _normalize_job_name(payload.get("name"), fallback=job_id)
    created_at = time.time()
    artifact_root = ARTIFACTS_ROOT / simulator
    run_dir = artifact_root / "runs" / job_id
    log_path = RUNLOGS_ROOT / f"{job_id}.log"
    stop_file = _stop_file_for_job(job_id)
    keep_last_epochs = max(0, int(payload.get("keep_last_epochs", 5)))
    seed_value = payload.get("seed", 42)
    seed = max(0, int(seed_value if seed_value is not None else 42))
    num_workers = max(0, int(payload["num_workers"]))
    prepare_workers = payload.get("prepare_workers")
    prepare_workers = num_workers if prepare_workers is None else max(0, int(prepare_workers))
    test_ratio = float(payload["test_ratio"])
    max_train_samples = payload.get("max_train_samples")
    max_test_samples = payload.get("max_test_samples")
    max_train_samples = int(max_train_samples) if max_train_samples is not None else None
    max_test_samples = int(max_test_samples) if max_test_samples is not None else None
    resume_from = payload.get("resume_from") or None
    if resume_from:
        _validate_resume_checkpoint(
            resume_from=str(resume_from),
            simulator=simulator,
            scenarios=scenarios,
            test_ratio=test_ratio,
            input_representation=resolved_input_representation,
            embedding_model=embedding_metadata.embedding_model,
            embedding_tokenizer=embedding_metadata.tokenizer_name,
        )
    prepared_name = _hash_prepared_name(
        simulator,
        scenarios,
        test_ratio,
        input_representation=resolved_input_representation,
        embedding_model=embedding_metadata.embedding_model,
        embedding_tokenizer=embedding_metadata.tokenizer_name,
    )
    return {
        "job_id": job_id,
        "name": job_name,
        "status": "queued",
        "simulator": simulator,
        "scenarios": scenarios,
        "gpu_id": gpu_id,
        "created_at": created_at,
        "started_at": None,
        "ended_at": None,
        "pid": None,
        "artifact_root": str(artifact_root),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "stop_file": str(stop_file),
        "config": {
            "epochs": int(payload["epochs"]),
            "eval_interval": int(payload["eval_interval"]),
            "keep_last_epochs": keep_last_epochs,
            "seed": seed,
            "batch_size": int(payload["batch_size"]),
            "test_batch_size": int(payload["test_batch_size"]),
            "learning_rate": float(payload["learning_rate"]),
            "weight_decay": float(payload["weight_decay"]),
            "num_workers": num_workers,
            "prepare_workers": prepare_workers,
            "test_ratio": test_ratio,
            "max_train_samples": max_train_samples,
            "max_test_samples": max_test_samples,
            "resume_from": resume_from,
            "input_representation": resolved_input_representation,
            "embedding_model": embedding_metadata.embedding_model,
            "embedding_tokenizer": embedding_metadata.tokenizer_name,
        },
        "command": [],
        "prepared_name": prepared_name,
        "terminated": False,
        "stop_requested": False,
        "stop_requested_at": None,
        "stop_signal_sent_at": None,
        "stop_force_kill_after": None,
        "forced_kill": False,
        "exit_reason": None,
        "latest_epoch": None,
        "latest_step": None,
        "steps_per_epoch": None,
        "global_step": None,
        "avg_loss": None,
        "steps_per_sec": None,
        "eta_seconds": None,
        "latest_test_epoch": None,
        "latest_metrics": None,
        "error_message": None,
        "checkpoints": [],
        "queue_priority": int(payload.get("queue_priority") or TRAINING_QUEUE_DEFAULT_PRIORITY),
        "queued_at": created_at,
    }


def queue_job(payload: dict[str, Any]) -> dict[str, Any]:
    entry = _queued_training_entry(payload)
    _append_launch_log(
        Path(entry["log_path"]),
        f"[queue] job_id={entry['job_id']} name={entry['name']} queued_at={entry['queued_at']:.3f}",
        f"[queue] simulator={entry['simulator']} scenarios={','.join(entry['scenarios'])} gpu={entry['gpu_id']}",
        "[queue] waiting for PierNet-worker to launch training subprocess...",
    )
    with _REGISTRY_LOCK:
        entries = _load_registry()
        entries.append(entry)
        _save_registry(entries)
    return entry


def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    if _training_queue_enabled():
        return queue_job(payload)
    return _launch_job(payload)


def _payload_from_queued_entry(entry: dict[str, Any]) -> dict[str, Any]:
    config = dict(entry.get("config") or {})
    payload = {
        **config,
        "name": entry.get("name"),
        "simulator": entry.get("simulator"),
        "scenarios": entry.get("scenarios") or [],
        "gpu_id": entry.get("gpu_id"),
        "_job_id": entry.get("job_id"),
        "_created_at": entry.get("created_at"),
    }
    if config.get("resume_from") is None:
        payload.pop("resume_from", None)
    return payload


def run_queued_job(job_id: str) -> dict[str, Any] | None:
    with _REGISTRY_LOCK:
        entries = _load_registry()
        entry = _find_job(entries, job_id)
        if entry.get("status") != "queued":
            return None
        payload = _payload_from_queued_entry(entry)
    return _launch_job(payload)


def mark_queued_job_error(job_id: str, message: str) -> None:
    with _REGISTRY_LOCK:
        entries = _load_registry()
        entry = _find_job(entries, job_id)
        if entry.get("status") != "queued" or entry.get("stop_requested"):
            return
        entry["status"] = "error"
        entry["ended_at"] = time.time()
        entry["error_message"] = message
        _append_launch_log(Path(entry["log_path"]), f"[error] queued training launch failed: {message}")
        _save_registry(entries)


def _launch_job(payload: dict[str, Any]) -> dict[str, Any] | None:
    gpu_id = int(payload["gpu_id"])
    gpu_map = {gpu["index"]: gpu for gpu in get_gpu_inventory()}
    gpu = gpu_map.get(gpu_id)
    if gpu is None:
        raise ValueError(f"GPU {gpu_id} not found")
    simulator = str(payload["simulator"])
    scenarios = _validate_scenarios(simulator, list(payload.get("scenarios") or []))
    requested_input_representation = "embedding"
    resolved_input_representation, embedding_metadata = inspect_router_input_representation(
        simulator=simulator,
        router_dir=ROUTER_DATA_DIR,
        scenarios=scenarios,
        input_representation=requested_input_representation,
    )
    artifact_root = ARTIFACTS_ROOT / simulator
    job_id = str(payload.get("_job_id") or _make_job_id())
    job_name = _normalize_job_name(payload.get("name"), fallback=job_id)
    stop_file = _stop_file_for_job(job_id)
    keep_last_epochs = max(0, int(payload.get("keep_last_epochs", 5)))
    seed_value = payload.get("seed", 42)
    seed = max(0, int(seed_value if seed_value is not None else 42))
    num_workers = max(0, int(payload["num_workers"]))
    prepare_workers = payload.get("prepare_workers")
    prepare_workers = num_workers if prepare_workers is None else max(0, int(prepare_workers))
    test_ratio = float(payload["test_ratio"])
    max_train_samples = payload.get("max_train_samples")
    max_test_samples = payload.get("max_test_samples")
    max_train_samples = int(max_train_samples) if max_train_samples is not None else None
    max_test_samples = int(max_test_samples) if max_test_samples is not None else None
    prepared_name = _hash_prepared_name(
        simulator,
        scenarios,
        test_ratio,
        input_representation=resolved_input_representation,
        embedding_model=embedding_metadata.embedding_model,
        embedding_tokenizer=embedding_metadata.tokenizer_name,
    )
    run_dir = artifact_root / "runs" / job_id
    log_path = RUNLOGS_ROOT / f"{job_id}.log"
    resume_from = payload.get("resume_from") or None
    if resume_from:
        _validate_resume_checkpoint(
            resume_from=str(resume_from),
            simulator=simulator,
            scenarios=scenarios,
            test_ratio=test_ratio,
            input_representation=resolved_input_representation,
            embedding_model=embedding_metadata.embedding_model,
            embedding_tokenizer=embedding_metadata.tokenizer_name,
        )
    command = [
        str(PYTHON_BIN),
        "-u",
        str(TRAIN_SCRIPT),
        "--simulator",
        simulator,
        "--device",
        "cuda:0",
        "--epochs",
        str(int(payload["epochs"])),
        "--eval-interval",
        str(int(payload["eval_interval"])),
        "--keep-last-epochs",
        str(keep_last_epochs),
        "--seed",
        str(seed),
        "--batch-size",
        str(int(payload["batch_size"])),
        "--test-batch-size",
        str(int(payload["test_batch_size"])),
        "--learning-rate",
        str(float(payload["learning_rate"])),
        "--weight-decay",
        str(float(payload["weight_decay"])),
        "--num-workers",
        str(num_workers),
        "--prepare-workers",
        str(prepare_workers),
        "--test-ratio",
        str(test_ratio),
        "--router-dir",
        str(ROUTER_DATA_DIR),
        "--artifact-root",
        str(artifact_root),
        "--prepared-name",
        prepared_name,
        "--run-name",
        job_id,
        "--input-representation",
        requested_input_representation,
        "--stop-file",
        str(stop_file),
    ]
    if scenarios:
        command.extend(["--scenarios", *scenarios])
    if resume_from:
        command.extend(["--resume-from", str(resume_from)])
    if max_train_samples is not None:
        command.extend(["--max-train-samples", str(max_train_samples)])
    if max_test_samples is not None:
        command.extend(["--max-test-samples", str(max_test_samples)])

    with _REGISTRY_LOCK:
        entries = _load_registry()
        queued_job_id = payload.get("_job_id")
        if queued_job_id:
            try:
                queued_entry = _find_job(entries, str(queued_job_id))
            except KeyError:
                return None
            if queued_entry.get("status") != "queued" or queued_entry.get("stop_requested"):
                return None

        current_gpu_map = {item["index"]: item for item in get_gpu_inventory()}
        current_gpu = current_gpu_map.get(gpu_id)
        if current_gpu is None:
            raise ValueError(f"GPU {gpu_id} not found")
        if not current_gpu["available"] and current_gpu.get("locked_by_job_id") != job_id:
            raise ValueError(f"GPU {gpu_id} is not available: {current_gpu['reason']}")
        gpu_lock_key = f"gpu:{gpu_id}"
        if not task_locks.acquire_lock(
            gpu_lock_key,
            job_id,
            ttl_seconds=GPU_LOCK_TTL_SECONDS,
            metadata={"job_type": "training", "simulator": simulator, "scenarios": scenarios},
        ):
            raise ValueError(f"GPU {gpu_id} is locked by another task")

        lock_committed = False
        try:
            launch_started_at = time.time()
            _append_launch_log(
                log_path,
                f"[launch] job_id={job_id} name={job_name} created_at={launch_started_at:.3f}",
                f"[launch] status=starting simulator={simulator} scenarios={','.join(scenarios)} gpu={gpu_id}",
                (
                    f"[launch] prepared_name={prepared_name} "
                    f"requested_input_representation={requested_input_representation} "
                    f"prepared_input_representation={resolved_input_representation}"
                ),
                (
                    f"[launch] embedding_model={embedding_metadata.embedding_model} "
                    f"tokenizer={embedding_metadata.tokenizer_name or embedding_metadata.embedding_model}"
                ),
                f"[launch] run_dir={run_dir}",
                f"[launch] log_path={log_path}",
                f"[launch] stop_file={stop_file}",
                f"[launch] keep_last_epochs={keep_last_epochs} seed={seed}",
                f"[launch] max_train_samples={max_train_samples} max_test_samples={max_test_samples}",
                "[launch] spawning training subprocess...",
            )

            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                    text=True,
                )
            except Exception as exc:
                task_locks.release_lock(gpu_lock_key, job_id)
                log_handle.write(f"[error] failed to spawn training subprocess: {exc}\n")
                log_handle.flush()
                log_handle.close()
                raise
            log_handle.write(f"[launch] subprocess pid={process.pid} cwd={PROJECT_ROOT}\n")
            log_handle.flush()
            log_handle.close()

            entry = {
                "job_id": job_id,
                "name": job_name,
                "status": "starting",
                "simulator": simulator,
                "scenarios": scenarios,
                "gpu_id": gpu_id,
                "created_at": _coerce_float(payload.get("_created_at"), time.time()) or time.time(),
                "started_at": time.time(),
                "ended_at": None,
                "pid": process.pid,
                "artifact_root": str(artifact_root),
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "stop_file": str(stop_file),
                "config": {
                    "epochs": int(payload["epochs"]),
                    "eval_interval": int(payload["eval_interval"]),
                    "keep_last_epochs": keep_last_epochs,
                    "seed": seed,
                    "batch_size": int(payload["batch_size"]),
                    "test_batch_size": int(payload["test_batch_size"]),
                    "learning_rate": float(payload["learning_rate"]),
                    "weight_decay": float(payload["weight_decay"]),
                    "num_workers": num_workers,
                    "prepare_workers": prepare_workers,
                    "test_ratio": test_ratio,
                    "max_train_samples": max_train_samples,
                    "max_test_samples": max_test_samples,
                    "resume_from": resume_from,
                    "input_representation": resolved_input_representation,
                    "embedding_model": embedding_metadata.embedding_model,
                    "embedding_tokenizer": embedding_metadata.tokenizer_name,
                },
                "command": command,
                "prepared_name": prepared_name,
                "terminated": False,
                "stop_requested": False,
                "stop_requested_at": None,
                "stop_signal_sent_at": None,
                "stop_force_kill_after": None,
                "forced_kill": False,
                "exit_reason": None,
                "latest_epoch": None,
                "latest_step": None,
                "steps_per_epoch": None,
                "global_step": None,
                "avg_loss": None,
                "steps_per_sec": None,
                "eta_seconds": None,
                "latest_test_epoch": None,
                "latest_metrics": None,
                "error_message": None,
                "checkpoints": [],
            }
            entries = [item for item in entries if item["job_id"] != job_id]
            entries.append(entry)
            _save_registry(entries)
            lock_committed = True
            return _refresh_entry(entry)
        except Exception:
            if not lock_committed:
                process_obj = locals().get("process")
                if process_obj is not None and getattr(process_obj, "poll", lambda: None)() is None:
                    try:
                        _safe_kill_process_group(int(process_obj.pid), signal.SIGKILL)
                    except Exception:
                        LOGGER.exception("Failed to kill unregistered training process for job=%s", job_id)
                task_locks.release_lock(gpu_lock_key, job_id)
            raise


@_with_registry_lock
def delete_checkpoint(job_id: str, checkpoint_name: str) -> dict[str, Any]:
    if not checkpoint_name.startswith("router_epoch_") or not checkpoint_name.endswith(".pt"):
        raise ValueError(f"only epoch checkpoints can be deleted individually: {checkpoint_name}")
    if Path(checkpoint_name).name != checkpoint_name or "\\" in checkpoint_name:
        raise ValueError(f"checkpoint name must be a file name: {checkpoint_name}")

    entries = _load_registry()
    entry = _find_job(entries, job_id)
    entry = _refresh_entry(entry)
    if entry.get("status") in TRAINING_ACTIVE_STATUSES or _pid_alive(entry.get("pid")):
        raise ValueError(f"training job is still active: {job_id}")

    run_dir = Path(entry["run_dir"]).resolve()
    checkpoint_path = (run_dir / checkpoint_name).resolve()
    try:
        checkpoint_path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("checkpoint path escapes run directory") from exc
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_name}")

    checkpoint_path.unlink()
    _clear_checkpoint_metadata_cache()
    entry["checkpoints"] = _checkpoint_entries(run_dir)
    _save_registry(entries)
    return entry


def _force_kill_after_grace(job_id: str, pid: int, deadline: float) -> None:
    time.sleep(max(0.0, deadline - time.time()))
    with _REGISTRY_LOCK:
        entries = _load_registry()
        try:
            entry = _find_job(entries, job_id)
        except KeyError:
            return
        if not entry.get("stop_requested") or not _pid_alive(pid):
            return
        _append_launch_log(
            Path(entry["log_path"]),
            f"[stop] graceful timeout exceeded; sending SIGKILL to process group pid={pid}",
        )
        _safe_kill_process_group(pid, signal.SIGKILL)
        entry["forced_kill"] = True
        entry["terminated"] = True
        entry["status"] = "terminated"
        entry["exit_reason"] = "platform_force_kill"
        entry["ended_at"] = time.time()
        entry["error_message"] = "Stopped by platform force kill after graceful timeout."
        _save_registry(entries)


@_with_registry_lock
def stop_job(job_id: str) -> dict[str, Any]:
    entries = _load_registry()
    entry = _find_job(entries, job_id)
    entry = _refresh_entry(entry)
    pid = entry.get("pid")
    if not pid or not _pid_alive(pid):
        if entry.get("status") in TRAINING_TERMINAL_STATUSES:
            _save_registry(entries)
            return entry
        entry["terminated"] = True
        entry["status"] = "terminated" if entry.get("status") in TRAINING_ACTIVE_STATUSES else "terminated"
        entry["ended_at"] = entry.get("ended_at") or time.time()
        _save_registry(entries)
        return entry

    now = time.time()
    _write_stop_request(entry)
    _append_launch_log(
        Path(entry["log_path"]),
        f"[stop] platform stop requested at={now:.3f}; sending SIGTERM to process group pid={pid}",
    )
    _safe_kill_process_group(pid, signal.SIGTERM)
    force_after = now + TRAINING_STOP_GRACE_SECONDS
    entry["stop_requested"] = True
    entry["stop_requested_at"] = now
    entry["stop_signal_sent_at"] = now
    entry["stop_force_kill_after"] = force_after
    entry["status"] = "stopping"
    entry["exit_reason"] = "platform_stop_requested"
    entry["error_message"] = PLATFORM_STOP_PENDING_DISPLAY
    _save_registry(entries)
    thread = Thread(
        target=_force_kill_after_grace,
        args=(job_id, int(pid), force_after),
        name=f"force-stop-{job_id}",
        daemon=True,
    )
    thread.start()
    return entry


def get_job_logs(job_id: str, limit: int = 300) -> list[str]:
    entry = get_job(job_id, refresh=True)
    return _tail_lines(Path(entry["log_path"]), limit=limit)


def get_curves(job_id: str, max_points: int = 2000) -> dict[str, Any]:
    entry = get_job(job_id, refresh=True)
    return training_progress.build_curves(job_id=job_id, run_dir=Path(entry["run_dir"]), max_points=max_points)
