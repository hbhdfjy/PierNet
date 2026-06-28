from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from threading import RLock, Thread
from typing import Any

from PierNet.synth.services import expert_models as uploaded_expert_models
from PierNet.training.router.data import inspect_router_input_representation
from PierNet.shared.runtime.paths import ARTIFACT_ROOT, DATA_ROOT, PROJECT_ROOT, RUNLOG_ROOT
from PierNet.shared.tasks import locks as task_locks
from PierNet.training.services import job_store as training_job_store
from PierNet.training.text2comp import text2comp_manager
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
AUTO_STOP_METRICS = {"accuracy", "precision", "recall", "f1", "pr_auc"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


QUICK_TRAINING_DEFAULTS = {
    "epochs": _env_int("PierNet_QUICK_TRAINING_EPOCHS", 0),
    "eval_interval": _env_int("PierNet_QUICK_TRAINING_EVAL_INTERVAL", 1),
    "keep_last_epochs": _env_int("PierNet_QUICK_TRAINING_KEEP_LAST_EPOCHS", 5),
    "seed": _env_int("PierNet_QUICK_TRAINING_SEED", 42),
    "batch_size": _env_int("PierNet_QUICK_TRAINING_BATCH_SIZE", 256),
    "test_batch_size": _env_int("PierNet_QUICK_TRAINING_TEST_BATCH_SIZE", 256),
    "learning_rate": _env_float("PierNet_QUICK_TRAINING_LEARNING_RATE", 2e-4),
    "weight_decay": _env_float("PierNet_QUICK_TRAINING_WEIGHT_DECAY", 1e-2),
    "num_workers": _env_int("PierNet_QUICK_TRAINING_NUM_WORKERS", 8),
    "prepare_workers": None,
    "test_ratio": _env_float("PierNet_QUICK_TRAINING_TEST_RATIO", 0.10),
    "max_train_samples": None,
    "max_test_samples": None,
    "resume_from": None,
    "input_representation": "embedding",
    "auto_stop_enabled": True,
    "auto_stop_metric": os.getenv("PierNet_QUICK_TRAINING_AUTO_STOP_METRIC", "f1"),
    "auto_stop_threshold": _env_float("PierNet_QUICK_TRAINING_AUTO_STOP_THRESHOLD", 0.98),
    "auto_stop_min_epochs": _env_int("PierNet_QUICK_TRAINING_AUTO_STOP_MIN_EPOCHS", 1),
}
QUICK_TEXT2COMP_DEFAULTS = {
    "epochs": _env_int("PierNet_QUICK_TEXT2COMP_EPOCHS", 1),
    "batch_size": _env_int("PierNet_QUICK_TEXT2COMP_BATCH_SIZE", 8),
    "test_batch_size": _env_int("PierNet_QUICK_TEXT2COMP_TEST_BATCH_SIZE", 8),
    "learning_rate": _env_float("PierNet_QUICK_TEXT2COMP_LEARNING_RATE", 1e-5),
    "weight_decay": _env_float("PierNet_QUICK_TEXT2COMP_WEIGHT_DECAY", 1e-2),
    "num_workers": _env_int("PierNet_QUICK_TEXT2COMP_NUM_WORKERS", 2),
    "test_ratio": _env_float("PierNet_QUICK_TEXT2COMP_TEST_RATIO", 0.1),
    "max_length": _env_int("PierNet_QUICK_TEXT2COMP_MAX_LENGTH", 512),
    "eval_interval": _env_int("PierNet_QUICK_TEXT2COMP_EVAL_INTERVAL", 1),
    "freeze_base": True,
    "max_samples": _env_int("PierNet_QUICK_TEXT2COMP_MAX_SAMPLES", 1024),
}
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
            "simple_pipeline": normalized.get("simple_pipeline") if isinstance(normalized.get("simple_pipeline"), dict) else {},
        }
    )
    pipeline = normalized.get("simple_pipeline") if isinstance(normalized.get("simple_pipeline"), dict) else {}
    normalized["pipeline_stage"] = pipeline.get("stage")
    normalized["router_status"] = normalized.get("router_status")
    normalized["text2comp_job_id"] = pipeline.get("text2comp_job_id")
    normalized["text2comp_status"] = pipeline.get("text2comp_status")
    normalized["text2comp_run_dir"] = pipeline.get("text2comp_run_dir")
    normalized["text2comp_model_path"] = pipeline.get("text2comp_model_path")
    normalized["text2comp_dataset_path"] = pipeline.get("text2comp_dataset_path")
    normalized["text2comp_error_message"] = pipeline.get("text2comp_error_message")
    normalized["uploaded_expert_id"] = pipeline.get("uploaded_expert_id")
    normalized["uploaded_expert_name"] = pipeline.get("uploaded_expert_name")
    normalized["uploaded_expert_input_dim"] = pipeline.get("uploaded_expert_input_dim")
    normalized["text2comp_output_dim"] = pipeline.get("text2comp_output_dim") or pipeline.get("uploaded_expert_input_dim")
    normalized["text2comp_target_source"] = pipeline.get("text2comp_target_source")
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


def _release_gpu_lock(entry: dict[str, Any], owner: str | None = None) -> None:
    gpu_id = entry.get("gpu_id")
    if gpu_id is None:
        return
    lock_owner = owner or str(entry.get("job_id") or "")
    if not lock_owner:
        return
    try:
        task_locks.release_lock(f"gpu:{gpu_id}", lock_owner)
    except Exception:
        LOGGER.warning("Failed to release GPU lock gpu=%s owner=%s", gpu_id, lock_owner, exc_info=True)


def _should_release_gpu_lock(entry: dict[str, Any]) -> bool:
    if entry.get("status") not in TRAINING_TERMINAL_STATUSES:
        return False
    config = entry.get("config") if isinstance(entry.get("config"), dict) else {}
    if not config.get("simple_pipeline_enabled"):
        return True
    pipeline = entry.get("simple_pipeline") if isinstance(entry.get("simple_pipeline"), dict) else {}
    if entry.get("status") == "done" and pipeline.get("stage") in {None, "router"} and not pipeline.get("text2comp_job_id"):
        return False
    return True


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


def _normalize_auto_stop_options(payload: dict[str, Any]) -> dict[str, Any]:
    metric = str(payload.get("auto_stop_metric") or "f1").strip().lower()
    if metric not in AUTO_STOP_METRICS:
        metric = "f1"
    try:
        threshold = float(payload.get("auto_stop_threshold", 0.98))
    except (TypeError, ValueError):
        threshold = 0.98
    try:
        min_epochs = int(payload.get("auto_stop_min_epochs", 1))
    except (TypeError, ValueError):
        min_epochs = 1
    return {
        "auto_stop_enabled": bool(payload.get("auto_stop_enabled", False)),
        "auto_stop_metric": metric,
        "auto_stop_threshold": min(max(threshold, 0.0), 1.0),
        "auto_stop_min_epochs": max(1, min_epochs),
    }


def _safe_text2comp_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "text2comp")).strip("_-")
    if not cleaned:
        cleaned = "text2comp"
    if not cleaned[0].isalnum():
        cleaned = f"model_{cleaned}"
    return cleaned[:80]


def _active_uploaded_experts() -> list[dict[str, Any]]:
    return [
        model
        for model in uploaded_expert_models.list_assembly_models()
        if model.get("status") == "active" and model.get("assembly_enabled") and int(model.get("input_dim") or 0) > 0
    ]


def _select_uploaded_expert(model_id: str | None = None) -> dict[str, Any]:
    models = _active_uploaded_experts()
    if model_id:
        for model in models:
            if str(model.get("model_id")) == str(model_id):
                return model
        raise ValueError(f"Uploaded Expert not available for simple training: {model_id}")
    if not models:
        raise ValueError("No active Uploaded Expert is available for simple training")
    return models[0]


def _scenario_h5_path(simulator: str, scenario: str) -> Path:
    return DATA_ROOT / simulator / f"{simulator}_{scenario}.h5"


def _label_from_h5_row(
    handle: Any,
    row_index: int,
    expected_dim: int | None = None,
) -> tuple[list[float], str] | None:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - numpy is part of the runtime
        raise RuntimeError("numpy is required to prepare Text2Comp data") from exc

    source = None
    source_name = ""
    if "params" in handle:
        source = np.asarray(handle["params"][row_index], dtype=np.float32).reshape(-1)
        source_name = "params"
    elif "timeseries" in handle:
        source = np.asarray(handle["timeseries"][row_index], dtype=np.float32).reshape(-1)
        source_name = "timeseries"
    if source is None or source.size == 0 or not np.isfinite(source).all():
        return None
    if expected_dim is not None and int(source.size) != expected_dim:
        return None
    return [float(x) for x in source], source_name


def _prepare_simple_text2comp_dataset(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - h5py is part of the runtime
        raise RuntimeError("h5py is required to prepare Text2Comp data") from exc

    simulator = str(entry.get("simulator") or "")
    scenarios = [str(item) for item in entry.get("scenarios") or []]
    config = entry.get("config") if isinstance(entry.get("config"), dict) else {}
    max_samples = max(1, int(config.get("simple_text2comp_max_samples") or QUICK_TEXT2COMP_DEFAULTS["max_samples"]))
    output_path = DATA_ROOT / "text2comp" / f"{simulator}_{entry['job_id']}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    output_dim: int | None = None
    target_source: str | None = None
    used_sources: list[str] = []
    with output_path.open("w", encoding="utf-8") as handle_out:
        for scenario in scenarios:
            h5_path = _scenario_h5_path(simulator, scenario)
            if not h5_path.exists():
                skipped += 1
                continue
            used_sources.append(str(h5_path))
            with h5py.File(h5_path, "r") as h5:
                if "params" in h5:
                    n_rows = int(h5["params"].shape[0])
                elif "timeseries" in h5:
                    n_rows = int(h5["timeseries"].shape[0])
                else:
                    skipped += 1
                    continue
                for row_index in range(n_rows):
                    if generated >= max_samples:
                        break
                    label_result = _label_from_h5_row(h5, row_index, output_dim)
                    if label_result is None:
                        skipped += 1
                        continue
                    label, source_name = label_result
                    if output_dim is None:
                        output_dim = len(label)
                        target_source = source_name
                    prompt = (
                        f"请根据 {simulator.upper()} / {scenario} 场景训练数据生成 Text2Comp 目标参数。"
                        f"目标来自训练数据字段 {target_source or source_name}，共 {output_dim} 维。"
                        f"样本编号：{row_index}。"
                    )
                    handle_out.write(json.dumps({"prompt": prompt, "label": label}, ensure_ascii=False) + "\n")
                    generated += 1
                if generated >= max_samples:
                    break

    if generated == 0 or output_dim is None:
        try:
            output_path.unlink()
        except OSError:
            pass
        raise ValueError(f"No Text2Comp samples could be prepared for simulator={simulator} scenarios={scenarios}")

    return {
        "path": str(output_path),
        "generated": generated,
        "skipped": skipped,
        "output_dim": output_dim,
        "target_source": target_source or "unknown",
        "sources": used_sources,
    }


def _simple_text2comp_model_path(job: dict[str, Any] | None) -> str | None:
    if not job:
        return None
    run_dir = Path(str(job.get("run_dir") or ""))
    for name in ("final_model.pt", "best_model.pt"):
        path = run_dir / name
        if path.exists():
            return str(path)
    return None


def _simple_pipeline_fields(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.setdefault("simple_pipeline", {})


def _start_simple_text2comp_stage(entry: dict[str, Any]) -> None:
    pipeline = _simple_pipeline_fields(entry)
    if pipeline.get("text2comp_job_id"):
        return
    dataset = _prepare_simple_text2comp_dataset(entry)
    pipeline["text2comp_dataset_path"] = dataset["path"]
    pipeline["text2comp_dataset_samples"] = dataset["generated"]
    pipeline["text2comp_output_dim"] = int(dataset["output_dim"])
    pipeline["text2comp_target_source"] = dataset["target_source"]
    _append_launch_log(
        Path(entry["log_path"]),
        (
            "[pipeline] stage=text2comp_prepare "
            f"dataset={dataset['path']} samples={dataset['generated']} "
            f"output_dim={dataset['output_dim']} target_source={dataset['target_source']}"
        ),
    )
    config = entry.get("config") if isinstance(entry.get("config"), dict) else {}
    text2comp_payload = {
        "name": f"{entry['name']}-Text2Comp",
        "expert_model": "expert_model",
        "dataset_path": dataset["path"],
        "gpu_id": int(entry.get("gpu_id") or 0),
        "output_dim": int(dataset["output_dim"]),
        "epochs": int(config.get("simple_text2comp_epochs") or QUICK_TEXT2COMP_DEFAULTS["epochs"]),
        "eval_interval": int(config.get("simple_text2comp_eval_interval") or QUICK_TEXT2COMP_DEFAULTS["eval_interval"]),
        "batch_size": int(config.get("simple_text2comp_batch_size") or QUICK_TEXT2COMP_DEFAULTS["batch_size"]),
        "test_batch_size": int(
            config.get("simple_text2comp_test_batch_size") or QUICK_TEXT2COMP_DEFAULTS["test_batch_size"]
        ),
        "learning_rate": float(
            config.get("simple_text2comp_learning_rate") or QUICK_TEXT2COMP_DEFAULTS["learning_rate"]
        ),
        "weight_decay": float(config.get("simple_text2comp_weight_decay") or QUICK_TEXT2COMP_DEFAULTS["weight_decay"]),
        "num_workers": int(config.get("simple_text2comp_num_workers") or QUICK_TEXT2COMP_DEFAULTS["num_workers"]),
        "test_ratio": float(config.get("simple_text2comp_test_ratio") or QUICK_TEXT2COMP_DEFAULTS["test_ratio"]),
        "max_length": int(config.get("simple_text2comp_max_length") or QUICK_TEXT2COMP_DEFAULTS["max_length"]),
        "freeze_base": bool(config.get("simple_text2comp_freeze_base", QUICK_TEXT2COMP_DEFAULTS["freeze_base"])),
    }
    child = text2comp_manager.create_job(text2comp_payload)
    pipeline.update(
        {
            "stage": "text2comp",
            "text2comp_job_id": child.get("job_id"),
            "text2comp_status": child.get("status"),
            "text2comp_run_dir": child.get("run_dir"),
            "text2comp_model_path": _simple_text2comp_model_path(child),
            "text2comp_error_message": child.get("error_message"),
        }
    )
    _append_launch_log(Path(entry["log_path"]), f"[pipeline] stage=text2comp_launch job_id={child.get('job_id')}")


def _sync_simple_pipeline(entry: dict[str, Any]) -> dict[str, Any]:
    config = entry.get("config") if isinstance(entry.get("config"), dict) else {}
    if not config.get("simple_pipeline_enabled"):
        return entry
    pipeline = _simple_pipeline_fields(entry)
    pipeline.setdefault("stage", "router")
    entry["pipeline_stage"] = pipeline.get("stage")
    entry["router_status"] = entry.get("status")
    if entry.get("status") not in TRAINING_TERMINAL_STATUSES:
        pipeline["stage"] = "router"
        entry["pipeline_stage"] = "router"
        return entry
    if entry.get("status") != "done":
        pipeline["stage"] = "error" if entry.get("status") == "error" else "router"
        entry["pipeline_stage"] = pipeline["stage"]
        return entry

    try:
        _start_simple_text2comp_stage(entry)
    except Exception as exc:
        pipeline["stage"] = "error"
        pipeline["text2comp_status"] = "error"
        pipeline["text2comp_error_message"] = str(exc)
        entry["status"] = "error"
        entry["error_message"] = f"Text2Comp 阶段启动失败：{exc}"
        entry["pipeline_stage"] = "error"
        _append_launch_log(Path(entry["log_path"]), f"[pipeline] stage=text2comp_error error={exc}")
        return entry

    child_id = pipeline.get("text2comp_job_id")
    if child_id:
        try:
            child = text2comp_manager.get_job(str(child_id), refresh=True)
        except KeyError:
            child = None
        if child:
            child_status = str(child.get("status") or "starting")
            for key in (
                "latest_epoch",
                "latest_step",
                "steps_per_epoch",
                "global_step",
                "avg_loss",
                "steps_per_sec",
                "eta_seconds",
                "latest_test_epoch",
                "latest_metrics",
            ):
                if child.get(key) is not None:
                    entry[key] = child.get(key)
            pipeline.update(
                {
                    "stage": "text2comp" if child_status not in {"done", "error", "terminated"} else child_status,
                    "text2comp_status": child_status,
                    "text2comp_run_dir": child.get("run_dir"),
                    "text2comp_model_path": _simple_text2comp_model_path(child),
                    "text2comp_error_message": child.get("error_message"),
                }
            )
            entry["pipeline_stage"] = pipeline["stage"]
            if child_status in {"starting", "running", "evaluating", "queued"}:
                entry["status"] = child_status if child_status != "queued" else "starting"
                entry["eta_seconds"] = child.get("eta_seconds")
                return entry
            if child_status == "done":
                entry["status"] = "done"
                entry["ended_at"] = child.get("ended_at") or entry.get("ended_at")
                entry["exit_reason"] = "completed"
                entry["error_message"] = None
                entry["pipeline_stage"] = "done"
                pipeline["stage"] = "done"
                return entry
            if child_status in {"error", "terminated"}:
                entry["status"] = "error" if child_status == "error" else "terminated"
                entry["ended_at"] = child.get("ended_at") or time.time()
                entry["error_message"] = child.get("error_message") or f"Text2Comp stage {child_status}"
                entry["pipeline_stage"] = pipeline["stage"]
                return entry
    return entry


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

    if _should_release_gpu_lock(entry):
        _release_gpu_lock(entry)
    entry = _sync_simple_pipeline(entry)
    if _should_release_gpu_lock(entry):
        _release_gpu_lock(entry)
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

    pipeline = entry.get("simple_pipeline") if isinstance(entry.get("simple_pipeline"), dict) else {}
    child_id = pipeline.get("text2comp_job_id")
    if child_id:
        try:
            child = text2comp_manager.get_job(str(child_id), refresh=True)
        except KeyError:
            child = None
        if child and child.get("status") in TRAINING_ACTIVE_STATUSES:
            raise ValueError(f"Text2Comp training job is still active: {child_id}")

    remaining = [item for item in entries if item["job_id"] != job_id]
    staged = _stage_job_artifacts_for_delete(entry)
    try:
        _save_registry(remaining)
        training_job_store.mark_deleted(job_id)
    except Exception:
        _restore_staged_job_artifacts(staged)
        raise

    _remove_job_log(entry)
    _release_gpu_lock(entry, job_id)
    _remove_job_stop_file(entry)
    if child_id:
        try:
            text2comp_manager.delete_job(str(child_id))
        except (KeyError, ValueError):
            LOGGER.debug("Text2Comp child job could not be deleted: %s", child_id, exc_info=True)
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


def _auto_select_gpu_id() -> int:
    inventory = get_gpu_inventory()
    if not inventory:
        raise ValueError("No GPU found")

    def rank(gpu: dict[str, Any]) -> tuple[int, int, int, int]:
        free_memory = int(gpu.get("memory_total_mib") or 0) - int(gpu.get("memory_used_mib") or 0)
        utilization = int(gpu.get("utilization_gpu") or 0)
        return (0 if gpu.get("available") else 1, -free_memory, utilization, int(gpu.get("index") or 0))

    return int(sorted(inventory, key=rank)[0]["index"])


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
    auto_stop_options = _normalize_auto_stop_options(payload)
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
            "simple_pipeline_enabled": bool(payload.get("simple_pipeline_enabled", False)),
            "simple_text2comp_epochs": int(payload.get("simple_text2comp_epochs", QUICK_TEXT2COMP_DEFAULTS["epochs"])),
            "simple_text2comp_max_samples": int(
                payload.get("simple_text2comp_max_samples", QUICK_TEXT2COMP_DEFAULTS["max_samples"])
            ),
            **auto_stop_options,
        },
        "simple_pipeline": {
            "stage": "router",
            "uploaded_expert_id": payload.get("uploaded_expert_id") or None,
        }
        if payload.get("simple_pipeline_enabled")
        else {},
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


def create_quick_job(payload: dict[str, Any]) -> dict[str, Any]:
    simulator = str(payload.get("simulator") or "modflow").strip() or "modflow"
    requested_scenarios = [str(item).strip() for item in list(payload.get("scenarios") or []) if str(item).strip()]
    if not requested_scenarios:
        raise ValueError("Please select at least one training scenario")
    scenarios = _validate_scenarios(simulator, requested_scenarios)
    requested_gpu_id = payload.get("gpu_id")
    gpu_id = _auto_select_gpu_id() if requested_gpu_id is None else int(requested_gpu_id)
    requested_seed = payload.get("seed")
    uploaded_expert_id = payload.get("uploaded_expert_id") or None
    quick_payload = dict(QUICK_TRAINING_DEFAULTS)
    quick_payload.update(
        {
            "name": _normalize_job_name(
                payload.get("name"),
                fallback=f"简洁训练-{simulator}-{time.strftime('%m%d-%H%M')}",
            ),
            "simulator": simulator,
            "scenarios": scenarios,
            "gpu_id": gpu_id,
            "resume_from": payload.get("resume_from") or None,
            "seed": requested_seed if requested_seed is not None else QUICK_TRAINING_DEFAULTS["seed"],
            "simple_pipeline_enabled": True,
            "uploaded_expert_id": uploaded_expert_id,
            "simple_text2comp_epochs": QUICK_TEXT2COMP_DEFAULTS["epochs"],
            "simple_text2comp_max_samples": QUICK_TEXT2COMP_DEFAULTS["max_samples"],
        }
    )
    return create_job(quick_payload)


def _payload_from_queued_entry(entry: dict[str, Any]) -> dict[str, Any]:
    config = dict(entry.get("config") or {})
    pipeline = entry.get("simple_pipeline") if isinstance(entry.get("simple_pipeline"), dict) else {}
    payload = {
        **config,
        "name": entry.get("name"),
        "simulator": entry.get("simulator"),
        "scenarios": entry.get("scenarios") or [],
        "gpu_id": entry.get("gpu_id"),
        "uploaded_expert_id": pipeline.get("uploaded_expert_id"),
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
    auto_stop_options = _normalize_auto_stop_options(payload)
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
    if auto_stop_options["auto_stop_enabled"]:
        command.extend(
            [
                "--auto-stop",
                "--auto-stop-metric",
                auto_stop_options["auto_stop_metric"],
                "--auto-stop-threshold",
                str(auto_stop_options["auto_stop_threshold"]),
                "--auto-stop-min-epochs",
                str(auto_stop_options["auto_stop_min_epochs"]),
            ]
        )

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
                _release_gpu_lock({"gpu_id": gpu_id}, job_id)
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
                    "simple_pipeline_enabled": bool(payload.get("simple_pipeline_enabled", False)),
                    "simple_text2comp_epochs": int(
                        payload.get("simple_text2comp_epochs", QUICK_TEXT2COMP_DEFAULTS["epochs"])
                    ),
                    "simple_text2comp_max_samples": int(
                        payload.get("simple_text2comp_max_samples", QUICK_TEXT2COMP_DEFAULTS["max_samples"])
                    ),
                    **auto_stop_options,
                },
                "simple_pipeline": {
                    "stage": "router",
                    "uploaded_expert_id": payload.get("uploaded_expert_id") or None,
                }
                if payload.get("simple_pipeline_enabled")
                else {},
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
                _release_gpu_lock({"gpu_id": gpu_id}, job_id)
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
    pipeline = entry.get("simple_pipeline") if isinstance(entry.get("simple_pipeline"), dict) else {}
    child_id = pipeline.get("text2comp_job_id")
    if child_id and entry.get("status") in {"starting", "running", "evaluating", "stopping"}:
        try:
            child = text2comp_manager.get_job(str(child_id), refresh=True)
            if child.get("status") in {"starting", "running", "evaluating", "queued"}:
                text2comp_manager.stop_job(str(child_id))
                pipeline["text2comp_status"] = "terminated"
                pipeline["stage"] = "terminated"
                entry["status"] = "terminated"
                entry["terminated"] = True
                entry["ended_at"] = time.time()
                entry["exit_reason"] = "platform_stop_requested"
                entry["error_message"] = None
                _release_gpu_lock(entry, job_id)
                _save_registry(entries)
                return entry
        except KeyError:
            pass
    pid = entry.get("pid")
    if not pid or not _pid_alive(pid):
        if entry.get("status") in TRAINING_TERMINAL_STATUSES:
            _save_registry(entries)
            return entry
        entry["terminated"] = True
        entry["status"] = "terminated" if entry.get("status") in TRAINING_ACTIVE_STATUSES else "terminated"
        entry["ended_at"] = entry.get("ended_at") or time.time()
        _release_gpu_lock(entry, job_id)
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
