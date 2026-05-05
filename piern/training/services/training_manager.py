from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import shutil
import signal
import sys
import subprocess
import time
import uuid
from collections import deque
from functools import lru_cache
from pathlib import Path
from threading import RLock, Thread
from typing import Any

from piern.training.router.data import (
    PRETRAINED_EMBEDDINGS,
    inspect_router_input_representation,
)
from piern.shared.runtime.paths import PROJECT_ROOT

PYTHON_BIN = Path(os.getenv("PIERN_TRAINING_PYTHON", sys.executable))
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "router" / "train_token_router.py"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "token_router"
RUNLOGS_ROOT = PROJECT_ROOT / ".runlogs"
CONTROL_ROOT = RUNLOGS_ROOT / "training-controls"
REGISTRY_PATH = ARTIFACTS_ROOT / "training_jobs.json"
ROUTER_MANIFEST_PATH = PROJECT_ROOT / "data" / ".manifests" / "router.json"
ROUTER_DATA_DIR = PROJECT_ROOT / "data" / "router"
GPU_FREE_MEMORY_THRESHOLD_MIB = 2048
GPU_AVAILABLE_UTIL_THRESHOLD = 20
TRAINING_ACTIVE_STATUSES = {"queued", "starting", "running", "evaluating", "stopping"}
TRAINING_TERMINAL_STATUSES = {"done", "error", "terminated", "external_terminated"}
TRAINING_STOP_GRACE_SECONDS = 45.0

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


def _load_registry() -> list[dict[str, Any]]:
    _ensure_dirs()
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_registry(entries: list[dict[str, Any]]) -> None:
    _ensure_dirs()
    tmp_path = REGISTRY_PATH.with_suffix(REGISTRY_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp_path.chmod(0o600)
    except OSError:
        LOGGER.debug("Failed to chmod temporary training registry %s", tmp_path, exc_info=True)
    tmp_path.replace(REGISTRY_PATH)
    try:
        REGISTRY_PATH.chmod(0o600)
    except OSError:
        LOGGER.debug("Failed to chmod training registry %s", REGISTRY_PATH, exc_info=True)


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


def _make_stop_token() -> str:
    return secrets.token_urlsafe(32)


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


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _safe_kill_process_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return


def _write_stop_request(entry: dict[str, Any]) -> None:
    stop_token = entry.get("stop_token")
    stop_file_value = entry.get("stop_file")
    if not stop_token or not stop_file_value:
        raise ValueError("training job has no platform stop token; cannot request authorized stop")

    stop_file = Path(stop_file_value)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": entry["job_id"],
        "token": stop_token,
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


def _tail_lines(path: Path, limit: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            file_size = handle.tell()
            if file_size == 0:
                return []
            chunk_size = 8192
            buffers: list[bytes] = []
            remaining = limit + 1
            pos = file_size
            while pos > 0 and len(buffers) < remaining:
                read_size = min(chunk_size, pos)
                pos -= read_size
                handle.seek(pos)
                chunk = handle.read(read_size)
                buffers.append(chunk)
            tail_bytes = b"".join(reversed(buffers))
            lines = tail_bytes.decode("utf-8", errors="replace").splitlines()
            return lines[-limit:]
    except OSError:
        return []


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=256)
def _load_checkpoint_metadata(path_str: str, mtime_ns: int, size_bytes: int) -> dict[str, Any] | None:
    del mtime_ns, size_bytes
    try:
        import torch

        checkpoint = torch.load(path_str, map_location="cpu", weights_only=True)
    except Exception:
        return None
    prepared_summary = checkpoint.get("prepared_summary") or {}
    config = checkpoint.get("config") or {}
    epoch = checkpoint.get("epoch")
    try:
        parsed_epoch = int(epoch) if epoch is not None else None
    except (TypeError, ValueError):
        parsed_epoch = None

    return {
        "epoch": parsed_epoch,
        "simulator": prepared_summary.get("simulator") or config.get("simulator"),
        "scenarios": sorted(prepared_summary.get("scenarios") or config.get("scenarios") or []),
        "test_ratio": prepared_summary.get("test_ratio") if prepared_summary.get("test_ratio") is not None else config.get("test_ratio"),
        "input_representation": prepared_summary.get("input_representation") or config.get("input_representation"),
        "embedding_model": prepared_summary.get("embedding_model") or config.get("embedding_model"),
        "embedding_tokenizer": prepared_summary.get("embedding_tokenizer") or config.get("embedding_tokenizer"),
    }


def _checkpoint_epoch_from_file(path_str: str, mtime_ns: int, size_bytes: int) -> int | None:
    metadata = _load_checkpoint_metadata(path_str, mtime_ns, size_bytes)
    if metadata is None:
        return None
    return metadata.get("epoch")


def _latest_training_point(run_dir: Path) -> dict[str, Any] | None:
    log_path = run_dir / "train_log.jsonl"
    lines = _tail_lines(log_path, 1)
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _latest_test_metrics(run_dir: Path) -> dict[str, Any] | None:
    latest = run_dir / "test_metrics_latest.json"
    return _read_json(latest)


def _checkpoint_entries(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not run_dir.exists():
        return entries
    for path in sorted(run_dir.glob("router*.pt")):
        stat = path.stat()
        epoch = None
        stem = path.stem
        if stem.startswith("router_epoch_"):
            try:
                epoch = int(stem.split("_")[-1])
            except ValueError:
                epoch = None
        else:
            epoch = _checkpoint_epoch_from_file(str(path), stat.st_mtime_ns, stat.st_size)
        entries.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "epoch": epoch,
            }
        )
    return sorted(entries, key=lambda item: item["mtime"], reverse=True)


def _validate_resume_checkpoint(
    *,
    resume_from: str,
    simulator: str,
    scenarios: list[str],
    test_ratio: float,
    input_representation: str,
    embedding_model: str = "",
    embedding_tokenizer: str = "",
) -> None:
    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise ValueError(f"resume checkpoint not found: {resume_from}")

    stat = checkpoint_path.stat()
    metadata = _load_checkpoint_metadata(str(checkpoint_path), stat.st_mtime_ns, stat.st_size)
    if metadata is None:
        raise ValueError(f"resume checkpoint is unreadable: {resume_from}")

    checkpoint_simulator = metadata.get("simulator")
    checkpoint_scenarios = sorted(metadata.get("scenarios") or [])
    checkpoint_ratio = metadata.get("test_ratio")
    checkpoint_input_representation = metadata.get("input_representation")
    checkpoint_embedding_model = str(metadata.get("embedding_model") or "")
    checkpoint_embedding_tokenizer = str(metadata.get("embedding_tokenizer") or "")

    if checkpoint_simulator and checkpoint_simulator != simulator:
        raise ValueError(
            f"resume checkpoint simulator mismatch: expected {simulator}, got {checkpoint_simulator}"
        )
    if checkpoint_scenarios and checkpoint_scenarios != sorted(scenarios):
        raise ValueError(
            "resume checkpoint scenario mismatch: "
            f"expected {sorted(scenarios)}, got {checkpoint_scenarios}"
        )
    if checkpoint_ratio is not None and not math.isclose(float(checkpoint_ratio), float(test_ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"resume checkpoint test_ratio mismatch: expected {test_ratio}, got {checkpoint_ratio}"
        )
    if checkpoint_input_representation and checkpoint_input_representation != input_representation:
        raise ValueError(
            "resume checkpoint input_representation mismatch: "
            f"expected {input_representation}, got {checkpoint_input_representation}"
        )
    if input_representation == PRETRAINED_EMBEDDINGS:
        if checkpoint_embedding_model and checkpoint_embedding_model != embedding_model:
            raise ValueError(
                "resume checkpoint embedding_model mismatch: "
                f"expected {embedding_model}, got {checkpoint_embedding_model}"
            )
        if checkpoint_embedding_tokenizer and checkpoint_embedding_tokenizer != embedding_tokenizer:
            raise ValueError(
                "resume checkpoint embedding_tokenizer mismatch: "
                f"expected {embedding_tokenizer}, got {checkpoint_embedding_tokenizer}"
            )




def _stage_job_artifacts_for_delete(entry: dict[str, Any]) -> list[tuple[Path, Path]]:
    staged: list[tuple[Path, Path]] = []
    run_dir_value = entry.get("run_dir")
    if not run_dir_value:
        return staged

    run_dir = Path(run_dir_value)
    if not run_dir.exists():
        return staged

    timestamp_ms = int(time.time() * 1000)
    target = run_dir.with_name(f".deleting-{run_dir.name}-{timestamp_ms}")
    suffix = 1
    while target.exists():
        target = run_dir.with_name(f".deleting-{run_dir.name}-{timestamp_ms}-{suffix}")
        suffix += 1

    run_dir.rename(target)
    staged.append((run_dir, target))
    _load_checkpoint_metadata.cache_clear()
    return staged


def _restore_staged_job_artifacts(staged: list[tuple[Path, Path]]) -> None:
    for original, target in reversed(staged):
        if target.exists() and not original.exists():
            try:
                target.rename(original)
            except OSError:
                LOGGER.exception("Failed to restore staged training artifact %s to %s", target, original)


def _delete_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        LOGGER.exception("Failed to delete training artifact directory %s", path)
    finally:
        _load_checkpoint_metadata.cache_clear()


def _delete_job_artifacts_in_background(paths: list[Path]) -> None:
    for path in paths:
        thread = Thread(target=_delete_tree, args=(path,), name=f"delete-{path.name}", daemon=True)
        thread.start()


def _remove_job_log(entry: dict[str, Any]) -> None:
    log_path_value = entry.get("log_path")
    if not log_path_value:
        return

    log_path = Path(log_path_value)
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        LOGGER.exception("Failed to delete training log %s", log_path)


def _remove_job_stop_file(entry: dict[str, Any]) -> None:
    stop_file_value = entry.get("stop_file")
    if not stop_file_value:
        return
    try:
        Path(stop_file_value).unlink(missing_ok=True)
    except OSError:
        LOGGER.exception("Failed to delete training stop file %s", stop_file_value)


def _refresh_entry(entry: dict[str, Any]) -> dict[str, Any]:
    entry["name"] = _normalize_job_name(entry.get("name"), fallback=entry["job_id"])
    run_dir_str = entry.get("run_dir")
    log_path_str = entry.get("log_path")
    if not run_dir_str or not log_path_str:
        return entry
    run_dir = Path(run_dir_str)
    log_path = Path(log_path_str)
    pid = entry.get("pid")
    alive = _pid_alive(pid)
    latest_point = _latest_training_point(run_dir)
    if latest_point:
        entry["latest_epoch"] = int(latest_point.get("epoch", 0)) or None
        entry["latest_step"] = int(latest_point.get("step", 0)) or None
        entry["steps_per_epoch"] = int(latest_point.get("steps_per_epoch", 0)) or None
        entry["global_step"] = int(latest_point.get("global_step", 0)) or None
        entry["avg_loss"] = float(latest_point.get("avg_loss", 0.0))
        entry["steps_per_sec"] = float(latest_point.get("steps_per_sec", 0.0))
        entry["eta_seconds"] = float(latest_point.get("eta_seconds", 0.0))

    latest_metrics = _latest_test_metrics(run_dir)
    if latest_metrics:
        entry["latest_test_epoch"] = int(latest_metrics.get("epoch", 0)) or None
        overall = latest_metrics.get("overall", {})
        entry["latest_metrics"] = {
            "accuracy": overall.get("accuracy"),
            "precision": overall.get("precision"),
            "recall": overall.get("recall"),
            "f1": overall.get("f1"),
            "pr_auc": overall.get("pr_auc"),
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
    except Exception:
        _restore_staged_job_artifacts(staged)
        raise

    _remove_job_log(entry)
    _remove_job_stop_file(entry)
    _delete_job_artifacts_in_background([target for _, target in staged])
    return entry


def _dataset_manifest() -> dict[str, Any] | None:
    if not ROUTER_MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(ROUTER_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to read router manifest at %s: %s", ROUTER_MANIFEST_PATH, exc)
        return None


def list_datasets() -> list[dict[str, Any]]:
    manifest = _dataset_manifest()
    if not manifest:
        return []
    grouped: dict[str, dict[str, Any]] = {}
    for scenario in manifest.get("scenarios", []):
        simulator = scenario["simulator"]
        bucket = grouped.setdefault(simulator, {"simulator": simulator, "total_count": 0, "scenarios": []})
        bucket["total_count"] += int(scenario["router_count"])
        bucket["scenarios"].append(scenario)
    for bucket in grouped.values():
        bucket["scenarios"].sort(key=lambda item: item["scenario"])
    return sorted(grouped.values(), key=lambda item: item["simulator"])


def get_gpu_inventory() -> list[dict[str, Any]]:
    try:
        raw_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        LOGGER.info("nvidia-smi unavailable; returning no visible GPUs: %s", exc)
        return []

    rows = [line.strip() for line in raw_output.splitlines() if line.strip()]
    jobs = list_jobs(refresh=True)
    locked = {
        int(job["gpu_id"]): job["job_id"]
        for job in jobs
        if job["status"] in TRAINING_ACTIVE_STATUSES
    }
    gpus: list[dict[str, Any]] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",", maxsplit=4)]
        if len(parts) != 5:
            LOGGER.warning("Skipping malformed nvidia-smi row: %s", row)
            continue
        idx_s, name, mem_used_s, mem_total_s, util_s = parts
        index = int(idx_s)
        memory_used = int(mem_used_s)
        memory_total = int(mem_total_s)
        utilization = int(util_s)
        available = True
        reason = None
        locked_by_job_id = locked.get(index)
        if locked_by_job_id:
            available = False
            reason = f"locked by {locked_by_job_id}"
        elif (memory_total - memory_used) < GPU_FREE_MEMORY_THRESHOLD_MIB:
            available = False
            reason = "memory busy"
        elif utilization >= GPU_AVAILABLE_UTIL_THRESHOLD:
            available = False
            reason = "utilization busy"
        gpus.append(
            {
                "index": index,
                "name": name,
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
                "utilization_gpu": utilization,
                "available": available,
                "locked_by_job_id": locked_by_job_id,
                "reason": reason,
            }
        )
    return gpus


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
    if not scenarios:
        return sorted(available)
    missing = sorted(set(scenarios) - available)
    if missing:
        raise ValueError(f"Unknown scenarios for {simulator}: {missing}")
    return sorted(scenarios)


def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    gpu_id = int(payload["gpu_id"])
    gpu_map = {gpu["index"]: gpu for gpu in get_gpu_inventory()}
    gpu = gpu_map.get(gpu_id)
    if gpu is None:
        raise ValueError(f"GPU {gpu_id} not found")
    if not gpu["available"]:
        raise ValueError(f"GPU {gpu_id} is not available: {gpu['reason']}")

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
    job_id = _make_job_id()
    job_name = _normalize_job_name(payload.get("name"), fallback=job_id)
    stop_token = _make_stop_token()
    stop_file = _stop_file_for_job(job_id)
    keep_last_epochs = max(0, int(payload.get("keep_last_epochs", 5)))
    test_ratio = float(payload["test_ratio"])
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
        "--batch-size",
        str(int(payload["batch_size"])),
        "--test-batch-size",
        str(int(payload["test_batch_size"])),
        "--learning-rate",
        str(float(payload["learning_rate"])),
        "--weight-decay",
        str(float(payload["weight_decay"])),
        "--num-workers",
        str(int(payload["num_workers"])),
        "--test-ratio",
        str(test_ratio),
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

    with _REGISTRY_LOCK:
        entries = _load_registry()
        current_gpu_map = {item["index"]: item for item in get_gpu_inventory()}
        current_gpu = current_gpu_map.get(gpu_id)
        if current_gpu is None:
            raise ValueError(f"GPU {gpu_id} not found")
        if not current_gpu["available"]:
            raise ValueError(f"GPU {gpu_id} is not available: {current_gpu['reason']}")

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
            f"[launch] keep_last_epochs={keep_last_epochs}",
            "[launch] spawning training subprocess...",
        )

        log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PIERN_TRAIN_STOP_TOKEN"] = stop_token
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
            "created_at": time.time(),
            "started_at": time.time(),
            "ended_at": None,
            "pid": process.pid,
            "artifact_root": str(artifact_root),
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "stop_file": str(stop_file),
            "stop_token": stop_token,
            "config": {
                "epochs": int(payload["epochs"]),
                "eval_interval": int(payload["eval_interval"]),
                "keep_last_epochs": keep_last_epochs,
                "batch_size": int(payload["batch_size"]),
                "test_batch_size": int(payload["test_batch_size"]),
                "learning_rate": float(payload["learning_rate"]),
                "weight_decay": float(payload["weight_decay"]),
                "num_workers": int(payload["num_workers"]),
                "test_ratio": test_ratio,
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
        entries.append(entry)
        _save_registry(entries)
        return _refresh_entry(entry)


@_with_registry_lock
def delete_checkpoint(job_id: str, checkpoint_name: str) -> dict[str, Any]:
    if not checkpoint_name.startswith("router_epoch_") or not checkpoint_name.endswith(".pt"):
        raise ValueError(f"only epoch checkpoints can be deleted individually: {checkpoint_name}")

    entries = _load_registry()
    entry = _find_job(entries, job_id)
    entry = _refresh_entry(entry)
    if entry.get("status") in TRAINING_ACTIVE_STATUSES or _pid_alive(entry.get("pid")):
        raise ValueError(f"training job is still active: {job_id}")

    run_dir = Path(entry["run_dir"])
    checkpoint_path = run_dir / checkpoint_name
    try:
        checkpoint_path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("checkpoint path escapes run directory") from exc
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_name}")

    checkpoint_path.unlink()
    _load_checkpoint_metadata.cache_clear()
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
        entry["terminated"] = True
        entry["status"] = "terminated" if entry.get("status") in TRAINING_ACTIVE_STATUSES else entry.get("status", "terminated")
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
    entry["error_message"] = "Platform stop requested; waiting for checkpoint save."
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


def _downsample(points: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    sampled = [points[idx] for idx in range(0, len(points), stride)]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def get_curves(job_id: str, max_points: int = 2000) -> dict[str, Any]:
    entry = get_job(job_id, refresh=True)
    run_dir = Path(entry["run_dir"])
    training_points: list[dict[str, Any]] = []
    train_log_path = run_dir / "train_log.jsonl"
    if train_log_path.exists():
        with train_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                training_points.append(
                    {
                        "epoch": int(payload["epoch"]),
                        "step": int(payload["step"]),
                        "global_step": int(payload.get("global_step", 0)),
                        "avg_loss": float(payload["avg_loss"]),
                        "steps_per_sec": float(payload["steps_per_sec"]),
                        "eta_seconds": float(payload["eta_seconds"]),
                    }
                )
    epoch_points_map: dict[int, dict[str, Any]] = {}
    for point in training_points:
        epoch_points_map[int(point["epoch"])] = point
    training_epoch_points = [epoch_points_map[epoch] for epoch in sorted(epoch_points_map)]

    test_points: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("test_metrics_epoch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        overall = payload.get("overall", {})
        test_points.append(
            {
                "epoch": int(payload["epoch"]),
                "accuracy": float(overall.get("accuracy", 0.0)),
                "precision": float(overall.get("precision", 0.0)),
                "recall": float(overall.get("recall", 0.0)),
                "f1": float(overall.get("f1", 0.0)),
                "pr_auc": float(overall.get("pr_auc", 0.0)),
                "per_scenario": payload.get("per_scenario", {}),
            }
        )

    return {
        "job_id": job_id,
        "training_points": _downsample(training_points, max_points=max_points),
        "training_epoch_points": _downsample(training_epoch_points, max_points=max_points),
        "test_points": _downsample(test_points, max_points=max_points),
        "checkpoints": _checkpoint_entries(run_dir),
    }
