"""GPU inventory helpers for training task scheduling."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterable
from typing import Any

LOGGER = logging.getLogger(__name__)


def _locked_gpu_map(
    *,
    jobs: Iterable[dict[str, Any]],
    lock_rows: Iterable[dict[str, Any]],
    active_statuses: set[str] | frozenset[str],
) -> dict[int, str]:
    locked: dict[int, str] = {}
    for job in jobs:
        if job.get("status") == "queued" or job.get("status") not in active_statuses:
            continue
        try:
            index = int(job.get("gpu_id"))
        except (TypeError, ValueError):
            continue
        if index >= 0:
            locked[index] = str(job.get("job_id"))
    for lock in lock_rows:
        try:
            index = int(str(lock["lock_key"]).split(":", 1)[1])
        except (IndexError, ValueError):
            continue
        locked.setdefault(index, str(lock["owner"]))
    return locked


def query_nvidia_smi() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=5,
    )


def build_gpu_inventory(
    *,
    jobs: Iterable[dict[str, Any]],
    lock_rows: Iterable[dict[str, Any]],
    active_statuses: set[str] | frozenset[str],
    free_memory_threshold_mib: int,
    utilization_threshold: int,
) -> list[dict[str, Any]]:
    try:
        raw_output = query_nvidia_smi()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        LOGGER.info("nvidia-smi unavailable; returning no visible GPUs: %s", exc)
        return []

    rows = [line.strip() for line in raw_output.splitlines() if line.strip()]
    locked = _locked_gpu_map(jobs=jobs, lock_rows=lock_rows, active_statuses=active_statuses)
    gpus: list[dict[str, Any]] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",", maxsplit=4)]
        if len(parts) != 5:
            LOGGER.warning("Skipping malformed nvidia-smi row: %s", row)
            continue
        idx_s, name, mem_used_s, mem_total_s, util_s = parts
        try:
            index = int(idx_s)
            memory_used = int(mem_used_s)
            memory_total = int(mem_total_s)
            utilization = int(util_s)
        except (TypeError, ValueError):
            LOGGER.warning("Skipping nvidia-smi row with non-integer fields: %s", row)
            continue
        available = True
        reason = None
        locked_by_job_id = locked.get(index)
        if locked_by_job_id:
            available = False
            reason = f"locked by {locked_by_job_id}"
        elif (memory_total - memory_used) < free_memory_threshold_mib:
            available = False
            reason = "memory busy"
        elif utilization >= utilization_threshold:
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
