"""Operational metrics for PierNet single-node deployments."""

from __future__ import annotations

import shutil
import time
from collections import Counter
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from PierNet.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT
from PierNet.shared.tasks import workers
from PierNet.shared.tasks.state import ACTIVE_STATUSES, TERMINAL_STATUSES, normalize_status
from PierNet.synth.services import job_store as synth_job_store
from PierNet.training.services import training_manager

router = APIRouter(prefix="/metrics", tags=["metrics"])


class StatusSummary(BaseModel):
    total: int
    queued: int
    active: int
    terminal: int
    counts: dict[str, int] = Field(default_factory=dict)


class QueueSummary(BaseModel):
    queued: int
    oldest_queued_age_seconds: float | None = None


class WorkerMetrics(BaseModel):
    total: int
    running: int
    stale: int
    stopped: int
    busy: int
    items: list[dict[str, Any]] = Field(default_factory=list)


class ResourceMetrics(BaseModel):
    disk: dict[str, Any] = Field(default_factory=dict)
    gpus: list[dict[str, Any]] = Field(default_factory=list)


class MetricsSummary(BaseModel):
    status: str
    generated_at: float
    jobs: dict[str, StatusSummary]
    queues: dict[str, QueueSummary]
    workers: WorkerMetrics
    resources: ResourceMetrics
    warnings: list[str] = Field(default_factory=list)


def _status_summary(jobs: list[dict[str, Any]]) -> StatusSummary:
    counts = Counter(normalize_status(job.get("status")) for job in jobs)
    return StatusSummary(
        total=len(jobs),
        queued=counts.get("queued", 0),
        active=sum(counts.get(status, 0) for status in ACTIVE_STATUSES),
        terminal=sum(counts.get(status, 0) for status in TERMINAL_STATUSES),
        counts=dict(sorted(counts.items())),
    )


def _queue_summary(jobs: list[dict[str, Any]], *, now: float) -> QueueSummary:
    queued = [job for job in jobs if normalize_status(job.get("status")) == "queued"]
    ages: list[float] = []
    for job in queued:
        created_at = job.get("created_at") or job.get("started_at") or job.get("queued_at")
        try:
            ages.append(max(0.0, now - float(created_at)))
        except (TypeError, ValueError):
            continue
    return QueueSummary(queued=len(queued), oldest_queued_age_seconds=max(ages) if ages else None)


def _disk_metrics() -> dict[str, Any]:
    target = DATA_ROOT if DATA_ROOT.exists() else PROJECT_ROOT
    usage = shutil.disk_usage(target)
    free_ratio = usage.free / usage.total if usage.total else 0.0
    return {
        "path": str(target),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_ratio": free_ratio,
    }


def build_metrics_summary() -> MetricsSummary:
    now = time.time()
    synth_jobs = synth_job_store.list_jobs(limit=10000)
    training_jobs = training_manager.list_jobs(refresh=True)
    worker_items = workers.list_workers()
    worker_counts = Counter(str(item.get("status") or "unknown") for item in worker_items)
    disk = _disk_metrics()
    try:
        gpus = training_manager.get_gpu_inventory()
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        gpus = []
        gpu_error = str(exc)
    else:
        gpu_error = None

    queue_summaries = {
        "synth": _queue_summary(synth_jobs, now=now),
        "training": _queue_summary(training_jobs, now=now),
    }
    queued_total = sum(item.queued for item in queue_summaries.values())
    running_workers = worker_counts.get("running", 0)
    warnings: list[str] = []
    if worker_counts.get("stale", 0):
        warnings.append(f"{worker_counts['stale']} worker(s) are stale")
    if queued_total and running_workers == 0:
        warnings.append("queued jobs exist but no running worker is reporting heartbeat")
    if disk.get("free_ratio", 1.0) < 0.1:
        warnings.append("disk free ratio is below 10%")
    if gpu_error:
        warnings.append(f"gpu inventory unavailable: {gpu_error}")

    status = "degraded" if warnings else "ok"
    return MetricsSummary(
        status=status,
        generated_at=now,
        jobs={
            "synth": _status_summary(synth_jobs),
            "training": _status_summary(training_jobs),
        },
        queues=queue_summaries,
        workers=WorkerMetrics(
            total=len(worker_items),
            running=worker_counts.get("running", 0),
            stale=worker_counts.get("stale", 0),
            stopped=worker_counts.get("stopped", 0),
            busy=sum(1 for item in worker_items if item.get("current_job_id")),
            items=worker_items,
        ),
        resources=ResourceMetrics(disk=disk, gpus=gpus),
        warnings=warnings,
    )


@router.get("/summary", response_model=MetricsSummary)
def metrics_summary() -> MetricsSummary:
    return build_metrics_summary()
