"""Durable synthesis task queue consumed by the maintenance worker."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

from piern.shared.tasks import locks as task_locks, workers
from piern.synth.services import generation_executor, job_manager, job_store, router_executor
from piern.synth.services.job_manager import JobRecord, publish

LOGGER = logging.getLogger(__name__)

QUEUE_LOCK_KEY = "worker:synth-queue"
QUEUE_LOCK_TTL_SECONDS = float(os.getenv("PIERN_WORKER_QUEUE_LOCK_TTL_SECONDS", "60"))
DISPATCH: dict[str, Callable[[JobRecord, dict], None]] = {
    "generate_templates": generation_executor.run_generate_templates_job,
    "fill_samples": generation_executor.run_fill_samples_job,
    "router": router_executor.run_router_build_job,
}


def queue_enabled() -> bool:
    return os.getenv("PIERN_WORKER_QUEUE_SYNTH", "1").strip().lower() not in {"0", "false", "no", "off"}


def _queued_jobs() -> list[dict]:
    jobs = [job for job in job_store.list_jobs(status="queued", limit=100) if job.get("job_type") in DISPATCH]
    return sorted(jobs, key=lambda item: float(item.get("created_at") or item.get("started_at") or 0))


def run_next_queued_job(*, worker_id: str | None = None) -> bool:
    if not queue_enabled():
        return False
    owner = worker_id or workers.default_worker_id()
    if not task_locks.acquire_lock(QUEUE_LOCK_KEY, owner, ttl_seconds=QUEUE_LOCK_TTL_SECONDS):
        return False
    try:
        for stored in _queued_jobs():
            latest = job_store.load_job(str(stored["job_id"]))
            if not latest or latest.get("status") != "queued":
                continue
            record = job_manager.record_from_stored(latest)
            record.status = "running"
            publish(
                record,
                {
                    "type": "started",
                    "ts": time.time(),
                    "message": f"Worker {owner} started {record.job_type}",
                },
            )
            dispatcher = DISPATCH[record.job_type]
            LOGGER.info("running queued synthesis job job_id=%s type=%s worker_id=%s", record.job_id, record.job_type, owner)
            with workers.heartbeat_while(worker_id=owner, kind="piern-worker", current_job_id=record.job_id):
                dispatcher(record, latest.get("request_json") or {})
            return True
    finally:
        task_locks.release_lock(QUEUE_LOCK_KEY, owner)
    return False
