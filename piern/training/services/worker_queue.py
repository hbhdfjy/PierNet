"""Training task queue consumed by the PiERN worker."""

from __future__ import annotations

import logging
import os

from piern.shared.tasks import locks as task_locks
from piern.shared.tasks import workers
from piern.training.services import training_manager

LOGGER = logging.getLogger(__name__)

QUEUE_LOCK_KEY = "worker:training-queue"
QUEUE_LOCK_TTL_SECONDS = float(os.getenv("PIERN_WORKER_QUEUE_LOCK_TTL_SECONDS", "60"))


def queue_enabled() -> bool:
    return os.getenv("PIERN_WORKER_QUEUE_TRAINING", "1").strip().lower() not in {"0", "false", "no", "off"}


def _is_transient_launch_error(exc: ValueError) -> bool:
    message = str(exc)
    return (
        message.startswith("GPU ")
        and (
            " is not available:" in message
            or " is locked by another task" in message
        )
    )


def run_next_queued_job(*, worker_id: str | None = None) -> bool:
    if not queue_enabled():
        return False
    owner = worker_id or workers.default_worker_id()
    if not task_locks.acquire_lock(QUEUE_LOCK_KEY, owner, ttl_seconds=QUEUE_LOCK_TTL_SECONDS):
        return False
    try:
        queued = [
            job
            for job in training_manager.list_jobs(refresh=False)
            if job.get("status") == "queued" and not job.get("stop_requested")
        ]
        queued.sort(key=lambda item: (int(item.get("queue_priority") or 0), float(item.get("created_at") or 0)))
        for job in queued:
            job_id = str(job["job_id"])
            try:
                workers.upsert_worker(worker_id=owner, kind="piern-worker", status="running", current_job_id=job_id)
                LOGGER.info("starting queued training job job_id=%s worker_id=%s", job_id, owner)
                with task_locks.refresh_lock_while(
                    QUEUE_LOCK_KEY, owner, ttl_seconds=QUEUE_LOCK_TTL_SECONDS
                ), workers.heartbeat_while(
                    worker_id=owner, kind="piern-worker", current_job_id=job_id, interval=5.0
                ):
                    launched = training_manager.run_queued_job(job_id)
                if launched is None:
                    LOGGER.info("queued training job was no longer launchable job_id=%s", job_id)
                    continue
                return True
            except ValueError as exc:
                if _is_transient_launch_error(exc):
                    LOGGER.info("queued training job not ready job_id=%s reason=%s", job_id, exc)
                    continue
                LOGGER.exception("queued training job has invalid launch configuration job_id=%s", job_id)
                training_manager.mark_queued_job_error(job_id, str(exc))
                return True
            except Exception as exc:
                LOGGER.exception("queued training job failed before launch job_id=%s", job_id)
                training_manager.mark_queued_job_error(job_id, str(exc))
                return True
            finally:
                workers.upsert_worker(worker_id=owner, kind="piern-worker", status="running", current_job_id=None)
    finally:
        task_locks.release_lock(QUEUE_LOCK_KEY, owner)
    return False
