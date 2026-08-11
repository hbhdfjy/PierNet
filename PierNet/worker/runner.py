"""Maintenance worker for shared runtime housekeeping and queued synthesis jobs."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time

from PierNet.shared.runtime.config import load_runtime_config
from PierNet.shared.tasks import locks, workers
from PierNet.synth.services import worker_queue as synth_worker_queue
from PierNet.training.services import training_manager
from PierNet.training.services import worker_queue as training_worker_queue
from PierNet.training.services.cache_cleanup import cleanup_training_cache

LOGGER = logging.getLogger(__name__)
TRAINING_STATE_REFRESH_INTERVAL_SECONDS = max(
    1.0,
    float(os.getenv("PierNet_WORKER_TRAINING_REFRESH_INTERVAL_SECONDS", "5")),
)


def refresh_training_state(*, worker_id: str) -> None:
    """Advance Router/Text2Comp pipelines outside request-handling threads."""
    with workers.heartbeat_while(
        worker_id=worker_id,
        kind="PierNet-worker",
        metadata={"phase": "training-state-refresh"},
        interval=5.0,
    ):
        training_manager.list_jobs(refresh=True)


def run(*, interval: float = 10.0, once: bool = False) -> int:
    stopping = False

    def _stop(signum, frame):
        del signum, frame
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    worker_id = workers.default_worker_id()
    runtime_config = load_runtime_config()
    last_cache_cleanup_at = 0.0
    last_training_refresh_at = 0.0
    workers.upsert_worker(worker_id=worker_id, kind="PierNet-worker", status="running")
    LOGGER.info("PierNet worker started worker_id=%s interval=%s once=%s", worker_id, interval, once)
    while not stopping:
        reaped = training_manager.reap_finished_processes()
        if reaped:
            LOGGER.info("reaped %s finished training child process(es)", reaped)
        refresh_now = time.monotonic()
        if once or refresh_now - last_training_refresh_at >= TRAINING_STATE_REFRESH_INTERVAL_SECONDS:
            try:
                refresh_training_state(worker_id=worker_id)
            except Exception:
                LOGGER.exception("training state refresh failed")
            finally:
                last_training_refresh_at = time.monotonic()
        if runtime_config.cache_cleanup_enabled:
            now = time.time()
            cleanup_interval = max(0.1, runtime_config.cache_cleanup_interval_hours) * 3600
            if now - last_cache_cleanup_at >= cleanup_interval:
                try:
                    cleanup_result = cleanup_training_cache(
                        router_jsonl_cache_dir=runtime_config.router_jsonl_cache_dir,
                        training_artifact_root=runtime_config.artifact_root / "token_router",
                        router_jsonl_ttl_days=runtime_config.router_jsonl_cache_ttl_days,
                        training_prepared_ttl_days=runtime_config.training_prepared_cache_ttl_days,
                        max_delete_bytes=int(runtime_config.cache_cleanup_max_delete_gb * 1024**3),
                        dry_run=runtime_config.cache_cleanup_dry_run,
                    )
                    LOGGER.info(
                        "training cache cleanup completed dry_run=%s candidates=%s deleted=%s "
                        "reclaimable_bytes=%s deleted_bytes=%s errors=%s",
                        cleanup_result.dry_run,
                        len(cleanup_result.candidates),
                        len(cleanup_result.deleted),
                        cleanup_result.reclaimable_bytes,
                        cleanup_result.deleted_bytes,
                        len(cleanup_result.errors),
                    )
                except Exception:
                    LOGGER.exception("training cache cleanup failed")
                finally:
                    last_cache_cleanup_at = now
        expired = locks.cleanup_expired()
        if expired:
            LOGGER.info("expired %s stale task lock(s)", expired)
        workers.upsert_worker(worker_id=worker_id, kind="PierNet-worker", status="running")
        ran_job = synth_worker_queue.run_next_queued_job(worker_id=worker_id)
        if not ran_job:
            ran_job = training_worker_queue.run_next_queued_job(worker_id=worker_id)
        if once:
            break
        time.sleep(1.0 if ran_job else max(1.0, interval))
    workers.mark_worker_stopped(worker_id)
    LOGGER.info("PierNet worker stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PierNet worker loop.")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(interval=args.interval, once=args.once)
