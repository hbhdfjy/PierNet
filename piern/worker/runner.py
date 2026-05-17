"""Maintenance worker for shared runtime housekeeping and queued synthesis jobs."""

from __future__ import annotations

import argparse
import logging
import signal
import time

from piern.shared.tasks import locks, workers
from piern.synth.services import worker_queue as synth_worker_queue
from piern.training.services import worker_queue as training_worker_queue

LOGGER = logging.getLogger(__name__)


def run(*, interval: float = 10.0, once: bool = False) -> int:
    stopping = False

    def _stop(signum, frame):
        del signum, frame
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    worker_id = workers.default_worker_id()
    workers.upsert_worker(worker_id=worker_id, kind="piern-worker", status="running")
    LOGGER.info("PiERN worker started worker_id=%s interval=%s once=%s", worker_id, interval, once)
    while not stopping:
        expired = locks.cleanup_expired()
        if expired:
            LOGGER.info("expired %s stale task lock(s)", expired)
        workers.upsert_worker(worker_id=worker_id, kind="piern-worker", status="running")
        ran_job = synth_worker_queue.run_next_queued_job()
        if not ran_job:
            ran_job = training_worker_queue.run_next_queued_job(worker_id=worker_id)
        if once:
            break
        time.sleep(1.0 if ran_job else max(1.0, interval))
    workers.mark_worker_stopped(worker_id)
    LOGGER.info("PiERN worker stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PiERN worker loop.")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(interval=args.interval, once=args.once)
