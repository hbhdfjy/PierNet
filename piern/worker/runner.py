"""Maintenance worker for shared runtime housekeeping.

The current platform still executes long jobs through existing service managers.
This worker gives deployments a stable process target and owns shared maintenance
tasks such as expiring stale cooperative locks until task execution is migrated
behind a queue.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time

from piern.shared.tasks import locks

LOGGER = logging.getLogger(__name__)


def run(*, interval: float = 10.0, once: bool = False) -> int:
    stopping = False

    def _stop(signum, frame):
        del signum, frame
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    LOGGER.info("PiERN maintenance worker started interval=%s once=%s", interval, once)
    while not stopping:
        expired = locks.cleanup_expired()
        if expired:
            LOGGER.info("expired %s stale task lock(s)", expired)
        if once:
            break
        time.sleep(max(1.0, interval))
    LOGGER.info("PiERN maintenance worker stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PiERN maintenance worker loop.")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(interval=args.interval, once=args.once)
