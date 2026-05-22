"""后台任务注册、SQLite 状态持久化与 SSE 事件分发。

每个 job 都会保留事件历史。新的订阅者先收到快照，再接收后续 queue。
进程重启后，SQLite 中未完成的任务会被标记为 external_terminated，前端仍能看到历史状态和日志。
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from piern.shared.tasks import locks as task_locks
from piern.shared.tasks.state import ACTIVE_STATUSES, TERMINAL_STATUSES, normalize_status
from piern.synth.services import job_store

logger = logging.getLogger(__name__)

SYNTH_ACTIVE_STATUSES = set(ACTIVE_STATUSES)
SYNTH_TERMINAL_STATUSES = set(TERMINAL_STATUSES - {"deleted"})
SYNTH_LOCK_TTL_SECONDS = float(os.getenv("PIERN_SYNTH_LOCK_TTL_SECONDS", str(24 * 3600)))


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    status: str
    loop: Optional[asyncio.AbstractEventLoop]
    scenario_totals: dict
    started_at: float
    events: list = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {"elapsed_sec": 0.0, "samples_per_sec": 0.0})
    finished_at: Optional[float] = None
    error_message: Optional[str] = None
    persisted: bool = False
    _subscribers: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    future: Optional[asyncio.Task] = None
    proc: Optional[subprocess.Popen] = None
    proc_uses_process_group: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock_keys: list[str] = field(default_factory=list)


_jobs: dict[str, JobRecord] = {}


def create_job(
    job_type: str,
    scenario_totals: dict = None,
    request: dict[str, Any] | None = None,
    lock_keys: list[str] | None = None,
    status: str = "running",
) -> JobRecord:
    """创建后台任务，并绑定当前 async loop 供 FastAPI SSE 推送。"""
    loop = asyncio.get_running_loop()
    job_id = _make_prefix(job_type) + str(uuid.uuid4())[:6]
    acquired_locks: list[str] = []
    for lock_key in lock_keys or []:
        if not task_locks.acquire_lock(
            lock_key,
            job_id,
            ttl_seconds=SYNTH_LOCK_TTL_SECONDS,
            metadata={"job_type": job_type},
        ):
            for acquired in acquired_locks:
                task_locks.release_lock(acquired, job_id)
            raise RuntimeError(f"资源已被其他任务占用: {lock_key}")
        acquired_locks.append(lock_key)
    record = JobRecord(
        job_id=job_id,
        job_type=job_type,
        status=normalize_status(status, fallback="running"),
        loop=loop,
        scenario_totals=scenario_totals or {},
        started_at=time.time(),
        lock_keys=acquired_locks,
    )
    _jobs[job_id] = record
    _persist_record(record, request=request)
    return record


def _make_prefix(job_type: str) -> str:
    prefixes = {
        "generate_templates": "tmpl-",
        "fill_samples": "fill-",
        "register": "reg-",
        "simulate": "sim-",
        "router": "router-",
    }
    return prefixes.get(job_type, "job-")


def get_job(job_id: str) -> Optional[JobRecord]:
    job = _jobs.get(job_id)
    if job is not None:
        return job
    stored = job_store.load_job(job_id)
    if not stored:
        return None
    return _record_from_stored(stored)


def _record_from_stored(stored: dict[str, Any]) -> JobRecord:
    return JobRecord(
        job_id=str(stored["job_id"]),
        job_type=str(stored.get("job_type") or "job"),
        status=normalize_status(stored.get("status")),
        loop=None,
        scenario_totals=stored.get("scenario_totals") or {},
        started_at=float(stored.get("started_at") or stored.get("created_at") or time.time()),
        events=list(stored.get("events") or []),
        progress=stored.get("progress") or {},
        stats=stored.get("stats") or {"elapsed_sec": 0.0, "samples_per_sec": 0.0},
        finished_at=stored.get("finished_at"),
        error_message=stored.get("error_message"),
        persisted=True,
    )


def record_from_stored(stored: dict[str, Any]) -> JobRecord:
    return _record_from_stored(stored)


def should_stop(record: JobRecord) -> bool:
    if record.stop_event.is_set() or record.status in SYNTH_TERMINAL_STATUSES:
        return True
    stored = job_store.load_job(record.job_id, include_events=False)
    return bool(stored and stored.get("status") in {"terminated", "external_terminated", "error"})


def _persist_record(record: JobRecord, request: dict[str, Any] | None = None) -> None:
    try:
        job_store.upsert_job(
            job_id=record.job_id,
            job_type=record.job_type,
            status=record.status,
            started_at=record.started_at,
            finished_at=record.finished_at,
            pid=record.proc.pid if record.proc is not None else None,
            request_json=request,
            scenario_totals=record.scenario_totals,
            progress=record.progress,
            stats=record.stats,
            error_message=record.error_message,
        )
    except Exception:
        logger.exception("Failed to persist synthesis job %s", record.job_id)


def _persist_event(record: JobRecord, event: dict[str, Any]) -> None:
    try:
        job_store.append_event(record.job_id, event)
        _persist_record(record)
    except Exception:
        logger.exception("Failed to persist synthesis job event job=%s event=%s", record.job_id, event.get("type"))


def _coerce_non_negative_int(value, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _normalize_progress(progress: dict) -> dict | None:
    scenario = str(progress.get("scenario") or "").strip()
    if not scenario:
        return None
    return {
        "scenario": scenario,
        "done": _coerce_non_negative_int(progress.get("done")),
        "total": _coerce_non_negative_int(progress.get("total")),
    }


def _progress_stats(record: JobRecord) -> dict:
    elapsed = max(0.0, time.time() - record.started_at)
    total_done = sum(
        _coerce_non_negative_int(item.get("done"))
        for item in record.progress.values()
        if isinstance(item, dict)
    )
    samples_per_sec = total_done / elapsed if elapsed > 0 else 0.0
    return {
        "elapsed_sec": elapsed,
        "samples_per_sec": samples_per_sec,
    }


def _apply_event_state(record: JobRecord, event: dict) -> None:
    event_type = event.get("type")
    if event_type in SYNTH_TERMINAL_STATUSES:
        record.status = normalize_status(event_type)
        record.finished_at = record.finished_at or float(event.get("ts") or time.time())
        if event_type in {"error", "terminated", "external_terminated"}:
            message = event.get("message")
            record.error_message = str(message) if message else record.error_message

    scenario_totals = event.get("scenario_totals")
    if isinstance(scenario_totals, dict):
        for scenario, total in scenario_totals.items():
            record.scenario_totals[str(scenario)] = _coerce_non_negative_int(total)

    progress = event.get("progress")
    if isinstance(progress, dict):
        normalized = _normalize_progress(progress)
        if normalized is not None:
            event["progress"] = normalized
            record.progress[normalized["scenario"]] = normalized
            event.setdefault("stats", _progress_stats(record))

    stats = event.get("stats")
    if isinstance(stats, dict):
        record.stats.update(stats)


def _completion_progress_events(record: JobRecord, ts: float) -> list[dict]:
    events: list[dict] = []
    for scenario, raw_total in sorted(record.scenario_totals.items()):
        total = _coerce_non_negative_int(raw_total)
        if total <= 0:
            continue
        current = record.progress.get(scenario)
        current_done = _coerce_non_negative_int(current.get("done")) if isinstance(current, dict) else 0
        current_total = _coerce_non_negative_int(current.get("total")) if isinstance(current, dict) else total
        if current_done >= total and current_total == total:
            continue
        events.append({
            "type": "log",
            "line": f"  {scenario}: {total}/{total}",
            "ts": ts,
            "progress": {"scenario": scenario, "done": total, "total": total},
        })
    return events


def publish(record: JobRecord, event: dict) -> None:
    """向任务追加事件，写入 SQLite，并扇出给当前所有订阅者。"""
    with record._lock:
        events = []
        if event.get("type") == "done":
            events.extend(_completion_progress_events(record, float(event.get("ts") or time.time())))
        events.append(event)

        dead = []
        for item in events:
            _apply_event_state(record, item)
            record.events.append(item)
            if record.status in SYNTH_TERMINAL_STATUSES:
                task_locks.release_owner(record.job_id)
                record.lock_keys = []
            _persist_event(record, item)
            if record.loop is None:
                continue
            for q in record._subscribers:
                try:
                    record.loop.call_soon_threadsafe(q.put_nowait, item)
                except Exception:
                    dead.append(q)
        for q in dead:
            if q in record._subscribers:
                record._subscribers.remove(q)


def subscribe(record: JobRecord) -> tuple[list[dict], asyncio.Queue]:
    """返回历史事件快照和新的订阅队列。"""
    q: asyncio.Queue = asyncio.Queue()
    with record._lock:
        snapshot = list(record.events)
        if record.status not in SYNTH_TERMINAL_STATUSES and record.loop is not None:
            record._subscribers.append(q)
    return snapshot, q


def unsubscribe(record: JobRecord, q: asyncio.Queue) -> None:
    """移除一个 SSE 订阅队列。"""
    with record._lock:
        try:
            record._subscribers.remove(q)
        except ValueError:
            pass


def terminate_job(job_id: str) -> bool:
    record = _jobs.get(job_id)
    if not record:
        stored = job_store.load_job(job_id)
        if not stored:
            return False
        if stored.get("status") not in SYNTH_TERMINAL_STATUSES:
            synthetic = _record_from_stored(stored)
            synthetic.status = "terminated"
            synthetic.finished_at = time.time()
            synthetic.error_message = "任务已由平台终止。"
            _persist_record(synthetic)
            job_store.append_event(
                synthetic.job_id,
                {"type": "terminated", "ts": synthetic.finished_at, "message": synthetic.error_message},
            )
            task_locks.release_owner(synthetic.job_id)
        return True

    record.status = "terminated"
    task_locks.release_owner(record.job_id)
    record.lock_keys = []
    record.stop_event.set()
    publish(record, {"type": "terminated", "ts": time.time(), "message": "任务已由平台终止。"})

    if record.proc is not None:
        try:
            if record.proc_uses_process_group:
                os.killpg(os.getpgid(record.proc.pid), signal.SIGKILL)
            else:
                record.proc.kill()
        except Exception as e:
            logger.warning("终止子进程失败，job=%s: %s", record.job_id, e)
        finally:
            record.proc = None
            record.proc_uses_process_group = False

    if record.future and not record.future.done():
        record.future.cancel()
    return True


def all_jobs(max_recent: int = 200) -> dict[str, JobRecord]:
    jobs = dict(_jobs)
    for stored in job_store.list_jobs(limit=max_recent, include_events=False):
        job_id = str(stored["job_id"])
        if job_id not in jobs:
            jobs[job_id] = _record_from_stored(stored)
    return jobs


def running_jobs(job_types: set[str] | None = None) -> list[JobRecord]:
    jobs = []
    for job in all_jobs().values():
        if job.status in SYNTH_ACTIVE_STATUSES and (job_types is None or job.job_type in job_types):
            jobs.append(job)
    return sorted(jobs, key=lambda item: item.started_at, reverse=True)


def assert_no_running_jobs(job_types: set[str], *, message: str) -> None:
    active = running_jobs(job_types)
    if active:
        ids = ", ".join(job.job_id for job in active)
        raise RuntimeError(f"{message}: {ids}")


def _recover_orphaned_jobs() -> None:
    try:
        recovered = job_store.mark_incomplete_external_terminated(active_job_ids=_jobs.keys())
    except Exception:
        logger.exception("Failed to recover orphaned synthesis jobs")
        return
    if recovered:
        for job_id in recovered:
            task_locks.release_owner(job_id)
        logger.info("Recovered orphaned synthesis jobs as external_terminated: %s", ", ".join(recovered))


_recover_orphaned_jobs()
