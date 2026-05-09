"""后台任务注册与 SSE 事件分发。

说明：
  每个 job 都会保留 events 历史，新的订阅者先收到快照，再接收后续 queue。
  后台线程通过 publish() 推送事件，不直接感知 SSE 订阅端。
  SSE 路由通过 subscribe() 获取 (snapshot, queue) 组合，支持断线后的历史回放。
  terminate_job() 负责同步停止子进程并广播 terminated 事件。
"""

import asyncio
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    job_type: str          # "generate_templates" | "fill_samples" | "register" | "simulate" | "router"
    status: str            # "running" | "done" | "error" | "terminated"
    loop: asyncio.AbstractEventLoop
    scenario_totals: dict
    started_at: float
    events: list = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {"elapsed_sec": 0.0, "samples_per_sec": 0.0})
    _subscribers: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    future: Optional[asyncio.Task] = None
    proc: Optional[subprocess.Popen] = None
    proc_uses_process_group: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)


_jobs: dict[str, JobRecord] = {}


def create_job(job_type: str, scenario_totals: dict = None) -> JobRecord:
    """创建后台任务，并绑定当前 async loop 供 FastAPI SSE 推送。"""
    loop = asyncio.get_running_loop()
    job_id = _make_prefix(job_type) + str(uuid.uuid4())[:6]
    record = JobRecord(
        job_id=job_id,
        job_type=job_type,
        status="running",
        loop=loop,
        scenario_totals=scenario_totals or {},
        started_at=time.time(),
    )
    _jobs[job_id] = record
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
    return _jobs.get(job_id)


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


def _apply_event_state(record: JobRecord, event: dict) -> None:
    event_type = event.get("type")
    if event_type in {"done", "error", "terminated"}:
        record.status = event_type

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
    """向任务追加事件，并扇出给当前所有订阅者。"""
    with record._lock:
        events = []
        if event.get("type") == "done":
            events.extend(_completion_progress_events(record, float(event.get("ts") or time.time())))
        events.append(event)

        dead = []
        for item in events:
            _apply_event_state(record, item)
            record.events.append(item)
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
        if record.status == "running":
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
        return False

    record.status = "terminated"
    record.stop_event.set()
    publish(record, {"type": "terminated", "ts": time.time()})

    if record.proc is not None:
        try:
            if record.proc_uses_process_group:
                os.killpg(os.getpgid(record.proc.pid), signal.SIGKILL)
            else:
                record.proc.kill()
        except Exception as e:
            logger.warning(f"终止子进程失败，job={record.job_id}: {e}")
        finally:
            record.proc = None
            record.proc_uses_process_group = False

    if record.future and not record.future.done():
        record.future.cancel()
    return True


def all_jobs() -> dict[str, JobRecord]:
    return dict(_jobs)
