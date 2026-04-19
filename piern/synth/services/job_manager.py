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


def publish(record: JobRecord, event: dict) -> None:
    """向任务追加事件，并扇出给当前所有订阅者。"""
    with record._lock:
        record.events.append(event)
        dead = []
        for q in record._subscribers:
            try:
                record.loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                dead.append(q)
        for q in dead:
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
