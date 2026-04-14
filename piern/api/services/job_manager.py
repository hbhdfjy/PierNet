"""统一管理所有后台任务（SSE 事件广播）。

设计：
  每个 job 维护一个 events 列表（永久保留所有历史事件）和一组订阅者 queue。
  后台线程调用 publish() 写事件：同时追加到 events 并推送给当前所有订阅者。
  SSE 端点调用 subscribe() 获得一个 queue，先回放历史事件，再实时接收新事件。
  刷新重连时重新 subscribe()，历史事件完整回放，不丢进度。
"""

import asyncio
import logging
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
    job_type: str          # "generate_templates" | "fill_samples" | "register" | "simulate"
    status: str            # "running" | "done" | "error" | "terminated"
    loop: asyncio.AbstractEventLoop
    scenario_totals: dict
    started_at: float
    # 历史事件缓冲（所有事件永久保留，支持回放）
    events: list = field(default_factory=list)
    # 当前活跃订阅者（每个 SSE 连接一个 queue）
    _subscribers: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    future: Optional[asyncio.Task] = None
    # 当前运行的子进程（仿真任务用），terminate 时 kill
    proc: Optional[subprocess.Popen] = None
    # 停止信号：线程任务通过 stop_event.is_set() 检查是否应终止
    stop_event: threading.Event = field(default_factory=threading.Event)


# 全局 job 注册表
_jobs: dict[str, JobRecord] = {}


def create_job(job_type: str, scenario_totals: dict = None) -> JobRecord:
    """创建并注册一个新 job。必须在 async 上下文中调用（FastAPI 路由内）。"""
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
    }
    return prefixes.get(job_type, "job-")


def get_job(job_id: str) -> Optional[JobRecord]:
    return _jobs.get(job_id)


def publish(record: JobRecord, event: dict) -> None:
    """后台线程调用：把事件追加到历史缓冲，并推送给所有当前订阅者。线程安全。"""
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


def subscribe(record: JobRecord) -> asyncio.Queue:
    """
    SSE 端点调用：返回一个新 queue，预填所有历史事件，再注册为订阅者。
    必须在 async 上下文（event loop 线程）中调用。

    策略：先回放历史（锁内拿快照，锁外 put），再注册为订阅者。
    这样历史和新事件之间不会有重复：注册之前的事件已在快照中，
    注册之后的新事件才会被推入 queue。
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=10000)

    # 第一步：在锁内拿历史快照（不注册订阅者）
    with record._lock:
        snapshot = list(record.events)

    # 第二步：在锁外回放历史
    for event in snapshot:
        q.put_nowait(event)

    # 第三步：在锁内注册订阅者（只有 running 状态才有新事件）
    # 注册后 publish 的新事件才会进入 q，不会与历史重复
    with record._lock:
        if record.status == "running":
            record._subscribers.append(q)

    return q


def unsubscribe(record: JobRecord, q: asyncio.Queue) -> None:
    """SSE 连接断开时移除订阅者。"""
    with record._lock:
        try:
            record._subscribers.remove(q)
        except ValueError:
            pass


def terminate_job(job_id: str) -> bool:
    record = _jobs.get(job_id)
    if not record:
        return False
    # 先改 status，再 publish：subscribe() 检查 status=="running" 才注册订阅者，
    # 改完后新连接不会被加入订阅列表，terminated 事件仍通过历史回放可见
    record.status = "terminated"
    # 设置停止信号，线程任务的回调里会检查并抛出 InterruptedError
    record.stop_event.set()
    publish(record, {"type": "terminated", "ts": time.time()})
    # kill 子进程（仿真任务）
    if record.proc is not None:
        try:
            import os, signal
            os.killpg(os.getpgid(record.proc.pid), signal.SIGKILL)
        except Exception as e:
            logger.warning(f"killpg 失败（job={record.job_id}）: {e}，尝试 proc.kill()")
            try:
                record.proc.kill()
            except Exception as e2:
                logger.error(f"proc.kill() 也失败（job={record.job_id}）: {e2}")
        record.proc = None
    if record.future and not record.future.done():
        record.future.cancel()
    return True


def all_jobs() -> dict[str, JobRecord]:
    return dict(_jobs)
