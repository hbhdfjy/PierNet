"""任务状态和 SSE 流路由：/api/generate/{id}/stream, status, delete。"""

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from piern.api.services import job_manager
from piern.api.services.job_manager import subscribe, unsubscribe
from piern.api.schemas.jobs import JobStatusResponse

router = APIRouter()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.get("/generate/{job_id}/stream")
async def stream_job(job_id: str):
    """
    SSE 流。每次连接都先回放所有历史事件，再实时接收新事件。
    刷新重连后进度完整恢复。
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"任务 {job_id} 不存在")

    async def _generator():
        # init 必须最先发，让前端拿到 scenario_totals 后再处理历史进度事件
        if job.scenario_totals:
            yield _sse({"type": "init", "scenario_totals": job.scenario_totals, "ts": time.time()})

        # subscribe() 预填历史事件并注册订阅者，在 init 之后调用
        q = subscribe(job)
        try:
            # 消费 queue（历史事件已预填，新事件实时推入）
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield _sse(event)
                    if event.get("type") in ("done", "error", "terminated"):
                        break
                except asyncio.TimeoutError:
                    # 心跳保活
                    yield _sse({"type": "heartbeat", "ts": time.time()})
                    # 若任务已结束但 queue 已空，退出
                    if job.status in ("done", "error", "terminated") and q.empty():
                        yield _sse({"type": job.status, "ts": time.time()})
                        break
        finally:
            unsubscribe(job, q)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/generate/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"任务 {job_id} 不存在")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        job_type=job.job_type,
        started_at=job.started_at,
        scenario_totals=job.scenario_totals,
    )


@router.delete("/generate/{job_id}")
def cancel_job(job_id: str):
    """终止任务。"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"任务 {job_id} 不存在")
    job_manager.terminate_job(job_id)
    return {"status": "terminated", "job_id": job_id}
