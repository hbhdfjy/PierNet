"""统一的 SSE 作业流接口：stream、status、delete。"""

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from piern.synth.services import job_manager
from piern.synth.services.job_manager import subscribe, unsubscribe
from piern.synth.api.schemas.jobs import JobStatusResponse

router = APIRouter()
_TERMINAL = {"done", "error", "terminated", "external_terminated"}


def _job_status_response(job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        job_type=job.job_type,
        started_at=job.started_at,
        scenario_totals=job.scenario_totals,
        progress=job.progress,
        stats=job.stats,
        finished_at=job.finished_at,
        error_message=job.error_message,
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.get("/generate/{job_id}/stream")
async def stream_job(job_id: str):
    """
    返回指定作业的 SSE 事件流。
    已结束作业会先回放历史事件，再结束连接。
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"任务 {job_id} 不存在")

    async def _generator():
        snapshot, q = subscribe(job)
        try:
            for event in snapshot:
                yield _sse(event)
                if event.get("type") in _TERMINAL:
                    return

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield _sse(event)
                    if event.get("type") in _TERMINAL:
                        break
                except asyncio.TimeoutError:
                    yield _sse({"type": "heartbeat", "ts": time.time()})
                    if job.status in _TERMINAL and q.empty():
                        yield _sse({"type": job.status, "ts": time.time()})
                        break
        finally:
            unsubscribe(job, q)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/generate/jobs", response_model=list[JobStatusResponse])
def list_jobs(
    job_type: str | None = Query(None),
    status: str | None = Query(None),
):
    """List recent generation jobs so reopened browsers can reconnect after backend restarts."""
    jobs = list(job_manager.all_jobs().values())
    if job_type:
        jobs = [job for job in jobs if job.job_type == job_type]
    if status:
        jobs = [job for job in jobs if job.status == status]
    jobs.sort(key=lambda job: job.started_at, reverse=True)
    return [_job_status_response(job) for job in jobs]


@router.get("/generate/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"任务 {job_id} 不存在")
    return _job_status_response(job)


@router.delete("/generate/{job_id}")
def cancel_job(job_id: str):
    """终止指定作业。"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"任务 {job_id} 不存在")
    job_manager.terminate_job(job_id)
    return {"status": "terminated", "job_id": job_id}
