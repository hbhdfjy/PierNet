"""Template generation and sample filling routes."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from piern.shared.runtime.paths import PROJECT_ROOT, TEMPLATES_DIR  # noqa: E402
from piern.synth.api.schemas.generation import FillSamplesRequest, GenerateTemplatesRequest, JobStartResponse  # noqa: E402
from piern.synth.services import generation_executor, job_manager, worker_queue  # noqa: E402
from piern.synth.services.job_manager import JobRecord, publish  # noqa: E402

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gen-worker")


def _request_payload(req) -> dict:
    if hasattr(req, "model_dump"):
        return req.model_dump(mode="json")
    return req.dict()


def _resource_lock_keys(kind: str, scenarios: list[str]) -> list[str]:
    selected = scenarios or ["all"]
    return [f"{kind}:{scenario}" for scenario in selected]


def _reject_active_jobs(job_types: set[str], message: str) -> None:
    try:
        job_manager.assert_no_running_jobs(job_types, message=message)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _use_worker_queue() -> bool:
    return worker_queue.queue_enabled() and os.getenv("PIERN_WORKER_QUEUE_SYNTH", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _start_job(
    *,
    job_type: str,
    scenario_totals: dict[str, int],
    payload: dict,
    lock_keys: list[str],
    direct_runner,
) -> JobRecord:
    queued = _use_worker_queue()
    try:
        record = job_manager.create_job(
            job_type,
            scenario_totals,
            request=payload,
            lock_keys=lock_keys,
            status="queued" if queued else "running",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if scenario_totals:
        publish(record, {"type": "init", "scenario_totals": dict(scenario_totals), "ts": time.time()})
    if queued:
        publish(record, {"type": "queued", "ts": time.time(), "message": "任务已进入 worker 队列"})
    else:
        _executor.submit(direct_runner, record, payload)
    return record


@router.post("/generate-templates", response_model=JobStartResponse)
async def start_generate_templates(req: GenerateTemplatesRequest):
    """阶段二：生成语言模板库。"""
    _reject_active_jobs({"generate_templates"}, "已有模板生成任务正在运行")
    scenario_totals = {sc: req.n_templates for sc in req.scenarios} if req.scenarios else {}
    payload = _request_payload(req)
    record = _start_job(
        job_type="generate_templates",
        scenario_totals=scenario_totals,
        payload=payload,
        lock_keys=_resource_lock_keys("template", req.scenarios),
        direct_runner=generation_executor.run_generate_templates_job,
    )
    return JobStartResponse(job_id=record.job_id, status=record.status, scenario_totals=scenario_totals)


@router.post("/fill-samples", response_model=JobStartResponse)
async def start_fill_samples(req: FillSamplesRequest):
    """阶段三：数值填充。"""
    _reject_active_jobs({"fill_samples", "router"}, "样本填充或路由构建任务正在运行")
    scenario_totals = {sc: req.n_samples for sc in req.scenarios} if req.scenarios else {}
    payload = _request_payload(req)
    record = _start_job(
        job_type="fill_samples",
        scenario_totals=scenario_totals,
        payload=payload,
        lock_keys=_resource_lock_keys("dataset", req.scenarios),
        direct_runner=generation_executor.run_fill_samples_job,
    )
    return JobStartResponse(job_id=record.job_id, status=record.status, scenario_totals=scenario_totals)


@router.get("/templates")
def get_templates_status():
    """扫描 data/templates/ 目录，返回各场景的模板库状态。"""
    if not TEMPLATES_DIR.exists():
        return []

    results = []
    for f in sorted(TEMPLATES_DIR.glob("*_templates.jsonl")):
        scenario = f.stem.replace("_templates", "")
        stat = f.stat()
        template_count = 0
        try:
            with open(f, "rb") as fh:
                template_count = fh.read().count(b"\n")
        except Exception:
            pass
        results.append(
            {
                "scenario": scenario,
                "template_count": template_count,
                "file_size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "path": str(f.relative_to(PROJECT_ROOT)),
            }
        )
    return results
