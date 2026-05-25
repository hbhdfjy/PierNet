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

from PierNet.shared.runtime.paths import PROJECT_ROOT  # noqa: E402
from PierNet.synth.api.schemas.generation import FillSamplesRequest, GenerateTemplatesRequest, JobStartResponse  # noqa: E402
from PierNet.synth.api.schemas.jobs import TemplateFileInfo  # noqa: E402
from PierNet.synth.services import file_manager, generation_executor, job_manager, worker_queue  # noqa: E402
from PierNet.synth.services.job_manager import JobRecord, publish  # noqa: E402
from PierNet.synth.text2comp.pipeline import (  # noqa: E402
    _scan_h5_files,
    _scenario_name_from_path,
    duplicate_stage_scenarios,
    load_config,
)

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gen-worker")


def _request_payload(req) -> dict:
    if hasattr(req, "model_dump"):
        return req.model_dump(mode="json")
    return req.dict()


def _resource_lock_keys(kind: str, scenarios: list[str]) -> list[str]:
    selected = scenarios or ["all"]
    return [f"{kind}:{scenario}" for scenario in selected]


def _config_path(config: str) -> Path:
    path = Path(config).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _base_dir_for_config(config_path: Path) -> Path:
    base_dir = config_path.parent.parent.parent
    return base_dir if (base_dir / "data").exists() else PROJECT_ROOT


def _duplicate_stage_scenarios(config: str, scenarios: list[str]) -> list[str]:
    cfg_path = _config_path(config)
    cfg = load_config(cfg_path)
    base_dir = _base_dir_for_config(cfg_path)
    selected = {item.strip() for item in scenarios if item.strip()}
    h5_files = [
        (h5_path, simulator, file_suffix)
        for h5_path, simulator, file_suffix in _scan_h5_files(cfg, base_dir)
        if not selected or _scenario_name_from_path(h5_path, file_suffix) in selected
    ]
    return duplicate_stage_scenarios(h5_files)


def _assert_unique_stage_scenarios(config: str, scenarios: list[str]) -> None:
    duplicates = _duplicate_stage_scenarios(config, scenarios)
    if duplicates:
        raise HTTPException(
            status_code=400,
            detail=(
                "同名场景分布在多个 simulator 中，阶段 2/3 的模板文件仍按 scenario 命名，无法安全区分："
                + "; ".join(duplicates)
            ),
        )


def _reject_active_jobs(job_types: set[str], message: str) -> None:
    try:
        job_manager.assert_no_running_jobs(job_types, message=message)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _use_worker_queue() -> bool:
    return worker_queue.queue_enabled() and os.getenv("PierNet_WORKER_QUEUE_SYNTH", "1").strip().lower() not in {
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
    _assert_unique_stage_scenarios(req.config, req.scenarios)
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
    _assert_unique_stage_scenarios(req.config, req.scenarios)
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


@router.get("/templates", response_model=list[TemplateFileInfo])
def get_templates_status():
    """返回各场景的模板库状态。"""
    return file_manager.list_template_files()
