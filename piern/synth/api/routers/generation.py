"""两阶段生成路由：/api/generate-templates, /api/fill-samples, /api/templates。"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter

# 确保项目根目录在 sys.path，使 scripts/ 可导入（无论 uvicorn 从哪里启动）
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from piern.shared.runtime.paths import PROJECT_ROOT, TEMPLATES_DIR  # noqa: E402
from piern.synth.api.schemas.generation import GenerateTemplatesRequest, FillSamplesRequest, JobStartResponse  # noqa: E402
from piern.synth.services import job_manager  # noqa: E402
from piern.synth.services.job_manager import JobRecord, publish  # noqa: E402
from scripts.text2comp.generate_templates import run_generate_templates  # noqa: E402
from scripts.text2comp.fill_samples import run_fill_samples  # noqa: E402

router = APIRouter()

# 后台线程池（生成任务是 CPU/IO 密集型，用独立线程池避免阻塞 event loop）
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gen-worker")


def _run_generate_templates(record: JobRecord, req: GenerateTemplatesRequest) -> None:
    """在后台线程中直接调用 run_generate_templates，通过回调推进度。"""

    scenario_totals = record.scenario_totals
    current_scenario: list[str] = [""]   # 用列表包装以在闭包中可写

    def on_scenario_start(scenario: str, total: int) -> None:
        if record.stop_event.is_set():
            raise InterruptedError("任务已终止")
        current_scenario[0] = scenario
        if scenario not in scenario_totals:
            scenario_totals[scenario] = total
            record.scenario_totals[scenario] = total
            publish(record, {"type": "init", "scenario_totals": dict(record.scenario_totals), "ts": time.time()})
        publish(record, {"type": "log", "line": f"[处理] {scenario}（共 {total} 条）", "ts": time.time()})

    def on_progress(scenario: str, done: int) -> None:
        if record.stop_event.is_set():
            raise InterruptedError("任务已终止")
        total = scenario_totals.get(scenario, 0)
        publish(record, {
            "type": "log",
            "line": f"  {scenario}: {done}/{total}",
            "ts": time.time(),
            "progress": {"scenario": scenario, "done": done, "total": total},
        })

    def on_log(line: str) -> None:
        publish(record, {"type": "log", "line": line, "ts": time.time()})

    try:
        run_generate_templates(
            cfg_path=req.config,
            n_templates=req.n_templates,
            scenarios=req.scenarios or None,
            skip_existing=req.skip_existing,
            append_existing=req.append_existing,
            language_mix=req.language_mix,
            transform_prob=req.transform_prob,
            max_workers=req.max_workers,
            on_scenario_start=on_scenario_start,
            on_progress=on_progress,
            on_log=on_log,
        )
        if not record.stop_event.is_set():
            record.status = "done"
            publish(record, {"type": "done", "ts": time.time(), "message": "模板生成完成"})
    except InterruptedError:
        pass  # 已由 terminate_job 设置 status="terminated" 并 publish terminated 事件
    except Exception as e:
        if not record.stop_event.is_set():
            record.status = "error"
            publish(record, {
                "type": "error",
                "ts": time.time(),
                "message": str(e),
                "scenario": current_scenario[0] or None,
            })


def _run_fill_samples(record: JobRecord, req: FillSamplesRequest) -> None:
    """在后台线程中直接调用 run_fill_samples，通过回调推进度。"""
    scenario_totals = record.scenario_totals
    current_scenario: list[str] = [""]

    def on_scenario_start(scenario: str, total: int) -> None:
        if record.stop_event.is_set():
            raise InterruptedError("任务已终止")
        current_scenario[0] = scenario
        if scenario not in scenario_totals:
            scenario_totals[scenario] = total
            record.scenario_totals[scenario] = total
            publish(record, {"type": "init", "scenario_totals": dict(record.scenario_totals), "ts": time.time()})
        publish(record, {"type": "log", "line": f"[处理] {scenario}（共 {total} 条）", "ts": time.time()})

    def on_progress(scenario: str, done: int) -> None:
        if record.stop_event.is_set():
            raise InterruptedError("任务已终止")
        total = scenario_totals.get(scenario, 0)
        publish(record, {
            "type": "log",
            "line": f"  {scenario}: {done}/{total}",
            "ts": time.time(),
            "progress": {"scenario": scenario, "done": done, "total": total},
        })

    def on_log(line: str) -> None:
        publish(record, {"type": "log", "line": line, "ts": time.time()})

    try:
        run_fill_samples(
            cfg_path=req.config,
            n_samples=req.n_samples,
            scenarios=req.scenarios or None,
            templates_dir=req.templates_dir or None,
            output_dir=req.output_dir or None,
            skip_existing=req.skip_existing,
            seed=req.seed,
            precision=req.precision,
            output_format=req.output_format,
            compression=req.compression,
            batch_size=req.batch_size,
            max_workers=req.max_workers,
            on_scenario_start=on_scenario_start,
            on_progress=on_progress,
            on_log=on_log,
        )
        if not record.stop_event.is_set():
            record.status = "done"
            publish(record, {"type": "done", "ts": time.time(), "message": "样本填充完成"})
    except InterruptedError:
        pass  # 已由 terminate_job 设置 status="terminated" 并 publish terminated 事件
    except Exception as e:
        if not record.stop_event.is_set():
            record.status = "error"
            publish(record, {
                "type": "error",
                "ts": time.time(),
                "message": str(e),
                "scenario": current_scenario[0] or None,
            })


@router.post("/generate-templates", response_model=JobStartResponse)
async def start_generate_templates(req: GenerateTemplatesRequest):
    """阶段一：生成语言模板库。"""
    scenario_totals = {sc: req.n_templates for sc in req.scenarios} if req.scenarios else {}
    record = job_manager.create_job("generate_templates", scenario_totals)
    if scenario_totals:
        publish(record, {"type": "init", "scenario_totals": dict(scenario_totals), "ts": time.time()})
    _executor.submit(_run_generate_templates, record, req)
    return JobStartResponse(job_id=record.job_id, scenario_totals=scenario_totals)


@router.post("/fill-samples", response_model=JobStartResponse)
async def start_fill_samples(req: FillSamplesRequest):
    """阶段二：数值填充（不调 LLM）。"""
    scenario_totals = {sc: req.n_samples for sc in req.scenarios} if req.scenarios else {}
    record = job_manager.create_job("fill_samples", scenario_totals)
    if scenario_totals:
        publish(record, {"type": "init", "scenario_totals": dict(scenario_totals), "ts": time.time()})
    _executor.submit(_run_fill_samples, record, req)
    return JobStartResponse(job_id=record.job_id, scenario_totals=scenario_totals)


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
        results.append({
            "scenario": scenario,
            "template_count": template_count,
            "file_size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "path": str(f.relative_to(PROJECT_ROOT)),
        })
    return results
