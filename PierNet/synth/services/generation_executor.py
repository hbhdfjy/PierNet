"""Worker-safe executors for template generation and sample filling."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PierNet.synth.api.schemas.generation import FillSamplesRequest, GenerateTemplatesRequest
from PierNet.synth.services import job_manager
from PierNet.synth.services.job_manager import JobRecord, publish
from scripts.text2comp.fill_samples import run_fill_samples
from scripts.text2comp.generate_templates import run_generate_templates


def _invalidate_text2comp_scenarios_cache() -> None:
    try:
        from PierNet.synth.api.routers.config import invalidate_text2comp_scenarios_cache

        invalidate_text2comp_scenarios_cache()
    except Exception:
        pass


def _model_validate(model, payload: dict[str, Any]):
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)
    return model(**payload)


def run_generate_templates_job(record: JobRecord, payload: dict[str, Any]) -> None:
    req = _model_validate(GenerateTemplatesRequest, payload)
    scenario_totals = record.scenario_totals
    current_scenario: list[str] = [""]

    def _check_stop() -> None:
        if job_manager.should_stop(record):
            raise InterruptedError("任务已终止")

    def on_scenario_start(scenario: str, total: int) -> None:
        _check_stop()
        current_scenario[0] = scenario
        if scenario not in scenario_totals:
            scenario_totals[scenario] = total
            record.scenario_totals[scenario] = total
            publish(record, {"type": "init", "scenario_totals": dict(record.scenario_totals), "ts": time.time()})
        publish(record, {"type": "log", "line": f"[处理] {scenario}（共 {total} 条）", "ts": time.time()})

    def on_progress(scenario: str, done: int) -> None:
        _check_stop()
        total = scenario_totals.get(scenario, 0)
        publish(
            record,
            {
                "type": "log",
                "line": f"  {scenario}: {done}/{total}",
                "ts": time.time(),
                "progress": {"scenario": scenario, "done": done, "total": total},
            },
        )

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
        if not job_manager.should_stop(record):
            record.status = "done"
            publish(record, {"type": "done", "ts": time.time(), "message": "模板生成完成"})
    except InterruptedError:
        record.status = "terminated"
        publish(record, {"type": "terminated", "ts": time.time(), "message": "任务已由平台终止。"})
    except Exception as exc:
        if not job_manager.should_stop(record):
            record.status = "error"
            publish(
                record,
                {
                    "type": "error",
                    "ts": time.time(),
                    "message": str(exc),
                    "scenario": current_scenario[0] or None,
                },
            )


def run_fill_samples_job(record: JobRecord, payload: dict[str, Any]) -> None:
    req = _model_validate(FillSamplesRequest, payload)
    scenario_totals = record.scenario_totals
    current_scenario: list[str] = [""]
    last_progress_publish: dict[str, tuple[int, float]] = {}

    def _check_stop() -> None:
        if job_manager.should_stop(record):
            raise InterruptedError("任务已终止")

    def on_scenario_start(scenario: str, total: int) -> None:
        _check_stop()
        current_scenario[0] = scenario
        if scenario not in scenario_totals:
            scenario_totals[scenario] = total
            record.scenario_totals[scenario] = total
            publish(record, {"type": "init", "scenario_totals": dict(record.scenario_totals), "ts": time.time()})
        publish(record, {"type": "log", "line": f"[处理] {scenario}（共 {total} 条）", "ts": time.time()})

    def on_progress(scenario: str, done: int) -> None:
        _check_stop()
        total = scenario_totals.get(scenario, 0)
        now = time.time()
        last_done, last_ts = last_progress_publish.get(scenario, (0, 0.0))
        min_rows = max(10000, int(total) // 100) if total else 10000
        if done < total and done - last_done < min_rows and now - last_ts < 2.0:
            return
        last_progress_publish[scenario] = (done, now)
        publish(
            record,
            {
                "type": "log",
                "line": f"  {scenario}: {done}/{total}",
                "ts": now,
                "progress": {"scenario": scenario, "done": done, "total": total},
            },
        )

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
            should_stop=lambda: job_manager.should_stop(record),
        )
        publish(record, {"type": "log", "line": "[收尾] 落盘中：样本数据已写入，正在刷新样本状态", "ts": time.time()})
        publish(record, {"type": "log", "line": "[收尾] 重建索引中：刷新样本场景清单", "ts": time.time()})
        _invalidate_text2comp_scenarios_cache()
        publish(record, {"type": "log", "line": "[收尾] 释放锁中：正在释放填充任务资源锁", "ts": time.time()})
        if not job_manager.should_stop(record):
            record.status = "done"
            publish(record, {"type": "done", "ts": time.time(), "message": "样本填充完成"})
    except InterruptedError:
        record.status = "terminated"
        publish(record, {"type": "terminated", "ts": time.time(), "message": "任务已由平台终止。"})
    except Exception as exc:
        if not job_manager.should_stop(record):
            record.status = "error"
            publish(
                record,
                {
                    "type": "error",
                    "ts": time.time(),
                    "message": str(exc),
                    "scenario": current_scenario[0] or None,
                },
            )
