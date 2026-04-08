"""Stage 1 物理仿真路由：场景扫描、单场景/批量仿真、历史记录。"""

import json
import re
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from piern.api.deps import PROJECT_ROOT
from piern.api.services import job_manager
from piern.api.services.job_manager import JobRecord, publish

router = APIRouter()

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sim-worker")

# power_system 拆分为 power_flow（潮流，IEEE14）和 transient（暂态，IEEE14）
# 两者共用 piern.simulators.power_system.pipeline，但 configs 目录独立
SIMULATORS = ["modflow", "simpeg", "power_flow", "transient", "gcam"]

# 虚拟 simulator 到实际 pipeline 模块的映射
_SIMULATOR_PIPELINE_MAP = {
    "power_flow": "power_system",
    "transient":  "power_system",
}

# 历史记录（内存，最多保留 200 条；重启后清空）
_history: deque = deque(maxlen=200)

# ── Pydantic 模型 ────────────────────────────────────────────────

class SimulationScenario(BaseModel):
    simulator: str
    scenario: str
    config_path: str
    h5_path: Optional[str] = None
    sample_count: int = 0
    output_shape: Optional[List[int]] = None
    file_size_bytes: int = 0
    target_samples: int = 0   # 来自 YAML 中 n_samples 字段（如有）


class SimulateRequest(BaseModel):
    simulator: str
    scenario: str
    n_samples: int = 100
    seed: int = 42
    config_path: str
    skip_existing: bool = False
    parallel: bool = False
    max_workers: int = 4


class BatchSimulateRequest(BaseModel):
    scenarios: List[str]
    n_samples: int = 100
    seed: int = 42
    skip_existing: bool = False
    parallel: bool = False
    max_workers: int = 4


class JobStartResponse(BaseModel):
    job_id: str
    status: str = "running"
    scenario_totals: dict = {}


class HistoryRecord(BaseModel):
    job_id: str
    simulator: str
    scenario: str
    n_samples: int
    skip_existing: bool
    started_at: float
    finished_at: Optional[float] = None
    status: str   # running | done | error | terminated
    elapsed_sec: Optional[float] = None
    final_sample_count: Optional[int] = None


# ── 场景扫描 ─────────────────────────────────────────────────────

def _read_h5_info(h5_path: Path):
    """返回 (sample_count, output_shape, file_size_bytes)。"""
    sample_count = 0
    output_shape = None
    file_size_bytes = 0
    try:
        file_size_bytes = h5_path.stat().st_size
        import h5py
        with h5py.File(str(h5_path), "r") as hf:
            # 优先读根属性 n_samples（写入时已保证正确）
            if "n_samples" in hf.attrs:
                sample_count = int(hf.attrs["n_samples"])
            # 直接读 timeseries dataset 获取形状
            if "timeseries" in hf:
                ts = hf["timeseries"]
                if sample_count == 0:
                    sample_count = int(ts.shape[0])
                if len(ts.shape) >= 3:
                    output_shape = list(ts.shape[1:])
    except Exception:
        pass
    return sample_count, output_shape, file_size_bytes


def _scan_scenarios() -> List[SimulationScenario]:
    results = []
    for sim in SIMULATORS:
        variants_dir = PROJECT_ROOT / "configs" / sim / "variants"
        if not variants_dir.exists():
            continue
        for cfg_file in sorted(variants_dir.glob("*.yaml")):
            scenario = cfg_file.stem
            config_path = str(cfg_file.relative_to(PROJECT_ROOT))
            h5_path = None
            sample_count = 0
            output_shape = None
            file_size_bytes = 0
            target_samples = 0

            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                output_file = cfg.get("output_file")
                target_samples = int(cfg.get("n_samples", 0))
                if output_file:
                    # 优先使用 YAML 中的 output_dir（power_flow/transient 指向 data/power_system/）
                    output_dir_str = cfg.get("output_dir")
                    if output_dir_str:
                        h5_candidate = PROJECT_ROOT / output_dir_str / output_file
                    else:
                        h5_candidate = PROJECT_ROOT / "data" / sim / output_file
                    if h5_candidate.exists():
                        h5_path = str(h5_candidate.relative_to(PROJECT_ROOT))
                        sample_count, output_shape, file_size_bytes = _read_h5_info(h5_candidate)
            except Exception:
                pass

            results.append(SimulationScenario(
                simulator=sim,
                scenario=scenario,
                config_path=config_path,
                h5_path=h5_path,
                sample_count=sample_count,
                output_shape=output_shape,
                file_size_bytes=file_size_bytes,
                target_samples=target_samples,
            ))
    return results


# 全局缓存场景列表（避免每次请求都扫描 YAML）
_scenario_cache: Optional[List[SimulationScenario]] = None
_scenario_cache_ts: float = 0
_CACHE_TTL = 3.0  # seconds


def _get_scenarios_cached() -> List[SimulationScenario]:
    global _scenario_cache, _scenario_cache_ts
    if _scenario_cache is None or time.time() - _scenario_cache_ts > _CACHE_TTL:
        _scenario_cache = _scan_scenarios()
        _scenario_cache_ts = time.time()
    return _scenario_cache


def _invalidate_cache():
    global _scenario_cache
    _scenario_cache = None


# ── 仿真执行逻辑 ─────────────────────────────────────────────────

def _get_run_pipeline(simulator: str):
    """动态导入对应 simulator 的 run_pipeline 函数。"""
    # 虚拟 simulator 映射到实际 pipeline 模块
    pipeline_module = _SIMULATOR_PIPELINE_MAP.get(simulator, simulator)
    if pipeline_module == "modflow":
        from piern.simulators.modflow.pipeline import run_pipeline
    elif pipeline_module == "simpeg":
        from piern.simulators.simpeg.pipeline import run_pipeline
    elif pipeline_module == "power_system":
        from piern.simulators.power_system.pipeline import run_pipeline
    elif pipeline_module == "gcam":
        from piern.simulators.gcam.pipeline import run_pipeline
    else:
        raise ValueError(f"未知 simulator: {simulator}")
    return run_pipeline


def _run_one_scenario(record: JobRecord, req: SimulateRequest, history_entry: dict) -> bool:
    """运行单个场景仿真，返回是否成功。"""
    # 发 init 事件，让前端进度环知道总量
    publish(record, {
        "type": "init",
        "scenario_totals": {req.scenario: req.n_samples},
        "ts": time.time(),
    })
    publish(record, {
        "type": "log",
        "line": f"[仿真] {req.simulator}/{req.scenario}  n={req.n_samples}" +
                (f"  [{req.max_workers}核并行]" if req.parallel else "") +
                ("  [跳过已有]" if req.skip_existing else ""),
        "ts": time.time(),
    })

    if req.parallel:
        # 并行模式：直接在当前线程调用 run_pipeline，避免 spawn+PIPE 死锁
        return _run_in_process_direct(record, req, history_entry)
    else:
        # 非并行模式：subprocess，保留实时日志流
        return _run_via_subprocess(record, req, history_entry)


def _run_via_subprocess(record: JobRecord, req: SimulateRequest, history_entry: dict) -> bool:
    """非并行模式：用 subprocess 运行，实时转发 stdout 日志。"""
    pipeline_module = _SIMULATOR_PIPELINE_MAP.get(req.simulator, req.simulator)
    cmd = [
        sys.executable, "-m", f"piern.simulators.{pipeline_module}.pipeline",
        "--config", req.config_path,
        "--n-samples", str(req.n_samples),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
        record.proc = proc

        for line in proc.stdout:
            if record.status == "terminated":
                proc.kill()
                break
            line = line.rstrip()
            if not line:
                continue
            event: dict = {"type": "log", "line": line, "ts": time.time()}
            m = re.search(r'(?:生成进度|增强进度)[：:]\s*(\d+)/(\d+)', line)
            if m:
                done = int(m.group(1))
                event["progress"] = {"scenario": req.scenario, "done": done, "total": req.n_samples}
            publish(record, event)

        proc.wait()
        record.proc = None
        success = (proc.returncode == 0) and (record.status != "terminated")
    except Exception as e:
        import traceback
        record.proc = None
        for ln in (f"[ERROR] {e}\n" + traceback.format_exc()).splitlines():
            publish(record, {"type": "log", "line": ln, "ts": time.time()})
        success = False

    _finalize_history(record, req, history_entry, success)
    return success


def _run_in_process_direct(record: JobRecord, req: SimulateRequest, history_entry: dict) -> bool:
    """并行模式：直接调用 run_pipeline()，通过回调接口推送结构化进度 SSE。"""
    import logging

    # 日志 handler：只转发文本日志行，不解析进度（进度由 progress_callback 负责）
    class SSEHandler(logging.Handler):
        def emit(self, lr: logging.LogRecord):
            if record.status == "terminated":
                return
            publish(record, {"type": "log", "line": self.format(lr), "ts": time.time()})

    handler = SSEHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    # 用实际 pipeline 模块名挂 logger（power_flow/transient → power_system）
    pipeline_module = _SIMULATOR_PIPELINE_MAP.get(req.simulator, req.simulator)
    sim_logger = logging.getLogger(f"piern.simulators.{pipeline_module}")
    sim_logger.setLevel(logging.INFO)
    sim_logger.addHandler(handler)
    main_logger = logging.getLogger("__main__")
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(handler)

    def progress_callback(done: int, total: int):
        if record.status == "terminated":
            return
        publish(record, {
            "type": "log",
            "line": f"进度：{done}/{total}",
            "ts": time.time(),
            "progress": {"scenario": req.scenario, "done": done, "total": total},
        })

    success = False
    try:
        if record.status == "terminated":
            return False
        import inspect
        run_pipeline = _get_run_pipeline(req.simulator)
        sig = inspect.signature(run_pipeline)
        kwargs: dict = {"cfg_path": str(PROJECT_ROOT / req.config_path), "n_samples": req.n_samples}
        if "parallel" in sig.parameters:
            kwargs["parallel"] = req.parallel
        if "max_workers" in sig.parameters:
            kwargs["max_workers"] = req.max_workers
        if "progress_callback" in sig.parameters:
            kwargs["progress_callback"] = progress_callback
        run_pipeline(**kwargs)
        success = (record.status != "terminated")
    except Exception as e:
        import traceback
        for ln in (f"[ERROR] {e}\n" + traceback.format_exc()).splitlines():
            publish(record, {"type": "log", "line": ln, "ts": time.time()})
        success = False
    finally:
        sim_logger.removeHandler(handler)
        main_logger.removeHandler(handler)

    _finalize_history(record, req, history_entry, success)
    return success


def _finalize_history(record: JobRecord, req: SimulateRequest, history_entry: dict, success: bool):
    history_entry["finished_at"] = time.time()
    history_entry["elapsed_sec"] = history_entry["finished_at"] - history_entry["started_at"]
    history_entry["status"] = "done" if success else ("terminated" if record.status == "terminated" else "error")
    # 先失效缓存，再重新扫描拿最新样本数（仿真已写完 HDF5）
    _invalidate_cache()
    if success:
        try:
            for s in _get_scenarios_cached():
                if s.scenario == req.scenario:
                    history_entry["final_sample_count"] = s.sample_count
                    break
        except Exception:
            pass


def _run_simulate(record: JobRecord, req: SimulateRequest) -> None:
    """单场景仿真后台线程。"""
    history_entry = {
        "job_id": record.job_id,
        "simulator": req.simulator,
        "scenario": req.scenario,
        "n_samples": req.n_samples,
        "skip_existing": req.skip_existing,
        "started_at": time.time(),
        "finished_at": None,
        "status": "running",
        "elapsed_sec": None,
        "final_sample_count": None,
    }
    _history.appendleft(history_entry)

    success = _run_one_scenario(record, req, history_entry)
    if success:
        # 补发满进度事件，让前端进度环和「共处理 N 条」显示最终样本数
        final_count = history_entry.get("final_sample_count") or req.n_samples
        publish(record, {
            "type": "log",
            "line": f"[完成] {req.scenario} 共 {final_count} 个样本",
            "ts": time.time(),
            "progress": {"scenario": req.scenario, "done": final_count, "total": final_count},
        })
        record.status = "done"
        publish(record, {"type": "done", "ts": time.time(), "message": "仿真完成"})
    else:
        record.status = "error"
        publish(record, {"type": "error", "ts": time.time(), "message": "仿真失败，请查看日志"})


def _run_batch_simulate(record: JobRecord, reqs: List[SimulateRequest]) -> None:
    """批量仿真后台线程：顺序执行多个场景。"""
    total = len(reqs)
    failed = []

    # 初始化所有场景的进度为 0
    publish(record, {
        "type": "init",
        "scenario_totals": {r.scenario: r.n_samples for r in reqs},
        "ts": time.time(),
    })
    publish(record, {
        "type": "log",
        "line": f"[批量仿真] 共 {total} 个场景，顺序执行",
        "ts": time.time(),
    })

    for i, req in enumerate(reqs):
        if record.status == "terminated":
            break
        publish(record, {
            "type": "log",
            "line": f"[{i+1}/{total}] 开始：{req.simulator}/{req.scenario}",
            "ts": time.time(),
        })
        history_entry = {
            "job_id": record.job_id + f"_{i}",
            "simulator": req.simulator,
            "scenario": req.scenario,
            "n_samples": req.n_samples,
            "skip_existing": req.skip_existing,
            "started_at": time.time(),
            "finished_at": None,
            "status": "running",
            "elapsed_sec": None,
            "final_sample_count": None,
        }
        _history.appendleft(history_entry)
        success = _run_one_scenario(record, req, history_entry)
        if success:
            final_count = history_entry.get("final_sample_count") or req.n_samples
            publish(record, {
                "type": "log",
                "line": f"[完成] {req.scenario} 共 {final_count} 个样本",
                "ts": time.time(),
                "progress": {"scenario": req.scenario, "done": final_count, "total": final_count},
            })
            # 通知前端刷新场景列表（已写入 HDF5）
            publish(record, {"type": "scenario_done", "scenario": req.scenario, "ts": time.time()})
        else:
            failed.append(req.scenario)
            publish(record, {
                "type": "log",
                "line": f"[警告] {req.scenario} 仿真失败，继续下一个",
                "ts": time.time(),
            })

    if record.status == "terminated":
        return

    if failed:
        record.status = "error"
        publish(record, {
            "type": "error",
            "ts": time.time(),
            "message": f"批量完成，{len(failed)} 个失败：{', '.join(failed)}",
        })
    else:
        record.status = "done"
        publish(record, {
            "type": "done",
            "ts": time.time(),
            "message": f"批量仿真完成，共 {total} 个场景",
        })


# ── API 端点 ─────────────────────────────────────────────────────

@router.get("/simulation/scenarios", response_model=List[SimulationScenario])
def get_simulation_scenarios(refresh: bool = False):
    """扫描场景配置，返回所有场景及 HDF5 状态。refresh=true 强制刷新缓存。"""
    if refresh:
        _invalidate_cache()
    return _get_scenarios_cached()


@router.post("/simulate", response_model=JobStartResponse)
async def start_simulate(req: SimulateRequest):
    """启动单场景仿真。"""
    record = job_manager.create_job("simulate", {req.scenario: req.n_samples})
    _executor.submit(_run_simulate, record, req)
    return JobStartResponse(
        job_id=record.job_id,
        status="running",
        scenario_totals={req.scenario: req.n_samples},
    )


@router.post("/simulate/batch", response_model=JobStartResponse)
async def start_batch_simulate(req: BatchSimulateRequest):
    """批量启动多场景仿真（顺序执行）。"""
    # 从缓存中查找各场景的 config_path
    all_scenarios = _get_scenarios_cached()
    scenario_map = {s.scenario: s for s in all_scenarios}

    reqs: List[SimulateRequest] = []
    missing = []
    for sc_name in req.scenarios:
        sc = scenario_map.get(sc_name)
        if sc is None:
            missing.append(sc_name)
            continue
        reqs.append(SimulateRequest(
            simulator=sc.simulator,
            scenario=sc.scenario,
            n_samples=req.n_samples,
            seed=req.seed,
            config_path=sc.config_path,
            skip_existing=req.skip_existing,
            parallel=req.parallel,
            max_workers=req.max_workers,
        ))

    if not reqs:
        raise HTTPException(status_code=400, detail=f"未找到场景：{missing}")

    scenario_totals = {r.scenario: r.n_samples for r in reqs}
    record = job_manager.create_job("simulate", scenario_totals)
    _executor.submit(_run_batch_simulate, record, reqs)
    return JobStartResponse(
        job_id=record.job_id,
        status="running",
        scenario_totals=scenario_totals,
    )


@router.get("/simulation/history")
def get_simulation_history(limit: int = 50):
    """返回最近的仿真历史记录（内存，重启后清空）。"""
    return list(_history)[:limit]


@router.delete("/simulation/history")
def clear_simulation_history():
    """清空历史记录。"""
    _history.clear()
    return {"ok": True}
