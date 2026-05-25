"""Stage 1 物理仿真路由：场景扫描、单场景/批量仿真。"""

import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from piern.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT, RUNLOG_ROOT
from piern.synth.api.routers.config import invalidate_text2comp_scenarios_cache
from piern.synth.services import job_manager
from piern.synth.services.job_manager import JobRecord, publish
from piern.synth.services.hdf5_data import (
    canonical_hdf5_path,
    list_hdf5_data_files,
    validate_hdf5_file,
    validate_name,
)

router = APIRouter()

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sim-worker")


def _display_path(path: Path) -> str:
    for root in (PROJECT_ROOT, DATA_ROOT):
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def _resolve_data_aware_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        return DATA_ROOT.joinpath(*parts[1:])
    return PROJECT_ROOT / path


def cleanup_stale_tmp_configs(tmp_configs_dir: Optional[Path] = None) -> int:
    tmp_dir = tmp_configs_dir or RUNLOG_ROOT / "tmp_configs"
    if not tmp_dir.exists():
        return 0
    deleted = 0
    for path in tmp_dir.glob("*.yaml"):
        try:
            path.unlink()
            deleted += 1
        except OSError:
            continue
    return deleted

SIMULATORS = ["modflow", "simpeg", "power_flow", "transient", "gcam"]

# ── Pydantic 模型 ────────────────────────────────────────────────

class SimulationScenario(BaseModel):
    simulator: str
    scenario: str
    config_path: str
    h5_path: Optional[str] = None
    sample_count: int = 0
    output_shape: Optional[List[int]] = None
    file_size_bytes: int = 0


class SimulateRequest(BaseModel):
    simulator: str
    scenario: str
    n_samples: int = Field(100, ge=1, le=1_000_000)
    seed: int = Field(42, ge=0, le=2_147_483_647)
    config_path: str
    skip_existing: bool = False
    parallel: bool = False
    max_workers: int = Field(4, ge=1, le=64)


class BatchSimulateRequest(BaseModel):
    scenarios: List[str]
    n_samples: int = Field(100, ge=1, le=1_000_000)
    seed: int = Field(42, ge=0, le=2_147_483_647)
    skip_existing: bool = False
    parallel: bool = False
    max_workers: int = Field(4, ge=1, le=64)


class JobStartResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running"] = "running"
    scenario_totals: dict[str, int] = Field(default_factory=dict)


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
            config_path = _display_path(cfg_file)
            h5_path = None
            sample_count = 0
            output_shape = None
            file_size_bytes = 0

            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                output_file = cfg.get("output_file")
                if output_file:
                    # 优先使用 YAML 中的 output_dir（各 simulator 写入自己的数据目录）
                    output_dir_str = cfg.get("output_dir")
                    if output_dir_str:
                        h5_candidate = _resolve_data_aware_path(str(output_dir_str)) / output_file
                    else:
                        h5_candidate = DATA_ROOT / sim / output_file
                    if h5_candidate.exists():
                        h5_path = _display_path(h5_candidate)
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
            ))
    return results


# 全局缓存场景列表（避免每次请求都扫描 YAML）
_scenario_cache: Optional[List[SimulationScenario]] = None
_scenario_cache_ts: float = 0
_CACHE_TTL = 3.0  # seconds

_SIM_PROGRESS_PATTERNS = (
    re.compile(r"(?:生成|增强)?进度[：:]\s*(\d+)\s*/\s*(\d+)"),
    re.compile(r"progress[：:]\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE),
    re.compile(r'^\s*(\d+)\s*/\s*(\d+)\s*$'),
)


def _extract_progress_counts(line: str) -> tuple[int, int] | None:
    for pattern in _SIM_PROGRESS_PATTERNS:
        m = pattern.search(line)
        if not m:
            continue
        done, total = int(m.group(1)), int(m.group(2))
        if total > 0 and done <= total:
            return done, total
    return None


def _get_scenarios_cached() -> List[SimulationScenario]:
    global _scenario_cache, _scenario_cache_ts
    if _scenario_cache is None or time.time() - _scenario_cache_ts > _CACHE_TTL:
        _scenario_cache = _scan_scenarios()
        _scenario_cache_ts = time.time()
    return _scenario_cache


def _invalidate_cache():
    global _scenario_cache
    _scenario_cache = None


def _scenario_key(simulator: str, scenario: str) -> str:
    return f"{simulator}/{scenario}"


def _scenario_key_for_req(req: SimulateRequest) -> str:
    return _scenario_key(req.simulator, req.scenario)


def _simulation_output_lock_key(req: SimulateRequest) -> str:
    output_path = _resolve_output_h5_path(req.config_path, req.simulator)
    if output_path is None:
        return f"raw:{_scenario_key_for_req(req)}"
    try:
        output_path = output_path.resolve(strict=False)
    except OSError:
        pass
    return f"raw-file:{_display_path(output_path)}"


def _simulation_lock_keys(reqs: List[SimulateRequest]) -> list[str]:
    seen: dict[str, str] = {}
    result: list[str] = []
    for req in reqs:
        key = _simulation_output_lock_key(req)
        scenario_key = _scenario_key_for_req(req)
        previous = seen.get(key)
        if previous and previous != scenario_key:
            raise HTTPException(
                status_code=400,
                detail=f"多个场景写入同一 HDF5 输出：{previous}, {scenario_key}",
            )
        if key not in seen:
            seen[key] = scenario_key
            result.append(key)
    return result


def _parse_scenario_selector(selector: str) -> tuple[str | None, str]:
    value = str(selector or "").strip()
    if "::" in value:
        simulator, scenario = value.split("::", 1)
        return simulator.strip() or None, scenario.strip()
    if "/" in value:
        simulator, scenario = value.split("/", 1)
        return simulator.strip() or None, scenario.strip()
    return None, value


def _simulate_request_payload(req: SimulateRequest) -> dict:
    return req.model_dump() if hasattr(req, "model_dump") else req.dict()


def _canonical_simulate_request(req: SimulateRequest) -> SimulateRequest:
    requested_key = _scenario_key(req.simulator, req.scenario)
    for scenario in _get_scenarios_cached():
        if _scenario_key(scenario.simulator, scenario.scenario) != requested_key:
            continue
        payload = _simulate_request_payload(req)
        payload["config_path"] = scenario.config_path
        return SimulateRequest(**payload)
    raise HTTPException(status_code=400, detail=f"未找到仿真场景: {requested_key}")


@router.get("/simulation/data-files")
def get_simulation_data_files():
    """列出 data/ 下已存在的 Stage 1 HDF5 文件及校验状态。"""
    return list_hdf5_data_files()


@router.post("/simulation/upload")
async def upload_simulation_data(
    request: Request,
    simulator: str = Query(...),
    scenario: str = Query(...),
    overwrite: bool = Query(False),
):
    """上传外部 HDF5 数据，保存后返回预检结果；注册时执行强校验。"""
    try:
        simulator = validate_name("simulator", simulator)
        scenario = validate_name("scenario", scenario)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target = canonical_hdf5_path(simulator, scenario)
    if target.exists() and not overwrite:
        rel = _display_path(target)
        raise HTTPException(status_code=409, detail=f"目标文件已存在: {rel}；如需覆盖请开启 overwrite")

    tmp_dir = RUNLOG_ROOT / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f".{simulator}_{scenario}_{time.time_ns()}.h5"

    bytes_written = 0
    try:
        with tmp_path.open("wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                handle.write(chunk)
                bytes_written += len(chunk)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"上传写入失败: {exc}") from exc

    if bytes_written == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传文件为空")

    tmp_validation = validate_hdf5_file(tmp_path)
    if not tmp_validation.get("valid"):
        tmp_path.unlink(missing_ok=True)
        return {
            "ok": False,
            "simulator": simulator,
            "scenario": scenario,
            "saved_path": "",
            "validation": tmp_validation,
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(target)
    _invalidate_cache()
    invalidate_text2comp_scenarios_cache()

    saved_validation = validate_hdf5_file(target)
    return {
        "ok": True,
        "simulator": simulator,
        "scenario": scenario,
        "saved_path": _display_path(target),
        "validation": saved_validation,
    }


# ── 仿真执行逻辑 ─────────────────────────────────────────────────

def _get_run_pipeline(simulator: str):
    """动态导入对应 simulator 的 run_pipeline 函数。"""
    try:
        if simulator == "modflow":
            from piern.simulators.modflow.pipeline import run_pipeline
        elif simulator == "simpeg":
            from piern.simulators.simpeg.pipeline import run_pipeline
        elif simulator == "power_flow":
            from piern.simulators.power_flow.pipeline import run_pipeline
        elif simulator == "transient":
            from piern.simulators.transient.pipeline import run_pipeline
        elif simulator == "gcam":
            from piern.simulators.gcam.pipeline import run_pipeline
        else:
            raise ValueError(f"未知 simulator: {simulator}")
    except ImportError as exc:
        raise ImportError(
            f"Simulator '{simulator}' dependencies are not installed: {exc}"
        ) from exc
    return run_pipeline


def _resolve_config_path(config_path: str) -> Path:
    cfg_path = Path(config_path)
    return cfg_path if cfg_path.is_absolute() else PROJECT_ROOT / cfg_path


def _resolve_output_h5_path(config_path: str, simulator: str) -> Optional[Path]:
    cfg_path = _resolve_config_path(config_path)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None

    output_file = cfg.get("output_file")
    if not output_file:
        return None

    output_dir_str = cfg.get("output_dir")
    if output_dir_str:
        return _resolve_data_aware_path(str(output_dir_str)) / output_file
    return DATA_ROOT / simulator / output_file


def _prepare_runtime_config(req: SimulateRequest) -> tuple[str, Optional[Path]]:
    cfg_path = _resolve_config_path(req.config_path)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return str(cfg_path), None

    if int(cfg.get("seed", req.seed)) == req.seed:
        return str(cfg_path), None

    cfg["seed"] = req.seed
    tmp_dir = RUNLOG_ROOT / "tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{req.simulator}_{req.scenario}_",
        suffix=".yaml",
        dir=str(tmp_dir),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return str(tmp_path), tmp_path


def _cleanup_runtime_config(tmp_cfg_path: Optional[Path]) -> None:
    if tmp_cfg_path is None:
        return
    try:
        tmp_cfg_path.unlink()
    except FileNotFoundError:
        pass


def _run_one_scenario(record: JobRecord, req: SimulateRequest, run_state: dict) -> bool:
    """执行单个场景仿真，支持跳过已有结果。"""
    publish(record, {
        "type": "init",
        "scenario_totals": {_scenario_key_for_req(req): req.n_samples},
        "ts": time.time(),
    })
    publish(record, {
        "type": "log",
        "line": f"[开始] {req.simulator}/{req.scenario}  n={req.n_samples}" +
                (f"  [{req.max_workers} 并行]" if req.parallel else "") +
                ("  [跳过已有结果]" if req.skip_existing else ""),
        "ts": time.time(),
    })

    if req.skip_existing:
        h5_path = _resolve_output_h5_path(req.config_path, req.simulator)
        if h5_path is not None and h5_path.exists():
            validation = validate_hdf5_file(h5_path)
            existing_count = int(validation.get("sample_count") or 0)
            display_path = _display_path(h5_path)
            if validation.get("valid") and existing_count >= req.n_samples:
                publish(record, {
                    "type": "log",
                    "line": (
                        f"[跳过] {req.scenario} 已达到目标 "
                        f"{existing_count}/{req.n_samples} 条样本：{display_path}"
                    ),
                    "ts": time.time(),
                })
                run_state["final_sample_count"] = existing_count
                return True
            if validation.get("valid"):
                publish(record, {
                    "type": "log",
                    "line": (
                        f"[重建] {req.scenario} 已有 {existing_count}/{req.n_samples} "
                        f"条样本，未达到目标：{display_path}"
                    ),
                    "ts": time.time(),
                })
            else:
                errors = "; ".join(str(e) for e in validation.get("errors") or ["无有效样本"])
                publish(record, {
                    "type": "log",
                    "line": f"[重建] {req.scenario} 现有 HDF5 不可用，重新生成：{display_path}（{errors}）",
                    "ts": time.time(),
                })

    if req.parallel:
        return _run_in_process_direct(record, req, run_state)
    return _run_via_subprocess(record, req, run_state)


def _run_via_subprocess(record: JobRecord, req: SimulateRequest, run_state: dict) -> bool:
    """通过子进程运行 simulator pipeline，并把 stdout 转发到 SSE。"""
    runtime_cfg_path, tmp_cfg_path = _prepare_runtime_config(req)
    cmd = [
        sys.executable, "-m", f"piern.simulators.{req.simulator}.pipeline",
        "--config", runtime_cfg_path,
        "--n-samples", str(req.n_samples),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
            start_new_session=True,
        )
        record.proc = proc
        record.proc_uses_process_group = True

        for line in proc.stdout:
            if record.status == "terminated":
                proc.kill()
                break
            line = line.rstrip()
            if not line:
                continue
            event: dict = {"type": "log", "line": line, "ts": time.time()}
            counts = _extract_progress_counts(line)
            if counts is not None:
                done, total = counts
                event["progress"] = {"scenario": _scenario_key_for_req(req), "done": done, "total": total}
            publish(record, event)

        proc.wait()
        success = (proc.returncode == 0) and (record.status != "terminated")
    except Exception as e:
        import traceback
        for ln in (f"[ERROR] {e}\n" + traceback.format_exc()).splitlines():
            publish(record, {"type": "log", "line": ln, "ts": time.time()})
        success = False
    finally:
        record.proc = None
        record.proc_uses_process_group = False
        _cleanup_runtime_config(tmp_cfg_path)

    return _finalize_simulation_result(record, req, run_state, success)


def _run_in_process_direct(record: JobRecord, req: SimulateRequest, run_state: dict) -> bool:
    """在当前进程直接调用 run_pipeline()，把日志和进度转发到 SSE。"""
    import logging

    class SSEHandler(logging.Handler):
        def emit(self, lr: logging.LogRecord):
            if record.status == "terminated":
                return
            publish(record, {"type": "log", "line": self.format(lr), "ts": time.time()})

    handler = SSEHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    sim_logger = logging.getLogger(f"piern.simulators.{req.simulator}")
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
            "line": f"进度 {done}/{total}",
            "ts": time.time(),
            "progress": {"scenario": _scenario_key_for_req(req), "done": done, "total": total},
        })

    runtime_cfg_path, tmp_cfg_path = _prepare_runtime_config(req)
    success = False
    try:
        if record.status == "terminated":
            return False
        import inspect
        run_pipeline = _get_run_pipeline(req.simulator)
        sig = inspect.signature(run_pipeline)
        kwargs: dict = {"cfg_path": runtime_cfg_path, "n_samples": req.n_samples}
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
        _cleanup_runtime_config(tmp_cfg_path)

    return _finalize_simulation_result(record, req, run_state, success)


def _finalize_simulation_result(
    record: JobRecord,
    req: SimulateRequest,
    run_state: dict,
    success: bool,
) -> bool:
    _invalidate_cache()
    if not success:
        return False

    h5_path = _resolve_output_h5_path(req.config_path, req.simulator)
    if h5_path is None:
        publish(record, {
            "type": "log",
            "line": f"[ERROR] 无法解析 {req.simulator}/{req.scenario} 的 HDF5 输出路径",
            "ts": time.time(),
        })
        return False
    if not h5_path.exists():
        publish(record, {
            "type": "log",
            "line": f"[ERROR] 仿真结束但未生成 HDF5：{_display_path(h5_path)}",
            "ts": time.time(),
        })
        return False

    validation = validate_hdf5_file(h5_path)
    if not validation.get("valid"):
        invalidate_text2comp_scenarios_cache()
        errors = "; ".join(str(e) for e in validation.get("errors") or ["未知错误"])
        publish(record, {
            "type": "log",
            "line": f"[ERROR] 仿真输出 HDF5 校验失败：{_display_path(h5_path)}（{errors}）",
            "ts": time.time(),
        })
        return False

    run_state["final_sample_count"] = int(validation.get("sample_count") or req.n_samples)
    invalidate_text2comp_scenarios_cache()
    return True


def _run_simulate(record: JobRecord, req: SimulateRequest) -> None:
    """单场景仿真后台线程。"""
    run_state = {
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

    success = _run_one_scenario(record, req, run_state)
    if success:
        final_count = run_state.get("final_sample_count") or req.n_samples
        publish(record, {
            "type": "log",
            "line": f"[完成] {req.scenario} 共 {final_count} 条样本",
            "ts": time.time(),
            "progress": {"scenario": _scenario_key_for_req(req), "done": final_count, "total": final_count},
        })
        record.status = "done"
        publish(record, {"type": "done", "ts": time.time(), "message": "仿真完成"})
    else:
        if record.status == "terminated":
            return
        record.status = "error"
        publish(record, {"type": "error", "ts": time.time(), "message": "仿真失败，请检查日志"})


def _run_batch_simulate(record: JobRecord, reqs: List[SimulateRequest]) -> None:
    """批量仿真后台线程：顺序执行多个场景。"""
    total = len(reqs)
    failed = []

    # 初始化所有场景的进度为 0
    publish(record, {
        "type": "init",
        "scenario_totals": {_scenario_key_for_req(r): r.n_samples for r in reqs},
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
        run_state = {
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
        success = _run_one_scenario(record, req, run_state)
        if success:
            final_count = run_state.get("final_sample_count") or req.n_samples
            publish(record, {
                "type": "log",
                "line": f"[完成] {req.scenario} 共 {final_count} 个样本",
                "ts": time.time(),
                "progress": {"scenario": _scenario_key_for_req(req), "done": final_count, "total": final_count},
            })
            # 通知前端刷新场景列表（已写入 HDF5）
            publish(record, {"type": "scenario_done", "scenario": _scenario_key_for_req(req), "ts": time.time()})
        else:
            failed_key = _scenario_key_for_req(req)
            failed.append(failed_key)
            publish(record, {
                "type": "log",
                "line": f"[警告] {failed_key} 仿真失败，继续下一个",
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
    req = _canonical_simulate_request(req)
    scenario_totals = {_scenario_key_for_req(req): req.n_samples}
    try:
        record = job_manager.create_job("simulate", scenario_totals, lock_keys=_simulation_lock_keys([req]))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _executor.submit(_run_simulate, record, req)
    return JobStartResponse(
        job_id=record.job_id,
        status="running",
        scenario_totals=scenario_totals,
    )


@router.post("/simulate/batch", response_model=JobStartResponse)
async def start_batch_simulate(req: BatchSimulateRequest):
    """批量启动多场景仿真（顺序执行）。"""
    # 从缓存中查找各场景的 config_path
    all_scenarios = _get_scenarios_cached()
    by_key = {_scenario_key(s.simulator, s.scenario): s for s in all_scenarios}
    by_scenario: dict[str, list[SimulationScenario]] = {}
    for scenario in all_scenarios:
        by_scenario.setdefault(scenario.scenario, []).append(scenario)

    reqs: List[SimulateRequest] = []
    missing = []
    ambiguous = []
    duplicates = []
    seen_keys: set[str] = set()
    for selector in req.scenarios:
        simulator, scenario = _parse_scenario_selector(selector)
        sc = by_key.get(_scenario_key(simulator, scenario)) if simulator else None
        if sc is None and not simulator:
            matches = by_scenario.get(scenario, [])
            if len(matches) == 1:
                sc = matches[0]
            elif len(matches) > 1:
                ambiguous.append(selector)
                continue
        if sc is None:
            missing.append(selector)
            continue
        scenario_key = _scenario_key(sc.simulator, sc.scenario)
        if scenario_key in seen_keys:
            duplicates.append(selector)
            continue
        seen_keys.add(scenario_key)
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

    if ambiguous:
        raise HTTPException(status_code=400, detail=f"场景选择存在歧义，请使用 simulator/scenario: {ambiguous}")
    if missing:
        raise HTTPException(status_code=400, detail=f"未找到场景：{missing}")
    if duplicates:
        raise HTTPException(status_code=400, detail=f"重复场景：{duplicates}")

    if not reqs:
        raise HTTPException(status_code=400, detail="未选择场景")

    scenario_totals = {_scenario_key_for_req(r): r.n_samples for r in reqs}
    try:
        record = job_manager.create_job("simulate", scenario_totals, lock_keys=_simulation_lock_keys(reqs))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _executor.submit(_run_batch_simulate, record, reqs)
    return JobStartResponse(
        job_id=record.job_id,
        status="running",
        scenario_totals=scenario_totals,
    )
