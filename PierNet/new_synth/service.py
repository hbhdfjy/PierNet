from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from PierNet.new_synth import store
from PierNet.new_synth.paths import NEW_SYNTH_CACHE_ROOT, workflow_paths
from PierNet.shared.runtime.paths import CONFIG_DIR, PROJECT_ROOT
from PierNet.shared.tasks import locks as task_locks
from PierNet.synth.api.routers import simulation
from PierNet.synth.services import expert_models
from PierNet.synth.services.hdf5_data import canonical_hdf5_path, validate_hdf5_file

MAX_GENERATION_SAMPLES = max(4, int(os.getenv("PIERN_NEW_SYNTH_MAX_SAMPLES", "100000")))
MIN_FREE_DISK_BYTES = max(
    0,
    int(os.getenv("PIERN_NEW_SYNTH_MIN_FREE_DISK_BYTES", str(2 * 1024**3))),
)
CACHE_TTL_SECONDS = max(
    3600,
    int(os.getenv("PIERN_NEW_SYNTH_CACHE_TTL_SECONDS", str(14 * 24 * 3600))),
)
_THREADS: dict[str, threading.Thread] = {}
_PROCESSES: dict[str, subprocess.Popen[str]] = {}
_THREAD_LOCK = threading.RLock()
logger = logging.getLogger(__name__)


class NewSynthError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def initialize() -> None:
    store.init_store()
    cleanup_expired_caches()


def cleanup_expired_caches(*, now: float | None = None) -> dict[str, int]:
    """Delete only rebuildable cache entries after their last access time expires."""
    current = float(now or time.time())
    deleted_files = 0
    deleted_bytes = 0
    if not NEW_SYNTH_CACHE_ROOT.exists():
        return {"deleted_files": 0, "deleted_bytes": 0}
    root = NEW_SYNTH_CACHE_ROOT.resolve()
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        last_used = max(stat.st_atime, stat.st_mtime)
        if current - last_used < CACHE_TTL_SECONDS:
            continue
        try:
            deleted_bytes += int(stat.st_size)
            path.unlink()
            deleted_files += 1
        except OSError:
            continue
    return {"deleted_files": deleted_files, "deleted_bytes": deleted_bytes}


def presets() -> dict[str, Any]:
    scenarios = []
    for item in simulation._get_scenarios_cached():
        scenarios.append(
            {
                "simulator": item.simulator,
                "scenario": item.scenario,
                "sample_count": item.sample_count,
                "has_data": bool(item.h5_path),
                "output_shape": item.output_shape,
            }
        )
    return {
        "source_types": ["upload", "simulation", "expert"],
        "accepted_uploads": [".h5", ".hdf5"],
        "max_upload_bytes": 1024**3,
        "max_generation_samples": MAX_GENERATION_SAMPLES,
        "router_mode": "binary",
        "simulations": scenarios,
        "experts": [
            {
                "model_id": item.get("model_id"),
                "name": item.get("name"),
                "input_dim": item.get("input_dim"),
                "output_dim": item.get("output_dim"),
                "active": item.get("status") == "active",
                "data_generation_enabled": bool(item.get("data_generation_enabled")),
            }
            for item in expert_models.list_models()
        ],
    }


def _public_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    source = dict(workflow.get("source") or {}) or None
    if source:
        for key in ("source_path", "canonical_path", "config_path", "generated_path"):
            source.pop(key, None)
    artifacts = dict(workflow.get("artifacts") or {}) or None
    if artifacts:
        for key in ("manifest_path", "template_path", "evaluation_path"):
            artifacts.pop(key, None)
    return {
        "workflow_id": workflow["workflow_id"],
        "name": workflow["name"],
        "status": workflow["status"],
        "current_step": workflow["current_step"],
        "created_at": workflow["created_at"],
        "updated_at": workflow["updated_at"],
        "source": source,
        "definition": workflow.get("definition"),
        "artifacts": artifacts,
        "error": workflow.get("error"),
        "cancel_requested": bool(workflow.get("cancel_requested")),
        "can_define": bool(source and source.get("ready")) and workflow["status"] != "running",
        "can_generate": bool(source and source.get("ready") and workflow.get("definition"))
        and workflow["status"] != "running",
        "can_open_training": workflow["status"] == "succeeded" and bool(artifacts),
    }


def list_workflows(owner_id: str) -> list[dict[str, Any]]:
    return [_public_workflow(item) for item in store.list_workflows(owner_id)]


def get_workflow(owner_id: str, workflow_id: str) -> dict[str, Any]:
    return _public_workflow(store.get_workflow(owner_id, workflow_id))


def create_workflow(owner_id: str, name: str) -> dict[str, Any]:
    workflow = store.create_workflow(owner_id, name)
    workflow_paths(workflow["workflow_id"], create=True)
    return _public_workflow(workflow)


def _ensure_mutable(workflow: dict[str, Any]) -> None:
    if workflow["status"] == "running":
        raise NewSynthError("workflow_running", "当前任务正在运行，请先等待完成或取消。")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_names(dataset: h5py.Dataset) -> list[str]:
    result: list[str] = []
    for item in dataset[:].reshape(-1):
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            result.append(bytes(item).decode("utf-8", errors="replace").strip())
        else:
            result.append(str(item).strip())
    return result


def _normalize_hdf5(source_path: Path, target_path: Path) -> dict[str, Any]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_name(f".{target_path.stem}.tmp{target_path.suffix}")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source_path, temporary)
    try:
        with h5py.File(temporary, "r+") as handle:
            missing = [name for name in ("timeseries", "params", "param_names") if name not in handle]
            if missing:
                raise NewSynthError(
                    "invalid_hdf5",
                    f"数据缺少必需字段：{', '.join(missing)}",
                )
            timeseries = handle["timeseries"]
            params = handle["params"]
            param_names = handle["param_names"]
            if timeseries.ndim != 3:
                raise NewSynthError("invalid_hdf5", "timeseries 必须是 [样本, 通道, 时间] 三维数组。")
            if params.ndim != 2:
                raise NewSynthError("invalid_hdf5", "params 必须是 [样本, 参数] 二维数组。")
            if param_names.ndim != 1:
                raise NewSynthError("invalid_hdf5", "param_names 必须是一维参数名称数组。")
            if int(timeseries.shape[0]) != int(params.shape[0]):
                raise NewSynthError("invalid_hdf5", "输入参数与物理输出的样本数量不一致。")
            if int(params.shape[1]) != int(param_names.shape[0]):
                raise NewSynthError("invalid_hdf5", "参数名称数量与参数维度不一致。")
            handle.attrs["n_samples"] = int(timeseries.shape[0])
            handle.attrs["n_channels"] = int(timeseries.shape[1])
            handle.attrs["n_timesteps"] = int(timeseries.shape[2])
            handle.attrs["n_params"] = int(params.shape[1])
        validation = validate_hdf5_file(temporary)
        if not validation.get("valid"):
            raise NewSynthError(
                "invalid_hdf5",
                "；".join(str(item) for item in validation.get("errors") or ["HDF5 校验失败"]),
            )
        temporary.replace(target_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    with h5py.File(target_path, "r") as handle:
        timeseries = handle["timeseries"]
        params = handle["params"]
        names = _decode_names(handle["param_names"])
        preview_count = min(3, int(params.shape[0]))
        stat_count = min(256, int(params.shape[0]))
        param_sample = np.asarray(params[:stat_count], dtype=np.float64)
        output_sample = np.asarray(timeseries[:stat_count], dtype=np.float64)
        metadata = {
            "sample_count": int(params.shape[0]),
            "input_dim": int(params.shape[1]),
            "input_shape": [int(params.shape[1])],
            "output_shape": [int(timeseries.shape[1]), int(timeseries.shape[2])],
            "param_names": names,
            "input_stats": {
                "min": float(param_sample.min()),
                "max": float(param_sample.max()),
                "mean": float(param_sample.mean()),
                "std": float(param_sample.std()),
            },
            "output_stats": {
                "min": float(output_sample.min()),
                "max": float(output_sample.max()),
                "mean": float(output_sample.mean()),
                "std": float(output_sample.std()),
            },
            "preview": [
                {
                    "expert_input": np.asarray(params[index], dtype=float).reshape(-1).tolist(),
                    "expert_output": np.asarray(timeseries[index], dtype=float).reshape(-1)[:24].tolist(),
                }
                for index in range(preview_count)
            ],
        }
    metadata["content_hash"] = _sha256(target_path)
    metadata["file_size_bytes"] = target_path.stat().st_size
    metadata["canonical_path"] = str(target_path)
    return metadata


def _default_definition(metadata: dict[str, Any], *, simulator: str, scenario: str) -> dict[str, Any]:
    parameters = [
        {
            "index": index,
            "name": name or f"param_{index + 1}",
            "display_name": name or f"参数 {index + 1}",
            "description": "",
            "unit": "",
        }
        for index, name in enumerate(metadata["param_names"])
    ]
    channels = int(metadata["output_shape"][0])
    outputs = [
        {
            "index": index,
            "name": f"output_{index + 1}",
            "display_name": f"输出通道 {index + 1}",
            "description": "",
            "unit": "",
        }
        for index in range(channels)
    ]
    return {
        "schema_name": "piernet.data-definition",
        "schema_version": 1,
        "version": 1,
        "simulator": simulator,
        "scenario": scenario,
        "task_description": f"根据给定参数完成 {simulator}/{scenario} 科学计算",
        "parameters": parameters,
        "outputs": outputs,
        "sampling": {"channels": None, "time_stride": 1, "max_time_points": None},
        "confirmed": False,
    }


def _load_llm_config() -> dict[str, Any]:
    config_path = CONFIG_DIR / "default.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise NewSynthError("llm_config_invalid", f"读取智能识别 API 配置失败：{exc}") from exc
    config = payload.get("llm") or {}
    return config if isinstance(config, dict) else {}


def _parse_llm_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise NewSynthError("llm_response_invalid", "智能识别 API 没有返回可解析的数据定义。")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise NewSynthError("llm_response_invalid", "智能识别 API 返回的 JSON 格式无效。") from exc
    if not isinstance(payload, dict):
        raise NewSynthError("llm_response_invalid", "智能识别 API 返回的数据定义不是对象。")
    return payload


def _request_definition_suggestion(definition: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    from PierNet.core.llm_client import LLMClient

    config = _load_llm_config()
    provider = str(config.get("provider") or "siliconflow")
    model = str(config.get("model") or "").strip()
    if not model:
        raise NewSynthError("llm_not_configured", "请先配置智能识别 API 的模型名称。")

    context = {
        "simulator": definition["simulator"],
        "scenario": definition["scenario"],
        "current_task_description": definition["task_description"],
        "parameters": [
            {
                "index": item["index"],
                "name": item["name"],
                "display_name": item.get("display_name", ""),
                "description": item.get("description", ""),
                "unit": item.get("unit", ""),
            }
            for item in definition["parameters"]
        ],
        "outputs": [
            {
                "index": item["index"],
                "name": item["name"],
                "display_name": item.get("display_name", ""),
                "description": item.get("description", ""),
                "unit": item.get("unit", ""),
            }
            for item in definition["outputs"]
        ],
        "source_shape": {
            "expert_input": source.get("input_shape"),
            "expert_output": source.get("output_shape"),
        },
    }
    prompt = (
        "请为科学计算训练数据补全中文数据定义。只返回一个 JSON 对象，不要使用 Markdown。"
        "必须保持 parameters 和 outputs 的数量、index、name 完全不变；不得推测或返回数值样本。"
        "返回字段为 task_description、parameters、outputs。每个数组元素仅包含 index、name、"
        "display_name、description、unit。task_description 不超过 80 个汉字，display_name 不超过"
        "12 个汉字，description 不超过 28 个汉字。信息不确定时使用空字符串，不要编造单位。\n\n"
        f"待补全结构：{json.dumps(context, ensure_ascii=False)}"
    )
    try:
        client = LLMClient(
            provider=provider,
            model=model,
            api_key=str(config.get("api_key") or "") or None,
            base_url=str(config.get("base_url") or "") or None,
            max_retries=2,
            timeout=60,
            thinking=str(config.get("thinking") or "disabled"),
        )
        reply = client.generate(
            prompt,
            system_prompt="你是科学计算数据字典助手，必须严格输出 JSON。",
            temperature=0.1,
            max_tokens=min(2048, max(512, int(config.get("max_tokens") or 1024))),
        )
    except NewSynthError:
        raise
    except Exception as exc:
        logger.warning("New synthesis definition suggestion failed: %s", exc)
        raise NewSynthError("llm_unavailable", "智能识别 API 调用失败，请检查连接和模型配置。") from exc
    return _parse_llm_json(reply)


def _placeholder_display_name(item: dict[str, Any], *, output: bool) -> bool:
    value = str(item.get("display_name") or "").strip()
    defaults = {
        "",
        str(item.get("name") or "").strip(),
        f"{'输出通道' if output else '参数'} {int(item['index']) + 1}",
    }
    return value in defaults


def _merge_definition_suggestion(definition: dict[str, Any], suggestion: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(definition))
    default_task = f"根据给定参数完成 {definition['simulator']}/{definition['scenario']} 科学计算"
    task = str(suggestion.get("task_description") or "").strip()
    if task and str(merged.get("task_description") or "").strip() in {"", default_task}:
        merged["task_description"] = task[:2000]

    limits = {"display_name": 180, "description": 1000, "unit": 64}
    for collection, output in (("parameters", False), ("outputs", True)):
        proposed = suggestion.get(collection)
        if not isinstance(proposed, list):
            continue
        by_name = {
            str(item.get("name")): item for item in proposed if isinstance(item, dict) and item.get("name") is not None
        }
        by_index = {
            int(item["index"]): item
            for item in proposed
            if isinstance(item, dict) and isinstance(item.get("index"), int)
        }
        for item in merged[collection]:
            candidate = by_name.get(str(item["name"])) or by_index.get(int(item["index"]))
            if not candidate:
                continue
            for field, limit in limits.items():
                value = str(candidate.get(field) or "").strip()
                if not value:
                    continue
                current = str(item.get(field) or "").strip()
                can_fill = not current or (field == "display_name" and _placeholder_display_name(item, output=output))
                if can_fill:
                    item[field] = value[:limit]
    return merged


def suggest_definition(owner_id: str, workflow_id: str, definition: dict[str, Any]) -> dict[str, Any]:
    workflow = store.get_workflow(owner_id, workflow_id)
    source = workflow.get("source") or {}
    canonical = workflow.get("definition") or {}
    if not source.get("ready") or not canonical:
        raise NewSynthError("source_required", "请先接入并校验数据来源。")
    for collection in ("parameters", "outputs"):
        current_items = list(definition.get(collection) or [])
        canonical_items = list(canonical.get(collection) or [])
        if len(current_items) != len(canonical_items):
            raise NewSynthError("invalid_definition", "数据定义维度与已校验的数据来源不一致。")
        for current, expected in zip(current_items, canonical_items, strict=True):
            if current.get("index") != expected.get("index") or current.get("name") != expected.get("name"):
                raise NewSynthError("invalid_definition", "智能补全不能修改参数顺序或字段名称。")
    suggestion = _request_definition_suggestion(definition, source)
    merged = _merge_definition_suggestion(definition, suggestion)
    store.append_event(workflow_id, "definition_suggested", {"message": "智能识别已补全数据定义"})
    return merged


def _record_source(
    owner_id: str,
    workflow_id: str,
    source_path: Path,
    *,
    source_type: str,
    simulator: str,
    scenario: str,
    filename: str,
    extra: dict[str, Any] | None = None,
    allow_running: bool = False,
) -> dict[str, Any]:
    workflow = store.get_workflow(owner_id, workflow_id)
    if not allow_running:
        _ensure_mutable(workflow)
    paths = workflow_paths(workflow_id)
    metadata = _normalize_hdf5(source_path, paths.canonical / "data.h5")
    source = {
        **metadata,
        "source_type": source_type,
        "filename": filename,
        "simulator": simulator,
        "scenario": scenario,
        "ready": True,
        "source_path": str(source_path),
        **(extra or {}),
    }
    definition = _default_definition(metadata, simulator=simulator, scenario=scenario)
    store.update_workflow(
        workflow_id,
        status="draft",
        current_step="definition",
        source=source,
        definition=definition,
        artifacts=None,
        error=None,
        cancel_requested=False,
    )
    store.append_event(
        workflow_id,
        "source_ready",
        {
            "message": "数据已经完成校验，可以确认数据定义。",
            "source_type": source_type,
            "sample_count": metadata["sample_count"],
        },
    )
    return get_workflow(owner_id, workflow_id)


def attach_uploaded_hdf5(
    owner_id: str,
    workflow_id: str,
    source_path: Path,
    original_name: str,
) -> dict[str, Any]:
    if source_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise NewSynthError("unsupported_file", "第一版新数据合成只接受 .h5 或 .hdf5 文件。")
    stem = source_path.stem
    simulator = "user_data"
    scenario = stem[:80] or "scenario"
    return _record_source(
        owner_id,
        workflow_id,
        source_path,
        source_type="upload",
        simulator=simulator,
        scenario=scenario,
        filename=original_name,
    )


def use_existing_simulation(
    owner_id: str,
    workflow_id: str,
    simulator_name: str,
    scenario_name: str,
) -> dict[str, Any]:
    source_path = canonical_hdf5_path(simulator_name, scenario_name)
    if not source_path.is_file():
        raise NewSynthError("simulation_data_missing", "该场景还没有可复用数据，请运行一次内置仿真。")
    return _record_source(
        owner_id,
        workflow_id,
        source_path,
        source_type="simulation",
        simulator=simulator_name,
        scenario=scenario_name,
        filename=source_path.name,
    )


def _find_simulation(simulator_name: str, scenario_name: str):
    for item in simulation._get_scenarios_cached():
        if item.simulator == simulator_name and item.scenario == scenario_name:
            return item
    raise NewSynthError("simulation_not_found", "没有找到对应的内置仿真场景。")


def start_simulation_source(
    owner_id: str,
    workflow_id: str,
    *,
    simulator_name: str,
    scenario_name: str,
    n_samples: int,
    seed: int,
    reuse_existing: bool,
) -> dict[str, Any]:
    workflow = store.get_workflow(owner_id, workflow_id)
    _ensure_mutable(workflow)
    existing = canonical_hdf5_path(simulator_name, scenario_name)
    if reuse_existing and existing.is_file():
        return use_existing_simulation(owner_id, workflow_id, simulator_name, scenario_name)
    scenario = _find_simulation(simulator_name, scenario_name)
    request = simulation.SimulateRequest(
        simulator=simulator_name,
        scenario=scenario_name,
        n_samples=n_samples,
        seed=seed,
        config_path=scenario.config_path,
        skip_existing=False,
        parallel=False,
        max_workers=1,
    )
    store.update_workflow(
        workflow_id,
        status="running",
        current_step="source",
        source={
            "source_type": "simulation",
            "simulator": simulator_name,
            "scenario": scenario_name,
            "ready": False,
            "progress": 0.0,
            "message": "正在运行内置仿真",
        },
        error=None,
        cancel_requested=False,
    )
    store.append_event(workflow_id, "source_started", {"message": "内置仿真已经开始"})
    thread = threading.Thread(
        target=_run_simulation_source,
        args=(workflow_id, request),
        name=f"new-synth-sim-{workflow_id}",
        daemon=True,
    )
    with _THREAD_LOCK:
        _THREADS[workflow_id] = thread
    thread.start()
    return get_workflow(owner_id, workflow_id)


def _run_simulation_source(workflow_id: str, request: simulation.SimulateRequest) -> None:
    workflow = store.get_workflow_unscoped(workflow_id)
    owner_id = str(workflow["owner_id"])
    runtime_config, temporary_config = simulation._prepare_runtime_config(request)
    output_path = simulation._resolve_output_h5_path(request.config_path, request.simulator)
    lock_key = (
        f"raw-file:{output_path.resolve(strict=False) if output_path else request.simulator + '/' + request.scenario}"
    )
    if not task_locks.acquire_lock(lock_key, workflow_id, ttl_seconds=24 * 3600, metadata={"job_type": "new_synth"}):
        _fail_workflow(workflow_id, "source_busy", "这个仿真场景正在被其他任务使用，请稍后重试。")
        simulation._cleanup_runtime_config(temporary_config)
        return
    command = [
        sys.executable,
        "-m",
        f"PierNet.simulators.{request.simulator}.pipeline",
        "--config",
        runtime_config,
        "--n-samples",
        str(request.n_samples),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        with _THREAD_LOCK:
            _PROCESSES[workflow_id] = process
        assert process.stdout is not None
        for line in process.stdout:
            if store.get_workflow_unscoped(workflow_id).get("cancel_requested"):
                process.terminate()
                raise InterruptedError("用户取消了内置仿真")
            counts = simulation._extract_progress_counts(line)
            if counts:
                done, total = counts
                progress = done / max(1, total)
                source = dict(store.get_workflow_unscoped(workflow_id).get("source") or {})
                source.update({"progress": progress, "message": f"正在生成数据 {done}/{total}"})
                store.update_workflow(workflow_id, source=source)
                store.append_event(
                    workflow_id,
                    "progress",
                    {"step": "source", "progress": progress, "message": source["message"]},
                )
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"内置仿真退出码为 {return_code}")
        if output_path is None or not output_path.is_file():
            raise FileNotFoundError("内置仿真没有生成预期的 HDF5 文件")
        _record_source(
            owner_id,
            workflow_id,
            output_path,
            source_type="simulation",
            simulator=request.simulator,
            scenario=request.scenario,
            filename=output_path.name,
            allow_running=True,
        )
    except InterruptedError as exc:
        store.update_workflow(
            workflow_id,
            status="cancelled",
            error={"code": "cancelled", "message": str(exc)},
            cancel_requested=False,
        )
        store.append_event(workflow_id, "cancelled", {"message": str(exc)})
    except Exception as exc:
        _fail_workflow(workflow_id, "simulation_failed", f"内置仿真失败：{exc}")
    finally:
        simulation._cleanup_runtime_config(temporary_config)
        task_locks.release_lock(lock_key, workflow_id)
        with _THREAD_LOCK:
            _PROCESSES.pop(workflow_id, None)
            _THREADS.pop(workflow_id, None)


def upload_expert_model(name: str, content: bytes) -> dict[str, Any]:
    try:
        return expert_models.upload_model(name, content)
    except (expert_models.ExpertModelError, ValueError, OSError) as exc:
        raise NewSynthError("invalid_expert", str(exc)) from exc


def start_expert_source(
    owner_id: str,
    workflow_id: str,
    *,
    model_id: str,
    scenario: str,
    prompt: str,
    input_dim: int | None,
) -> dict[str, Any]:
    workflow = store.get_workflow(owner_id, workflow_id)
    _ensure_mutable(workflow)
    model = expert_models.get_model(model_id)
    internal_scenario = f"ns_{workflow_id[-8:]}_{scenario}"[:120]
    store.update_workflow(
        workflow_id,
        status="running",
        current_step="source",
        source={
            "source_type": "expert",
            "model_id": model_id,
            "model_name": model.get("name"),
            "scenario": scenario,
            "ready": False,
            "progress": None,
            "message": "正在调用专家模型生成数据",
        },
        error=None,
        cancel_requested=False,
    )
    thread = threading.Thread(
        target=_run_expert_source,
        args=(workflow_id, model_id, internal_scenario, scenario, prompt, input_dim),
        name=f"new-synth-expert-{workflow_id}",
        daemon=True,
    )
    with _THREAD_LOCK:
        _THREADS[workflow_id] = thread
    thread.start()
    return get_workflow(owner_id, workflow_id)


def _run_expert_source(
    workflow_id: str,
    model_id: str,
    internal_scenario: str,
    display_scenario: str,
    prompt: str,
    input_dim: int | None,
) -> None:
    workflow = store.get_workflow_unscoped(workflow_id)
    try:
        result = expert_models.generate_dataset(
            model_id=model_id,
            scenario=internal_scenario,
            prompt=prompt,
            input_dim=input_dim,
            overwrite=True,
        )
        if store.get_workflow_unscoped(workflow_id).get("cancel_requested"):
            raise InterruptedError("用户取消了专家数据生成")
        _record_source(
            str(workflow["owner_id"]),
            workflow_id,
            Path(result["saved_path"]),
            source_type="expert",
            simulator="uploaded_expert",
            scenario=display_scenario,
            filename=Path(result["saved_path"]).name,
            extra={"model_id": model_id, "input_plan": result.get("input_plan")},
            allow_running=True,
        )
    except InterruptedError as exc:
        store.update_workflow(
            workflow_id,
            status="cancelled",
            error={"code": "cancelled", "message": str(exc)},
            cancel_requested=False,
        )
        store.append_event(workflow_id, "cancelled", {"message": str(exc)})
    except Exception as exc:
        _fail_workflow(workflow_id, "expert_generation_failed", f"专家模型数据生成失败：{exc}")
    finally:
        with _THREAD_LOCK:
            _THREADS.pop(workflow_id, None)


def save_definition(owner_id: str, workflow_id: str, definition: dict[str, Any]) -> dict[str, Any]:
    workflow = store.get_workflow(owner_id, workflow_id)
    _ensure_mutable(workflow)
    source = workflow.get("source") or {}
    if not source.get("ready"):
        raise NewSynthError("source_required", "请先接入并校验数据。")
    input_dim = int(source["input_dim"])
    output_channels = int(source["output_shape"][0])
    parameters = list(definition.get("parameters") or [])
    outputs = list(definition.get("outputs") or [])
    if len(parameters) != input_dim or sorted(int(item["index"]) for item in parameters) != list(range(input_dim)):
        raise NewSynthError("invalid_definition", "参数定义必须与专家输入参数数量和顺序完全一致。")
    if len(outputs) != output_channels or sorted(int(item["index"]) for item in outputs) != list(
        range(output_channels)
    ):
        raise NewSynthError("invalid_definition", "输出定义必须覆盖全部物理输出通道。")
    sampling = dict(definition.get("sampling") or {})
    channels = sampling.get("channels")
    if channels is not None:
        normalized = sorted({int(item) for item in channels})
        if not normalized or normalized[0] < 0 or normalized[-1] >= output_channels:
            raise NewSynthError("invalid_definition", "输出通道选择超出数据范围。")
        sampling["channels"] = normalized
    previous = workflow.get("definition") or {}
    saved = {
        **definition,
        "schema_name": "piernet.data-definition",
        "schema_version": 1,
        "version": int(previous.get("version") or 0) + 1,
        "sampling": sampling,
        "confirmed": True,
        "confirmed_at": time.time(),
    }
    paths = workflow_paths(workflow_id)
    definition_path = paths.definitions / "definition.json"
    definition_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    store.update_workflow(
        workflow_id,
        definition=saved,
        status="draft",
        current_step="generation",
        artifacts=None,
        error=None,
    )
    store.append_event(workflow_id, "definition_saved", {"message": "数据定义已经确认"})
    return get_workflow(owner_id, workflow_id)


def _format_value(value: float) -> str:
    return f"{float(value):.8g}"


def _prompt_templates(task_description: str, parameter_names: list[str]) -> list[str]:
    values = "；".join(f"{name}={{value_{index}}}" for index, name in enumerate(parameter_names))
    return [
        f"请完成{task_description}。已知参数：{values}。请给出计算所需参数。",
        f"我要执行{task_description}，输入条件为：{values}。",
        f"根据这些条件准备科学计算输入：{values}。任务是{task_description}。",
        f"帮我求解{task_description}。参数为：{values}。",
    ]


def _render_prompt(template: str, values: np.ndarray) -> str:
    result = template
    for index, value in enumerate(values):
        result = result.replace(f"{{value_{index}}}", _format_value(float(value)))
    return result


def _router_context(prompt: str, route_prefix: str, *, positive: bool, rng: random.Random) -> str:
    base = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if positive:
        return base + route_prefix
    if len(route_prefix) <= 1:
        return base
    return base + route_prefix[: rng.randint(0, len(route_prefix) - 1)]


def _stable_dataset_id(kind: str, source_hash: str, definition: dict[str, Any], config: dict[str, Any]) -> str:
    serialized = json.dumps(
        {"kind": kind, "source": source_hash, "definition": definition, "config": config},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    prefix = {"text2comp": "t2c", "router": "router", "evaluation": "eval"}[kind]
    return f"{prefix}-{digest}"


def _dataset_payload(
    *,
    dataset_id: str,
    workflow: dict[str, Any],
    kind: str,
    path: Path,
    root_path: Path,
    sample_count: int,
    schema_name: str,
    content_hash: str,
    paired_dataset_id: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    definition = workflow["definition"]
    return {
        "dataset_id": dataset_id,
        "workflow_id": workflow["workflow_id"],
        "owner_id": workflow["owner_id"],
        "kind": kind,
        "name": f"{workflow['name']} · {'Text2Comp' if kind == 'text2comp' else 'Router' if kind == 'router' else '评测'}",
        "simulator": definition["simulator"],
        "scenario": definition["scenario"],
        "schema_name": schema_name,
        "schema_version": 1,
        "sample_count": sample_count,
        "size_bytes": path.stat().st_size,
        "path": str(path),
        "root_path": str(root_path),
        "content_hash": content_hash,
        "paired_dataset_id": paired_dataset_id,
        "metadata": metadata,
        "created_at": time.time(),
    }


def _set_progress(workflow_id: str, progress: float | None, message: str, phase: str) -> None:
    workflow = store.get_workflow_unscoped(workflow_id)
    artifacts = dict(workflow.get("artifacts") or {})
    artifacts.update({"progress": progress, "message": message, "phase": phase})
    store.update_workflow(workflow_id, artifacts=artifacts, current_step="generation")
    store.append_event(
        workflow_id,
        "progress",
        {"step": "generation", "phase": phase, "progress": progress, "message": message},
    )


def _preflight_generation(workflow: dict[str, Any], config: dict[str, Any]) -> int:
    source = workflow["source"]
    selected = min(
        int(source["sample_count"]),
        int(config.get("max_samples") or source["sample_count"]),
        MAX_GENERATION_SAMPLES,
    )
    prompt_count = selected * int(config["variants_per_sample"])
    estimated = prompt_count * (800 + int(source["input_dim"]) * 24)
    estimated += prompt_count * (1 + int(config["negative_ratio"])) * 1400
    free = shutil.disk_usage(workflow_paths(workflow["workflow_id"]).root).free
    required = max(MIN_FREE_DISK_BYTES, estimated * 2)
    if free < required:
        raise NewSynthError(
            "insufficient_storage",
            f"预计需要至少 {required / 1024**3:.1f} GB 可用空间，当前空间不足。",
        )
    return selected


def start_generation(owner_id: str, workflow_id: str, config: dict[str, Any]) -> dict[str, Any]:
    workflow = store.get_workflow(owner_id, workflow_id)
    _ensure_mutable(workflow)
    if not (workflow.get("source") or {}).get("ready"):
        raise NewSynthError("source_required", "请先接入并校验数据。")
    if not (workflow.get("definition") or {}).get("confirmed"):
        raise NewSynthError("definition_required", "请先确认数据定义。")
    selected = _preflight_generation(workflow, config)
    initial_artifacts = {
        "progress": 0.0,
        "phase": "templates",
        "message": "正在准备语言模板",
        "planned_source_samples": selected,
        "negative_ratio": int(config["negative_ratio"]),
    }
    store.update_workflow(
        workflow_id,
        status="running",
        current_step="generation",
        artifacts=initial_artifacts,
        error=None,
        cancel_requested=False,
    )
    store.append_event(workflow_id, "generation_started", {"message": "训练数据生成已经开始"})
    thread = threading.Thread(
        target=_run_generation,
        args=(workflow_id, dict(config)),
        name=f"new-synth-generate-{workflow_id}",
        daemon=True,
    )
    with _THREAD_LOCK:
        _THREADS[workflow_id] = thread
    thread.start()
    return get_workflow(owner_id, workflow_id)


def _run_generation(workflow_id: str, config: dict[str, Any]) -> None:
    workflow = store.get_workflow_unscoped(workflow_id)
    paths = workflow_paths(workflow_id)
    definition = workflow["definition"]
    source = workflow["source"]
    try:
        selected_count = _preflight_generation(workflow, config)
        parameter_names = [str(item["name"]) for item in definition["parameters"]]
        templates = _prompt_templates(definition["task_description"], parameter_names)
        template_path = paths.templates / "templates.jsonl"
        with template_path.open("w", encoding="utf-8") as handle:
            for index, template in enumerate(templates):
                handle.write(
                    json.dumps(
                        {
                            "schema_name": "piernet.prompt-template",
                            "schema_version": 1,
                            "simulator": definition["simulator"],
                            "scenario": definition["scenario"],
                            "template_id": f"template-{index + 1}",
                            "input_template": template,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        _set_progress(workflow_id, 0.12, "语言模板已准备，正在生成 Text2Comp 数据", "text2comp")

        generation_config = {
            "max_samples": selected_count,
            "variants_per_sample": int(config["variants_per_sample"]),
            "negative_ratio": int(config["negative_ratio"]),
            "seed": int(config["seed"]),
        }
        text2comp_id = _stable_dataset_id("text2comp", source["content_hash"], definition, generation_config)
        router_id = _stable_dataset_id("router", source["content_hash"], definition, generation_config)
        evaluation_id = _stable_dataset_id("evaluation", source["content_hash"], definition, generation_config)
        text2comp_path = paths.text2comp / "train.jsonl"
        router_scenario_dir = paths.router / "by_scenario"
        router_scenario_dir.mkdir(parents=True, exist_ok=True)
        router_path = router_scenario_dir / f"{definition['scenario']}.jsonl"
        evaluation_path = paths.evaluation / "evaluation.jsonl"
        rng = np.random.default_rng(int(config["seed"]))
        py_rng = random.Random(int(config["seed"]))
        route_prefix = f"{definition['simulator']}/{definition['scenario']} 专家计算："

        with h5py.File(source["canonical_path"], "r") as handle:
            params = handle["params"]
            timeseries = handle["timeseries"]
            indices = rng.choice(int(params.shape[0]), size=selected_count, replace=False)
            variants = int(config["variants_per_sample"])
            negative_ratio = int(config["negative_ratio"])
            channels = definition["sampling"].get("channels")
            if channels is None:
                channels = list(range(int(timeseries.shape[1])))
            stride = int(definition["sampling"].get("time_stride") or 1)
            max_time = definition["sampling"].get("max_time_points")
            time_indices = list(range(0, int(timeseries.shape[2]), stride))
            if max_time is not None:
                time_indices = time_indices[: int(max_time)]

            text_count = 0
            router_count = 0
            with (
                text2comp_path.open("w", encoding="utf-8") as text_handle,
                router_path.open("w", encoding="utf-8") as router_handle,
                evaluation_path.open("w", encoding="utf-8") as eval_handle,
            ):
                for position, source_index in enumerate(indices.tolist()):
                    if store.get_workflow_unscoped(workflow_id).get("cancel_requested"):
                        raise InterruptedError("用户取消了训练数据生成")
                    expert_input = np.asarray(params[source_index], dtype=np.float32).reshape(-1)
                    for variant in range(variants):
                        template = templates[(position + variant) % len(templates)]
                        prompt = _render_prompt(template, expert_input)
                        base_metadata = {
                            "workflow_id": workflow_id,
                            "dataset_id": text2comp_id,
                            "simulator": definition["simulator"],
                            "scenario": definition["scenario"],
                            "sample_index": int(source_index),
                            "label_semantics": "expert_input",
                            "parameter_names": parameter_names,
                            "definition_version": int(definition["version"]),
                        }
                        text_handle.write(
                            json.dumps(
                                {
                                    "schema_name": "piernet.text2comp",
                                    "schema_version": 1,
                                    "artifact_type": "training_dataset",
                                    "prompt": prompt,
                                    "label": expert_input.astype(float).tolist(),
                                    "expert_input": expert_input.astype(float).tolist(),
                                    "metadata": base_metadata,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        text_count += 1

                        router_meta = {
                            "workflow_id": workflow_id,
                            "dataset_id": router_id,
                            "simulator": definition["simulator"],
                            "scenario": definition["scenario"],
                            "route_label": f"{definition['simulator']}/{definition['scenario']}",
                            "class_names": ["not_target", "target"],
                            "source_sample_index": int(source_index),
                        }
                        router_handle.write(
                            json.dumps(
                                {
                                    "schema_name": "piernet.router.binary",
                                    "schema_version": 1,
                                    "artifact_type": "training_dataset",
                                    "context": _router_context(prompt, route_prefix, positive=True, rng=py_rng),
                                    "label": 1,
                                    "metadata": router_meta,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        router_count += 1
                        for _ in range(negative_ratio):
                            router_handle.write(
                                json.dumps(
                                    {
                                        "schema_name": "piernet.router.binary",
                                        "schema_version": 1,
                                        "artifact_type": "training_dataset",
                                        "context": _router_context(prompt, route_prefix, positive=False, rng=py_rng),
                                        "label": 0,
                                        "metadata": router_meta,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            router_count += 1

                    if position < 64:
                        physical = np.asarray(timeseries[source_index, channels, :][:, time_indices], dtype=np.float32)
                        eval_handle.write(
                            json.dumps(
                                {
                                    "schema_name": "piernet.expert-evaluation",
                                    "schema_version": 1,
                                    "artifact_type": "evaluation_dataset",
                                    "expert_input": expert_input.astype(float).tolist(),
                                    "expected_expert_output": physical.astype(float).tolist(),
                                    "metadata": {
                                        "workflow_id": workflow_id,
                                        "dataset_id": evaluation_id,
                                        "simulator": definition["simulator"],
                                        "scenario": definition["scenario"],
                                        "sample_index": int(source_index),
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    if position % max(1, selected_count // 10) == 0:
                        _set_progress(
                            workflow_id,
                            0.12 + 0.72 * ((position + 1) / selected_count),
                            f"正在生成训练样本 {position + 1}/{selected_count}",
                            "datasets",
                        )

        _set_progress(workflow_id, 0.9, "正在校验并登记训练数据", "validation")
        text_hash = _sha256(text2comp_path)
        router_hash = _sha256(router_path)
        evaluation_hash = _sha256(evaluation_path)
        evaluation_count = min(64, selected_count)
        text_payload = _dataset_payload(
            dataset_id=text2comp_id,
            workflow=workflow,
            kind="text2comp",
            path=text2comp_path,
            root_path=paths.text2comp,
            sample_count=text_count,
            schema_name="piernet.text2comp",
            content_hash=text_hash,
            paired_dataset_id=router_id,
            metadata={
                "label_semantics": "expert_input",
                "input_dim": int(source["input_dim"]),
                "parameter_names": parameter_names,
                "definition_version": int(definition["version"]),
            },
        )
        router_payload = _dataset_payload(
            dataset_id=router_id,
            workflow=workflow,
            kind="router",
            path=router_path,
            root_path=paths.router,
            sample_count=router_count,
            schema_name="piernet.router.binary",
            content_hash=router_hash,
            paired_dataset_id=text2comp_id,
            metadata={
                "class_names": ["not_target", "target"],
                "negative_ratio": int(config["negative_ratio"]),
                "route_prefix": route_prefix,
                "definition_version": int(definition["version"]),
            },
        )
        evaluation_payload = _dataset_payload(
            dataset_id=evaluation_id,
            workflow=workflow,
            kind="evaluation",
            path=evaluation_path,
            root_path=paths.evaluation,
            sample_count=evaluation_count,
            schema_name="piernet.expert-evaluation",
            content_hash=evaluation_hash,
            paired_dataset_id=text2comp_id,
            metadata={"output_semantics": "expert_output_ground_truth"},
        )
        store.register_dataset(text_payload)
        store.register_dataset(router_payload)
        store.register_dataset(evaluation_payload)
        manifest = {
            "schema_name": "piernet.new-synth-manifest",
            "schema_version": 1,
            "workflow_id": workflow_id,
            "source_hash": source["content_hash"],
            "definition_version": int(definition["version"]),
            "generation_config": generation_config,
            "datasets": [text_payload, router_payload, evaluation_payload],
            "created_at": time.time(),
        }
        manifest_path = paths.artifacts / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts = {
            "progress": 1.0,
            "phase": "complete",
            "message": "训练数据已经生成并登记",
            "manifest_path": str(manifest_path),
            "template_path": str(template_path),
            "evaluation_path": str(evaluation_path),
            "text2comp": {
                "dataset_id": text2comp_id,
                "sample_count": text_count,
                "schema_name": "piernet.text2comp",
                "label_semantics": "expert_input",
            },
            "router": {
                "dataset_id": router_id,
                "sample_count": router_count,
                "schema_name": "piernet.router.binary",
                "class_names": ["not_target", "target"],
            },
            "evaluation": {
                "dataset_id": evaluation_id,
                "sample_count": evaluation_count,
                "schema_name": "piernet.expert-evaluation",
            },
        }
        store.update_workflow(
            workflow_id,
            status="succeeded",
            current_step="complete",
            artifacts=artifacts,
            error=None,
            cancel_requested=False,
        )
        store.append_event(
            workflow_id,
            "generation_finished",
            {
                "message": "训练数据已经生成，可以进入训练。",
                "text2comp_dataset_id": text2comp_id,
                "router_dataset_id": router_id,
            },
        )
    except InterruptedError as exc:
        store.update_workflow(
            workflow_id,
            status="cancelled",
            error={"code": "cancelled", "message": str(exc)},
            cancel_requested=False,
        )
        store.append_event(workflow_id, "cancelled", {"message": str(exc)})
    except NewSynthError as exc:
        _fail_workflow(workflow_id, exc.code, exc.message)
    except Exception as exc:
        _fail_workflow(workflow_id, "generation_failed", f"训练数据生成失败：{exc}")
    finally:
        with _THREAD_LOCK:
            _THREADS.pop(workflow_id, None)


def _fail_workflow(workflow_id: str, code: str, message: str) -> None:
    store.update_workflow(
        workflow_id,
        status="failed",
        error={"code": code, "message": message},
        cancel_requested=False,
    )
    store.append_event(workflow_id, "failed", {"code": code, "message": message})


def cancel_workflow(owner_id: str, workflow_id: str) -> dict[str, Any]:
    workflow = store.get_workflow(owner_id, workflow_id)
    if workflow["status"] != "running":
        raise NewSynthError("not_running", "当前没有正在运行的任务。")
    store.update_workflow(workflow_id, cancel_requested=True)
    with _THREAD_LOCK:
        process = _PROCESSES.get(workflow_id)
    if process and process.poll() is None:
        process.terminate()
    store.append_event(workflow_id, "cancel_requested", {"message": "正在安全停止任务"})
    return get_workflow(owner_id, workflow_id)


def retry_generation(owner_id: str, workflow_id: str, config: dict[str, Any]) -> dict[str, Any]:
    workflow = store.get_workflow(owner_id, workflow_id)
    if workflow["status"] not in {"failed", "cancelled"}:
        raise NewSynthError("retry_not_available", "当前任务不需要重试。")
    if workflow["current_step"] == "source":
        raise NewSynthError("source_retry_required", "请重新选择或上传数据来源。")
    return start_generation(owner_id, workflow_id, config)


def list_registered_datasets(*, kind: str | None = None) -> list[dict[str, Any]]:
    return store.list_datasets(kind=kind)


def get_registered_dataset(dataset_id: str) -> dict[str, Any]:
    return store.get_dataset(dataset_id)
