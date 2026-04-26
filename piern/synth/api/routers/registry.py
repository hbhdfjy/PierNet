"""Registry CRUD 路由：/api/registry, /api/register。"""

import os
import re
import sys
import subprocess
import threading
import time

import yaml
from fastapi import APIRouter, HTTPException

from piern.shared.runtime.paths import REGISTRY_PATH, PROJECT_ROOT
from piern.synth.api.schemas.registry import RegisterRequest
from piern.synth.services import job_manager
from piern.synth.services.job_manager import publish

router = APIRouter()

_REGISTER_PROGRESS_RE = re.compile("^\[注册\]\s+(\S+)\s+→\s+字段组:\s+(.+)$")
_SAVE_MARKERS = ("已保存",)


def _load_registry_raw() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_registry_raw(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)


def _validate_registry_entry(key: str, body: dict) -> None:
    if not isinstance(body, dict):
        raise HTTPException(400, "registry 条目必须是 JSON 对象")

    output_info = body.get("output_info")
    if output_info is not None:
        if not isinstance(output_info, list) or len(output_info) == 0:
            raise HTTPException(400, "至少需要保留 1 个 output_info 输出定义")

    obs = body.get("observation_config")
    if obs is None:
        return
    if not isinstance(obs, dict):
        raise HTTPException(400, "observation_config 必须是 JSON 对象")

    channel_level = str(obs.get("channel_level", "row") or "row").lower()
    if channel_level not in {"row", "output", "output_info"}:
        raise HTTPException(400, "channel_level 必须是 row、output 或 output_info")
    is_output_level = channel_level in {"output", "output_info"}
    obs["channel_level"] = "output_info" if is_output_level else "row"

    fixed_channels = obs.get("fixed_channels", None)
    if fixed_channels is None:
        return
    if not isinstance(fixed_channels, list):
        raise HTTPException(400, "fixed_channels 必须是 null 或通道索引列表")
    if len(fixed_channels) == 0:
        raise HTTPException(
            400,
            "至少选择 1 个输出通道；如需全选，请将 fixed_channels 设为 null",
        )
    if is_output_level and not isinstance(output_info, list):
        raise HTTPException(400, "按输出维度采样需要先定义 output_info")

    output_names = set()
    if isinstance(output_info, list):
        output_names = {
            str(item.get("name", "")).strip()
            for item in output_info
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }

    invalid = []
    invalid_output = []
    for value in fixed_channels:
        if isinstance(value, bool):
            invalid.append(value)
        elif isinstance(value, int):
            if value < 0:
                invalid.append(value)
            elif is_output_level and isinstance(output_info, list) and value >= len(output_info):
                invalid_output.append(value)
        elif isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                invalid.append(value)
            elif is_output_level:
                if cleaned in output_names:
                    continue
                try:
                    idx = int(cleaned)
                except ValueError:
                    invalid_output.append(value)
                else:
                    if idx < 0 or idx >= len(output_info):
                        invalid_output.append(value)
        else:
            invalid.append(value)
    if invalid:
        raise HTTPException(
            400,
            f"fixed_channels 包含无效通道值: {invalid}",
        )
    if invalid_output:
        raise HTTPException(
            400,
            f"output_info 通道选择超出范围或不存在: {invalid_output}",
        )


@router.get("/registry")
def get_registry():
    """读取 registry.yaml，返回完整内容。"""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(500, f"读取 registry 失败: {e}")


@router.put("/registry/{key:path}")
def update_registry_entry(key: str, body: dict):
    """更新（或新建）registry 中的记录。

    key = "simulator"              → 更新 simulator 级字段（body 为完整条目）
    key = "simulator/scenario"     → 更新 simulator.scenarios.scenario（body 为字符串或 {"scenario_description": "..."}）
    """
    try:
        data = _load_registry_raw()
        if "/" in key:
            simulator, scenario = key.split("/", 1)
            sim_entry = data.setdefault(simulator, {})
            scenarios = sim_entry.setdefault("scenarios", {})
            # body 可以是 {"scenario_description": "..."} 或直接字符串
            desc = body.get("scenario_description", "") if isinstance(body, dict) else str(body)
            scenarios[scenario] = desc
        else:
            _validate_registry_entry(key, body)
            # simulator 级：合并 scenarios 子字段（不丢失已有场景描述）
            existing_scenarios = data.get(key, {}).get("scenarios", {}) if isinstance(data.get(key), dict) else {}
            data[key] = body
            if existing_scenarios and isinstance(body, dict):
                # body 中的 scenarios 优先，但保留 body 中没有的场景描述
                merged = {**existing_scenarios, **(body.get("scenarios") or {})}
                data[key]["scenarios"] = merged
        _save_registry_raw(data)
        return {"ok": True, "key": key}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"保存 registry 失败: {e}")


@router.delete("/registry/{key:path}")
def delete_registry_entry(key: str):
    """删除 registry 中的记录。

    key = "simulator"          → 删除整个 simulator 条目
    key = "simulator/scenario" → 只删除 simulator.scenarios.scenario
    """
    try:
        data = _load_registry_raw()
        if "/" in key:
            simulator, scenario = key.split("/", 1)
            sim_entry = data.get(simulator, {})
            scenarios = sim_entry.get("scenarios", {})
            if scenario not in scenarios:
                raise HTTPException(404, f"场景 '{key}' 不存在")
            del scenarios[scenario]
            if not scenarios:
                sim_entry.pop("scenarios", None)
        else:
            if key not in data:
                raise HTTPException(404, f"key '{key}' 不存在")
            del data[key]
        _save_registry_raw(data)
        return {"ok": True, "key": key}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"删除 registry 失败: {e}")


@router.post("/register")
async def start_register(req: RegisterRequest):
    """启动 auto_register 后台任务，返回 job_id 供 SSE 订阅。"""
    record = job_manager.create_job("register", {})
    job_id = record.job_id

    cmd = [
        sys.executable, "-m", "piern.synth.text2comp.auto_register",
        "--config", req.config,
        "--output", req.output,
    ]
    if req.scenarios:
        cmd += ["--scenarios"] + req.scenarios
    if req.fields:
        cmd += ["--fields"] + req.fields
    if req.overwrite:
        cmd.append("--overwrite")
    if req.simulator_level:
        cmd.append("--simulator-level")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            start_new_session=True,
        )
        record.proc = proc
        record.proc_uses_process_group = True
    except FileNotFoundError as e:
        raise HTTPException(500, f"启动注册任务失败: {e}")

    def _reader():
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                event: dict = {"type": "log", "line": line, "ts": time.time()}

                m = _REGISTER_PROGRESS_RE.search(line)
                if m:
                    event["register_progress"] = {"key": m.group(1), "fields": m.group(2)}
                if any(marker in line for marker in _SAVE_MARKERS):
                    event["saved"] = True

                publish(record, event)
        except Exception as e:
            if not record.stop_event.is_set():
                publish(record, {"type": "error", "message": str(e), "ts": time.time()})
        finally:
            try:
                proc.wait()
            finally:
                record.proc = None
                record.proc_uses_process_group = False

            if record.stop_event.is_set():
                return

            rc = proc.returncode
            final = {
                "type": "done" if rc == 0 else "error",
                "return_code": rc,
                "ts": time.time(),
                "message": "注册完成" if rc == 0 else f"注册失败，退出码: {rc}",
            }
            publish(record, final)
            record.status = "done" if rc == 0 else "error"

    threading.Thread(target=_reader, daemon=True).start()
    return {"job_id": job_id, "status": "running"}
