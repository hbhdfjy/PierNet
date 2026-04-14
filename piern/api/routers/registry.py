"""Registry CRUD 路由：/api/registry, /api/register。"""

import os
import re
import sys
import subprocess
import threading
import time

import yaml
from fastapi import APIRouter, HTTPException

from piern.api.deps import REGISTRY_PATH, PROJECT_ROOT
from piern.api.schemas.registry import RegisterRequest
from piern.api.services import job_manager
from piern.api.services.job_manager import publish

router = APIRouter()


def _load_registry_raw() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_registry_raw(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)


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
            # simulator 级：合并 scenarios 子字段（不丢失已有场景描述）
            existing_scenarios = data.get(key, {}).get("scenarios", {}) if isinstance(data.get(key), dict) else {}
            data[key] = body
            if existing_scenarios and isinstance(body, dict):
                # body 中的 scenarios 优先，但保留 body 中没有的场景描述
                merged = {**existing_scenarios, **(body.get("scenarios") or {})}
                data[key]["scenarios"] = merged
        _save_registry_raw(data)
        return {"ok": True, "key": key}
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
    """启动 auto_register 子进程，返回 job_id（复用 SSE 流机制）。"""
    record = job_manager.create_job("register", {})
    job_id = record.job_id

    cmd = [
        sys.executable, "-m", "piern.text2comp.auto_register",
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
        )
    except FileNotFoundError as e:
        raise HTTPException(500, f"启动子进程失败: {e}")

    def _reader():
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                event: dict = {"type": "log", "line": line, "ts": time.time()}

                m = re.search(r"\[注册\]\s+(\S+)\s+→\s+字段组:\s+(.+)", line)
                if m:
                    event["register_progress"] = {"key": m.group(1), "fields": m.group(2)}
                if "已保存" in line:
                    event["saved"] = True

                publish(record, event)
        except Exception as e:
            publish(record, {"type": "error", "message": str(e), "ts": time.time()})
        finally:
            proc.wait()
            rc = proc.returncode
            final = {
                "type": "done" if rc == 0 else "error",
                "return_code": rc,
                "ts": time.time(),
                "message": "注册完成" if rc == 0 else f"进程退出码: {rc}",
            }
            publish(record, final)
            record.status = "done" if rc == 0 else "error"

    threading.Thread(target=_reader, daemon=True).start()
    return {"job_id": job_id, "status": "running"}
