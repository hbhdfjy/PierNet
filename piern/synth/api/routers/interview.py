"""多智能体交互式注册路由：/api/interview/*。"""

import asyncio
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from piern.shared.runtime.paths import CONFIG_DIR, PROJECT_ROOT, REGISTRY_PATH
from piern.synth.text2comp.interview_agent import (
    create_session as _iv_create,
    process_user_message as _iv_message,
    process_confirm as _iv_confirm,
    get_session as _iv_get,
    delete_session as _iv_delete,
)

router = APIRouter()


class InterviewStartRequest(BaseModel):
    simulator: str
    scenario: str
    hdf5_path: Optional[str] = None
    mode: str = "simulator"   # "simulator" | "scenario"


class InterviewMessageRequest(BaseModel):
    message: str


class InterviewConfirmRequest(BaseModel):
    confirmed: bool
    edited_data: Optional[dict] = None


def _load_llm_cfg() -> dict:
    """从 default.yaml → generation.yaml 读取 LLM 配置。"""
    default_yaml = CONFIG_DIR / "default.yaml"
    try:
        with open(default_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        gen_cfg_path = cfg.get("generation_config")
        if gen_cfg_path:
            gen_file = PROJECT_ROOT / gen_cfg_path
            if gen_file.exists():
                with open(gen_file, "r", encoding="utf-8") as f:
                    base = yaml.safe_load(f) or {}
                cfg = {**base, **cfg}
        return cfg.get("llm", {})
    except Exception:
        return {}


@router.post("/interview/start")
async def start_interview(req: InterviewStartRequest):
    """创建新的面试会话，返回第一个问题。"""
    if not req.simulator or not req.scenario:
        raise HTTPException(400, "simulator 和 scenario 不能为空")

    llm_cfg = _load_llm_cfg()
    hdf5_path: Optional[str] = None
    if req.hdf5_path:
        p = PROJECT_ROOT / req.hdf5_path
        hdf5_path = str(p) if p.exists() else None

    loop = asyncio.get_event_loop()
    try:
        session_id, resp = await loop.run_in_executor(
            None,
            lambda: _iv_create(
                simulator=req.simulator,
                scenario=req.scenario,
                hdf5_path=hdf5_path,
                llm_cfg=llm_cfg,
                registry_path=REGISTRY_PATH,
                mode=req.mode,
            )
        )
    except Exception as e:
        raise HTTPException(500, f"会话创建失败: {e}")

    return {"session_id": session_id, **resp.to_dict()}


@router.post("/interview/{session_id}/message")
async def send_interview_message(session_id: str, req: InterviewMessageRequest):
    """发送用户消息，获取 Agent 回复。"""
    session = _iv_get(session_id)
    if not session:
        raise HTTPException(404, f"会话 {session_id} 不存在")
    if session.status == "confirming":
        raise HTTPException(409, "当前步骤等待确认，请调用 /confirm 端点")
    if session.status == "done":
        raise HTTPException(409, "会话已完成")

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None, lambda: _iv_message(session_id, req.message)
        )
    except Exception as e:
        raise HTTPException(500, f"消息处理失败: {e}")

    return resp.to_dict()


@router.get("/interview/{session_id}/state")
def get_interview_state(session_id: str):
    """获取会话完整状态快照。"""
    session = _iv_get(session_id)
    if not session:
        raise HTTPException(404, f"会话 {session_id} 不存在")

    return {
        "session_id": session.session_id,
        "simulator": session.simulator,
        "scenario": session.scenario,
        "step": session.step,
        "status": session.status,
        "history": session.history,
        "collected_data": {
            "domain_context": session.domain_context,
            "output_description": session.output_description,
            "param_info": session.param_info,
            "output_info": session.output_info,
            "observation_config": session.observation_config,
        },
        "pending_extraction": session.pending_extraction or None,
        "hdf5_loaded": session.timeseries_shape is not None,
        "timeseries_shape": list(session.timeseries_shape) if session.timeseries_shape else None,
        "github_url": session.github_url,
        "prefilled_steps": list(session.prefilled_steps),
    }


@router.post("/interview/{session_id}/confirm")
async def confirm_interview_step(session_id: str, req: InterviewConfirmRequest):
    """确认或拒绝当前步骤的提取结果。"""
    session = _iv_get(session_id)
    if not session:
        raise HTTPException(404, f"会话 {session_id} 不存在")
    if session.status != "confirming":
        raise HTTPException(409, "当前不在确认状态，请先发送消息")

    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None, lambda: _iv_confirm(session_id, req.confirmed, req.edited_data)
        )
    except Exception as e:
        raise HTTPException(500, f"确认处理失败: {e}")

    return resp.to_dict()


@router.delete("/interview/{session_id}")
def cancel_interview(session_id: str):
    """取消并删除会话。"""
    if not _iv_delete(session_id):
        raise HTTPException(404, f"会话 {session_id} 不存在")
    return {"cancelled": True, "session_id": session_id}
