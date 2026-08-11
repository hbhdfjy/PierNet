from __future__ import annotations

import asyncio
import json
import re
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from PierNet.new_synth import service, store
from PierNet.new_synth.paths import workflow_paths
from PierNet.new_synth.schemas import (
    BuiltinSourceRequest,
    DefinitionRequest,
    ExpertGenerateRequest,
    GenerateRequest,
    RunResponse,
    SessionResponse,
    WorkflowCreateRequest,
    WorkflowSnapshot,
    WorkflowSummary,
)

router = APIRouter(prefix="/new-synth", tags=["new-synth"])
SESSION_COOKIE = "piern_new_synth_session"
MAX_UPLOAD_BYTES = 1024**3


def _session_id(session_id: str | None = Cookie(default=None, alias=SESSION_COOKIE)) -> str:
    if not session_id or not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise HTTPException(status_code=401, detail="请先建立新数据合成会话")
    return session_id


def _api_error(exc: service.NewSynthError) -> HTTPException:
    conflict_codes = {
        "insufficient_storage",
        "not_running",
        "retry_not_available",
        "simulation_data_missing",
        "source_busy",
        "workflow_running",
    }
    if exc.code == "llm_unavailable":
        status_code = 502
    elif exc.code in conflict_codes:
        status_code = 409
    else:
        status_code = 400
    return HTTPException(status_code=status_code, detail={"code": exc.code, "message": exc.message})


def _safe_name(value: str | None, fallback: str) -> str:
    name = Path(value or fallback).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or fallback


async def _save_upload(request: Request, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    length = request.headers.get("content-length")
    if length and int(length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="上传文件不能超过 1GB")
    written = 0
    try:
        with target.open("wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="上传文件不能超过 1GB")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传文件为空")
    return written


@router.post("/session", response_model=SessionResponse)
def create_session(
    request: Request,
    response: Response,
    session_id: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    value = session_id if session_id and re.fullmatch(r"[a-f0-9]{32}", session_id) else uuid.uuid4().hex
    response.set_cookie(
        SESSION_COOKIE,
        value,
        max_age=30 * 24 * 3600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )
    return SessionResponse(session_id=value)


@router.get("/presets")
def get_presets():
    return service.presets()


@router.get("/workflows", response_model=list[WorkflowSummary])
def list_workflows(owner_id: str = Depends(_session_id)):
    return service.list_workflows(owner_id)


@router.post("/workflows", response_model=WorkflowSnapshot)
def create_workflow(req: WorkflowCreateRequest, owner_id: str = Depends(_session_id)):
    return service.create_workflow(owner_id, req.name)


@router.get("/workflows/{workflow_id}", response_model=WorkflowSnapshot)
def get_workflow(workflow_id: str, owner_id: str = Depends(_session_id)):
    try:
        return service.get_workflow(owner_id, workflow_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc


@router.post("/workflows/{workflow_id}/source/upload", response_model=WorkflowSnapshot)
async def upload_source(
    workflow_id: str,
    request: Request,
    encoded_filename: str | None = Header(default=None, alias="X-File-Name"),
    owner_id: str = Depends(_session_id),
):
    try:
        service.get_workflow(owner_id, workflow_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc
    original_name = unquote(encoded_filename or "data.h5")
    target = workflow_paths(workflow_id).source / _safe_name(original_name, "data.h5")
    await _save_upload(request, target)
    try:
        return service.attach_uploaded_hdf5(owner_id, workflow_id, target, original_name)
    except service.NewSynthError as exc:
        raise _api_error(exc) from exc


@router.post("/workflows/{workflow_id}/source/simulation", response_model=WorkflowSnapshot)
def select_simulation_source(
    workflow_id: str,
    req: BuiltinSourceRequest,
    owner_id: str = Depends(_session_id),
):
    try:
        return service.start_simulation_source(
            owner_id,
            workflow_id,
            simulator_name=req.simulator,
            scenario_name=req.scenario,
            n_samples=req.n_samples,
            seed=req.seed,
            reuse_existing=req.reuse_existing,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc
    except service.NewSynthError as exc:
        raise _api_error(exc) from exc


@router.post("/experts/upload")
async def upload_expert(
    request: Request,
    name: str = Query(..., min_length=1, max_length=180),
    encoded_filename: str | None = Header(default=None, alias="X-File-Name"),
    _owner_id: str = Depends(_session_id),
):
    with tempfile.TemporaryDirectory(prefix="piern-new-synth-expert-") as directory:
        temporary = Path(directory) / _safe_name(unquote(encoded_filename or name), "expert.bin")
        try:
            await _save_upload(request, temporary)
            content = temporary.read_bytes()
            return {"ok": True, "model": service.upload_expert_model(name, content)}
        except service.NewSynthError as exc:
            raise _api_error(exc) from exc


@router.post("/workflows/{workflow_id}/source/expert", response_model=WorkflowSnapshot)
def generate_expert_source(
    workflow_id: str,
    req: ExpertGenerateRequest,
    owner_id: str = Depends(_session_id),
):
    try:
        return service.start_expert_source(
            owner_id,
            workflow_id,
            model_id=req.model_id,
            scenario=req.scenario,
            prompt=req.prompt,
            input_dim=req.input_dim,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务或专家模型不存在") from exc
    except service.NewSynthError as exc:
        raise _api_error(exc) from exc


@router.put("/workflows/{workflow_id}/definition", response_model=WorkflowSnapshot)
def save_definition(
    workflow_id: str,
    req: DefinitionRequest,
    owner_id: str = Depends(_session_id),
):
    try:
        return service.save_definition(owner_id, workflow_id, req.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc
    except service.NewSynthError as exc:
        raise _api_error(exc) from exc


@router.post("/workflows/{workflow_id}/definition/suggest", response_model=DefinitionRequest)
def suggest_definition(
    workflow_id: str,
    req: DefinitionRequest,
    owner_id: str = Depends(_session_id),
):
    try:
        return service.suggest_definition(owner_id, workflow_id, req.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc
    except service.NewSynthError as exc:
        raise _api_error(exc) from exc


@router.post("/workflows/{workflow_id}/generate", response_model=RunResponse)
def start_generation(
    workflow_id: str,
    req: GenerateRequest,
    owner_id: str = Depends(_session_id),
):
    try:
        snapshot = service.start_generation(owner_id, workflow_id, req.model_dump())
        return RunResponse(
            workflow_id=workflow_id,
            status=snapshot["status"],
            message="训练数据生成已经开始",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc
    except service.NewSynthError as exc:
        raise _api_error(exc) from exc


@router.post("/workflows/{workflow_id}/retry", response_model=RunResponse)
def retry_generation(
    workflow_id: str,
    req: GenerateRequest,
    owner_id: str = Depends(_session_id),
):
    try:
        snapshot = service.retry_generation(owner_id, workflow_id, req.model_dump())
        return RunResponse(
            workflow_id=workflow_id,
            status=snapshot["status"],
            message="正在重新生成训练数据",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc
    except service.NewSynthError as exc:
        raise _api_error(exc) from exc


@router.post("/workflows/{workflow_id}/cancel", response_model=WorkflowSnapshot)
def cancel_workflow(workflow_id: str, owner_id: str = Depends(_session_id)):
    try:
        return service.cancel_workflow(owner_id, workflow_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc
    except service.NewSynthError as exc:
        raise _api_error(exc) from exc


@router.get("/workflows/{workflow_id}/datasets")
def workflow_datasets(workflow_id: str, owner_id: str = Depends(_session_id)):
    try:
        service.get_workflow(owner_id, workflow_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc
    result = []
    for item in store.list_datasets(workflow_id=workflow_id, owner_id=owner_id):
        public = dict(item)
        public.pop("owner_id", None)
        public.pop("root_path", None)
        path = Path(str(public.get("path") or ""))
        try:
            public["path"] = str(path.relative_to(Path.cwd()))
        except ValueError:
            public["path"] = path.name
        result.append(public)
    return result


async def _event_stream(
    request: Request,
    owner_id: str,
    workflow_id: str,
    after_id: int,
) -> AsyncIterator[str]:
    last_id = after_id
    while not await request.is_disconnected():
        try:
            store.get_workflow(owner_id, workflow_id)
        except KeyError:
            yield f"event: error\ndata: {json.dumps({'message': '数据合成任务不存在'}, ensure_ascii=False)}\n\n"
            return
        events = store.list_events(workflow_id, after_id=last_id)
        for event in events:
            last_id = int(event["id"])
            yield (f"id: {last_id}\nevent: {event['event_type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n")
        if not events:
            yield ": keepalive\n\n"
        await asyncio.sleep(0.8)


@router.get("/workflows/{workflow_id}/events")
def workflow_events(
    request: Request,
    workflow_id: str,
    after_id: int = 0,
    owner_id: str = Depends(_session_id),
):
    try:
        store.get_workflow(owner_id, workflow_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="数据合成任务不存在") from exc
    return StreamingResponse(
        _event_stream(request, owner_id, workflow_id, after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health")
def health():
    return {"ok": True, "module": "new-synth"}
