from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import unquote

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse

from PierNet.studio import service, store
from PierNet.studio.paths import project_paths
from PierNet.studio.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    MappingRequest,
    ProjectCreateRequest,
    ProjectSnapshot,
    ProjectSummary,
    RunResponse,
    SessionResponse,
)

router = APIRouter(prefix="/studio", tags=["studio"])
SESSION_COOKIE = "piern_studio_session"
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024


def _session_id(
    session_id: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> str:
    if not session_id or not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise HTTPException(status_code=401, detail="请先建立 Studio 会话")
    return session_id


def _studio_error(exc: service.StudioError) -> HTTPException:
    status = (
        409
        if exc.code
        in {
            "compatibility_required",
            "insufficient_storage",
            "mapping_required",
            "not_running",
            "project_model_mismatch",
            "project_not_ready",
            "project_cleanup_failed",
            "project_deleting",
            "project_running",
            "run_quota_reached",
        }
        else 400
    )
    return HTTPException(status_code=status, detail={"code": exc.code, "message": exc.message})


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


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(owner_id: str = Depends(_session_id)):
    return service.list_projects(owner_id)


@router.post("/projects", response_model=ProjectSnapshot)
def create_project(req: ProjectCreateRequest, owner_id: str = Depends(_session_id)):
    return service.create_project(owner_id, req.name, req.goal)


@router.get("/projects/{project_id}", response_model=ProjectSnapshot)
def get_project(project_id: str, owner_id: str = Depends(_session_id)):
    try:
        return service.get_project(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc


@router.delete("/projects/{project_id}", response_model=DeleteResponse)
def delete_project(project_id: str, owner_id: str = Depends(_session_id)):
    try:
        return service.delete_project(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


def _safe_upload_name(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or fallback


async def _save_upload(request: Request, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="上传文件不能超过 1GB")
    try:
        with target.open("wb") as output:
            async for chunk in request.stream():
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="上传文件不能超过 1GB")
                output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传文件为空")
    return written


@router.post("/projects/{project_id}/data", response_model=ProjectSnapshot)
async def upload_data(
    project_id: str,
    request: Request,
    encoded_filename: str | None = Header(default=None, alias="X-File-Name"),
    owner_id: str = Depends(_session_id),
):
    try:
        service.get_project(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    paths = project_paths(project_id)
    original_name = unquote(encoded_filename or "data.bin")
    filename = _safe_upload_name(original_name, "data.bin")
    target = paths.source / filename
    await _save_upload(request, target)
    try:
        return service.attach_data(owner_id, project_id, target, original_name)
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


@router.post("/projects/{project_id}/expert", response_model=ProjectSnapshot)
async def upload_expert(
    project_id: str,
    request: Request,
    encoded_filename: str | None = Header(default=None, alias="X-File-Name"),
    owner_id: str = Depends(_session_id),
):
    try:
        service.get_project(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    paths = project_paths(project_id)
    original_name = unquote(encoded_filename or "expert.bin")
    filename = _safe_upload_name(original_name, "expert.bin")
    target = paths.source / f"expert-{filename}"
    await _save_upload(request, target)
    try:
        return service.attach_expert(owner_id, project_id, target, original_name)
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


@router.post("/projects/{project_id}/mapping", response_model=ProjectSnapshot)
def apply_mapping(
    project_id: str,
    req: MappingRequest,
    owner_id: str = Depends(_session_id),
):
    try:
        return service.apply_mapping(owner_id, project_id, req.input_fields, req.output_fields)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


@router.post(
    "/projects/{project_id}/inspect",
    response_model=ProjectSnapshot,
)
def inspect_project(project_id: str, owner_id: str = Depends(_session_id)):
    try:
        return service.inspect_resources(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


@router.post(
    "/projects/{project_id}/compatibility-check",
    response_model=ProjectSnapshot,
)
def check_compatibility(project_id: str, owner_id: str = Depends(_session_id)):
    try:
        return service.inspect_and_check(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


@router.post("/projects/{project_id}/run", response_model=RunResponse)
def run_project(project_id: str, owner_id: str = Depends(_session_id)):
    try:
        return service.start_run(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


@router.post("/projects/{project_id}/retry", response_model=RunResponse)
def retry_project(project_id: str, owner_id: str = Depends(_session_id)):
    try:
        return service.start_run(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


@router.post("/projects/{project_id}/cancel", response_model=RunResponse)
def cancel_project(project_id: str, owner_id: str = Depends(_session_id)):
    try:
        return service.cancel_run(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


@router.post("/projects/{project_id}/chat", response_model=ChatResponse)
def chat(
    project_id: str,
    req: ChatRequest,
    owner_id: str = Depends(_session_id),
):
    try:
        return service.chat(owner_id, project_id, req.message)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    except service.StudioError as exc:
        raise _studio_error(exc) from exc


async def _event_stream(
    request: Request,
    owner_id: str,
    project_id: str,
    after_id: int,
) -> AsyncIterator[str]:
    last_id = after_id
    last_keepalive = time.monotonic()
    while not await request.is_disconnected():
        try:
            store.get_project(owner_id, project_id)
        except KeyError:
            yield f"event: error\ndata: {json.dumps({'message': '项目不存在'}, ensure_ascii=False)}\n\n"
            return
        events = store.list_events(project_id, after_id=last_id)
        for event in events:
            last_id = int(event["id"])
            yield (f"id: {last_id}\nevent: {event['event_type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n")
        if time.monotonic() - last_keepalive > 15:
            yield ": keepalive\n\n"
            last_keepalive = time.monotonic()
        await asyncio.sleep(0.8)


@router.get("/projects/{project_id}/events")
def project_events(
    request: Request,
    project_id: str,
    after_id: int = 0,
    owner_id: str = Depends(_session_id),
):
    try:
        store.get_project(owner_id, project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    return StreamingResponse(
        _event_stream(request, owner_id, project_id, after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health")
def studio_health():
    service.initialize()
    return {"status": "ok", "service": "piern-studio"}
