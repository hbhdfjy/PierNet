"""Unified cross-platform task and audit API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from piern.shared.audit import store as audit_store
from piern.shared.tasks.state import normalize_status
from piern.synth.services import job_manager as synth_jobs
from piern.synth.services import job_store as synth_job_store
from piern.training.services import job_store as training_job_store
from piern.training.services import training_manager

router = APIRouter(prefix="/jobs", tags=["jobs"])


class UnifiedJobSummary(BaseModel):
    job_id: str
    platform: str
    job_type: str
    status: str
    name: str | None = None
    created_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    source: str


class UnifiedJobDetail(UnifiedJobSummary):
    request: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)


class UnifiedJobEventResponse(BaseModel):
    job_id: str
    platform: str
    events: list[dict[str, Any]]


class UnifiedJobLogResponse(BaseModel):
    job_id: str
    platform: str
    lines: list[str]


def _synth_summary(stored: dict[str, Any]) -> UnifiedJobSummary:
    return UnifiedJobSummary(
        job_id=str(stored["job_id"]),
        platform="synth",
        job_type=str(stored.get("job_type") or "job"),
        status=normalize_status(stored.get("status")),
        name=str(stored.get("job_id")),
        created_at=stored.get("created_at"),
        started_at=stored.get("started_at"),
        finished_at=stored.get("finished_at"),
        progress=stored.get("progress") or {},
        stats=stored.get("stats") or {},
        error_message=stored.get("error_message"),
        source="synth_jobs",
    )


def _training_summary(entry: dict[str, Any]) -> UnifiedJobSummary:
    config = entry.get("config") if isinstance(entry.get("config"), dict) else {}
    progress: dict[str, Any] = {}
    latest_step = entry.get("latest_step")
    steps_per_epoch = entry.get("steps_per_epoch")
    if latest_step is not None or steps_per_epoch is not None:
        progress["training"] = {
            "scenario": "training",
            "done": int(latest_step or 0),
            "total": int(steps_per_epoch or latest_step or 0),
        }
    return UnifiedJobSummary(
        job_id=str(entry["job_id"]),
        platform="training",
        job_type="training",
        status=normalize_status(entry.get("status")),
        name=entry.get("name"),
        created_at=entry.get("created_at"),
        started_at=entry.get("started_at"),
        finished_at=entry.get("ended_at"),
        progress=progress,
        stats={
            "steps_per_sec": entry.get("steps_per_sec"),
            "eta_seconds": entry.get("eta_seconds"),
            "simulator": entry.get("simulator"),
            "scenarios": entry.get("scenarios") or [],
            "epochs": config.get("epochs"),
        },
        error_message=entry.get("error_message"),
        source="training_jobs",
    )


def _find_synth(job_id: str) -> dict[str, Any] | None:
    return synth_job_store.load_job(job_id)


def _find_training(job_id: str) -> dict[str, Any] | None:
    try:
        return training_manager.get_job(job_id, refresh=True)
    except KeyError:
        return None


def _resolve_job(job_id: str) -> tuple[str, dict[str, Any]]:
    training = _find_training(job_id)
    if training is not None:
        return "training", training
    synth = _find_synth(job_id)
    if synth is not None:
        return "synth", synth
    raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("", response_model=list[UnifiedJobSummary])
def list_unified_jobs(
    platform: str | None = Query(None, pattern="^(synth|training)$"),
    status: str | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
):
    items: list[UnifiedJobSummary] = []
    if platform in (None, "training"):
        for entry in training_manager.list_jobs(refresh=True)[:limit]:
            summary = _training_summary(entry)
            if status is None or summary.status == normalize_status(status):
                items.append(summary)
    if platform in (None, "synth"):
        for stored in synth_job_store.list_jobs(limit=limit):
            summary = _synth_summary(stored)
            if status is None or summary.status == normalize_status(status):
                items.append(summary)
    items.sort(key=lambda item: item.started_at or item.created_at or 0, reverse=True)
    return items[:limit]


@router.get("/audit/events")
def list_audit_events(limit: int = Query(200, ge=1, le=1000), action: str | None = None, target: str | None = None):
    return {"items": audit_store.list_events(limit=limit, action=action, target=target)}


@router.get("/{job_id}", response_model=UnifiedJobDetail)
def get_unified_job(job_id: str, log_limit: int = Query(300, ge=0, le=5000)):
    platform, payload = _resolve_job(job_id)
    if platform == "training":
        summary = _training_summary(payload)
        try:
            events = training_job_store.list_events(job_id)
        except Exception:
            events = []
        logs = training_manager.get_job_logs(job_id, limit=log_limit) if log_limit else []
        return UnifiedJobDetail(**summary.model_dump(), request=payload.get("config") or {}, events=events, logs=logs)
    summary = _synth_summary(payload)
    events = payload.get("events") or []
    logs = [str(event.get("line")) for event in events if isinstance(event, dict) and event.get("type") == "log"]
    return UnifiedJobDetail(
        **summary.model_dump(),
        request=payload.get("request_json") or {},
        events=events,
        logs=logs[-log_limit:] if log_limit else [],
    )


@router.get("/{job_id}/events", response_model=UnifiedJobEventResponse)
def get_unified_job_events(job_id: str, limit: int = Query(1000, ge=1, le=10000)):
    platform, payload = _resolve_job(job_id)
    if platform == "training":
        events = training_job_store.list_events(job_id)[-limit:]
    else:
        events = (payload.get("events") or [])[-limit:]
    return UnifiedJobEventResponse(job_id=job_id, platform=platform, events=events)


@router.get("/{job_id}/logs", response_model=UnifiedJobLogResponse)
def get_unified_job_logs(job_id: str, limit: int = Query(300, ge=20, le=5000)):
    platform, payload = _resolve_job(job_id)
    if platform == "training":
        lines = training_manager.get_job_logs(job_id, limit=limit)
    else:
        events = payload.get("events") or []
        lines = [str(event.get("line")) for event in events if isinstance(event, dict) and event.get("type") == "log"]
        lines = lines[-limit:]
    return UnifiedJobLogResponse(job_id=job_id, platform=platform, lines=lines)


@router.post("/{job_id}/stop", response_model=UnifiedJobSummary)
def stop_unified_job(job_id: str):
    platform, _ = _resolve_job(job_id)
    if platform == "training":
        try:
            return _training_summary(training_manager.stop_job(job_id))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not synth_jobs.terminate_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    stored = synth_job_store.load_job(job_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _synth_summary(stored)


@router.delete("/{job_id}", status_code=204)
def delete_unified_job(job_id: str):
    platform, _ = _resolve_job(job_id)
    if platform == "training":
        try:
            training_manager.delete_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=204)
    raise HTTPException(status_code=409, detail="合成任务历史暂不支持通过统一接口删除。")
