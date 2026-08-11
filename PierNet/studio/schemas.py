from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StageStatus = Literal["waiting", "running", "succeeded", "failed", "cancelled"]
ProjectStatus = Literal["draft", "ready", "running", "failed", "cancelled"]


class SessionResponse(BaseModel):
    session_id: str


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    goal: str = Field(min_length=1, max_length=1000)


class MappingRequest(BaseModel):
    input_fields: list[str] = Field(min_length=1)
    output_fields: list[str] = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class StageSnapshot(BaseModel):
    id: str
    title: str
    status: StageStatus
    progress: float | None = Field(default=None, ge=0, le=1)
    message: str
    retryable: bool = False
    started_at: float | None = None
    finished_at: float | None = None


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    goal: str
    status: ProjectStatus
    current_stage: str
    created_at: float
    updated_at: float


class ProjectSnapshot(ProjectSummary):
    stages: list[StageSnapshot]
    data: dict[str, Any] | None = None
    expert: dict[str, Any] | None = None
    inspection: dict[str, Any] | None = None
    compatibility: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    recommended_prompt: str | None = None
    can_run: bool = False
    can_chat: bool = False


class CompatibilityResponse(BaseModel):
    compatible: bool
    report: dict[str, Any]


class RunResponse(BaseModel):
    project_id: str
    status: ProjectStatus
    message: str


class DeleteResponse(BaseModel):
    project_id: str
    deleted: bool
    message: str


class ChatResponse(BaseModel):
    chat_id: str
    project_id: str
    message: str
    answer: str
    routed: bool
    confidence: float
    inputs: list[float]
    output: Any
    chart: dict[str, Any]
    latency_ms: float
    created_at: float


class EventSnapshot(BaseModel):
    id: int
    event_type: str
    payload: dict[str, Any]
    created_at: float
