"""任务状态相关 Pydantic 模型。"""

from pydantic import BaseModel, Field
from typing import Any, Literal, Optional


JobStatus = Literal[
    "queued",
    "starting",
    "running",
    "evaluating",
    "stopping",
    "done",
    "error",
    "terminated",
    "external_terminated",
]


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    job_type: Optional[str] = None
    started_at: Optional[float] = None
    scenario_totals: dict[str, int] = Field(default_factory=dict)
    progress: dict[str, dict[str, Any]] = Field(default_factory=dict)
    stats: dict[str, float] = Field(default_factory=dict)
    finished_at: Optional[float] = None
    error_message: Optional[str] = None
    lock_keys: list[str] = Field(default_factory=list)


class TemplateFileInfo(BaseModel):
    scenario: str
    simulator: Optional[str] = None
    template_count: int
    file_size_bytes: int
    mtime: float
    path: str
