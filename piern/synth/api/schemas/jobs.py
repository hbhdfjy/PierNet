"""任务状态相关 Pydantic 模型。"""

from pydantic import BaseModel, Field
from typing import Any, Optional


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    job_type: Optional[str] = None
    started_at: Optional[float] = None
    scenario_totals: dict[str, int] = Field(default_factory=dict)
    progress: dict[str, dict[str, Any]] = Field(default_factory=dict)
    stats: dict[str, float] = Field(default_factory=dict)


class TemplateFileInfo(BaseModel):
    scenario: str
    template_count: int
    file_size_bytes: int
    mtime: float
    path: str


class SampleFileInfo(BaseModel):
    scenario: str
    sample_count: int
    file_size_bytes: int
    mtime: float
    path: str
