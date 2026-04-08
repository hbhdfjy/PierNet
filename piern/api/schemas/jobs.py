"""任务状态相关 Pydantic 模型。"""

from pydantic import BaseModel
from typing import Optional


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    job_type: Optional[str] = None
    started_at: Optional[float] = None
    scenario_totals: dict[str, int] = {}


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
