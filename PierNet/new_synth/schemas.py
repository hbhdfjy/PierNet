from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

WorkflowStatus = Literal["draft", "running", "succeeded", "failed", "cancelled"]
WorkflowStep = Literal["source", "definition", "generation", "complete"]


class SessionResponse(BaseModel):
    session_id: str


class WorkflowCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class BuiltinSourceRequest(BaseModel):
    simulator: str = Field(min_length=1, max_length=128)
    scenario: str = Field(min_length=1, max_length=128)
    n_samples: int = Field(default=32, ge=4, le=100_000)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    reuse_existing: bool = True


class ExpertGenerateRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=180)
    scenario: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=20_000)
    input_dim: int | None = Field(default=None, ge=1, le=1_000_000)


class ParameterDefinition(BaseModel):
    index: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=128)
    display_name: str = Field(default="", max_length=180)
    description: str = Field(default="", max_length=1000)
    unit: str = Field(default="", max_length=64)


class OutputDefinition(BaseModel):
    index: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=128)
    display_name: str = Field(default="", max_length=180)
    description: str = Field(default="", max_length=1000)
    unit: str = Field(default="", max_length=64)


class SamplingDefinition(BaseModel):
    channels: list[int] | None = None
    time_stride: int = Field(default=1, ge=1, le=1_000_000)
    max_time_points: int | None = Field(default=None, ge=1, le=1_000_000)


class DefinitionRequest(BaseModel):
    simulator: str = Field(min_length=1, max_length=128)
    scenario: str = Field(min_length=1, max_length=128)
    task_description: str = Field(min_length=1, max_length=2000)
    parameters: list[ParameterDefinition] = Field(min_length=1)
    outputs: list[OutputDefinition] = Field(min_length=1)
    sampling: SamplingDefinition = Field(default_factory=SamplingDefinition)

    @field_validator("simulator", "scenario")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in cleaned
        ):
            raise ValueError("只能使用字母、数字、下划线或短横线")
        return cleaned


class GenerateRequest(BaseModel):
    max_samples: int | None = Field(default=None, ge=4, le=5_000_000)
    variants_per_sample: int = Field(default=2, ge=1, le=8)
    negative_ratio: int = Field(default=1, ge=1, le=10)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class WorkflowSummary(BaseModel):
    workflow_id: str
    name: str
    status: WorkflowStatus
    current_step: WorkflowStep
    created_at: float
    updated_at: float


class WorkflowSnapshot(WorkflowSummary):
    source: dict[str, Any] | None = None
    definition: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel_requested: bool = False
    can_define: bool = False
    can_generate: bool = False
    can_open_training: bool = False


class RunResponse(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    message: str


class DatasetSnapshot(BaseModel):
    dataset_id: str
    workflow_id: str
    kind: Literal["text2comp", "router", "evaluation"]
    name: str
    simulator: str
    scenario: str
    schema_name: str
    schema_version: int
    sample_count: int
    size_bytes: int
    path: str
    paired_dataset_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float


class EventSnapshot(BaseModel):
    id: int
    event_type: str
    payload: dict[str, Any]
    created_at: float
