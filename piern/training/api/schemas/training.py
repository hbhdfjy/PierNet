from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TrainingJobStatus = Literal["queued", "starting", "running", "evaluating", "done", "error", "terminated"]
TrainingJobInputRepresentation = Literal["auto", "char", "embedding", "char_tokens", "pretrained_embeddings"]
TrainingJobRequestedInputRepresentation = Literal["auto", "char", "embedding"]


class TrainingDatasetScenario(BaseModel):
    scenario: str
    simulator: str
    router_count: int
    file_size_bytes: int
    mtime: float
    path: str


class TrainingDatasetInfo(BaseModel):
    simulator: str
    total_count: int
    scenarios: list[TrainingDatasetScenario]


class GPUInfo(BaseModel):
    index: int
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_gpu: int
    available: bool
    locked_by_job_id: str | None = None
    reason: str | None = None


class TrainingJobConfig(BaseModel):
    epochs: int = 0
    eval_interval: int = 1
    batch_size: int = 256
    test_batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    num_workers: int = 8
    test_ratio: float = 0.10
    resume_from: str | None = None
    input_representation: TrainingJobInputRepresentation = "auto"


class TrainingJobCreateRequest(BaseModel):
    name: str | None = None
    simulator: str = "modflow"
    scenarios: list[str] = Field(default_factory=list)
    gpu_id: int
    epochs: int = 0
    eval_interval: int = 1
    batch_size: int = 256
    test_batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    num_workers: int = 8
    test_ratio: float = 0.10
    resume_from: str | None = None
    input_representation: TrainingJobRequestedInputRepresentation = "auto"


class TrainingMetricsSummary(BaseModel):
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    pr_auc: float | None = None


class TrainingJobSummary(BaseModel):
    job_id: str
    name: str
    status: TrainingJobStatus
    simulator: str
    scenarios: list[str]
    gpu_id: int
    created_at: float
    started_at: float | None = None
    ended_at: float | None = None
    pid: int | None = None
    artifact_root: str
    run_dir: str
    log_path: str
    config: TrainingJobConfig
    latest_epoch: int | None = None
    latest_step: int | None = None
    steps_per_epoch: int | None = None
    global_step: int | None = None
    avg_loss: float | None = None
    steps_per_sec: float | None = None
    eta_seconds: float | None = None
    latest_test_epoch: int | None = None
    latest_metrics: TrainingMetricsSummary | None = None
    error_message: str | None = None


class TrainingCheckpointInfo(BaseModel):
    name: str
    path: str
    size_bytes: int
    mtime: float
    epoch: int | None = None


class TrainingJobDetail(TrainingJobSummary):
    command: list[str]
    checkpoints: list[TrainingCheckpointInfo] = Field(default_factory=list)
    prepared_name: str | None = None


class TrainingLogResponse(BaseModel):
    job_id: str
    lines: list[str]


class TrainingPoint(BaseModel):
    epoch: int
    step: int
    global_step: int
    avg_loss: float
    steps_per_sec: float
    eta_seconds: float


class TrainingTestPoint(BaseModel):
    epoch: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    per_scenario: dict[str, dict[str, float | int]]


class TrainingCurvesResponse(BaseModel):
    job_id: str
    training_points: list[TrainingPoint]
    training_epoch_points: list[TrainingPoint]
    test_points: list[TrainingTestPoint]
    checkpoints: list[TrainingCheckpointInfo]


class TrainingOverviewResponse(BaseModel):
    datasets: list[TrainingDatasetInfo]
    gpus: list[GPUInfo]
    jobs: list[TrainingJobSummary]
    running_job_count: int
    completed_job_count: int

