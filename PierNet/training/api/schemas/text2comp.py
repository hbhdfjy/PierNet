"""
文生计算模块训练 API schemas
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Text2CompJobStatus = Literal["queued", "starting", "running", "evaluating", "done", "error", "terminated"]


class ExpertModelSummary(BaseModel):
    """专家模型概要"""
    name: str
    domain: str
    output_dim: int
    description: str | None = None


class Text2CompDatasetInfo(BaseModel):
    """数据集信息"""
    path: str
    simulator: str
    scenario: str
    n_samples: int | None = None
    file_size_bytes: int | None = None
    mtime: float | None = None


class Text2CompGPUInfo(BaseModel):
    """GPU信息"""
    index: int
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_gpu: int
    available: bool
    locked_by_job_id: str | None = None
    reason: str | None = None


class Text2CompJobConfig(BaseModel):
    """训练配置"""
    expert_model: str
    output_dim: int = 128
    epochs: int = 50
    eval_interval: int = 5
    batch_size: int = 32
    test_batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    num_workers: int = 4
    test_ratio: float = 0.1
    max_length: int = 2048
    precision: int = 4
    resume_from: str | None = None


class Text2CompJobCreateRequest(BaseModel):
    """创建训练任务请求"""
    name: str | None = None
    expert_model: str  # diff-sorp, diff-reaction, burgers, modflow, etc.
    dataset_path: str | None = None  # 如果为None，自动查找
    gpu_id: int
    output_dim: int = Field(default=0, ge=0, le=1_000_000)
    epochs: int = 50
    eval_interval: int = 5
    batch_size: int = 32
    test_batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    num_workers: int = 4
    test_ratio: float = 0.1
    max_length: int = 2048
    precision: int = 4
    resume_from: str | None = None
    freeze_base: bool = False


class Text2CompMetricsSummary(BaseModel):
    """评估指标"""
    rmse: float | None = None
    mae: float | None = None
    mse: float | None = None


class Text2CompJobSummary(BaseModel):
    """训练任务概要"""
    job_id: str
    name: str
    status: Text2CompJobStatus
    simulator: str  # 前端期望字段名（兼容expert_model）
    scenario: str = ""  # 前端期望字段名
    expert_model: str | None = None  # 后端兼容字段
    dataset_path: str | None = None  # 后端兼容字段
    gpu_id: int
    created_at: float
    started_at: float | None = None
    ended_at: float | None = None
    pid: int | None = None
    artifact_root: str = ""
    run_dir: str = ""
    log_path: str = ""
    config: Text2CompJobConfig | dict = Field(default_factory=dict)
    latest_epoch: int | None = None
    latest_step: int | None = None
    steps_per_epoch: int | None = None
    global_step: int | None = None
    avg_loss: float | None = None
    steps_per_sec: float | None = None
    eta_seconds: float | None = None
    latest_test_epoch: int | None = None
    latest_metrics: Text2CompMetricsSummary | None = None
    error_message: str | None = None


class Text2CompCheckpointInfo(BaseModel):
    """检查点信息"""
    name: str
    path: str
    size_bytes: int
    mtime: float
    epoch: int | None = None
    is_best: bool = False


class Text2CompJobDetail(Text2CompJobSummary):
    """训练任务详情"""
    command: list[str]
    checkpoints: list[Text2CompCheckpointInfo] = Field(default_factory=list)


class Text2CompLogResponse(BaseModel):
    """日志响应"""
    job_id: str
    lines: list[str]


class Text2CompTrainingPoint(BaseModel):
    """训练曲线点"""
    epoch: int
    step: int
    global_step: int
    avg_loss: float
    steps_per_sec: float
    eta_seconds: float


class Text2CompTestPoint(BaseModel):
    """测试评估点"""
    epoch: int
    rmse: float
    mae: float
    mse: float


class Text2CompCurvesResponse(BaseModel):
    """训练曲线响应"""
    job_id: str
    training_points: list[Text2CompTrainingPoint]
    training_epoch_points: list[Text2CompTrainingPoint]
    test_points: list[Text2CompTestPoint]
    checkpoints: list[Text2CompCheckpointInfo]


class Text2CompTrainRequest(BaseModel):
    """启动Text2Comp训练请求（前端兼容）"""
    name: str | None = None
    simulator: str
    scenario: str
    train_path: str
    output_dim: int = 128
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gpu_id: int
    base_model: str | None = None


class Text2CompTrainResponse(BaseModel):
    """启动Text2Comp训练响应"""
    ok: bool
    job_id: str = ""
    model_path: str = ""
    config: dict = Field(default_factory=dict)
    error: str | None = None


class Text2CompOverviewResponse(BaseModel):
    """总览响应"""
    expert_models: list[ExpertModelSummary]
    datasets: list[Text2CompDatasetInfo]
    gpus: list[Text2CompGPUInfo]
    jobs: list[Text2CompJobSummary]
    running_job_count: int
    completed_job_count: int
