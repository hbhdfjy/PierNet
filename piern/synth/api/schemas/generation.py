"""生成任务相关 Pydantic 模型。"""

from pydantic import BaseModel, Field
from typing import Optional


class GenerateTemplatesRequest(BaseModel):
    scenarios: list[str] = []
    n_templates: int = Field(100, ge=1, le=10000)
    skip_existing: bool = False
    append_existing: bool = False   # True=追加到已有文件（补齐到 n_templates）
    config: str = "configs/text2comp/default.yaml"
    # 覆盖 generation.yaml 的值
    language_mix: Optional[float] = Field(None, ge=0.0, le=1.0)
    transform_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_workers: Optional[int] = Field(None, ge=1, le=64)


class FillSamplesRequest(BaseModel):
    scenarios: list[str] = []
    n_samples: int = Field(100, ge=1, le=1000000)
    skip_existing: bool = False
    config: str = "configs/text2comp/default.yaml"
    templates_dir: str = ""
    output_dir: str = ""
    output_format: str = Field("parquet", pattern="^(parquet|jsonl|both)$")
    compression: str = Field("zstd", pattern="^(zstd|snappy|gzip|brotli|none)$")
    batch_size: int = Field(8192, ge=1, le=1000000)
    seed: Optional[int] = None
    precision: int = Field(4, ge=1, le=10)


class JobStartResponse(BaseModel):
    job_id: str
    status: str = "running"
    scenario_totals: dict[str, int] = {}


class GenerateRequest(BaseModel):
    """旧版一步生成请求（兼容）。"""
    scenarios: list[str] = []
    n_samples: int = 10
    language_mix: float = 0.5
    transform_prob: float = 0.1
    model: str = "deepseek-ai/DeepSeek-V3.2"
    skip_existing: bool = False
    config: str = "configs/text2comp/default.yaml"
