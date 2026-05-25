"""生成任务相关 Pydantic 模型。"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def _normalize_scenarios(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value.split(",") if isinstance(value, str) else value
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        scenario = str(item).strip()
        if not scenario or scenario in seen:
            continue
        seen.add(scenario)
        normalized.append(scenario)
    return normalized


class GenerateTemplatesRequest(BaseModel):
    scenarios: list[str] = Field(default_factory=list)

    @field_validator("scenarios", mode="before")
    @classmethod
    def normalize_scenarios(cls, value: Any) -> list[str]:
        return _normalize_scenarios(value)

    n_templates: int = Field(100, ge=1, le=10000)
    skip_existing: bool = False
    append_existing: bool = False  # True=追加到已有文件（补齐到 n_templates）
    config: str = "configs/text2comp/default.yaml"
    # 覆盖 generation.yaml 的值
    language_mix: Optional[float] = Field(None, ge=0.0, le=1.0)
    transform_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_workers: Optional[int] = Field(None, ge=1, le=64)


class FillSamplesRequest(BaseModel):
    scenarios: list[str] = Field(default_factory=list)

    @field_validator("scenarios", mode="before")
    @classmethod
    def normalize_scenarios(cls, value: Any) -> list[str]:
        return _normalize_scenarios(value)

    n_samples: int = Field(100, ge=1, le=1000000)
    skip_existing: bool = False
    config: str = "configs/text2comp/default.yaml"
    templates_dir: str = ""
    output_dir: str = ""
    output_format: str = Field("parquet", pattern="^(parquet|jsonl|both)$")
    compression: str = Field("zstd", pattern="^(zstd|snappy|gzip|brotli|none)$")
    batch_size: int = Field(8192, ge=1, le=1000000)
    max_workers: Optional[int] = Field(None, ge=1, le=64)
    seed: Optional[int] = Field(None, ge=0, le=2_147_483_647)
    precision: int = Field(4, ge=1, le=10)


JobStartStatus = Literal["queued", "running"]


class JobStartResponse(BaseModel):
    job_id: str
    status: JobStartStatus = "running"
    scenario_totals: dict[str, int] = Field(default_factory=dict)
