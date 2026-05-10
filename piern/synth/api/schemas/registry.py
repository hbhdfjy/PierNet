"""Registry 相关 Pydantic 模型。"""

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    scenarios: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    overwrite: bool = False
    simulator_level: bool = False
    config: str = "configs/text2comp/default.yaml"
    output: str = "configs/text2comp/registry.yaml"
