"""Registry 相关 Pydantic 模型。"""

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    scenarios: list[str] = []
    fields: list[str] = []
    overwrite: bool = False
    simulator_level: bool = False
    config: str = "configs/text2comp/default.yaml"
    output: str = "configs/text2comp/registry.yaml"
