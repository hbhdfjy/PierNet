"""配置相关 Pydantic 模型。"""

from pydantic import BaseModel, Field


class LLMConfigRequest(BaseModel):
    provider: str = "siliconflow"
    model: str = ""
    api_key: str = ""      # 空字符串=不修改
    base_url: str = ""
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=64, le=8192)
    thinking: str = "disabled"
