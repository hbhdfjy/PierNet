"""配置相关 Pydantic 模型。"""

from pydantic import BaseModel


class LLMConfigRequest(BaseModel):
    provider: str = "siliconflow"
    model: str = ""
    api_key: str = ""      # 空字符串=不修改
    base_url: str = ""
    temperature: float = 1.0
    max_tokens: int = 1024
