"""配置相关 Pydantic 模型。"""

from typing import Optional
from pydantic import BaseModel


class LLMConfigRequest(BaseModel):
    provider: str = "siliconflow"
    model: str = ""
    api_key: str = ""      # 空字符串=不修改
    base_url: str = ""
    temperature: float = 1.0
    max_tokens: int = 1024


class DataDirEntry(BaseModel):
    """单个数据目录配置条目。"""
    key: str                              # 配置键名，如 "modflow"
    path: str                             # 相对路径，如 "data/modflow"
    simulator: str                        # 默认 simulator 类型
    file_suffix: str = ""                 # 文件名后缀去除，如 "_groundwater_timeseries"
    transient_simulator: str = ""         # 暂态 simulator 类型（可选）
    transient_keywords: list[str] = []    # 触发暂态识别的关键词


class DataDirsRequest(BaseModel):
    entries: list[DataDirEntry]
