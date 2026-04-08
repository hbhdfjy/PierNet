"""
PiERN Stage 2 API 入口（向后兼容）。

启动：
  uvicorn api_server:app --reload --port 8000
"""

from piern.api.main import app  # noqa: F401
