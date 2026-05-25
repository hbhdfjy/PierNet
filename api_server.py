"""Backward-compatible import target for the unified PierNet FastAPI app.

Run locally with:
  uvicorn api_server:app --reload --port 8000
"""

from PierNet.api.main import app  # noqa: F401
