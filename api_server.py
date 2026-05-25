"""Backward-compatible import target for the unified PiERN FastAPI app.

Run locally with:
  uvicorn api_server:app --reload --port 8000
"""

from piern.api.main import app  # noqa: F401
