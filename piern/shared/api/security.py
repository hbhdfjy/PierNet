"""Optional single-token protection for mutating API calls."""

from __future__ import annotations

import hmac
import os
from typing import Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
DEFAULT_EXEMPT_PREFIXES = ("/api/health", "/docs", "/openapi.json")


def configured_auth_token() -> str:
    return os.getenv("PIERN_AUTH_TOKEN", "").strip()


def auth_enabled() -> bool:
    return bool(configured_auth_token())


def _extract_token(request: Request) -> str:
    header = request.headers.get("authorization", "").strip()
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-piern-token", "").strip()


def token_allowed(provided: str, expected: str | None = None) -> bool:
    expected_value = configured_auth_token() if expected is None else expected
    return bool(expected_value) and hmac.compare_digest(provided, expected_value)


class ApiTokenMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, exempt_prefixes: Iterable[str] = DEFAULT_EXEMPT_PREFIXES):
        super().__init__(app)
        self.exempt_prefixes = tuple(exempt_prefixes)

    async def dispatch(self, request: Request, call_next):
        expected = configured_auth_token()
        if not expected:
            return await call_next(request)
        if request.method.upper() not in MUTATING_METHODS:
            return await call_next(request)
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        if any(request.url.path.startswith(prefix) for prefix in self.exempt_prefixes):
            return await call_next(request)
        if token_allowed(_extract_token(request), expected):
            return await call_next(request)
        return JSONResponse(
            status_code=401,
            content={
                "code": "UNAUTHORIZED",
                "message": "需要有效的 PIERN_AUTH_TOKEN 才能执行写操作。",
                "details": {"auth": "Bearer token or X-PIERN-Token"},
                "request_id": None,
            },
        )


def install_security(app: FastAPI) -> None:
    app.add_middleware(ApiTokenMiddleware)
