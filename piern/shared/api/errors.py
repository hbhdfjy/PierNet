"""Structured API errors and request-id plumbing."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

LOGGER = logging.getLogger(__name__)
_REQUEST_ID: ContextVar[str | None] = ContextVar("piern_request_id", default=None)


def _client_host(request: Request) -> str | None:
    return request.client.host if request.client else None


def _log_request(
    request: Request,
    *,
    request_id: str,
    status_code: int,
    duration_ms: float,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": "api_request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "client": _client_host(request),
    }
    if error:
        payload["error"] = error
    LOGGER.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


class ApiError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = _REQUEST_ID.set(request_id)
        start = time.perf_counter()
        response = None
        error: str | None = None
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            status_code = response.status_code if response is not None else 500
            if response is not None:
                response.headers["x-request-id"] = request_id
            _log_request(
                request,
                request_id=request_id,
                status_code=status_code,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=error,
            )
            _REQUEST_ID.reset(token)


def current_request_id() -> str | None:
    return _REQUEST_ID.get()


def api_error(
    status_code: int,
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "details": details or {},
        },
    )


def _code_from_status(status_code: int) -> str:
    if status_code == 400:
        return "BAD_REQUEST"
    if status_code == 401:
        return "UNAUTHORIZED"
    if status_code == 403:
        return "FORBIDDEN"
    if status_code == 404:
        return "NOT_FOUND"
    if status_code == 409:
        return "CONFLICT"
    if status_code == 422:
        return "VALIDATION_ERROR"
    return "HTTP_ERROR" if status_code < 500 else "INTERNAL_ERROR"


def _normalize_detail(detail: Any, *, status_code: int) -> ApiError:
    request_id = current_request_id()
    if isinstance(detail, dict):
        code = detail.get("code")
        message = detail.get("message")
        raw_details = detail.get("details")
        if isinstance(code, str) and isinstance(message, str):
            return ApiError(
                code=code,
                message=message,
                details=raw_details if isinstance(raw_details, dict) else {},
                request_id=request_id,
            )
    if isinstance(detail, str):
        return ApiError(
            code=_code_from_status(status_code),
            message=detail,
            details={"status_code": status_code},
            request_id=request_id,
        )
    return ApiError(
        code=_code_from_status(status_code),
        message="请求参数或服务状态不正确",
        details={"detail": detail, "status_code": status_code},
        request_id=request_id,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    del request
    error = _normalize_detail(exc.detail, status_code=exc.status_code)
    return JSONResponse(status_code=exc.status_code, content=error.model_dump())


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    del request
    error = ApiError(
        code="VALIDATION_ERROR",
        message="请求参数校验失败",
        details={"errors": exc.errors()},
        request_id=current_request_id(),
    )
    return JSONResponse(status_code=422, content=error.model_dump())


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled API error path=%s", request.url.path)
    error = ApiError(
        code="INTERNAL_ERROR",
        message="服务内部错误",
        details={},
        request_id=current_request_id(),
    )
    return JSONResponse(status_code=500, content=error.model_dump())


def install_error_handlers(app: FastAPI) -> None:
    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
