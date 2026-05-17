"""Audit middleware for mutating API calls."""

from __future__ import annotations

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from piern.shared.audit import store as audit_store

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _client_host(request: Request) -> str | None:
    return request.client.host if request.client else None


def _actor(_: Request) -> str:
    return "anonymous"


def _target(path: str) -> str:
    pieces = [piece for piece in path.strip("/").split("/") if piece]
    if len(pieces) >= 2 and pieces[0] == "api":
        return "/".join(pieces[1:4])
    return path


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            continue_audit = request.method.upper() in MUTATING_METHODS and request.url.path.startswith("/api/")
            if continue_audit:
                try:
                    audit_store.append_event(
                        actor=_actor(request),
                        action=f"{request.method.upper()} {request.url.path}",
                        target=_target(request.url.path),
                        method=request.method.upper(),
                        path=request.url.path,
                        status_code=response.status_code if response is not None else 500,
                        request_id=response.headers.get("x-request-id") if response is not None else None,
                        client=_client_host(request),
                        details={"query": str(request.url.query) if request.url.query else ""},
                    )
                except Exception:
                    # Auditing must never break the user request path.
                    pass


def install_audit(app: FastAPI) -> None:
    app.add_middleware(AuditMiddleware)
