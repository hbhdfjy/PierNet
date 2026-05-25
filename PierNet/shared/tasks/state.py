"""Shared task status model and transition validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

QUEUED = "queued"
STARTING = "starting"
RUNNING = "running"
EVALUATING = "evaluating"
STOPPING = "stopping"
DONE = "done"
ERROR = "error"
TERMINATED = "terminated"
EXTERNAL_TERMINATED = "external_terminated"
DELETED = "deleted"

STATUS_ALIASES = {
    "succeeded": DONE,
    "success": DONE,
    "failed": ERROR,
    "failure": ERROR,
    "cancelled": TERMINATED,
    "canceled": TERMINATED,
    "canceling": STOPPING,
}

ACTIVE_STATUSES = frozenset({QUEUED, STARTING, RUNNING, EVALUATING, STOPPING})
TERMINAL_STATUSES = frozenset({DONE, ERROR, TERMINATED, EXTERNAL_TERMINATED, DELETED})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES

ALLOWED_TRANSITIONS = {
    QUEUED: frozenset({STARTING, RUNNING, STOPPING, ERROR, TERMINATED, EXTERNAL_TERMINATED}),
    STARTING: frozenset({RUNNING, EVALUATING, STOPPING, DONE, ERROR, TERMINATED, EXTERNAL_TERMINATED}),
    RUNNING: frozenset({EVALUATING, STOPPING, DONE, ERROR, TERMINATED, EXTERNAL_TERMINATED}),
    EVALUATING: frozenset({RUNNING, STOPPING, DONE, ERROR, TERMINATED, EXTERNAL_TERMINATED}),
    STOPPING: frozenset({TERMINATED, ERROR, EXTERNAL_TERMINATED, DONE}),
    DONE: frozenset({DELETED}),
    ERROR: frozenset({DELETED}),
    TERMINATED: frozenset({DELETED}),
    EXTERNAL_TERMINATED: frozenset({DELETED}),
    DELETED: frozenset(),
}


class IllegalStatusTransition(ValueError):
    """Raised when a task status transition violates the shared state machine."""


def normalize_status(value: object, *, fallback: str = EXTERNAL_TERMINATED) -> str:
    status = str(value or fallback).strip().lower()
    status = STATUS_ALIASES.get(status, status)
    return status if status in ALL_STATUSES else fallback


def is_active_status(value: object) -> bool:
    return normalize_status(value) in ACTIVE_STATUSES


def is_terminal_status(value: object) -> bool:
    return normalize_status(value) in TERMINAL_STATUSES


def is_transition_allowed(previous: object, next_status: object, *, allow_self: bool = True) -> bool:
    previous_norm = normalize_status(previous)
    next_norm = normalize_status(next_status)
    if allow_self and previous_norm == next_norm:
        return True
    return next_norm in ALLOWED_TRANSITIONS.get(previous_norm, frozenset())


def validate_transition(previous: object, next_status: object, *, allow_self: bool = True) -> str:
    previous_norm = normalize_status(previous)
    next_norm = normalize_status(next_status)
    if not is_transition_allowed(previous_norm, next_norm, allow_self=allow_self):
        raise IllegalStatusTransition(f"illegal task status transition: {previous_norm} -> {next_norm}")
    return next_norm


def active_statuses(extra: Iterable[str] = ()) -> set[str]:
    return {normalize_status(item) for item in [*ACTIVE_STATUSES, *extra]}


def terminal_statuses(extra: Iterable[str] = ()) -> set[str]:
    return {normalize_status(item) for item in [*TERMINAL_STATUSES, *extra]}


@dataclass(frozen=True)
class StatusChange:
    previous: str
    next: str
    allowed: bool


def describe_transition(previous: object, next_status: object) -> StatusChange:
    previous_norm = normalize_status(previous)
    next_norm = normalize_status(next_status)
    return StatusChange(
        previous=previous_norm,
        next=next_norm,
        allowed=is_transition_allowed(previous_norm, next_norm),
    )
