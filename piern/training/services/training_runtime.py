from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from piern.training.router.data import PRETRAINED_EMBEDDINGS

TRAINING_ACTIVE_STATUSES = {"queued", "starting", "running", "evaluating", "stopping"}
TRAINING_TERMINAL_STATUSES = {"done", "error", "terminated", "external_terminated"}
TRAINING_ALL_STATUSES = TRAINING_ACTIVE_STATUSES | TRAINING_TERMINAL_STATUSES


def coerce_int(value: Any, fallback: int | None = None) -> int | None:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def coerce_float(value: Any, fallback: float | None = None) -> float | None:
    if value is None:
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) else fallback


def coerce_status(value: Any) -> str:
    status = str(value or "external_terminated")
    return status if status in TRAINING_ALL_STATUSES else "external_terminated"


def normalize_training_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if normalized.get("input_representation") != PRETRAINED_EMBEDDINGS:
        normalized["input_representation"] = PRETRAINED_EMBEDDINGS
    return normalized


def normalize_per_scenario_metrics(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, float]] = {}
    for scenario, metrics in value.items():
        if not isinstance(metrics, dict):
            continue
        clean: dict[str, float] = {}
        for key, raw_value in metrics.items():
            parsed = coerce_float(raw_value)
            if parsed is not None:
                clean[str(key)] = parsed
        if clean:
            normalized[str(scenario)] = clean
    return normalized


def coerce_training_curve_point(payload: Any, *, include_steps_per_epoch: bool = False) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    epoch = coerce_int(payload.get("epoch"))
    step = coerce_int(payload.get("step"))
    avg_loss = coerce_float(payload.get("avg_loss"))
    steps_per_sec = coerce_float(payload.get("steps_per_sec"))
    eta_seconds = coerce_float(payload.get("eta_seconds"))
    if epoch is None or step is None or avg_loss is None or steps_per_sec is None or eta_seconds is None:
        return None
    point: dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "global_step": coerce_int(payload.get("global_step"), 0) or 0,
        "avg_loss": avg_loss,
        "steps_per_sec": steps_per_sec,
        "eta_seconds": eta_seconds,
    }
    if include_steps_per_epoch:
        steps_per_epoch = coerce_int(payload.get("steps_per_epoch"))
        if steps_per_epoch is not None:
            point["steps_per_epoch"] = steps_per_epoch
    return point


def tail_lines(path: Path, limit: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            file_size = handle.tell()
            if file_size == 0:
                return []
            chunk_size = 8192
            buffers: list[bytes] = []
            remaining = limit + 1
            pos = file_size
            while pos > 0 and len(buffers) < remaining:
                read_size = min(chunk_size, pos)
                pos -= read_size
                handle.seek(pos)
                chunk = handle.read(read_size)
                buffers.append(chunk)
            tail_bytes = b"".join(reversed(buffers))
            lines = tail_bytes.decode("utf-8", errors="replace").splitlines()
            return lines[-limit:]
    except OSError:
        return []


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
