from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


GPU_UNAVAILABLE_MARKERS = (
    "gpu",
    "not available",
)


class Text2CompGPUUnavailableError(ValueError):
    pass


def _is_gpu_unavailable_error(exc: ValueError) -> bool:
    message = str(exc).lower()
    return all(marker in message for marker in GPU_UNAVAILABLE_MARKERS)


def _rank_available_gpus(inventory: list[dict[str, Any]], preferred_gpu_id: int) -> list[int]:
    available = [item for item in inventory if item.get("available")]
    available.sort(
        key=lambda item: (
            0 if int(item.get("index", -1)) == preferred_gpu_id else 1,
            -(int(item.get("memory_total_mib") or 0) - int(item.get("memory_used_mib") or 0)),
            int(item.get("utilization_gpu") or 0),
            int(item.get("index") or 0),
        )
    )
    return [int(item["index"]) for item in available]


def create_text2comp_job_with_gpu_retry(
    payload: dict[str, Any],
    *,
    preferred_gpu_id: int,
    create_job: Callable[[dict[str, Any]], dict[str, Any]],
    get_gpu_inventory: Callable[[], list[dict[str, Any]]],
    attempts: int = 8,
    delay_seconds: float = 1.0,
    allow_fallback: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Start Text2Comp after a Router stage without failing on a short GPU handoff race."""

    attempts = max(1, int(attempts))
    last_error: ValueError | None = None
    for attempt in range(attempts):
        inventory = get_gpu_inventory()
        candidate_ids = _rank_available_gpus(inventory, preferred_gpu_id)
        if not allow_fallback:
            candidate_ids = [gpu_id for gpu_id in candidate_ids if gpu_id == preferred_gpu_id]
        if not candidate_ids:
            last_error = ValueError("No GPU is currently available for the Text2Comp stage")
        for gpu_id in candidate_ids:
            try:
                return create_job({**payload, "gpu_id": gpu_id})
            except ValueError as exc:
                if not _is_gpu_unavailable_error(exc):
                    raise
                last_error = exc
        if attempt + 1 < attempts:
            sleep(max(0.0, delay_seconds))
    message = str(last_error or "Unable to allocate a GPU for the Text2Comp stage")
    raise Text2CompGPUUnavailableError(message)
