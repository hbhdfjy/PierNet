from __future__ import annotations

from typing import Any

import pytest

from PierNet.training.services.conversation_pipeline import (
    Text2CompGPUUnavailableError,
    create_text2comp_job_with_gpu_retry,
)


def _gpu(index: int, *, available: bool, used: int = 0, total: int = 80_000) -> dict[str, Any]:
    return {
        "index": index,
        "available": available,
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_gpu": 0,
    }


def test_text2comp_gpu_retry_prefers_router_gpu() -> None:
    created: list[dict[str, Any]] = []

    def create(payload: dict[str, Any]) -> dict[str, Any]:
        created.append(payload)
        return {"job_id": "text-job", "gpu_id": payload["gpu_id"]}

    result = create_text2comp_job_with_gpu_retry(
        {"name": "test"},
        preferred_gpu_id=3,
        create_job=create,
        get_gpu_inventory=lambda: [_gpu(2, available=True), _gpu(3, available=True, used=10_000)],
        delay_seconds=0,
    )

    assert result["gpu_id"] == 3
    assert created == [{"name": "test", "gpu_id": 3}]


def test_text2comp_gpu_retry_uses_available_fallback() -> None:
    result = create_text2comp_job_with_gpu_retry(
        {},
        preferred_gpu_id=3,
        create_job=lambda payload: {"job_id": "text-job", "gpu_id": payload["gpu_id"]},
        get_gpu_inventory=lambda: [
            _gpu(3, available=False, used=79_000),
            _gpu(1, available=True, used=20_000),
            _gpu(2, available=True, used=40_000),
        ],
        delay_seconds=0,
    )

    assert result["gpu_id"] == 1


def test_text2comp_gpu_retry_survives_transient_launch_race() -> None:
    calls = 0
    sleeps: list[float] = []

    def create(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("GPU 3 is not available")
        return {"job_id": "text-job", "gpu_id": payload["gpu_id"]}

    result = create_text2comp_job_with_gpu_retry(
        {},
        preferred_gpu_id=3,
        create_job=create,
        get_gpu_inventory=lambda: [_gpu(3, available=True)],
        attempts=2,
        delay_seconds=0.25,
        sleep=sleeps.append,
    )

    assert result["gpu_id"] == 3
    assert calls == 2
    assert sleeps == [0.25]


def test_text2comp_gpu_retry_can_wait_for_reserved_gpu_without_fallback() -> None:
    inventories = iter(
        [
            [_gpu(3, available=False), _gpu(1, available=True)],
            [_gpu(3, available=True), _gpu(1, available=True)],
        ]
    )
    selected: list[int] = []

    def create(payload: dict[str, Any]) -> dict[str, Any]:
        selected.append(payload["gpu_id"])
        return {"job_id": "text-job", "gpu_id": payload["gpu_id"]}

    result = create_text2comp_job_with_gpu_retry(
        {},
        preferred_gpu_id=3,
        create_job=create,
        get_gpu_inventory=lambda: next(inventories),
        attempts=2,
        delay_seconds=0,
        allow_fallback=False,
    )

    assert result["gpu_id"] == 3
    assert selected == [3]


def test_text2comp_gpu_retry_does_not_hide_validation_errors() -> None:
    with pytest.raises(ValueError, match="Invalid training data"):
        create_text2comp_job_with_gpu_retry(
            {},
            preferred_gpu_id=3,
            create_job=lambda _payload: (_ for _ in ()).throw(ValueError("Invalid training data")),
            get_gpu_inventory=lambda: [_gpu(3, available=True)],
            delay_seconds=0,
        )


def test_text2comp_gpu_retry_reports_recoverable_resource_wait() -> None:
    with pytest.raises(Text2CompGPUUnavailableError, match="No GPU is currently available"):
        create_text2comp_job_with_gpu_retry(
            {},
            preferred_gpu_id=3,
            create_job=lambda payload: payload,
            get_gpu_inventory=lambda: [_gpu(3, available=False)],
            attempts=2,
            delay_seconds=0,
        )
