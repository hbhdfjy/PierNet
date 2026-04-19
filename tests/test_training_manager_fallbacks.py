from __future__ import annotations

from pathlib import Path

from piern.training.services import training_manager


def test_list_datasets_returns_empty_when_router_manifest_is_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(training_manager, "ROUTER_MANIFEST_PATH", tmp_path / "missing-router.json")

    assert training_manager.list_datasets() == []
    overview = training_manager.get_overview()
    assert overview["datasets"] == []


def test_get_gpu_inventory_returns_empty_when_nvidia_smi_is_unavailable(monkeypatch):
    monkeypatch.setattr(training_manager, "list_jobs", lambda refresh=True: [])

    def _raise(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(training_manager.subprocess, "check_output", _raise)

    assert training_manager.get_gpu_inventory() == []
