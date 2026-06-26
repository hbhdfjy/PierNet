from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from PierNet.training.text2comp import text2comp_manager


def _patch_job_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> object:
        captured["command"] = command
        captured["kwargs"] = kwargs

        class Process:
            pid = 4321

        return Process()

    monkeypatch.setattr(text2comp_manager, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(text2comp_manager, "RUNLOGS_ROOT", tmp_path / "runlogs")
    monkeypatch.setattr(text2comp_manager, "get_gpu_inventory", lambda: [{"index": 0, "available": True, "reason": None}])
    monkeypatch.setattr(text2comp_manager, "validate_training_data", lambda path, expected_dim: {"is_valid": True})
    monkeypatch.setattr(text2comp_manager, "_load_registry", lambda: [])
    monkeypatch.setattr(text2comp_manager, "_save_registry", lambda entries: captured.setdefault("entries", entries))
    monkeypatch.setattr(text2comp_manager, "_refresh_entry", lambda entry: entry)
    monkeypatch.setattr(text2comp_manager.subprocess, "Popen", fake_popen)

    return captured


def _train_data_arg(command: list[str]) -> str:
    return command[command.index("--train-data") + 1]


def test_validate_training_data_uses_stdlib_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "uploaded_train.jsonl"
    rows = [
        {"prompt": "sample 1", "label": [0.0, 1.0, 2.0, 3.0]},
        {"prompt": "sample 2", "label": [4.0, 5.0, 6.0, 7.0]},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = text2comp_manager.validate_training_data(str(dataset), expected_dim=4)

    assert result["is_valid"] is True
    assert result["valid_samples"] == 2
    assert result["actual_dims"] == [4]


def test_create_job_accepts_dataset_path_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _patch_job_runtime(monkeypatch, tmp_path)

    job = text2comp_manager.create_job(
        {
            "expert_model": "diff-sorp",
            "dataset_path": "/tmp/diff-sorp_train.jsonl",
            "gpu_id": 0,
            "epochs": 1,
        }
    )

    assert _train_data_arg(captured["command"]) == "/tmp/diff-sorp_train.jsonl"
    assert job["train_data_path"] == "/tmp/diff-sorp_train.jsonl"
    assert job["dataset_path"] == "/tmp/diff-sorp_train.jsonl"
    assert job["expert_model"] == "diff-sorp"


def test_create_job_auto_selects_matching_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _patch_job_runtime(monkeypatch, tmp_path)
    dataset_path = str(tmp_path / "diff-sorp_train.jsonl")
    monkeypatch.setattr(
        text2comp_manager,
        "list_datasets",
        lambda: [
            {"path": str(tmp_path / "burgers_train.jsonl"), "simulator": "burgers", "name": "burgers_train", "scenario": "burgers_train"},
            {"path": dataset_path, "simulator": "diff-sorp", "name": "diff-sorp_train", "scenario": "diff-sorp_train"},
        ],
    )

    job = text2comp_manager.create_job(
        {
            "expert_model": "diff-sorp",
            "gpu_id": 0,
            "epochs": 1,
        }
    )

    assert _train_data_arg(captured["command"]) == dataset_path
    assert job["dataset_path"] == dataset_path
    assert job["scenario"] == "diff-sorp_train"


def test_create_job_rejects_missing_training_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_job_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(text2comp_manager, "list_datasets", lambda: [])

    with pytest.raises(ValueError, match="No training data found"):
        text2comp_manager.create_job(
            {
                "expert_model": "diff-sorp",
                "gpu_id": 0,
                "epochs": 1,
            }
        )
