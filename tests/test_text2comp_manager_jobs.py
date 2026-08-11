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
    monkeypatch.setattr(
        text2comp_manager,
        "get_gpu_inventory",
        lambda **_kwargs: [{"index": 0, "available": True, "reason": None}],
    )
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


def test_create_job_validates_explicit_text2comp_output_dim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = _patch_job_runtime(monkeypatch, tmp_path)
    validation: dict[str, Any] = {}

    def capture_validation(path: str, expected_dim: int) -> dict[str, Any]:
        validation.update(path=path, expected_dim=expected_dim)
        return {"is_valid": True}

    monkeypatch.setattr(text2comp_manager, "validate_training_data", capture_validation)

    text2comp_manager.create_job(
        {
            "expert_model": "modflow",
            "dataset_path": "/tmp/modflow_expert_input.jsonl",
            "output_dim": 18,
            "gpu_id": 0,
            "epochs": 1,
        }
    )

    command = captured["command"]
    assert validation == {"path": "/tmp/modflow_expert_input.jsonl", "expected_dim": 18}
    assert command[command.index("--output-dim") + 1] == "18"


def test_create_job_passes_formal_quality_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _patch_job_runtime(monkeypatch, tmp_path)

    text2comp_manager.create_job(
        {
            "expert_model": "diff-sorp",
            "dataset_path": "/tmp/diff-sorp_train.jsonl",
            "gpu_id": 0,
            "epochs": 50,
            "head_learning_rate": 1e-4,
            "trainable_base_layers": 2,
            "normalize_labels": True,
            "min_samples": 100,
            "min_epochs": 10,
            "early_stop_patience": 8,
            "target_normalized_rmse": 0.15,
            "max_normalized_rmse": 0.25,
            "require_quality": True,
        }
    )

    command = captured["command"]
    assert command[command.index("--trainable-base-layers") + 1] == "2"
    assert command[command.index("--head-learning-rate") + 1] == "0.0001"
    assert command[command.index("--min-samples") + 1] == "100"
    assert "--normalize-labels" in command
    assert "--require-quality" in command


def test_create_job_auto_selects_matching_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _patch_job_runtime(monkeypatch, tmp_path)
    dataset_path = str(tmp_path / "diff-sorp_train.jsonl")
    monkeypatch.setattr(
        text2comp_manager,
        "list_datasets",
        lambda: [
            {
                "path": str(tmp_path / "burgers_train.jsonl"),
                "simulator": "burgers",
                "name": "burgers_train",
                "scenario": "burgers_train",
            },
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


@pytest.mark.parametrize("status", ["done", "error", "terminated"])
def test_refresh_entry_never_revives_terminal_job_when_pid_is_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = run_dir / "train.log"
    log_path.write_text("[done] training complete\n", encoding="utf-8")
    monkeypatch.setattr(text2comp_manager, "_pid_alive", lambda _pid: True)
    entry = {
        "job_id": "text2comp-finished",
        "name": "finished",
        "status": status,
        "pid": 4321,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "expert_model": "modflow",
    }

    refreshed = text2comp_manager._refresh_entry(entry)

    assert refreshed["status"] == status


def test_refresh_entry_recovers_completed_job_after_pid_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "final_model.pt").write_bytes(b"model")
    log_path = run_dir / "train.log"
    log_path.write_text("[done] training complete\n", encoding="utf-8")
    monkeypatch.setattr(text2comp_manager, "_pid_alive", lambda _pid: True)
    entry = {
        "job_id": "text2comp-finished",
        "name": "finished",
        "status": "running",
        "pid": 4321,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "expert_model": "modflow",
    }

    refreshed = text2comp_manager._refresh_entry(entry)

    assert refreshed["status"] == "done"
    assert refreshed["ended_at"] is not None
