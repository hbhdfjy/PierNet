from __future__ import annotations

import sys
from pathlib import Path

from piern.training.router import train as router_train
from scripts.router import train_token_router


def _capture_training(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run_training(config):
        captured["config"] = config
        return tmp_path / "run"

    monkeypatch.setattr(train_token_router, "run_training", fake_run_training)
    return captured


def test_train_token_router_defaults_artifact_root_to_selected_simulator(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(train_token_router, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(router_train, "ARTIFACT_ROOT", tmp_path / "artifacts")
    captured = _capture_training(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_token_router.py", "--simulator", "simpeg", "--epochs", "1"],
    )

    train_token_router.main()

    config = captured["config"]
    assert config.router_dir == str(tmp_path / "data" / "router")
    assert config.artifact_root == str(tmp_path / "artifacts" / "token_router" / "simpeg")


def test_train_token_router_preserves_explicit_artifact_root(monkeypatch, tmp_path: Path):
    explicit_artifact_root = tmp_path / "custom-artifacts"
    captured = _capture_training(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_token_router.py",
            "--simulator",
            "simpeg",
            "--artifact-root",
            str(explicit_artifact_root),
        ],
    )

    train_token_router.main()

    assert captured["config"].artifact_root == str(explicit_artifact_root)
