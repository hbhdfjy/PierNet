from __future__ import annotations

from pathlib import Path

import torch

from piern.training.api.schemas.training import TrainingJobSummary
from piern.training.services import training_manager
from piern.training.router.data import DEFAULT_QWEN_EMBEDDING_MODEL
from piern.training.router.train import _prune_epoch_checkpoints


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


def test_pid_alive_treats_zombie_process_as_dead(monkeypatch):
    monkeypatch.setattr(training_manager.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(training_manager.Path, "read_text", lambda self, **kwargs: "123 (python) Z 1")

    assert training_manager._pid_alive(123) is False


def test_validate_resume_checkpoint_rejects_input_representation_mismatch(tmp_path: Path):
    checkpoint_path = tmp_path / "router_latest.pt"
    torch.save(
        {
            "prepared_summary": {
                "simulator": "modflow",
                "scenarios": ["coastal_seawater"],
                "test_ratio": 0.10,
                "input_representation": "pretrained_embeddings",
                "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
                "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
            }
        },
        checkpoint_path,
    )

    try:
        training_manager._validate_resume_checkpoint(
            resume_from=str(checkpoint_path),
            simulator="modflow",
            scenarios=["coastal_seawater"],
            test_ratio=0.10,
            input_representation="char_tokens",
        )
    except ValueError as exc:
        assert "input_representation mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError for resume checkpoint representation mismatch")


def test_training_job_summary_exposes_embedding_config():
    summary = TrainingJobSummary(
        job_id="train-test",
        name="train-test",
        status="starting",
        simulator="modflow",
        scenarios=["coastal_seawater"],
        gpu_id=0,
        created_at=1.0,
        artifact_root="/tmp/artifacts",
        run_dir="/tmp/run",
        log_path="/tmp/run.log",
        config={
            "epochs": 1,
            "eval_interval": 1,
            "keep_last_epochs": 5,
            "batch_size": 2,
            "test_batch_size": 2,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "num_workers": 0,
            "prepare_workers": 2,
            "test_ratio": 0.1,
            "resume_from": None,
            "input_representation": "pretrained_embeddings",
            "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
            "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
        },
    )

    assert summary.config.input_representation == "pretrained_embeddings"
    assert summary.config.embedding_model == DEFAULT_QWEN_EMBEDDING_MODEL
    assert summary.config.embedding_tokenizer == DEFAULT_QWEN_EMBEDDING_MODEL
    assert summary.config.keep_last_epochs == 5
    assert summary.config.prepare_workers == 2


def test_prune_epoch_checkpoints_keeps_latest_epochs(tmp_path: Path):
    for epoch in range(1, 7):
        (tmp_path / f"router_epoch_{epoch:04d}.pt").write_text(str(epoch), encoding="utf-8")
    (tmp_path / "router_latest.pt").write_text("latest", encoding="utf-8")

    _prune_epoch_checkpoints(tmp_path, keep_last_epochs=3)

    assert sorted(path.name for path in tmp_path.glob("router_epoch_*.pt")) == [
        "router_epoch_0004.pt",
        "router_epoch_0005.pt",
        "router_epoch_0006.pt",
    ]
    assert (tmp_path / "router_latest.pt").exists()



def test_delete_job_removes_finished_entry(monkeypatch, tmp_path: Path):
    registry_path = tmp_path / 'training_jobs.json'
    run_dir = tmp_path / 'artifacts' / 'modflow' / 'runs' / 'train-done'
    log_path = tmp_path / 'run.log'
    checkpoint_path = run_dir / 'router_latest.pt'
    metrics_path = run_dir / 'test_metrics_latest.json'

    monkeypatch.setattr(training_manager, 'REGISTRY_PATH', registry_path)
    monkeypatch.setattr(training_manager, '_refresh_entry', lambda entry: entry)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text('checkpoint', encoding='utf-8')
    metrics_path.write_text('{}', encoding='utf-8')
    log_path.write_text('log', encoding='utf-8')

    registry_path.write_text(
        training_manager.json.dumps([
            {
                'job_id': 'train-done',
                'name': 'done-job',
                'status': 'done',
                'pid': None,
                'created_at': 1.0,
                'run_dir': str(run_dir),
                'log_path': str(log_path),
            }
        ]),
        encoding='utf-8',
    )

    removed = training_manager.delete_job('train-done')

    assert removed['job_id'] == 'train-done'
    assert training_manager._load_registry() == []
    assert not run_dir.exists()
    assert not log_path.exists()



def test_delete_job_rejects_running_entry(monkeypatch, tmp_path: Path):
    registry_path = tmp_path / 'training_jobs.json'
    monkeypatch.setattr(training_manager, 'REGISTRY_PATH', registry_path)
    monkeypatch.setattr(training_manager, '_refresh_entry', lambda entry: entry)

    registry_path.write_text(
        training_manager.json.dumps([
            {
                'job_id': 'train-running',
                'name': 'running-job',
                'status': 'running',
                'pid': None,
                'created_at': 1.0,
                'run_dir': str(tmp_path / 'run'),
                'log_path': str(tmp_path / 'run.log'),
            }
        ]),
        encoding='utf-8',
    )

    try:
        training_manager.delete_job('train-running')
    except ValueError as exc:
        assert 'still active' in str(exc)
    else:
        raise AssertionError('expected ValueError for active job deletion')



def test_delete_checkpoint_removes_epoch_weight(monkeypatch, tmp_path: Path):
    registry_path = tmp_path / 'training_jobs.json'
    run_dir = tmp_path / 'artifacts' / 'modflow' / 'runs' / 'train-done'
    log_path = tmp_path / 'run.log'
    checkpoint_path = run_dir / 'router_epoch_0003.pt'

    monkeypatch.setattr(training_manager, 'REGISTRY_PATH', registry_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text('checkpoint', encoding='utf-8')
    log_path.write_text('log', encoding='utf-8')
    registry_path.write_text(
        training_manager.json.dumps([
            {
                'job_id': 'train-done',
                'name': 'done-job',
                'status': 'done',
                'pid': None,
                'created_at': 1.0,
                'run_dir': str(run_dir),
                'log_path': str(log_path),
            }
        ]),
        encoding='utf-8',
    )

    updated = training_manager.delete_checkpoint('train-done', 'router_epoch_0003.pt')

    assert updated['job_id'] == 'train-done'
    assert not checkpoint_path.exists()
    assert training_manager._load_registry()[0]['checkpoints'] == []


def test_delete_checkpoint_rejects_primary_weight(monkeypatch, tmp_path: Path):
    registry_path = tmp_path / 'training_jobs.json'
    monkeypatch.setattr(training_manager, 'REGISTRY_PATH', registry_path)
    registry_path.write_text('[]', encoding='utf-8')

    try:
        training_manager.delete_checkpoint('train-done', 'router_latest.pt')
    except ValueError as exc:
        assert 'only epoch checkpoints' in str(exc)
    else:
        raise AssertionError('expected ValueError for primary checkpoint deletion')
