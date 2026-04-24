from __future__ import annotations

from pathlib import Path

import torch

from piern.training.api.schemas.training import TrainingJobSummary
from piern.training.services import training_manager
from piern.training.router.data import DEFAULT_QWEN_EMBEDDING_MODEL


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
            "batch_size": 2,
            "test_batch_size": 2,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "num_workers": 0,
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

