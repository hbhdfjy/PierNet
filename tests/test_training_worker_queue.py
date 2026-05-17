from pathlib import Path
from types import SimpleNamespace

from piern.shared.tasks import locks, workers
from piern.training.services import job_store as training_job_store
from piern.training.services import training_manager
from piern.training.services import worker_queue as training_worker_queue


def _use_tmp_runtime(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(training_job_store, "TRAINING_JOB_STORE_PATH", tmp_path / "training_jobs.sqlite")
    monkeypatch.setattr(training_job_store, "_INITIALIZED", False)
    monkeypatch.setattr(locks, "LOCK_STORE_PATH", tmp_path / "locks.sqlite")
    monkeypatch.setattr(locks, "_INITIALIZED", False)
    monkeypatch.setattr(workers, "WORKER_STORE_PATH", tmp_path / "workers.sqlite")
    monkeypatch.setattr(workers, "_INITIALIZED", False)
    monkeypatch.setattr(training_manager, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(training_manager, "RUNLOGS_ROOT", tmp_path / "runlogs")
    monkeypatch.setattr(training_manager, "CONTROL_ROOT", tmp_path / "runlogs" / "training-controls")
    training_job_store.init_store()


def _mock_training_prereqs(monkeypatch):
    monkeypatch.setattr(
        training_manager,
        "list_datasets",
        lambda: [{"simulator": "modflow", "scenarios": [{"scenario": "coastal_seawater"}]}],
    )
    monkeypatch.setattr(
        training_manager,
        "get_gpu_inventory",
        lambda: [
            {
                "index": 0,
                "name": "GPU",
                "memory_used_mib": 1,
                "memory_total_mib": 10,
                "utilization_gpu": 0,
                "available": True,
                "locked_by_job_id": None,
                "reason": None,
            }
        ],
    )
    monkeypatch.setattr(
        training_manager,
        "inspect_router_input_representation",
        lambda **kwargs: (
            "pretrained_embeddings",
            SimpleNamespace(embedding_model="/models/qwen", tokenizer_name="/models/qwen"),
        ),
    )


def _payload() -> dict:
    return {
        "name": "queued-train",
        "simulator": "modflow",
        "scenarios": ["coastal_seawater"],
        "gpu_id": 0,
        "epochs": 1,
        "eval_interval": 1,
        "keep_last_epochs": 2,
        "seed": 42,
        "batch_size": 2,
        "test_batch_size": 2,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "num_workers": 0,
        "prepare_workers": 0,
        "test_ratio": 0.1,
        "max_train_samples": 10,
        "max_test_samples": 10,
        "resume_from": None,
    }


def test_training_create_job_queues_when_worker_queue_enabled(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "1")

    entry = training_manager.create_job(_payload())
    jobs = training_manager.list_jobs(refresh=True)

    assert entry["status"] == "queued"
    assert entry["started_at"] is None
    assert jobs[0]["status"] == "queued"
    assert "waiting for piern-worker" in Path(entry["log_path"]).read_text(encoding="utf-8")


def test_training_worker_launches_queued_job(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "1")
    entry = training_manager.create_job(_payload())
    launched = {}

    def fake_launch(payload):
        launched.update(payload)
        return {"job_id": payload["_job_id"], "status": "starting"}

    monkeypatch.setattr(training_manager, "_launch_job", fake_launch)

    assert training_worker_queue.run_next_queued_job(worker_id="worker-test") is True
    assert launched["_job_id"] == entry["job_id"]
    assert launched["simulator"] == "modflow"
    assert workers.list_workers()[0]["worker_id"] == "worker-test"
