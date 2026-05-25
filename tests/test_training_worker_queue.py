from contextlib import contextmanager
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


def test_training_queue_accepts_busy_existing_gpu(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "1")
    monkeypatch.setattr(
        training_manager,
        "get_gpu_inventory",
        lambda: [
            {
                "index": 0,
                "name": "GPU",
                "memory_used_mib": 9,
                "memory_total_mib": 10,
                "utilization_gpu": 80,
                "available": False,
                "locked_by_job_id": "train-running",
                "reason": "locked by train-running",
            }
        ],
    )

    entry = training_manager.create_job(_payload())

    assert entry["status"] == "queued"
    assert entry["gpu_id"] == 0
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


def test_training_worker_refreshes_queue_lock_while_launching(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "1")
    entry = training_manager.create_job(_payload())
    calls: list[tuple[str, str, float]] = []

    @contextmanager
    def fake_refresh(lock_key, owner, *, ttl_seconds, interval=None):
        calls.append((lock_key, owner, ttl_seconds))
        yield

    def fake_run_queued_job(job_id: str):
        assert job_id == entry["job_id"]
        return {"job_id": job_id, "status": "starting"}

    monkeypatch.setattr(locks, "refresh_lock_while", fake_refresh)
    monkeypatch.setattr(training_manager, "run_queued_job", fake_run_queued_job)

    assert training_worker_queue.run_next_queued_job(worker_id="worker-test") is True

    assert calls == [
        (training_worker_queue.QUEUE_LOCK_KEY, "worker-test", training_worker_queue.QUEUE_LOCK_TTL_SECONDS)
    ]


def test_training_worker_skips_not_ready_job_and_launches_next(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "1")
    first = training_manager.create_job({**_payload(), "name": "first"})
    second = training_manager.create_job({**_payload(), "name": "second"})
    attempts: list[str] = []

    def fake_launch(payload):
        attempts.append(payload["_job_id"])
        if payload["_job_id"] == first["job_id"]:
            raise ValueError("GPU 0 is not available: locked by train-other")
        return {"job_id": payload["_job_id"], "status": "starting"}

    monkeypatch.setattr(training_manager, "_launch_job", fake_launch)

    assert training_worker_queue.run_next_queued_job(worker_id="worker-test") is True
    assert attempts == [first["job_id"], second["job_id"]]

    jobs = {job["job_id"]: job for job in training_manager.list_jobs(refresh=False)}
    assert jobs[first["job_id"]]["status"] == "queued"
    assert jobs[second["job_id"]]["status"] == "queued"


def test_training_worker_marks_invalid_queued_payload_as_error(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "1")
    entry = training_manager.create_job(_payload())

    def fake_run_queued_job(job_id: str):
        assert job_id == entry["job_id"]
        raise ValueError("resume checkpoint not found: /missing/router_latest.pt")

    monkeypatch.setattr(training_manager, "run_queued_job", fake_run_queued_job)

    assert training_worker_queue.run_next_queued_job(worker_id="worker-test") is True

    stored = training_manager.get_job(entry["job_id"], refresh=False)
    assert stored["status"] == "error"
    assert "resume checkpoint not found" in stored["error_message"]
    assert "queued training launch failed" in Path(stored["log_path"]).read_text(encoding="utf-8")


def test_training_worker_does_not_overwrite_cancelled_queued_job(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "1")
    entry = training_manager.create_job(_payload())

    def fake_run_queued_job(job_id: str):
        assert job_id == entry["job_id"]
        training_manager.stop_job(job_id)
        raise ValueError("resume checkpoint not found: /missing/router_latest.pt")

    monkeypatch.setattr(training_manager, "run_queued_job", fake_run_queued_job)

    assert training_worker_queue.run_next_queued_job(worker_id="worker-test") is True

    stored = training_manager.get_job(entry["job_id"], refresh=False)
    assert stored["status"] == "terminated"
    assert stored["terminated"] is True
    assert stored["error_message"] is None
    assert "queued training launch failed" not in Path(stored["log_path"]).read_text(encoding="utf-8")


def test_training_worker_continues_when_queued_job_was_cancelled(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "1")
    first = training_manager.create_job({**_payload(), "name": "first"})
    second = training_manager.create_job({**_payload(), "name": "second"})
    attempts: list[str] = []

    def fake_run_queued_job(job_id: str):
        attempts.append(job_id)
        if job_id == first["job_id"]:
            return None
        return {"job_id": job_id, "status": "starting"}

    monkeypatch.setattr(training_manager, "run_queued_job", fake_run_queued_job)

    assert training_worker_queue.run_next_queued_job(worker_id="worker-test") is True
    assert attempts == [first["job_id"], second["job_id"]]


def test_training_launch_passes_runtime_router_dir(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    router_dir = tmp_path / "runtime-data" / "router"
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        captured["cwd"] = kwargs.get("cwd")
        return FakeProcess()

    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "0")
    monkeypatch.setattr(training_manager, "ROUTER_DATA_DIR", router_dir)
    monkeypatch.setattr(training_manager, "_refresh_entry", lambda entry: entry)
    monkeypatch.setattr(training_manager.subprocess, "Popen", fake_popen)

    entry = training_manager.create_job(_payload())

    command = captured["command"]
    assert command[command.index("--router-dir") + 1] == str(router_dir)
    assert entry["command"] == command


def test_launch_job_does_not_start_cancelled_queued_entry(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PIERN_WORKER_QUEUE_TRAINING", "1")
    entry = training_manager.create_job(_payload())
    payload = training_manager._payload_from_queued_entry(entry)

    entries = training_manager._load_registry()
    entries[0]["status"] = "terminated"
    entries[0]["terminated"] = True
    entries[0]["ended_at"] = 2.0
    training_manager._save_registry(entries)

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("cancelled queued job must not spawn a subprocess")

    monkeypatch.setattr(training_manager.subprocess, "Popen", fail_popen)

    assert training_manager._launch_job(payload) is None
    stored = training_manager.get_job(entry["job_id"], refresh=False)
    assert stored["status"] == "terminated"
    assert stored["pid"] is None
