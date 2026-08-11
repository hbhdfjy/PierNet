from contextlib import contextmanager
from types import SimpleNamespace

from PierNet.training.api.routers import training as training_api
from PierNet.training.services import training_manager
from PierNet.training.text2comp import text2comp_manager
from PierNet.worker import runner


def test_training_jobs_api_reads_snapshot_without_refresh(monkeypatch):
    calls: list[bool] = []

    def fake_list_jobs(*, refresh: bool):
        calls.append(refresh)
        return []

    monkeypatch.setattr(training_manager, "list_jobs", fake_list_jobs)

    assert training_api.get_training_jobs() == []
    assert calls == [False]


def test_training_overview_and_gpu_inventory_use_snapshots(monkeypatch):
    calls: list[bool] = []

    def fake_list_jobs(*, refresh: bool):
        calls.append(refresh)
        return []

    monkeypatch.setattr(training_manager, "list_jobs", fake_list_jobs)
    monkeypatch.setattr(training_manager, "list_datasets", lambda: [])
    monkeypatch.setattr(training_manager.task_locks, "list_locks", lambda **kwargs: [])
    monkeypatch.setattr(training_manager.training_gpu, "query_nvidia_smi", lambda: "")

    overview = training_manager.get_overview()

    assert overview["jobs"] == []
    assert calls == [False, False]


def test_worker_refreshes_training_state_with_heartbeat(monkeypatch):
    calls: list[bool] = []
    heartbeats: list[tuple[str, str]] = []

    @contextmanager
    def fake_heartbeat_while(*, worker_id, kind, metadata, interval):
        del interval
        heartbeats.append((worker_id, metadata["phase"]))
        assert kind == "PierNet-worker"
        yield worker_id

    def fake_list_jobs(*, refresh: bool):
        calls.append(refresh)
        return []

    monkeypatch.setattr(runner.workers, "heartbeat_while", fake_heartbeat_while)
    monkeypatch.setattr(runner.training_manager, "list_jobs", fake_list_jobs)

    runner.refresh_training_state(worker_id="worker-test")

    assert calls == [True]
    assert heartbeats == [("worker-test", "training-state-refresh")]


def test_text2comp_create_job_reuses_locked_registry_snapshot(monkeypatch, tmp_path):
    inventory_calls: list[list[str] | None] = []

    def fake_gpu_inventory(*, jobs=None):
        inventory_calls.append(None if jobs is None else [str(job["job_id"]) for job in jobs])
        return [
            {
                "index": 0,
                "available": True,
                "reason": None,
            }
        ]

    monkeypatch.setattr(text2comp_manager, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(text2comp_manager, "RUNLOGS_ROOT", tmp_path / "runlogs")
    monkeypatch.setattr(text2comp_manager, "REGISTRY_PATH", tmp_path / "training_jobs.json")
    monkeypatch.setattr(text2comp_manager, "get_gpu_inventory", fake_gpu_inventory)
    monkeypatch.setattr(
        text2comp_manager,
        "get_all_experts",
        lambda: {"demo": {"output_dim": 1}},
    )
    monkeypatch.setattr(
        text2comp_manager,
        "validate_training_data",
        lambda *_args, **_kwargs: {
            "valid_samples": 1,
            "invalid_samples": 0,
            "expected_dim": 1,
            "actual_dims": [1],
            "is_valid": True,
        },
    )
    monkeypatch.setattr(text2comp_manager, "_load_registry", lambda: [])
    monkeypatch.setattr(text2comp_manager, "_save_registry", lambda _entries: None)
    monkeypatch.setattr(text2comp_manager, "_refresh_entry", lambda entry: entry)
    monkeypatch.setattr(
        text2comp_manager.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(pid=12345),
    )

    job = text2comp_manager.create_job(
        {
            "expert_model": "demo",
            "dataset_path": str(tmp_path / "train.jsonl"),
            "gpu_id": 0,
            "output_dim": 1,
        }
    )

    assert job["status"] == "starting"
    assert inventory_calls == [None, []]
