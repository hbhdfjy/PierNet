from contextlib import contextmanager
import hashlib
import json

import pytest
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from PierNet.shared.tasks import locks, workers
from PierNet.training.services import job_store as training_job_store
from PierNet.training.services import training_manager
from PierNet.training.services import worker_queue as training_worker_queue


def test_prepared_cache_hash_preserves_builtin_legacy_key():
    values = {
        "simulator": "gcam",
        "scenarios": ["energy_transition", "carbon_pricing", "climate_feedback"],
        "test_ratio": 0.1,
        "input_representation": "pretrained_embeddings",
        "embedding_model": "/models/qwen",
        "embedding_tokenizer": "/models/qwen",
    }
    legacy_payload = json.dumps(
        {
            "simulator": values["simulator"],
            "scenarios": sorted(values["scenarios"]),
            "test_ratio": values["test_ratio"],
            "input_representation": values["input_representation"],
            "embedding_model": values["embedding_model"],
            "embedding_tokenizer": values["embedding_tokenizer"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    expected = f"gcam-{hashlib.blake2b(legacy_payload.encode('utf-8'), digest_size=6).hexdigest()}"

    assert training_manager._hash_prepared_name(**values, dataset_id=None) == expected
    assert training_manager._hash_prepared_name(**values, dataset_id="") == expected
    assert training_manager._hash_prepared_name(**values, dataset_id="router-custom") != expected


def test_reap_finished_training_processes_keeps_running_children(monkeypatch):
    class FakeProcess:
        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            return self.returncode

    running = FakeProcess(None)
    finished = FakeProcess(0)
    tracked = {101: running, 202: finished}
    monkeypatch.setattr(training_manager, "_LAUNCHED_PROCESSES", tracked)

    assert training_manager.reap_finished_processes() == 1
    assert tracked == {101: running}


def test_simple_pipeline_registers_text2comp_with_training_simulator(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        training_manager,
        "_prepare_simple_text2comp_dataset",
        lambda entry: {
            "path": str(tmp_path / "gcam.jsonl"),
            "generated": 100,
            "output_dim": 18,
            "target_source": "params",
        },
    )

    def fake_create_job(payload):
        captured.update(payload)
        return {
            "job_id": "text2comp-gcam",
            "status": "starting",
            "run_dir": str(tmp_path / "text2comp-run"),
            "error_message": None,
        }

    monkeypatch.setattr(training_manager.text2comp_manager, "create_job", fake_create_job)
    entry = {
        "job_id": "train-gcam",
        "name": "GCAM 简洁训练",
        "simulator": "gcam",
        "gpu_id": 0,
        "log_path": str(tmp_path / "train-gcam.log"),
        "config": {},
        "simple_pipeline": {},
    }

    training_manager._start_simple_text2comp_stage(entry)

    assert captured["expert_model"] == "gcam"
    assert captured["output_dim"] == 18
    assert captured["epochs"] == training_manager.QUICK_TEXT2COMP_DEFAULTS["epochs"]
    assert captured["normalize_labels"] is True
    assert captured["require_quality"] is True
    assert captured["trainable_base_layers"] == 2
    assert captured["head_learning_rate"] == training_manager.QUICK_TEXT2COMP_DEFAULTS["head_learning_rate"]


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
    monkeypatch.setattr(
        training_manager.uploaded_expert_models,
        "list_assembly_models",
        lambda: [
            {
                "model_id": "uploaded-identity",
                "name": "uploaded_identity",
                "status": "active",
                "assembly_enabled": True,
                "input_dim": 4,
                "output_dim": 4,
            }
        ],
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
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")

    entry = training_manager.create_job(_payload())
    jobs = training_manager.list_jobs(refresh=True)

    assert entry["status"] == "queued"
    assert entry["started_at"] is None
    assert jobs[0]["status"] == "queued"
    assert "waiting for PierNet-worker" in Path(entry["log_path"]).read_text(encoding="utf-8")


def test_training_queue_accepts_busy_existing_gpu(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
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
    assert "waiting for PierNet-worker" in Path(entry["log_path"]).read_text(encoding="utf-8")


def test_training_worker_launches_queued_job(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
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
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
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
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
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
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
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
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
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
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
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

    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "0")
    monkeypatch.setattr(training_manager, "ROUTER_DATA_DIR", router_dir)
    monkeypatch.setattr(training_manager, "_refresh_entry", lambda entry: entry)
    monkeypatch.setattr(training_manager.subprocess, "Popen", fake_popen)

    entry = training_manager.create_job(_payload())

    command = captured["command"]
    assert command[command.index("--router-dir") + 1] == str(router_dir)
    assert command.count("--prepare-workers") == 1
    assert command[command.index("--prepare-workers") + 1] == "0"
    assert entry["command"] == command


def test_refresh_entry_tolerates_gpu_lock_release_failure(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    entry = training_manager._queued_training_entry(_payload())
    entry["status"] = "starting"
    entry["pid"] = 99999999
    entry["started_at"] = 2.0
    run_dir = Path(entry["run_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "router_final.pt").write_bytes(b"ok")
    Path(entry["log_path"]).parent.mkdir(parents=True)
    Path(entry["log_path"]).write_text("[done]\n", encoding="utf-8")
    training_manager._save_registry([entry])

    def fail_release(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(training_manager.task_locks, "release_lock", fail_release)

    refreshed = training_manager.get_job(entry["job_id"], refresh=True)

    assert refreshed["status"] == "done"
    assert refreshed["exit_reason"] == "completed"


def test_prepare_simple_text2comp_dataset_uses_training_data_params(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    data_root = tmp_path / "data"
    monkeypatch.setattr(training_manager, "DATA_ROOT", data_root)
    h5_path = data_root / "modflow" / "modflow_coastal_seawater.h5"
    h5_path.parent.mkdir(parents=True)

    import h5py
    import numpy as np

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("params", data=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        h5.create_dataset("timeseries", data=np.zeros((2, 5, 12), dtype=np.float32))
    template_path = data_root / "templates" / "coastal_seawater_templates.jsonl"
    template_path.parent.mkdir(parents=True)
    template_path.write_text("{}\n", encoding="utf-8")
    template = SimpleNamespace(scenario="coastal_seawater", channel_indices=[0], time_indices=[0])
    monkeypatch.setattr(training_manager, "load_templates", lambda path: [template])
    monkeypatch.setattr(
        training_manager,
        "fill_sample",
        lambda template, params, timeseries, sample_index: {
            "input": f"hydraulic parameters: {params.tolist()}",
            "params_transformed": params.tolist(),
        },
    )

    entry = {
        "job_id": "train-data-driven",
        "simulator": "modflow",
        "scenarios": ["coastal_seawater"],
        "config": {"simple_text2comp_max_samples": 2},
    }

    result = training_manager._prepare_simple_text2comp_dataset(entry)

    assert result["output_dim"] == 3
    assert result["target_source"] == "params_transformed"
    rows = [json.loads(line) for line in Path(result["path"]).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["label"] == [1.0, 2.0, 3.0]
    assert rows[0]["prompt"] == "hydraulic parameters: [1.0, 2.0, 3.0]"
    assert "Uploaded Expert" not in rows[0]["prompt"]


def test_simple_pipeline_keeps_gpu_lock_while_text2comp_runs(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    entry = training_manager._queued_training_entry(
        {
            **_payload(),
            "simple_pipeline_enabled": True,
            "uploaded_expert_id": "uploaded-identity",
        }
    )
    entry["status"] = "starting"
    entry["pid"] = 99999999
    entry["started_at"] = 2.0
    run_dir = Path(entry["run_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "router_final.pt").write_bytes(b"ok")
    Path(entry["log_path"]).parent.mkdir(parents=True)
    Path(entry["log_path"]).write_text("[done]\n", encoding="utf-8")
    training_manager._save_registry([entry])

    release_calls = []

    def fake_release(lock_key, owner):
        release_calls.append((lock_key, owner))
        return True

    def fake_start_simple_text2comp_stage(active_entry):
        pipeline = active_entry.setdefault("simple_pipeline", {})
        pipeline.update(
            {
                "stage": "text2comp",
                "text2comp_job_id": "text2comp-child",
                "text2comp_status": "running",
                "uploaded_expert_id": "uploaded-identity",
            }
        )

    monkeypatch.setattr(training_manager.task_locks, "release_lock", fake_release)
    monkeypatch.setattr(training_manager, "_start_simple_text2comp_stage", fake_start_simple_text2comp_stage)
    monkeypatch.setattr(
        training_manager.text2comp_manager,
        "get_job",
        lambda job_id, refresh=True: {"job_id": job_id, "status": "running", "run_dir": str(tmp_path / "text2comp")},
    )

    refreshed = training_manager.get_job(entry["job_id"], refresh=True)

    assert refreshed["status"] == "running"
    assert refreshed["pipeline_stage"] == "text2comp"
    assert release_calls == []


def test_stop_simple_pipeline_text2comp_releases_gpu_lock(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    entry = training_manager._queued_training_entry(
        {
            **_payload(),
            "simple_pipeline_enabled": True,
            "uploaded_expert_id": "uploaded-identity",
        }
    )
    entry["status"] = "running"
    entry["pid"] = None
    entry["started_at"] = 2.0
    entry["simple_pipeline"] = {
        "stage": "text2comp",
        "text2comp_job_id": "text2comp-child",
        "text2comp_status": "running",
        "uploaded_expert_id": "uploaded-identity",
    }
    Path(entry["run_dir"]).mkdir(parents=True)
    Path(entry["log_path"]).parent.mkdir(parents=True)
    Path(entry["log_path"]).write_text("[pipeline] stage=text2comp\n", encoding="utf-8")
    training_manager._save_registry([entry])

    release_calls = []
    stop_calls = []

    monkeypatch.setattr(training_manager, "_refresh_entry", lambda active_entry: active_entry)
    monkeypatch.setattr(
        training_manager.text2comp_manager,
        "get_job",
        lambda job_id, refresh=True: {"job_id": job_id, "status": "running"},
    )
    monkeypatch.setattr(training_manager.text2comp_manager, "stop_job", lambda job_id: stop_calls.append(job_id))
    monkeypatch.setattr(
        training_manager.task_locks,
        "release_lock",
        lambda lock_key, owner: release_calls.append((lock_key, owner)) or True,
    )

    stopped = training_manager.stop_job(entry["job_id"])

    assert stopped["status"] == "terminated"
    assert stopped["simple_pipeline"]["stage"] == "terminated"
    assert stop_calls == ["text2comp-child"]
    assert release_calls == [("gpu:0", entry["job_id"])]


def test_simple_pipeline_releases_gpu_lock_after_text2comp_done(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    entry = training_manager._queued_training_entry(
        {
            **_payload(),
            "simple_pipeline_enabled": True,
            "uploaded_expert_id": "uploaded-identity",
        }
    )
    entry["status"] = "starting"
    entry["pid"] = 99999999
    entry["started_at"] = 2.0
    run_dir = Path(entry["run_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "router_final.pt").write_bytes(b"ok")
    Path(entry["log_path"]).parent.mkdir(parents=True)
    Path(entry["log_path"]).write_text("[done]\n", encoding="utf-8")
    training_manager._save_registry([entry])

    release_calls = []

    def fake_release(lock_key, owner):
        release_calls.append((lock_key, owner))
        return True

    def fake_start_simple_text2comp_stage(active_entry):
        pipeline = active_entry.setdefault("simple_pipeline", {})
        pipeline.update(
            {
                "stage": "text2comp",
                "text2comp_job_id": "text2comp-child",
                "text2comp_status": "done",
                "uploaded_expert_id": "uploaded-identity",
            }
        )

    monkeypatch.setattr(training_manager.task_locks, "release_lock", fake_release)
    monkeypatch.setattr(training_manager, "_start_simple_text2comp_stage", fake_start_simple_text2comp_stage)
    monkeypatch.setattr(
        training_manager.text2comp_manager,
        "get_job",
        lambda job_id, refresh=True: {
            "job_id": job_id,
            "status": "done",
            "run_dir": str(tmp_path / "text2comp"),
            "ended_at": 9.0,
        },
    )

    refreshed = training_manager.get_job(entry["job_id"], refresh=True)

    assert refreshed["status"] == "done"
    assert refreshed["pipeline_stage"] == "done"
    assert release_calls == [("gpu:0", entry["job_id"])]


def test_launch_job_does_not_start_cancelled_queued_entry(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
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


def test_quick_training_requires_selected_scenarios(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")

    with pytest.raises(ValueError, match="select at least one training scenario"):
        training_manager.create_quick_job({"simulator": "modflow", "scenarios": []})


def test_quick_training_job_queues_with_platform_defaults(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")

    entry = training_manager.create_quick_job(
        {"name": "quick-train", "simulator": "modflow", "scenarios": ["coastal_seawater"]}
    )

    assert entry["status"] == "queued"
    assert entry["name"] == "quick-train"
    assert entry["gpu_id"] == 0
    assert entry["scenarios"] == ["coastal_seawater"]
    assert entry["config"]["input_representation"] == "pretrained_embeddings"
    assert entry["config"]["auto_stop_enabled"] is True
    assert entry["config"]["auto_stop_metric"] == "f1"
    assert entry["config"]["auto_stop_threshold"] == 0.98
    assert entry["config"]["auto_stop_min_epochs"] == training_manager.QUICK_TRAINING_DEFAULTS["auto_stop_min_epochs"]
    assert entry["config"]["simple_pipeline_enabled"] is True
    assert entry["config"]["simple_quality_gate_enabled"] is True
    assert entry["config"]["simple_router_min_f1"] == training_manager.QUICK_ROUTER_MIN_F1
    assert entry["config"]["simple_text2comp_epochs"] == training_manager.QUICK_TEXT2COMP_DEFAULTS["epochs"]
    assert entry["config"]["simple_text2comp_normalize_labels"] is True
    assert entry["config"]["simple_text2comp_require_quality"] is True
    assert entry["simple_pipeline"]["stage"] == "router"
    assert "waiting for PierNet-worker" in Path(entry["log_path"]).read_text(encoding="utf-8")


def test_quick_training_job_preserves_coarse_options(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
    validated_resume = {}

    def fake_validate_resume_checkpoint(**kwargs):
        validated_resume.update(kwargs)

    monkeypatch.setattr(training_manager, "_validate_resume_checkpoint", fake_validate_resume_checkpoint)

    entry = training_manager.create_quick_job(
        {
            "name": "quick-options",
            "simulator": "modflow",
            "scenarios": ["coastal_seawater"],
            "gpu_id": 0,
            "resume_from": "/tmp/router_latest.pt",
            "seed": 123,
        }
    )

    assert entry["status"] == "queued"
    assert entry["gpu_id"] == 0
    assert entry["config"]["seed"] == 123
    assert entry["config"]["resume_from"] == "/tmp/router_latest.pt"
    assert entry["config"]["simple_pipeline_enabled"] is True
    assert entry["simple_pipeline"]["uploaded_expert_id"] is None
    assert validated_resume["resume_from"] == "/tmp/router_latest.pt"
    assert validated_resume["scenarios"] == ["coastal_seawater"]


def test_quick_training_job_accepts_uploaded_expert_selection(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")

    entry = training_manager.create_quick_job(
        {
            "name": "quick-uploaded",
            "simulator": "modflow",
            "scenarios": ["coastal_seawater"],
            "uploaded_expert_id": "uploaded-identity",
        }
    )
    payload = training_manager._payload_from_queued_entry(entry)

    assert entry["simple_pipeline"]["uploaded_expert_id"] == "uploaded-identity"
    assert payload["uploaded_expert_id"] == "uploaded-identity"
    assert payload["simple_pipeline_enabled"] is True


def test_quick_training_worker_preserves_custom_router_dataset(monkeypatch, tmp_path):
    _use_tmp_runtime(monkeypatch, tmp_path)
    _mock_training_prereqs(monkeypatch)
    monkeypatch.setenv("PierNet_WORKER_QUEUE_TRAINING", "1")
    router_dir = tmp_path / "new-synth" / "router"
    dataset_id = "router-custom-uploaded"
    monkeypatch.setattr(
        training_manager,
        "resolve_router_dataset",
        lambda requested_id: {
            "dataset_id": requested_id,
            "simulator": "uploaded_expert",
            "scenario": "custom_scenario",
            "root_path": str(router_dir),
        },
    )

    entry = training_manager.create_quick_job({"dataset_id": dataset_id, "gpu_id": 0})
    payload = training_manager._payload_from_queued_entry(entry)

    def fail_builtin_validation(*_args, **_kwargs):
        raise AssertionError("custom datasets must not use the built-in simulator whitelist")

    monkeypatch.setattr(training_manager, "_validate_scenarios", fail_builtin_validation)
    launched = {}

    def fake_launch(queued_payload):
        launched["source"] = training_manager._resolve_router_training_source(queued_payload)
        return {"job_id": queued_payload["_job_id"], "status": "starting"}

    monkeypatch.setattr(training_manager, "_launch_job", fake_launch)

    assert payload["dataset_id"] == dataset_id
    assert training_worker_queue.run_next_queued_job(worker_id="worker-test") is True
    assert launched["source"] == (
        dataset_id,
        "uploaded_expert",
        ["custom_scenario"],
        router_dir,
    )
