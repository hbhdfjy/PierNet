from piern.training.services import job_store


def _use_tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(job_store, "TRAINING_JOB_STORE_PATH", tmp_path / "training_jobs.sqlite")
    monkeypatch.setattr(job_store, "_INITIALIZED", False)
    job_store.init_store()


def test_training_job_store_records_status_transitions(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)

    entry = {
        "job_id": "train-test",
        "status": "starting",
        "simulator": "modflow",
        "gpu_id": 0,
        "pid": 123,
        "created_at": 10.0,
        "started_at": 11.0,
        "ended_at": None,
        "config": {"epochs": 1},
        "command": ["python"],
        "scenarios": ["a"],
        "error_message": None,
    }
    job_store.upsert_job(entry)
    job_store.upsert_job({**entry, "status": "running"})

    snapshots = job_store.list_job_snapshots()
    events = job_store.list_events("train-test")

    assert snapshots[0]["job_id"] == "train-test"
    assert snapshots[0]["status"] == "running"
    assert [event["event_type"] for event in events] == ["created", "status_changed"]
    assert events[-1]["payload"]["from"] == "starting"
    assert events[-1]["payload"]["to"] == "running"


def test_training_job_store_deleted_jobs_are_hidden_by_default(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)

    entry = {
        "job_id": "train-delete",
        "status": "done",
        "created_at": 10.0,
        "config": {},
        "command": [],
        "scenarios": [],
    }
    job_store.upsert_job(entry)
    job_store.mark_deleted("train-delete")

    assert job_store.list_job_snapshots() == []
    assert job_store.list_job_snapshots(include_deleted=True)[0]["job_id"] == "train-delete"
    assert job_store.list_events("train-delete")[-1]["event_type"] == "deleted"
