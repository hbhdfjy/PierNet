from piern.shared.tasks import workers


def test_worker_heartbeat_registry_marks_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(workers, "WORKER_STORE_PATH", tmp_path / "workers.sqlite")
    monkeypatch.setattr(workers, "_INITIALIZED", False)

    worker_id = workers.upsert_worker(worker_id="worker-test", kind="piern-worker")

    items = workers.list_workers(stale_after_seconds=0)
    assert worker_id == "worker-test"
    assert items[0]["worker_id"] == "worker-test"
    assert items[0]["status"] in {"running", "stale"}

    workers.mark_worker_stopped("worker-test")
    assert workers.list_workers()[0]["status"] == "stopped"
