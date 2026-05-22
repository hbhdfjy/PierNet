import asyncio

from piern.shared.tasks import locks as task_locks
from piern.synth.services import job_manager, job_store


def _use_tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(job_store, "JOB_STORE_PATH", tmp_path / "jobs.sqlite")
    monkeypatch.setattr(job_store, "_INITIALIZED", False)
    monkeypatch.setattr(task_locks, "LOCK_STORE_PATH", tmp_path / "job_locks.sqlite")
    monkeypatch.setattr(task_locks, "_INITIALIZED", False)
    job_manager._jobs.clear()
    job_store.init_store()
    task_locks.init_store()


def test_job_store_persists_job_events(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)

    job_store.upsert_job(
        job_id="fill-test",
        job_type="fill_samples",
        status="running",
        started_at=10.0,
        scenario_totals={"a": 2},
        progress={},
        stats={"elapsed_sec": 0.0, "samples_per_sec": 0.0},
    )
    job_store.append_event("fill-test", {"type": "log", "ts": 11.0, "line": "hello"})

    loaded = job_store.load_job("fill-test")

    assert loaded is not None
    assert loaded["job_type"] == "fill_samples"
    assert loaded["scenario_totals"] == {"a": 2}
    assert loaded["events"] == [{"type": "log", "ts": 11.0, "line": "hello"}]


def test_job_store_can_load_status_without_full_event_history(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)

    job_store.upsert_job(
        job_id="fill-test",
        job_type="fill_samples",
        status="running",
        started_at=10.0,
        scenario_totals={"a": 3},
        progress={},
        stats={"elapsed_sec": 0.0, "samples_per_sec": 0.0},
    )
    for idx in range(3):
        job_store.append_event("fill-test", {"type": "log", "ts": 11.0 + idx, "line": f"line-{idx}"})

    status_only = job_store.load_job("fill-test", include_events=False)
    last_id, first_batch = job_store.load_events_after("fill-test", 0, limit=2)
    last_id, second_batch = job_store.load_events_after("fill-test", last_id, limit=2)

    assert status_only is not None
    assert status_only["events"] == []
    assert [event["line"] for event in first_batch] == ["line-0", "line-1"]
    assert [event["line"] for event in second_batch] == ["line-2"]
    assert last_id > 0


def test_incomplete_jobs_recover_as_external_terminated(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    job_store.upsert_job(
        job_id="fill-orphan",
        job_type="fill_samples",
        status="running",
        started_at=10.0,
        scenario_totals={},
        progress={},
        stats={},
    )

    updated = job_store.mark_incomplete_external_terminated()
    loaded = job_store.load_job("fill-orphan")

    assert updated == ["fill-orphan"]
    assert loaded is not None
    assert loaded["status"] == "external_terminated"
    assert loaded["error_message"]
    assert loaded["events"][-1]["type"] == "external_terminated"


def test_job_manager_rehydrates_finished_job_from_store(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)

    async def run_job():
        record = job_manager.create_job("fill_samples", {"coastal": 3})
        job_manager.publish(
            record,
            {
                "type": "log",
                "ts": 11.0,
                "line": "  coastal: 1/3",
                "progress": {"scenario": "coastal", "done": 1, "total": 3},
            },
        )
        job_manager.publish(record, {"type": "done", "ts": 12.0, "message": "ok"})
        return record.job_id

    job_id = asyncio.run(run_job())
    job_manager._jobs.clear()

    restored = job_manager.get_job(job_id)

    assert restored is not None
    assert restored.persisted is True
    assert restored.status == "done"
    assert restored.progress["coastal"] == {"scenario": "coastal", "done": 3, "total": 3}
    assert restored.events[-1]["type"] == "done"


def test_rehydrated_terminal_job_releases_locks(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)

    async def run_job():
        record = job_manager.create_job("fill_samples", {"coastal": 1}, lock_keys=["dataset:coastal"])
        return record.job_id

    job_id = asyncio.run(run_job())
    locks = task_locks.list_locks()
    assert len(locks) == 1
    assert locks[0]["lock_key"] == "dataset:coastal"
    assert locks[0]["owner"] == job_id
    assert locks[0]["metadata"] == {"job_type": "fill_samples"}
    job_manager._jobs.clear()

    restored = job_manager.get_job(job_id)
    assert restored is not None
    assert restored.lock_keys == []

    job_manager.publish(restored, {"type": "done", "ts": 12.0, "message": "ok"})

    assert task_locks.list_locks() == []



def test_queued_jobs_survive_api_process_recovery(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    job_store.upsert_job(
        job_id="fill-queued",
        job_type="fill_samples",
        status="queued",
        started_at=10.0,
        scenario_totals={"coastal": 2},
        progress={},
        stats={},
    )

    updated = job_store.mark_incomplete_external_terminated()
    loaded = job_store.load_job("fill-queued")

    assert updated == []
    assert loaded is not None
    assert loaded["status"] == "queued"
