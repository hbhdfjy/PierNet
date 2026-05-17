import asyncio

from piern.synth.services import job_manager, job_store


def _use_tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(job_store, "JOB_STORE_PATH", tmp_path / "jobs.sqlite")
    monkeypatch.setattr(job_store, "_INITIALIZED", False)
    job_manager._jobs.clear()
    job_store.init_store()


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
