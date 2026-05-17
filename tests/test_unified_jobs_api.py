from fastapi import FastAPI
from fastapi.testclient import TestClient

from piern.shared.api.routers import jobs as jobs_router
from piern.synth.services import job_manager, job_store


def _use_tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(job_store, "JOB_STORE_PATH", tmp_path / "jobs.sqlite")
    monkeypatch.setattr(job_store, "_INITIALIZED", False)
    job_manager._jobs.clear()
    job_store.init_store()


def test_audit_route_is_not_shadowed_by_job_id(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_router.audit_store, "AUDIT_STORE_PATH", tmp_path / "audit.sqlite")
    monkeypatch.setattr(jobs_router.audit_store, "_INITIALIZED", False)
    monkeypatch.setattr(jobs_router.training_manager, "list_jobs", lambda refresh=True: [])

    app = FastAPI()
    app.include_router(jobs_router.router, prefix="/api")
    client = TestClient(app)

    response = client.get("/api/jobs/audit/events")

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_unified_jobs_lists_queued_synthesis_jobs(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr(jobs_router.training_manager, "list_jobs", lambda refresh=True: [])
    job_store.upsert_job(
        job_id="fill-queued",
        job_type="fill_samples",
        status="queued",
        started_at=10.0,
        scenario_totals={"coastal": 2},
        progress={},
        stats={},
    )

    app = FastAPI()
    app.include_router(jobs_router.router, prefix="/api")
    client = TestClient(app)

    response = client.get("/api/jobs", params={"platform": "synth"})

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["job_id"] == "fill-queued"
    assert payload[0]["status"] == "queued"
    assert payload[0]["platform"] == "synth"
