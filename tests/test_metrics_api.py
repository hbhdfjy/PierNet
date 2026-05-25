from fastapi import FastAPI
from fastapi.testclient import TestClient

from PierNet.shared.api.routers import metrics
from PierNet.shared.tasks import workers
from PierNet.synth.services import job_store


def _use_tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(job_store, "JOB_STORE_PATH", tmp_path / "jobs.sqlite")
    monkeypatch.setattr(job_store, "_INITIALIZED", False)
    monkeypatch.setattr(workers, "WORKER_STORE_PATH", tmp_path / "workers.sqlite")
    monkeypatch.setattr(workers, "_INITIALIZED", False)
    job_store.init_store()


def test_metrics_summary_reports_queues_workers_and_resources(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr(metrics.training_manager, "list_jobs", lambda refresh=True: [{"job_id": "train-1", "status": "running", "created_at": 10.0}])
    monkeypatch.setattr(metrics.training_manager, "get_gpu_inventory", lambda: [{"index": 0, "available": True}])
    monkeypatch.setattr(metrics, "_disk_metrics", lambda: {"path": str(tmp_path), "total_bytes": 100, "used_bytes": 40, "free_bytes": 60, "free_ratio": 0.6})

    job_store.upsert_job(
        job_id="fill-1",
        job_type="fill_samples",
        status="queued",
        started_at=10.0,
        scenario_totals={},
        progress={},
        stats={},
    )
    workers.upsert_worker(worker_id="worker-1", kind="PierNet-worker", current_job_id="fill-1")

    app = FastAPI()
    app.include_router(metrics.router, prefix="/api")
    client = TestClient(app)

    response = client.get("/api/metrics/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["jobs"]["synth"]["queued"] == 1
    assert payload["jobs"]["training"]["active"] == 1
    assert payload["queues"]["synth"]["queued"] == 1
    assert payload["workers"]["running"] == 1
    assert payload["workers"]["busy"] == 1
    assert payload["resources"]["disk"]["free_bytes"] >= 0
    assert payload["resources"]["gpus"][0]["index"] == 0
