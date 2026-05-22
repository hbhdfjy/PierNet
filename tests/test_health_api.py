from fastapi.testclient import TestClient

from piern.api.main import app


def test_health_live_and_storage_endpoints():
    client = TestClient(app)

    live = client.get("/api/health/live")
    storage = client.get("/api/health/storage")

    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert storage.status_code == 200
    assert storage.json()["free_bytes"] >= 0


def test_health_ready_reports_core_paths():
    client = TestClient(app)

    response = client.get("/api/health/ready")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] in {"ok", "degraded"}
    assert "project_root" in payload["checks"]
    assert payload["checks"]["project_root"]["writable_checked"] is False
    assert payload["checks"]["data_root"]["writable_checked"] is True
    assert "runlog_root" in payload["checks"]
    assert "runtime_config" in payload["checks"]
