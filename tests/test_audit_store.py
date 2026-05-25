from fastapi import FastAPI
from fastapi.testclient import TestClient

from PierNet.shared.api.audit import install_audit
from PierNet.shared.audit import store as audit_store


def _use_tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(audit_store, "AUDIT_STORE_PATH", tmp_path / "audit.sqlite")
    monkeypatch.setattr(audit_store, "_INITIALIZED", False)
    audit_store.init_store()


def test_audit_middleware_records_mutating_api_calls(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    app = FastAPI()
    install_audit(app)

    @app.post("/api/demo")
    def mutate():
        return {"ok": True}

    client = TestClient(app)
    response = client.post("/api/demo")

    events = audit_store.list_events(limit=10)
    assert response.status_code == 200
    assert len(events) == 1
    assert events[0]["actor"] == "anonymous"
    assert events[0]["action"] == "POST /api/demo"
    assert events[0]["target"] == "demo"
    assert events[0]["status_code"] == 200
