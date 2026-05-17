from fastapi import FastAPI
from fastapi.testclient import TestClient

from piern.shared.api.security import install_security, token_allowed


def test_token_allowed_uses_constant_time_compare():
    assert token_allowed("secret", expected="secret")
    assert not token_allowed("bad", expected="secret")


def test_optional_api_token_middleware_blocks_mutations_when_configured(monkeypatch):
    monkeypatch.setenv("PIERN_AUTH_TOKEN", "secret-token")
    app = FastAPI()
    install_security(app)

    @app.post("/api/demo")
    def mutate():
        return {"ok": True}

    client = TestClient(app)
    assert client.post("/api/demo").status_code == 401
    assert client.post("/api/demo", headers={"authorization": "Bearer secret-token"}).status_code == 200
