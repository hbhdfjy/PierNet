from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from PierNet.shared.api.static import SPAStaticFiles


def make_static_client(tmp_path: Path) -> TestClient:
    (tmp_path / "index.html").write_text("<html><body>PierNet Shell</body></html>", encoding="utf-8")
    app = FastAPI()
    app.mount("/", SPAStaticFiles(directory=str(tmp_path), html=True), name="static")
    return TestClient(app)


def test_spa_fallback_serves_browser_routes(tmp_path: Path) -> None:
    client = make_static_client(tmp_path)

    response = client.get("/synth/fill")

    assert response.status_code == 200
    assert "PierNet Shell" in response.text
    assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"


def test_spa_fallback_does_not_swallow_api_root(tmp_path: Path) -> None:
    client = make_static_client(tmp_path)

    assert client.get("/api").status_code == 404
    assert client.get("/api/").status_code == 404


def test_spa_fallback_does_not_swallow_missing_assets(tmp_path: Path) -> None:
    client = make_static_client(tmp_path)

    assert client.get("/assets/missing.js").status_code == 404
    assert client.get("/favicon.ico").status_code == 404
