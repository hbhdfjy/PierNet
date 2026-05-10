from fastapi.testclient import TestClient

from piern.api.main import app


def test_http_errors_are_structured():
    client = TestClient(app)

    response = client.get("/api/generate/missing-job/status")
    payload = response.json()

    assert response.status_code == 404
    assert payload["code"] == "NOT_FOUND"
    assert "任务 missing-job 不存在" in payload["message"]
    assert payload["request_id"]
    assert response.headers["x-request-id"] == payload["request_id"]


def test_validation_errors_are_structured():
    client = TestClient(app)

    response = client.get("/api/samples", params={"page": "bad"})
    payload = response.json()

    assert response.status_code == 422
    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["message"] == "请求参数校验失败"
    assert payload["details"]["errors"]


def test_request_id_header_is_preserved_on_success():
    client = TestClient(app)

    response = client.get("/api/health/live", headers={"x-request-id": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "test-request-id"
