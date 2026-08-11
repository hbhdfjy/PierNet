#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import requests


def _fixture_one(root: Path) -> tuple[Path, Path, list[int], list[int]]:
    rng = np.random.default_rng(1201)
    inputs = rng.uniform(-1.5, 1.5, size=(48, 3)).astype(np.float32)
    t = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    outputs = (
        inputs[:, 0, None] * np.sin(t * np.pi)
        + inputs[:, 1, None] * t
        + inputs[:, 2, None]
    )
    data_path = root / "thermal-response.npz"
    np.savez_compressed(
        data_path,
        inputs=inputs,
        outputs=outputs,
        input_names=np.asarray(["amplitude", "drift", "baseline"]),
        output_names=np.asarray([f"state_{index + 1}" for index in range(8)]),
    )
    expert_path = root / "thermal_expert.py"
    expert_path.write_text(
        """import numpy as np


def predict(inputs):
    values = np.asarray(inputs, dtype=np.float32)
    t = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    return (
        values[:, 0, None] * np.sin(t * np.pi)
        + values[:, 1, None] * t
        + values[:, 2, None]
    )
""",
        encoding="utf-8",
    )
    return data_path, expert_path, [3], [8]


def _fixture_two(root: Path) -> tuple[Path, Path, list[int], list[int]]:
    rng = np.random.default_rng(2402)
    inputs = rng.uniform(0.2, 2.0, size=(40, 5)).astype(np.float32)
    x = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    grid = x[:, None] + y[None, :]
    outputs = (
        inputs[:, 0, None, None] * grid
        + inputs[:, 1, None, None] * x[:, None]
        + inputs[:, 2, None, None] * y[None, :]
        + inputs[:, 3, None, None]
        - inputs[:, 4, None, None]
    )
    data_path = root / "field-response.npz"
    np.savez_compressed(
        data_path,
        inputs=inputs,
        outputs=outputs,
        input_names=np.asarray(["scale", "x_bias", "y_bias", "source", "loss"]),
        output_names=np.asarray([f"cell_{index + 1}" for index in range(24)]),
    )
    expert_path = root / "field_expert.py"
    expert_path.write_text(
        """import numpy as np


def predict(inputs):
    values = np.asarray(inputs, dtype=np.float32)
    x = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    grid = x[:, None] + y[None, :]
    return (
        values[:, 0, None, None] * grid
        + values[:, 1, None, None] * x[:, None]
        + values[:, 2, None, None] * y[None, :]
        + values[:, 3, None, None]
        - values[:, 4, None, None]
    )
""",
        encoding="utf-8",
    )
    return data_path, expert_path, [5], [4, 6]


def _request(session: requests.Session, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = session.request(method, url, timeout=180, **kwargs)
    if not response.ok:
        raise RuntimeError(f"{method} {url} -> {response.status_code}: {response.text}")
    return response.json()


def _upload(
    session: requests.Session,
    base_url: str,
    project_id: str,
    kind: str,
    path: Path,
) -> dict[str, Any]:
    return _request(
        session,
        "POST",
        f"{base_url}/projects/{project_id}/{kind}",
        data=path.read_bytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-File-Name": quote(path.name),
        },
    )


def _wait_ready(
    session: requests.Session,
    base_url: str,
    project_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_message = ""
    while time.monotonic() < deadline:
        project = _request(session, "GET", f"{base_url}/projects/{project_id}")
        stage = next(
            (item for item in project["stages"] if item["id"] == project["current_stage"]),
            None,
        )
        message = stage["message"] if stage else project["status"]
        if message != last_message:
            print(f"{project_id}: {message}", flush=True)
            last_message = message
        if project["status"] == "ready":
            return project
        if project["status"] in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(project.get("error"), ensure_ascii=False))
        time.sleep(1.5)
    raise TimeoutError(f"{project_id} did not finish in {timeout_seconds}s")


def _run_fixture(
    session: requests.Session,
    base_url: str,
    fixture: tuple[Path, Path, list[int], list[int]],
    *,
    name: str,
    goal: str,
    train: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    data_path, expert_path, input_shape, output_shape = fixture
    project = _request(
        session,
        "POST",
        f"{base_url}/projects",
        json={"name": name, "goal": goal},
    )
    project_id = project["project_id"]
    _upload(session, base_url, project_id, "data", data_path)
    _upload(session, base_url, project_id, "expert", expert_path)
    checked = _request(
        session,
        "POST",
        f"{base_url}/projects/{project_id}/compatibility-check",
    )
    assert checked["compatibility"]["compatible"] is True
    assert checked["data"]["input_shape"] == input_shape
    assert checked["data"]["output_shape"] == output_shape
    result: dict[str, Any] = {
        "project_id": project_id,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "compatibility": checked["compatibility"],
    }
    if not train:
        return result
    _request(session, "POST", f"{base_url}/projects/{project_id}/run")
    ready = _wait_ready(session, base_url, project_id, timeout_seconds)
    chat = _request(
        session,
        "POST",
        f"{base_url}/projects/{project_id}/chat",
        json={"message": ready["recommended_prompt"]},
    )
    output = np.asarray(chat["output"])
    assert list(output.shape) == output_shape
    assert np.isfinite(output).all()
    result.update(
        {
            "status": ready["status"],
            "recommended_prompt": ready["recommended_prompt"],
            "router_confidence": chat["confidence"],
            "latency_ms": chat["latency_ms"],
            "output_preview": output.reshape(-1)[:8].astype(float).tolist(),
            "metrics": ready["artifacts"]["metrics"],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/studio")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    session = requests.Session()
    session_payload = _request(session, "POST", f"{args.base_url}/session")
    with tempfile.TemporaryDirectory(prefix="piern-studio-e2e-") as temp_name:
        root = Path(temp_name)
        results = [
            _run_fixture(
                session,
                args.base_url,
                _fixture_one(root),
                name="热响应曲线演示",
                goal="根据三个用户输入参数预测八个时刻的系统状态",
                train=args.train,
                timeout_seconds=args.timeout,
            ),
            _run_fixture(
                session,
                args.base_url,
                _fixture_two(root),
                name="二维场预测演示",
                goal="根据五个用户输入参数预测四行六列的二维场",
                train=args.train,
                timeout_seconds=args.timeout,
            ),
        ]
    summary = {
        "session_id": session_payload["session_id"],
        "trained": args.train,
        "projects": results,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
