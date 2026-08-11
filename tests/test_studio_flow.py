from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from PierNet.studio import paths, service, store
from PierNet.studio.api.router import router
from PierNet.studio.data_io import DataInspectionError, canonicalize_data, discover_data_file
from PierNet.studio.expert import check_compatibility, prepare_expert_package
from PierNet.studio.training import (
    extract_named_values,
    merge_text2comp_inputs,
    prepare_training_data,
)


@pytest.mark.parametrize(
    ("input_dim", "output_shape", "body"),
    [
        (
            3,
            (8,),
            """
import numpy as np

def predict(inputs):
    t = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    return inputs[:, 0, None] * np.sin(t) + inputs[:, 1, None] * t + inputs[:, 2, None]
""",
        ),
        (
            5,
            (4, 6),
            """
import numpy as np

def predict(inputs):
    x = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    grid = x[:, None] + y[None, :]
    return (
        inputs[:, 0, None, None] * grid
        + inputs[:, 1, None, None] * x[:, None]
        + inputs[:, 2, None, None] * y[None, :]
        + inputs[:, 3, None, None]
        - inputs[:, 4, None, None]
    )
""",
        ),
    ],
)
def test_user_data_and_expert_are_compatible(
    tmp_path: Path,
    input_dim: int,
    output_shape: tuple[int, ...],
    body: str,
) -> None:
    rng = np.random.default_rng(42 + input_dim)
    inputs = rng.normal(size=(12, input_dim)).astype(np.float32)
    expert_source = tmp_path / "expert.py"
    expert_source.write_text(body.strip() + "\n", encoding="utf-8")
    expert = prepare_expert_package(expert_source, tmp_path / "expert-package")

    module_globals: dict[str, object] = {}
    exec(body, module_globals)
    outputs = module_globals["predict"](inputs)
    assert outputs.shape == (12, *output_shape)

    source = tmp_path / "user-data.npz"
    np.savez_compressed(
        source,
        inputs=inputs,
        outputs=outputs,
        input_names=np.asarray([f"parameter_{index + 1}" for index in range(input_dim)]),
        output_names=np.asarray([f"result_{index + 1}" for index in range(int(np.prod(output_shape)))]),
    )
    metadata = canonicalize_data(source, tmp_path / "canonical" / "data.npz")
    report = check_compatibility(
        Path(metadata["canonical_path"]),
        expert,
        work_dir=tmp_path / "logs",
    )

    assert metadata["input_shape"] == [input_dim]
    assert metadata["output_shape"] == list(output_shape)
    assert report["compatible"] is True
    assert report["sample_mse"] < 1e-12

    prepared = prepare_training_data(
        Path(metadata["canonical_path"]),
        tmp_path / "training",
        goal="预测用户上传系统的下一状态",
    )
    assert prepared["prompt_count"] >= 12
    assert "parameter_1=" in prepared["recommended_prompt"]


def test_named_values_report_missing_fields() -> None:
    values, missing = extract_named_values(
        "请计算 temperature=18.5，pressure: 2e-3",
        ["temperature", "pressure", "duration"],
    )
    assert values.tolist() == pytest.approx([18.5, 0.002, 0.0])
    assert missing == ["duration"]


def test_text2comp_only_generates_missing_values() -> None:
    parsed = np.asarray([2.0, 0.0, 0.5], dtype=np.float32)
    generated = np.asarray([1.99, 1.25, 0.51], dtype=np.float32)

    resolved = merge_text2comp_inputs(
        parsed,
        generated,
        ["elasticity", "load", "damping"],
        ["load"],
    )
    explicit = merge_text2comp_inputs(
        parsed,
        generated,
        ["elasticity", "load", "damping"],
        [],
    )

    assert resolved.tolist() == pytest.approx([2.0, 1.25, 0.5])
    assert explicit.tolist() == pytest.approx([2.0, 0.0, 0.5])


def test_data_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr("../outside.npz", b"not-an-npz")

    with pytest.raises(DataInspectionError, match="不安全路径"):
        discover_data_file(archive, tmp_path / "extracted")


def test_expert_manifest_is_generated_for_single_python_file(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text(
        "def predict(inputs):\n    return inputs\n",
        encoding="utf-8",
    )
    expert = prepare_expert_package(source, tmp_path / "package")
    manifest = json.loads(Path(expert["manifest_path"]).read_text(encoding="utf-8"))

    assert manifest == {
        "runtime": "python",
        "entrypoint": "model.py",
        "callable": "predict",
        "batch_mode": "auto",
    }


def test_multifile_expert_discovers_predict_without_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "expert.zip"
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr("package/helper.py", "OFFSET = 3.0\n")
        payload.writestr(
            "package/model.py",
            "from helper import OFFSET\ndef predict(inputs):\n    return inputs * 2 + OFFSET\n",
        )
    expert = prepare_expert_package(archive, tmp_path / "package")
    inputs = np.arange(12, dtype=np.float32).reshape(4, 3)
    source = tmp_path / "data.npz"
    np.savez_compressed(source, inputs=inputs, outputs=inputs * 2 + 3)
    metadata = canonicalize_data(source, tmp_path / "canonical.npz")

    report = check_compatibility(
        Path(metadata["canonical_path"]),
        expert,
        work_dir=tmp_path / "logs",
    )

    assert expert["entrypoint"] == "model.py"
    assert report["compatible"] is True


def test_single_sample_expert_is_adapted_to_batch_calls(tmp_path: Path) -> None:
    source = tmp_path / "single.py"
    source.write_text(
        "import numpy as np\n"
        "def predict(sample):\n"
        "    if sample.ndim != 1:\n"
        "        raise ValueError('single sample only')\n"
        "    return np.asarray([sample.sum(), sample.mean()], dtype=np.float32)\n",
        encoding="utf-8",
    )
    expert = prepare_expert_package(source, tmp_path / "expert")
    inputs = np.arange(15, dtype=np.float32).reshape(5, 3)
    outputs = np.stack([inputs.sum(axis=1), inputs.mean(axis=1)], axis=1)
    data = tmp_path / "data.npz"
    np.savez_compressed(data, inputs=inputs, outputs=outputs)
    metadata = canonicalize_data(data, tmp_path / "canonical.npz")

    report = check_compatibility(
        Path(metadata["canonical_path"]),
        expert,
        work_dir=tmp_path / "logs",
    )

    assert report["compatible"] is True
    assert expert["batch_mode"] == "per_sample"


def test_api_uses_raw_uploads_and_isolates_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(paths, "STUDIO_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(paths, "STUDIO_ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "STUDIO_RUNLOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(paths, "STUDIO_DB_PATH", tmp_path / "logs" / "projects.sqlite")
    monkeypatch.setattr(store, "STUDIO_DB_PATH", tmp_path / "logs" / "projects.sqlite")
    store._INITIALIZED = False
    service.initialize()

    inputs = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    outputs = inputs[:, :1] + inputs[:, 1:2] * 2
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        inputs=inputs,
        outputs=outputs,
        input_names=np.asarray(["a", "b", "c"]),
        output_names=np.asarray(["value"]),
    )
    expert = b"def predict(inputs):\n    return inputs[:, :1] + inputs[:, 1:2] * 2\n"

    app = FastAPI()
    app.include_router(router, prefix="/api")
    owner = TestClient(app)
    stranger = TestClient(app)
    owner_session = owner.post("/api/studio/session")
    assert owner_session.status_code == 200
    owner_id = owner_session.json()["session_id"]
    assert stranger.post("/api/studio/session").status_code == 200
    presets = owner.get("/api/studio/presets")
    assert presets.status_code == 200
    assert presets.json()["expert"]["call"] == "predict(inputs) -> outputs"
    assert "existing_models" not in presets.json()
    created = owner.post(
        "/api/studio/projects",
        json={"name": "隔离测试", "goal": "根据三个参数计算一个输出"},
    )
    assert created.status_code == 200
    project_id = created.json()["project_id"]

    data_response = owner.post(
        f"/api/studio/projects/{project_id}/data",
        content=buffer.getvalue(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-File-Name": "user-data.npz",
        },
    )
    assert data_response.status_code == 200
    expert_response = owner.post(
        f"/api/studio/projects/{project_id}/expert",
        content=expert,
        headers={
            "Content-Type": "application/octet-stream",
            "X-File-Name": "user-expert.py",
        },
    )
    assert expert_response.status_code == 200
    inspected = owner.post(f"/api/studio/projects/{project_id}/inspect")
    assert inspected.status_code == 200
    assert inspected.json()["stages"][1]["status"] == "succeeded"
    checked = owner.post(f"/api/studio/projects/{project_id}/compatibility-check")
    assert checked.status_code == 200
    assert checked.json()["compatibility"]["compatible"] is True

    store.update_project(project_id, status="ready", current_stage="validation")
    reinspected = owner.post(f"/api/studio/projects/{project_id}/inspect")
    assert reinspected.status_code == 200
    assert reinspected.json()["current_stage"] == "validation"

    assert stranger.get(f"/api/studio/projects/{project_id}").status_code == 404
    assert stranger.delete(f"/api/studio/projects/{project_id}").status_code == 404
    store.update_project(project_id, status="running")
    assert owner.delete(f"/api/studio/projects/{project_id}").status_code == 409
    store.update_project(project_id, status="ready")

    project_storage = paths.project_paths(project_id, create=False)
    assert project_storage.data_root.exists()
    assert project_storage.artifact_root.exists()
    assert project_storage.logs.exists()
    deleted = owner.delete(f"/api/studio/projects/{project_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert owner.get(f"/api/studio/projects/{project_id}").status_code == 404
    assert not project_storage.data_root.exists()
    assert not project_storage.artifact_root.exists()
    assert not project_storage.logs.exists()
    actions = {event["action"] for event in store.list_audit_events(owner_id, project_id=project_id)}
    assert {
        "project_created",
        "data_uploaded",
        "expert_uploaded",
        "compatibility_finished",
        "project_deleted",
    } <= actions
    store._INITIALIZED = False


def test_run_guards_enforce_quota_storage_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AliveThread:
        @staticmethod
        def is_alive() -> bool:
            return True

    service._THREADS.clear()
    service._THREADS["studio-other"] = AliveThread()  # type: ignore[assignment]
    monkeypatch.setattr(service, "MAX_CONCURRENT_RUNS", 1)
    with pytest.raises(service.StudioError, match="正在构建"):
        service._enforce_run_capacity("studio-new")
    service._THREADS.clear()

    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(
        service,
        "project_paths",
        lambda _project_id: SimpleNamespace(data_root=data_root),
    )
    monkeypatch.setattr(service, "MIN_FREE_DISK_BYTES", 1024)
    monkeypatch.setattr(
        service.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=512),
    )
    with pytest.raises(service.StudioError, match="存储空间不足"):
        service._preflight_run("studio-new")

    monkeypatch.setattr(service.time, "monotonic", lambda: 10.0)
    with pytest.raises(TimeoutError, match="运行时长上限"):
        service._run_should_cancel("studio-new", 9.0)


def test_manifest_must_belong_to_current_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembly = tmp_path / "assembly"
    assembly.mkdir()
    manifest_path = assembly / "manifest.json"
    manifest_path.write_text(
        json.dumps({"project_id": "studio-other"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        service,
        "project_paths",
        lambda _project_id: SimpleNamespace(assembly=assembly),
    )

    with pytest.raises(service.StudioError, match="项目不一致"):
        service._load_project_manifest(
            "studio-current",
            {"manifest_path": str(manifest_path)},
        )


def test_running_project_recovers_as_retryable_after_service_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(paths, "STUDIO_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(paths, "STUDIO_ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "STUDIO_RUNLOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(paths, "STUDIO_DB_PATH", tmp_path / "logs" / "projects.sqlite")
    monkeypatch.setattr(store, "STUDIO_DB_PATH", tmp_path / "logs" / "projects.sqlite")
    store._INITIALIZED = False
    service.initialize()
    project = store.create_project("owner", "恢复测试", "验证任务重启恢复")
    project_id = project["project_id"]
    store.update_stage(
        project_id,
        "training",
        status="running",
        progress=0.35,
        message="正在训练",
    )
    store.update_project(project_id, status="running")

    store._INITIALIZED = False
    service.initialize()
    recovered = store.get_project("owner", project_id)

    assert recovered["status"] == "failed"
    assert recovered["error"]["code"] == "service_restarted"
    assert recovered["stages"]["training"]["retryable"] is True
    store._INITIALIZED = False
