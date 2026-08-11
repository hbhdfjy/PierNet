import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from PierNet.training.services import assembly_registration
from PierNet.training.api.routers import training


def _completed_job(tmp_path: Path) -> dict:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "router_final.pt").write_bytes(b"router")
    text2comp = tmp_path / "text2comp.pt"
    text2comp.write_bytes(b"text2comp")
    return {
        "job_id": "train-test",
        "name": "conversation test",
        "status": "done",
        "pipeline_stage": "done",
        "simulator": "modflow",
        "run_dir": str(run_dir),
        "text2comp_model_path": str(text2comp),
        "text2comp_output_dim": 2,
        "config": {"simple_pipeline_enabled": True},
        "router_metrics": {"f1": 0.99},
        "text2comp_metrics": {"normalized_rmse": 0.1},
    }


def _expert_root(tmp_path: Path) -> Path:
    root = tmp_path / "experts" / "modflow"
    root.mkdir(parents=True)
    (root / "expert_model.pt").write_bytes(b"expert")
    (root / "manifest.json").write_text(
        yaml.safe_dump({"param_mean": [0.0, 0.0], "target_shape": [1, 3]}),
        encoding="utf-8",
    )
    return root.parent


def test_register_training_job_writes_runtime_profile(monkeypatch, tmp_path: Path) -> None:
    llm = tmp_path / "llm"
    llm.mkdir()
    store = tmp_path / "profiles.yaml"
    monkeypatch.setattr(assembly_registration, "DEFAULT_LLM_PATH", llm)
    monkeypatch.setattr(assembly_registration, "BUILTIN_EXPERT_ROOT", _expert_root(tmp_path))
    monkeypatch.setattr(assembly_registration, "PROFILE_STORE_PATH", store)

    first = assembly_registration.register_training_job(_completed_job(tmp_path))
    second = assembly_registration.register_training_job(_completed_job(tmp_path))

    assert first["model_id"] == "conversation_train-test"
    assert first["executor"] == "training_job_profile"
    assert first["expert_kind"] == "modflow_dnn"
    assert first["source"] == "simple_training"
    assert first["task_label"] == "地下水流动与水头预测"
    assert first["min_user_numeric_values"] == 2
    assert "[预测结果]" not in first["system_prompt"]
    assert "至少 2 个数值参数" in first["system_prompt"]
    assert second["model_id"] == first["model_id"]
    saved = yaml.safe_load(store.read_text(encoding="utf-8"))["profiles"]
    assert len(saved) == 1
    assert saved[0]["router_path"].endswith("router_final.pt")


def test_register_training_job_rejects_incomplete_pipeline(monkeypatch, tmp_path: Path) -> None:
    job = _completed_job(tmp_path)
    job["status"] = "running"

    with pytest.raises(ValueError, match="completed simple training pipeline"):
        assembly_registration.register_training_job(job)


def test_register_training_job_checks_expert_input_dimension(monkeypatch, tmp_path: Path) -> None:
    llm = tmp_path / "llm"
    llm.mkdir()
    monkeypatch.setattr(assembly_registration, "DEFAULT_LLM_PATH", llm)
    monkeypatch.setattr(assembly_registration, "BUILTIN_EXPERT_ROOT", _expert_root(tmp_path))
    job = _completed_job(tmp_path)
    job["text2comp_output_dim"] = 3

    with pytest.raises(ValueError, match="does not match built-in Expert input"):
        assembly_registration.register_training_job(job)


def test_register_load_selects_gpu_with_most_free_memory(monkeypatch) -> None:
    from PierNet.training.api.routers import assembly

    monkeypatch.setattr(training.training_manager, "get_job", lambda *_args, **_kwargs: {"job_id": "train-test"})
    monkeypatch.setattr(
        training.assembly_registration,
        "register_training_job",
        lambda _job: {"model_id": "conversation_train-test", "force_split": False},
    )
    monkeypatch.setattr(
        assembly,
        "get_gpu_info",
        lambda: [
            SimpleNamespace(index=0, available=True, memory_free_mb=1024),
            SimpleNamespace(index=3, available=True, memory_free_mb=8192),
        ],
    )
    unloaded = []
    load_requests = []

    async def fake_unload():
        unloaded.append(True)

    async def fake_load(request):
        load_requests.append(request)
        return {"status": "loaded"}

    monkeypatch.setattr(assembly, "unload_all", fake_unload)
    monkeypatch.setattr(assembly, "load_all_models", fake_load)
    monkeypatch.setattr(
        assembly,
        "_get_assembly_profile",
        lambda model_id: {"model_id": model_id, "name": "conversation"},
    )

    result = asyncio.run(
        training.register_and_load_training_job(
            "train-test",
            training.RegisterLoadTrainingJobRequest(),
        )
    )

    assert unloaded == [True]
    assert load_requests[0].llm_gpu_id == 3
    assert load_requests[0].router_gpu_id == 3
    assert result["status"] == "ready"
