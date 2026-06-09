from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path

import h5py
import pytest

from PierNet.synth.services import expert_models


def _patch_expert_roots(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    monkeypatch.setattr(expert_models, "DATA_ROOT", data_root)
    monkeypatch.setattr(expert_models, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(expert_models, "EXPERT_MODEL_ROOT", data_root / "expert_models")
    monkeypatch.setattr(expert_models, "EXPERT_MODEL_FILES", data_root / "expert_models" / "files")
    monkeypatch.setattr(expert_models, "EXPERT_METADATA_PATH", data_root / "expert_models" / "models.json")
    monkeypatch.setattr(
        expert_models,
        "EXPERT_CONFIG_ROOT",
        project_root / "configs" / expert_models.EXPERT_SIMULATOR / "variants",
    )
    return data_root, project_root


def _zip_bundle(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_build_input_plan_parses_chinese_linear_sweep() -> None:
    result = expert_models.build_input_plan("有5个点，每个点从0开始依次加10")

    assert result["plan"]["kind"] == "linear_sweep"
    assert result["plan"]["count"] == 5
    assert result["plan"]["start"] == 0.0
    assert result["plan"]["step"] == 10.0
    assert result["preview"] == [[0.0], [10.0], [20.0], [30.0], [40.0]]


def test_build_input_plan_accepts_explicit_json_values() -> None:
    result = expert_models.build_input_plan('{"values": [[1.0, 2.0], [3.0, 4.0]]}')

    assert result["plan"]["kind"] == "explicit_values"
    assert result["plan"]["count"] == 2
    assert result["plan"]["input_dim"] == 2
    assert result["preview"] == [[1.0, 2.0], [3.0, 4.0]]


def test_describe_constraints_exposes_upload_contract() -> None:
    payload = expert_models.describe_constraints()

    assert payload["interface"] == "def predict(inputs: list[float]) -> float | list[float]"
    assert "constraints" in payload
    assert payload["manifest_name"] == expert_models.MANIFEST_NAME
    assert ".zip" in payload["supported_upload_suffixes"]
    assert payload["max_input_dim"] == expert_models.MAX_INPUT_DIM
    assert payload["max_model_bytes"] == 1024 * 1024 * 1024
    assert payload["max_bundle_files"] == expert_models.MAX_BUNDLE_FILES


def test_upload_model_accepts_declared_example_input_for_multidim(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)
    model_source = """EXAMPLE_INPUT = [0.0, 0.0]

def predict(inputs):
    return float(inputs[0]) + float(inputs[1])
"""

    model = expert_models.upload_model("two_dim.py", model_source.encode("utf-8"))

    assert model["package_type"] == "python_file"
    assert model["example_input"] == [0.0, 0.0]
    assert model["example_input_dim"] == 2
    assert model["smoke_output_dim"] == 1


def test_upload_bundle_and_generate_dataset_with_assets(monkeypatch, tmp_path: Path) -> None:
    data_root, project_root = _patch_expert_roots(monkeypatch, tmp_path)
    manifest = {
        "schema_version": 1,
        "runtime": "python",
        "entrypoint": "adapter.py",
        "callable": "predict",
        "example_input": [1.0, 2.0],
    }
    adapter = """import json
from pathlib import Path


def predict(inputs):
    config = json.loads(Path("coefficients.json").read_text(encoding="utf-8"))
    return [float(inputs[0]) * float(config["scale"]) + float(inputs[1])]
"""
    content = _zip_bundle(
        {
            expert_models.MANIFEST_NAME: json.dumps(manifest),
            "adapter.py": adapter,
            "coefficients.json": '{"scale": 2.0}',
        }
    )

    model = expert_models.upload_model("bundle.zip", content)

    assert model["package_type"] == "zip"
    assert model["entrypoint"] == "adapter.py"
    assert model["callable"] == "predict"
    assert model["example_input"] == [1.0, 2.0]
    assert model["example_input_dim"] == 2
    assert model["smoke_output_dim"] == 1
    assert model["asset_count"] == 3

    result = expert_models.generate_dataset(
        model_id=model["model_id"],
        scenario="bundle_case",
        prompt='{"values": [[1.0, 2.0], [3.0, 4.0]]}',
    )

    h5_path = data_root / "expert_model" / "expert_model_bundle_case.h5"
    cfg_path = project_root / "configs" / "expert_model" / "variants" / "bundle_case.yaml"
    assert result["validation"]["valid"] is True
    assert h5_path.exists()
    assert cfg_path.exists()
    with h5py.File(h5_path, "r") as hf:
        assert hf["params"].shape == (2, 2)
        assert hf["timeseries"].shape == (2, 1, 1)
        assert hf["timeseries"][:, 0, 0].tolist() == [4.0, 10.0]
        assert int(hf.attrs["n_params"]) == 2


def test_upload_bundle_rejects_path_traversal(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)
    content = _zip_bundle({"../adapter.py": "def predict(inputs): return 1.0"})

    with pytest.raises(expert_models.ExpertModelError, match="非法路径"):
        expert_models.upload_model("bad.zip", content)


def test_upload_bundle_rejects_absolute_path(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)
    content = _zip_bundle({"/adapter.py": "def predict(inputs): return 1.0"})

    with pytest.raises(expert_models.ExpertModelError, match="绝对路径"):
        expert_models.upload_model("bad.zip", content)


def test_upload_and_generate_expert_dataset_uses_stage1_hdf5_contract(monkeypatch, tmp_path: Path) -> None:
    data_root, project_root = _patch_expert_roots(monkeypatch, tmp_path)
    model_source = """def predict(inputs):
    x = float(inputs[0])
    return [x, x + 1.0]
"""
    model = expert_models.upload_model("linear.py", model_source.encode("utf-8"))
    assert model["example_input_dim"] == 1
    assert model["smoke_output_dim"] == 2

    result = expert_models.generate_dataset(
        model_id=model["model_id"],
        scenario="expert_case",
        prompt="有4个点，每个点从0开始依次加10",
    )

    h5_path = data_root / "expert_model" / "expert_model_expert_case.h5"
    cfg_path = project_root / "configs" / "expert_model" / "variants" / "expert_case.yaml"
    assert result["validation"]["valid"] is True
    assert h5_path.exists()
    assert cfg_path.exists()
    with h5py.File(h5_path, "r") as hf:
        assert hf["params"].shape == (4, 1)
        assert hf["timeseries"].shape == (4, 2, 1)
        assert hf["params"][:, 0].tolist() == [0.0, 10.0, 20.0, 30.0]
        assert hf["timeseries"][:, 0, 0].tolist() == [0.0, 10.0, 20.0, 30.0]
        assert hf["timeseries"][:, 1, 0].tolist() == [1.0, 11.0, 21.0, 31.0]
        assert int(hf.attrs["n_samples"]) == 4
        assert int(hf.attrs["n_channels"]) == 2
        assert int(hf.attrs["n_timesteps"]) == 1
        assert int(hf.attrs["n_params"]) == 1



def test_upload_model_registers_system_metadata(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)
    model_source = """EXAMPLE_INPUT = [1.0, 2.0]

def predict(inputs):
    return [float(inputs[0]) + float(inputs[1]), float(inputs[0]) * 2.0]
"""

    model = expert_models.upload_model("registered.py", model_source.encode("utf-8"))

    assert model["status"] == "active"
    assert model["runtime"] == "python"
    assert model["input_dim"] == 2
    assert model["output_dim"] == 2
    assert model["assembly_enabled"] is True
    assert model["data_generation_enabled"] is True
    assert len(model["checksum"]) == 64
    listed = expert_models.list_models()
    assert listed[0]["model_id"] == model["model_id"]
    assert listed[0]["exists"] is True
    assert expert_models.predict_model(model["model_id"], [1.0, 3.0]) == [4.0, 2.0]


def test_update_revalidate_and_delete_model(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)
    model = expert_models.upload_model(
        "managed.py",
        b"EXAMPLE_INPUT = [0.0]\n\ndef predict(inputs):\n    return float(inputs[0])\n",
    )

    updated = expert_models.update_model(
        model["model_id"],
        {"status": "disabled", "assembly_enabled": False, "data_generation_enabled": False},
    )
    assert updated["status"] == "disabled"
    assert updated["assembly_enabled"] is False
    assert updated["data_generation_enabled"] is False
    with pytest.raises(expert_models.ExpertModelError, match="未启用"):
        expert_models.generate_dataset(model_id=model["model_id"], scenario="disabled", prompt="有1个点")

    Path(updated["path"]).write_text("def broken(inputs):\n    return 1.0\n", encoding="utf-8")
    invalid = expert_models.revalidate_model(model["model_id"])
    assert invalid["status"] == "invalid"
    assert "predict" in str(invalid["last_error"])

    deleted = expert_models.delete_model(model["model_id"])
    assert deleted["model_id"] == model["model_id"]
    assert expert_models.list_models() == []


def test_upload_bundle_manifest_optional_metadata_and_dim_inference(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)
    manifest = {
        "schema_version": 1,
        "runtime": "python",
        "entrypoint": "adapter.py",
        "callable": "predict",
        "example_input": [1.0, 2.0, 3.0],
        "name": "bundle_declared",
        "domain": "custom_domain",
        "assembly_enabled": True,
        "data_generation_enabled": False,
    }
    content = _zip_bundle(
        {
            expert_models.MANIFEST_NAME: json.dumps(manifest),
            "adapter.py": "def predict(inputs):\n    return [float(sum(inputs)), float(len(inputs))]\n",
        }
    )

    model = expert_models.upload_model("bundle.zip", content)

    assert model["name"] == "bundle_declared"
    assert model["domain"] == "custom_domain"
    assert model["simulator"] == "custom_domain"
    assert model["input_dim"] == 3
    assert model["output_dim"] == 2
    assert model["data_generation_enabled"] is False


def test_upload_bundle_rejects_missing_manifest(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)
    content = _zip_bundle({"adapter.py": "def predict(inputs): return 1.0"})

    with pytest.raises(expert_models.ExpertModelError, match=expert_models.MANIFEST_NAME):
        expert_models.upload_model("missing_manifest.zip", content)


def test_upload_rejects_missing_predict(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)

    with pytest.raises(expert_models.ExpertModelError, match="predict"):
        expert_models.upload_model("missing_predict.py", b"def not_predict(inputs):\n    return 1.0\n")


def test_upload_rejects_invalid_output(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)

    with pytest.raises(expert_models.ExpertModelError, match="float"):
        expert_models.upload_model("bad_output.py", b"def predict(inputs):\n    return {'x': 1.0}\n")


def test_assembly_uploaded_experts_list_and_dimension_guard(monkeypatch, tmp_path: Path) -> None:
    _patch_expert_roots(monkeypatch, tmp_path)
    from fastapi import HTTPException
    from PierNet.training.api.routers import assembly

    model = expert_models.upload_model(
        "assembly.py",
        b"EXAMPLE_INPUT = [0.0, 0.0]\n\ndef predict(inputs):\n    return [float(inputs[0]) + float(inputs[1])]\n",
    )

    experts = asyncio.run(assembly.list_uploaded_experts())
    assert [item.model_id for item in experts] == [model["model_id"]]
    assert experts[0].input_dim == 2

    with pytest.raises(HTTPException) as exc:
        assembly._load_uploaded_expert(model["model_id"], expected_input_dim=3)
    assert exc.value.status_code == 400
    assert "维度不匹配" in str(exc.value.detail)

    loaded = assembly._load_uploaded_expert(model["model_id"], expected_input_dim=2)
    assert loaded.model_id == model["model_id"]
    assert assembly._LOADED_MODELS["expert_executor"] == "uploaded"
    assert assembly._LOADED_MODELS["uploaded_expert_id"] == model["model_id"]
