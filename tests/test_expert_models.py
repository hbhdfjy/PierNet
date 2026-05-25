from __future__ import annotations

from pathlib import Path

import h5py

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


def test_upload_and_generate_expert_dataset_uses_stage1_hdf5_contract(monkeypatch, tmp_path: Path) -> None:
    data_root, project_root = _patch_expert_roots(monkeypatch, tmp_path)
    model_source = """def predict(inputs):
    x = float(inputs[0])
    return [x, x + 1.0]
"""
    model = expert_models.upload_model("linear.py", model_source.encode("utf-8"))

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
