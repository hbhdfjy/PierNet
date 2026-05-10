from __future__ import annotations

import h5py
import numpy as np
import yaml

from piern.synth.services import hdf5_data


def _write_valid_hdf5(path, *, n=3, c=2, t=4, p=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("timeseries", data=np.ones((n, c, t), dtype=np.float32))
        hf.create_dataset("params", data=np.ones((n, p), dtype=np.float32))
        hf.create_dataset("param_names", data=np.array([f"param_{i}".encode("utf-8") for i in range(p)]))
        hf.attrs["n_samples"] = n
        hf.attrs["n_channels"] = c
        hf.attrs["n_timesteps"] = t
        hf.attrs["n_params"] = p


def test_validate_hdf5_file_accepts_stage1_contract(tmp_path):
    path = tmp_path / "modflow_demo.h5"
    _write_valid_hdf5(path, n=5, c=3, t=7, p=4)

    result = hdf5_data.validate_hdf5_file(path)

    assert result["valid"] is True
    assert result["sample_count"] == 5
    assert result["output_shape"] == [3, 7]
    assert result["params_shape"] == [5, 4]
    assert result["n_params"] == 4


def test_canonical_hdf5_path_accepts_new_big_scene(monkeypatch, tmp_path):
    monkeypatch.setattr(hdf5_data, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hdf5_data, "DATA_ROOT", tmp_path / "data")

    path = hdf5_data.canonical_hdf5_path("new_big_scene", "case_001")

    assert path == hdf5_data.DATA_ROOT / "new_big_scene" / "new_big_scene_case_001.h5"


def test_validate_hdf5_file_rejects_shape_attr_mismatch(tmp_path):
    path = tmp_path / "modflow_bad.h5"
    _write_valid_hdf5(path, n=5, c=3, t=7, p=4)
    with h5py.File(path, "a") as hf:
        hf.attrs["n_samples"] = 4

    result = hdf5_data.validate_hdf5_file(path)

    assert result["valid"] is False
    assert any("n_samples" in msg for msg in result["errors"])


def test_validate_hdf5_file_rejects_non_finite_values(tmp_path):
    path = tmp_path / "modflow_nan.h5"
    _write_valid_hdf5(path)
    with h5py.File(path, "a") as hf:
        hf["params"][0, 0] = np.nan

    result = hdf5_data.validate_hdf5_file(path)

    assert result["valid"] is False
    assert any("NaN" in msg or "Inf" in msg for msg in result["errors"])


def test_collect_registration_hdf5_validations_honors_selected_scenarios(tmp_path, monkeypatch):
    monkeypatch.setattr(hdf5_data, "PROJECT_ROOT", tmp_path)
    data_root = tmp_path / "data"
    _write_valid_hdf5(data_root / "modflow" / "modflow_good.h5")

    config_path = tmp_path / "configs" / "text2comp" / "default.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump({"data_root": "data", "scenarios": {"modflow": ["good", "missing"]}}),
        encoding="utf-8",
    )

    selected = hdf5_data.collect_registration_hdf5_validations(str(config_path), scenarios=["good"])
    all_items = hdf5_data.collect_registration_hdf5_validations(str(config_path))

    assert len(selected) == 1
    assert selected[0]["valid"] is True
    assert {item["scenario"] for item in all_items} == {"good", "missing"}
    assert any(item["scenario"] == "missing" and not item["valid"] for item in all_items)


def test_collect_registration_hdf5_validations_infers_selected_when_config_has_no_scenarios(tmp_path, monkeypatch):
    monkeypatch.setattr(hdf5_data, "PROJECT_ROOT", tmp_path)
    data_root = tmp_path / "data"
    _write_valid_hdf5(data_root / "gcam" / "gcam_carbon_pricing.h5")

    config_path = tmp_path / "configs" / "text2comp" / "default.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump({"data_root": "data"}), encoding="utf-8")

    result = hdf5_data.collect_registration_hdf5_validations(
        str(config_path),
        scenarios=["carbon_pricing", "missing_scenario"],
    )

    by_name = {item["scenario"]: item for item in result}
    assert by_name["carbon_pricing"]["simulator"] == "gcam"
    assert by_name["carbon_pricing"]["valid"] is True
    assert by_name["missing_scenario"]["valid"] is False
    assert "无法根据场景名" in by_name["missing_scenario"]["errors"][0]
