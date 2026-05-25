from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from piern.synth.services import hdf5_data
from piern.synth.text2comp import pipeline as text2comp_pipeline


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
    monkeypatch.setattr(hdf5_data, "DATA_ROOT", data_root)
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
    monkeypatch.setattr(hdf5_data, "DATA_ROOT", data_root)
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


def test_collect_registration_hdf5_validations_uses_runtime_data_root(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    monkeypatch.setattr(hdf5_data, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(hdf5_data, "DATA_ROOT", data_root)
    _write_valid_hdf5(data_root / "simpeg" / "simpeg_external.h5")

    config_path = project_root / "configs" / "text2comp" / "default.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump({"data_root": "data", "scenarios": {"simpeg": ["external"]}}),
        encoding="utf-8",
    )

    result = hdf5_data.collect_registration_hdf5_validations(str(config_path))

    assert len(result) == 1
    assert result[0]["simulator"] == "simpeg"
    assert result[0]["scenario"] == "external"
    assert result[0]["path"].endswith("simpeg_external.h5")
    assert result[0]["valid"] is True
    assert result[0]["sample_count"] == 3


def test_text2comp_stage_scenario_guard_rejects_output_collisions() -> None:
    h5_files = [
        (Path("/data/gcam/gcam_shared.h5"), "gcam", None),
        (Path("/data/simpeg/simpeg_shared.h5"), "simpeg", None),
    ]

    duplicates = text2comp_pipeline.duplicate_stage_scenarios(h5_files)

    assert duplicates == ["shared (gcam, simpeg)"]
    with pytest.raises(ValueError, match="同名场景"):
        text2comp_pipeline.assert_unique_stage_scenarios(h5_files)


def test_text2comp_scan_h5_files_uses_runtime_data_root(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    monkeypatch.setattr(text2comp_pipeline, "DATA_ROOT", data_root)
    _write_valid_hdf5(data_root / "power_flow" / "power_flow_external.h5")

    found = text2comp_pipeline._scan_h5_files({"data_root": "data"}, project_root)

    assert found == [(data_root / "power_flow" / "power_flow_external.h5", "power_flow", None)]


def test_text2comp_scan_h5_files_includes_hdf5_suffix(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    monkeypatch.setattr(text2comp_pipeline, "DATA_ROOT", data_root)
    _write_valid_hdf5(data_root / "simpeg" / "simpeg_external.hdf5")

    found = text2comp_pipeline._scan_h5_files({"data_root": "data"}, project_root)

    assert found == [(data_root / "simpeg" / "simpeg_external.hdf5", "simpeg", None)]


def test_collect_registration_hdf5_validations_infers_hdf5_suffix(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    monkeypatch.setattr(hdf5_data, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(hdf5_data, "DATA_ROOT", data_root)
    _write_valid_hdf5(data_root / "gcam" / "gcam_external.hdf5")

    config_path = project_root / "configs" / "text2comp" / "default.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump({"data_root": "data"}), encoding="utf-8")

    result = hdf5_data.collect_registration_hdf5_validations(str(config_path), scenarios=["external"])

    assert len(result) == 1
    assert result[0]["simulator"] == "gcam"
    assert result[0]["scenario"] == "external"
    assert result[0]["path"].endswith("gcam_external.hdf5")
    assert result[0]["valid"] is True


def test_list_hdf5_data_files_discovers_uppercase_suffix(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setattr(hdf5_data, "DATA_ROOT", data_root)
    _write_valid_hdf5(data_root / "simpeg" / "simpeg_external.HDF5")

    result = hdf5_data.list_hdf5_data_files()

    assert [(item["simulator"], item["scenario"]) for item in result] == [("simpeg", "external")]
    assert result[0]["path"].endswith("simpeg_external.HDF5")
