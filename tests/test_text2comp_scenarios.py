from __future__ import annotations

import json
from pathlib import Path

import h5py
from fastapi import HTTPException
from pydantic import ValidationError
import pytest
import numpy as np
import yaml

from piern.shared.storage.portable import PartitionInfo
from piern.synth.api.schemas.config import LLMConfigRequest
from piern.synth.api.routers import config as config_router


def _write_hdf5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("timeseries", data=np.ones((3, 2, 4), dtype=np.float32))
        hf.create_dataset("params", data=np.ones((3, 2), dtype=np.float32))


def test_text2comp_scenarios_report_parquet_only_samples(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "text2comp"
    config_dir.mkdir(parents=True)
    (config_dir / "default.yaml").write_text(
        "data_root: data\nregistry: configs/text2comp/registry.yaml\n",
        encoding="utf-8",
    )
    (config_dir / "registry.yaml").write_text("{}\n", encoding="utf-8")

    scenario = "pytest_parquet_only_status"
    partition = PartitionInfo(
        kind="text2comp",
        simulator="simpeg",
        scenario=scenario,
        path=tmp_path / "partition",
        row_count=42,
        file_size_bytes=100,
        mtime=1.0,
        metadata={},
    )

    monkeypatch.setattr(config_router, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_router, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_router, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        config_router.portable,
        "discover_partitions",
        lambda kind: [partition] if kind == "text2comp" else [],
    )
    config_router.invalidate_text2comp_scenarios_cache()

    try:
        result = config_router.get_text2comp_scenarios()
    finally:
        config_router.invalidate_text2comp_scenarios_cache()

    item = result["simpeg"][0]
    assert item["name"] == scenario
    assert item["has_h5"] is False
    assert item["has_jsonl"] is False
    assert item["has_parquet"] is True
    assert item["has_samples"] is True
    assert item["existing_jsonl_count"] == 0
    assert item["existing_parquet_count"] == 42
    assert item["existing_sample_count"] == 42
    assert item["existing_storage"] == "parquet"


def test_text2comp_scenarios_count_prefixed_jsonl_by_metadata(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    config_dir = project_root / "configs" / "text2comp"
    config_dir.mkdir(parents=True)
    (config_dir / "default.yaml").write_text(
        "data_root: data\nregistry: configs/text2comp/registry.yaml\n",
        encoding="utf-8",
    )
    (config_dir / "registry.yaml").write_text(
        "gcam:\n  scenarios:\n    shared: shared case\n",
        encoding="utf-8",
    )
    sample_dir = data_root / "text2comp"
    sample_dir.mkdir(parents=True)
    sample = {"input": "gcam", "metadata": {"simulator": "gcam", "scenario": "shared"}}
    (sample_dir / "gcam_shared.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setattr(config_router, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_router, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(config_router, "DATA_ROOT", data_root)
    monkeypatch.setattr(config_router.file_manager, "DATA_DIR", sample_dir)
    monkeypatch.setattr(config_router.portable, "discover_partitions", lambda kind: [])
    config_router.invalidate_text2comp_scenarios_cache()

    try:
        result = config_router.get_text2comp_scenarios()
    finally:
        config_router.invalidate_text2comp_scenarios_cache()

    item = result["gcam"][0]
    assert item["name"] == "shared"
    assert item["existing_jsonl_count"] == 1
    assert item["existing_sample_count"] == 1
    assert item["existing_storage"] == "jsonl"
    assert item["has_samples"] is True



def test_text2comp_scenarios_survives_ambiguous_jsonl_status(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    config_dir = project_root / "configs" / "text2comp"
    config_dir.mkdir(parents=True)
    (config_dir / "default.yaml").write_text(
        "data_root: data" + chr(10) + "registry: configs/text2comp/registry.yaml" + chr(10),
        encoding="utf-8",
    )
    (config_dir / "registry.yaml").write_text(
        "simpeg:" + chr(10) + "  scenarios:" + chr(10) + "    shared: shared case" + chr(10),
        encoding="utf-8",
    )
    sample_dir = data_root / "text2comp"
    sample_dir.mkdir(parents=True)
    row = {"input": "simpeg", "metadata": {"simulator": "simpeg", "scenario": "shared"}}
    (sample_dir / "shared.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")
    (sample_dir / "mixed.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")

    monkeypatch.setattr(config_router, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_router, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(config_router, "DATA_ROOT", data_root)
    monkeypatch.setattr(config_router.file_manager, "DATA_DIR", sample_dir)
    monkeypatch.setattr(config_router.portable, "discover_partitions", lambda kind: [])
    config_router.invalidate_text2comp_scenarios_cache()

    try:
        result = config_router.get_text2comp_scenarios()
    finally:
        config_router.invalidate_text2comp_scenarios_cache()

    item = result["simpeg"][0]
    assert item["name"] == "shared"
    assert item["has_jsonl"] is True
    assert item["existing_jsonl_count"] == 0
    assert item["existing_storage"] == "jsonl"


def test_text2comp_scenarios_resolve_default_data_root_to_runtime_data_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    config_dir = project_root / "configs" / "text2comp"
    config_dir.mkdir(parents=True)
    (config_dir / "default.yaml").write_text(
        "data_root: data\nregistry: configs/text2comp/registry.yaml\n",
        encoding="utf-8",
    )
    (config_dir / "registry.yaml").write_text("{}\n", encoding="utf-8")
    _write_hdf5(data_root / "modflow" / "modflow_external.h5")

    monkeypatch.setattr(config_router, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_router, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(config_router, "DATA_ROOT", data_root)
    monkeypatch.setattr(config_router.portable, "discover_partitions", lambda kind: [])
    config_router.invalidate_text2comp_scenarios_cache()

    try:
        result = config_router.get_text2comp_scenarios()
    finally:
        config_router.invalidate_text2comp_scenarios_cache()

    item = result["modflow"][0]
    assert item["name"] == "external"
    assert item["has_h5"] is True
    assert item["sample_count"] == 3
    assert item["output_shape"] == [2, 4]


def test_text2comp_scenarios_include_hdf5_suffix(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    config_dir = project_root / "configs" / "text2comp"
    config_dir.mkdir(parents=True)
    (config_dir / "default.yaml").write_text(
        "data_root: data" + chr(10) + "registry: configs/text2comp/registry.yaml" + chr(10),
        encoding="utf-8",
    )
    (config_dir / "registry.yaml").write_text("{}" + chr(10), encoding="utf-8")
    _write_hdf5(data_root / "simpeg" / "simpeg_external.hdf5")

    monkeypatch.setattr(config_router, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_router, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(config_router, "DATA_ROOT", data_root)
    monkeypatch.setattr(config_router.portable, "discover_partitions", lambda kind: [])
    config_router.invalidate_text2comp_scenarios_cache()

    try:
        result = config_router.get_text2comp_scenarios()
    finally:
        config_router.invalidate_text2comp_scenarios_cache()

    item = result["simpeg"][0]
    assert item["name"] == "external"
    assert item["h5_file"] == "simpeg_external.hdf5"
    assert item["has_h5"] is True
    assert item["sample_count"] == 3


def test_llm_config_request_rejects_invalid_generation_bounds() -> None:
    valid = {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-V3",
        "api_key": "",
        "base_url": "",
        "temperature": 1.0,
        "max_tokens": 1024,
    }

    for override in [
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"max_tokens": 0},
        {"max_tokens": 8193},
    ]:
        with pytest.raises(ValidationError):
            LLMConfigRequest(**{**valid, **override})


def test_save_llm_config_rejects_empty_model_without_clearing_key(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "text2comp"
    config_dir.mkdir(parents=True)
    default_yaml = config_dir / "default.yaml"
    default_yaml.write_text(
        yaml.safe_dump(
            {
                "llm": {
                    "provider": "siliconflow",
                    "model": "old-model",
                    "api_key": "sk-old-secret",
                    "base_url": "https://old.example/v1",
                    "temperature": 1.0,
                    "max_tokens": 1024,
                    "thinking": "disabled",
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_router, "CONFIG_DIR", config_dir)

    try:
        config_router.save_llm_config(
            LLMConfigRequest(provider="deepseek", model="", api_key="", base_url="", temperature=0.7, max_tokens=512)
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "模型名称不能为空" in str(exc.detail)
    else:
        raise AssertionError("expected HTTPException for empty LLM model")

    saved = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))["llm"]
    assert saved["provider"] == "siliconflow"
    assert saved["model"] == "old-model"
    assert saved["api_key"] == "sk-old-secret"
    assert saved["base_url"] == "https://old.example/v1"
