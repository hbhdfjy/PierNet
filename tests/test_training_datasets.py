import json

import pytest

from PierNet.training.api.schemas.training import TrainingDatasetInfo
from PierNet.training.services import training_datasets


@pytest.fixture(autouse=True)
def _isolate_new_synth_registry(monkeypatch):
    monkeypatch.setattr(training_datasets, "list_router_datasets", lambda: [])


def test_training_datasets_groups_router_manifest(tmp_path):
    manifest = tmp_path / "router.json"
    manifest.write_text(
        json.dumps(
            {
                "scenarios": [
                    {"simulator": "b", "scenario": "s2", "router_count": 2},
                    {"simulator": "a", "scenario": "s1", "router_count": 3},
                    {"simulator": "a", "scenario": "s0", "router_count": 1},
                ]
            }
        ),
        encoding="utf-8",
    )

    datasets = training_datasets.list_datasets(router_manifest_path=manifest, default_router_manifest_path=tmp_path / "default.json")

    assert [item["simulator"] for item in datasets] == ["a", "b"]
    assert datasets[0]["total_count"] == 4
    assert [item["scenario"] for item in datasets[0]["scenarios"]] == ["s0", "s1"]


def test_training_datasets_merges_manifest_without_duplicate_scenarios():
    primary = {"storage": "parquet", "scenarios": [{"simulator": "a", "scenario": "s0", "router_count": 1}]}
    fallback = {
        "scenarios": [
            {"simulator": "a", "scenario": "s0", "router_count": 10},
            {"simulator": "a", "scenario": "s1", "router_count": 2},
        ]
    }

    merged = training_datasets.merge_router_manifests(primary, fallback)

    assert merged["storage"] == "mixed"
    assert merged["total"] == 3
    assert [item["scenario"] for item in merged["scenarios"]] == ["s0", "s1"]


def test_training_datasets_merges_by_simulator_and_scenario():
    primary = {"storage": "parquet", "scenarios": [{"simulator": "a", "scenario": "shared", "router_count": 1}]}
    fallback = {
        "scenarios": [
            {"simulator": "a", "scenario": "shared", "router_count": 10},
            {"simulator": "b", "scenario": "shared", "router_count": 2},
        ]
    }

    merged = training_datasets.merge_router_manifests(primary, fallback)

    assert merged["total"] == 3
    assert [(item["simulator"], item["scenario"]) for item in merged["scenarios"]] == [
        ("a", "shared"),
        ("b", "shared"),
    ]


def test_training_datasets_normalizes_legacy_minimal_manifest(tmp_path):
    manifest = tmp_path / "router.json"
    manifest.write_text(
        json.dumps(
            {
                "scenarios": [
                    {"simulator": "modflow", "scenario": "coastal", "router_count": "7"},
                    {"simulator": "modflow", "scenario": "", "router_count": 3},
                    {"simulator": "modflow", "router_count": 2},
                    "not-a-scenario",
                ]
            }
        ),
        encoding="utf-8",
    )

    datasets = training_datasets.list_datasets(
        router_manifest_path=manifest,
        default_router_manifest_path=tmp_path / "default.json",
    )
    model = TrainingDatasetInfo(**datasets[0])

    assert model.simulator == "modflow"
    assert model.total_count == 7
    assert len(model.scenarios) == 1
    assert model.scenarios[0].scenario == "coastal"
    assert model.scenarios[0].router_count == 7
    assert model.scenarios[0].file_size_bytes == 0
    assert model.scenarios[0].mtime == 0.0
    assert model.scenarios[0].path == ""


def test_training_datasets_caches_expensive_legacy_discovery(monkeypatch, tmp_path):
    calls = 0

    def discover(*, include_label_counts=True):
        nonlocal calls
        calls += 1
        assert include_label_counts is False
        return {"scenarios": [{"simulator": "legacy", "scenario": "base", "router_count": 3}]}

    monkeypatch.setattr(training_datasets.portable, "router_manifest_like", discover)
    monkeypatch.setattr(training_datasets, "list_router_datasets", lambda: [])
    monkeypatch.setattr(training_datasets, "_legacy_manifest_cache_ready", False)
    default_manifest = tmp_path / "router.json"

    first = training_datasets.list_datasets(
        router_manifest_path=default_manifest,
        default_router_manifest_path=default_manifest,
    )
    second = training_datasets.list_datasets(
        router_manifest_path=default_manifest,
        default_router_manifest_path=default_manifest,
    )

    assert first == second
    assert calls == 1


def test_simple_training_datasets_only_exposes_complete_workflows(tmp_path):
    gcam_h5 = tmp_path / "gcam" / "gcam_carbon_pricing.h5"
    gcam_h5.parent.mkdir(parents=True)
    gcam_h5.write_bytes(b"hdf5")
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "carbon_pricing_templates.jsonl").write_text("{}\n", encoding="utf-8")

    internal_h5 = tmp_path / "expert_model" / "expert_model_demo.h5"
    internal_h5.parent.mkdir(parents=True)
    internal_h5.write_bytes(b"hdf5")
    (templates / "demo_templates.jsonl").write_text("{}\n", encoding="utf-8")

    paired_path = tmp_path / "new_synth" / "text2comp.jsonl"
    paired_path.parent.mkdir()
    paired_path.write_text("{}\n", encoding="utf-8")
    smoke_path = tmp_path / "new_synth" / "smoke-text2comp.jsonl"
    smoke_path.write_text("{}\n", encoding="utf-8")
    duplicate_path = tmp_path / "new_synth" / "duplicate-text2comp.jsonl"
    duplicate_path.write_text("{}\n", encoding="utf-8")

    datasets = [
        {
            "source": "new_synth",
            "dataset_id": "router-ready",
            "text2comp_dataset_id": "t2c-ready",
            "display_name": "用户时序数据 · Router",
            "simulator": "uploaded_expert",
            "total_count": 400,
            "scenarios": [{"scenario": "custom", "router_count": 400}],
        },
        {
            "source": "new_synth",
            "dataset_id": "router-duplicate",
            "text2comp_dataset_id": "t2c-duplicate",
            "display_name": "GCAM · carbon_pricing · Router",
            "simulator": "uploaded_expert",
            "total_count": 400,
            "scenarios": [{"scenario": "custom_scenario", "router_count": 400}],
        },
        {
            "source": "new_synth",
            "dataset_id": "router-smoke",
            "text2comp_dataset_id": "t2c-smoke",
            "display_name": "端到端测试 · Router",
            "simulator": "e2e_mobile",
            "total_count": 8,
            "scenarios": [{"scenario": "real_hdf5", "router_count": 8}],
        },
        {
            "source": "legacy",
            "simulator": "expert_model",
            "total_count": 2_000,
            "scenarios": [{"scenario": "demo", "router_count": 2_000}],
        },
        {
            "source": "legacy",
            "simulator": "gcam",
            "total_count": 1_000,
            "scenarios": [{"scenario": "carbon_pricing", "router_count": 1_000}],
        },
        {
            "source": "legacy",
            "simulator": "modflow",
            "total_count": 1_000,
            "scenarios": [{"scenario": "missing_inputs", "router_count": 1_000}],
        },
    ]
    text2comp_datasets = [
        {
            "dataset_id": "t2c-ready",
            "sample_count": 200,
            "path": str(paired_path),
        },
        {
            "dataset_id": "t2c-smoke",
            "sample_count": 4,
            "path": str(smoke_path),
        },
        {
            "dataset_id": "t2c-duplicate",
            "sample_count": 200,
            "path": str(duplicate_path),
        },
    ]

    ready = training_datasets.list_simple_datasets(
        datasets,
        text2comp_datasets=text2comp_datasets,
        data_root=tmp_path,
        min_router_samples=100,
        min_text2comp_samples=100,
    )

    assert [item["display_name"] for item in ready] == ["用户时序数据", "GCAM"]
    assert [item["simulator"] for item in ready] == ["uploaded_expert", "gcam"]
    assert all("Router" not in item["display_name"] for item in ready)
    assert all(item["simulator"] != "expert_model" for item in ready)
