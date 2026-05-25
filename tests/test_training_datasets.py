import json

from PierNet.training.api.schemas.training import TrainingDatasetInfo
from PierNet.training.services import training_datasets


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
