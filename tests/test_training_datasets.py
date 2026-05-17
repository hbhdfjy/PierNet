import json

from piern.training.services import training_datasets


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
