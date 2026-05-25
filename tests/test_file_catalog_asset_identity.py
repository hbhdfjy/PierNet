from __future__ import annotations

import json
from pathlib import Path

import pytest

from PierNet.shared.storage.portable import PartitionInfo
from PierNet.synth.services import file_catalog


def _partition(kind: str, simulator: str, scenario: str, path: Path) -> PartitionInfo:
    return PartitionInfo(
        kind=kind,
        simulator=simulator,
        scenario=scenario,
        path=path,
        row_count=1,
        file_size_bytes=1,
        mtime=1.0,
        metadata={},
    )


def test_catalog_sample_and_router_asset_ids_include_simulator(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(file_catalog, "DATA_DIR", tmp_path / "text2comp")
    monkeypatch.setattr(file_catalog, "ROUTER_DIR", tmp_path / "router")
    monkeypatch.setattr(
        file_catalog.manifest_store,
        "ensure_sample_manifest",
        lambda: {
            "items": [
                {"scenario": "shared", "simulator": "simpeg", "sample_count": 1},
                {"scenario": "shared", "simulator": "modflow", "sample_count": 2},
            ],
            "summary": {},
        },
    )
    monkeypatch.setattr(
        file_catalog.manifest_store,
        "ensure_router_manifest",
        lambda: {
            "scenarios": [
                {"scenario": "shared", "simulator": "simpeg", "router_count": 1},
                {"scenario": "shared", "simulator": "modflow", "router_count": 2},
            ],
            "splits": {},
        },
    )

    sample_assets = file_catalog._sample_assets()
    router_assets = file_catalog._router_assets()

    assert {asset["title"]: file_catalog.decode_asset_id(asset["id"]) for asset in sample_assets} == {
        "simpeg/shared": ["sample", "simpeg", "shared"],
        "modflow/shared": ["sample", "modflow", "shared"],
    }
    assert {asset["title"]: file_catalog.decode_asset_id(asset["id"]) for asset in router_assets} == {
        "simpeg/shared": ["router_scenario", "simpeg", "shared"],
        "modflow/shared": ["router_scenario", "modflow", "shared"],
    }


def test_delete_sample_asset_with_simulator_removes_only_matching_partition(monkeypatch, tmp_path: Path) -> None:
    invalidated: list[bool] = []
    deleted: list[tuple[str, str | None, str]] = []
    asset_id = file_catalog.encode_asset_id("sample", "simpeg", "case_a")
    partitions = [
        _partition("text2comp", "simpeg", "case_a", tmp_path / "simpeg"),
        _partition("text2comp", "modflow", "case_a", tmp_path / "modflow"),
    ]

    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog.file_manager, "delete_sample_file", lambda scenario, simulator=None: True)
    monkeypatch.setattr(file_catalog.portable, "discover_partitions", lambda kind: partitions if kind == "text2comp" else [])

    def delete_partition(kind: str, scenario: str, simulator: str | None = None) -> bool:
        deleted.append((kind, simulator, scenario))
        return True

    monkeypatch.setattr(file_catalog.portable, "delete_partition", delete_partition)
    monkeypatch.setattr(file_catalog, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append(True))

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "sample", "deleted": 2}
    assert deleted == [("text2comp", "simpeg", "case_a")]
    assert invalidated == [True]



def test_delete_router_scenario_asset_resolves_metadata_scenario_for_prefixed_jsonl(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    scenario_jsonl = scenario_dir / "gcam_shared.jsonl"
    scenario_meta = scenario_dir / "gcam_shared.meta.json"
    scenario_jsonl.write_text(
        json.dumps({"label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    scenario_meta.write_text("{}", encoding="utf-8")
    asset_id = file_catalog.encode_asset_id("router_scenario", "gcam", "shared")

    monkeypatch.setattr(file_catalog, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(file_catalog, "ROUTER_SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog, "_assert_no_active_training_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog.manifest_store, "rebuild_router_manifest", lambda: {})
    monkeypatch.setattr(file_catalog.portable, "discover_partitions", lambda kind: [])

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "router_scenario", "deleted": 2, "train_count": 0}
    assert not scenario_jsonl.exists()
    assert not scenario_meta.exists()


def test_delete_router_scenario_asset_with_simulator_removes_only_matching_partition(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    scenario_jsonl = scenario_dir / "case_a.jsonl"
    scenario_meta = scenario_dir / "case_a.meta.json"
    scenario_jsonl.write_text("{}\n", encoding="utf-8")
    scenario_meta.write_text("{}", encoding="utf-8")
    calls: list[tuple[str, str | None, str]] = []
    asset_id = file_catalog.encode_asset_id("router_scenario", "simpeg", "case_a")
    partitions = [
        _partition("router", "simpeg", "case_a", tmp_path / "simpeg"),
        _partition("router", "modflow", "case_a", tmp_path / "modflow"),
    ]

    monkeypatch.setattr(file_catalog, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(file_catalog, "ROUTER_SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog, "_assert_no_active_training_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog.manifest_store, "rebuild_router_manifest", lambda: {})
    monkeypatch.setattr(file_catalog.portable, "discover_partitions", lambda kind: partitions if kind == "router" else [])

    def delete_partition(kind: str, scenario: str, simulator: str | None = None) -> bool:
        calls.append((kind, simulator, scenario))
        return True

    monkeypatch.setattr(file_catalog.portable, "delete_partition", delete_partition)

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "router_scenario", "deleted": 3, "train_count": 0}
    assert not scenario_jsonl.exists()
    assert not scenario_meta.exists()
    assert calls == [("router", "simpeg", "case_a")]


def test_delete_sample_file_rewrites_mixed_identity_jsonl(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    sample_path = data_dir / "mixed.jsonl"
    rows = [
        {"input": "remove", "metadata": {"simulator": "gcam", "scenario": "shared"}},
        {"input": "keep", "metadata": {"simulator": "simpeg", "scenario": "shared"}},
    ]
    sample_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    monkeypatch.setattr(file_catalog.file_manager, "DATA_DIR", data_dir)
    monkeypatch.setattr(file_catalog.file_manager.manifest_store, "rebuild_sample_manifest", lambda: {})

    assert file_catalog.file_manager.delete_sample_file("shared", simulator="gcam") is True

    remaining = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    assert remaining == [rows[1]]


def test_delete_router_scenario_rewrites_mixed_identity_jsonl(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    scenario_jsonl = scenario_dir / "mixed.jsonl"
    scenario_meta = scenario_dir / "mixed.meta.json"
    rows = [
        {"label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}},
        {"label": 0, "metadata": {"simulator": "simpeg", "scenario": "shared"}},
    ]
    scenario_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    scenario_meta.write_text(json.dumps({"output_count": 2}, ensure_ascii=False), encoding="utf-8")
    asset_id = file_catalog.encode_asset_id("router_scenario", "gcam", "shared")

    monkeypatch.setattr(file_catalog, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(file_catalog, "ROUTER_SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog, "_assert_no_active_training_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog.manifest_store, "rebuild_router_manifest", lambda: {})
    monkeypatch.setattr(file_catalog.portable, "discover_partitions", lambda kind: [])

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "router_scenario", "deleted": 1, "train_count": 1}
    remaining = [json.loads(line) for line in scenario_jsonl.read_text(encoding="utf-8").splitlines()]
    assert remaining == [rows[1]]
    train_rows = [json.loads(line) for line in (router_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert train_rows == [rows[1]]
    assert json.loads(scenario_meta.read_text(encoding="utf-8"))["output_count"] == 1



def test_rewrite_router_train_skips_bad_lines_without_dropping_following_rows(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    rows = [
        {"label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}},
        {"label": 0, "metadata": {"simulator": "simpeg", "scenario": "shared"}},
    ]
    scenario_jsonl = scenario_dir / "mixed.jsonl"
    scenario_jsonl.write_text(
        json.dumps(rows[0], ensure_ascii=False)
        + "\n{bad json\n"
        + json.dumps(rows[1], ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(file_catalog, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(file_catalog, "ROUTER_SCENARIO_DIR", scenario_dir)

    total = file_catalog._rewrite_router_train_from_scenarios()

    assert total == 2
    train_rows = [json.loads(line) for line in (router_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert train_rows == rows


def test_resolve_sample_file_rejects_direct_and_prefixed_duplicate_identity(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    row = {"input": "simpeg", "metadata": {"simulator": "simpeg", "scenario": "shared"}}
    (data_dir / "shared.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")
    (data_dir / "mixed.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")

    monkeypatch.setattr(file_catalog.file_manager, "DATA_DIR", data_dir)

    with pytest.raises(ValueError, match="ambiguous sample files"):
        file_catalog.file_manager.resolve_sample_file("shared", simulator="simpeg")


def test_resolve_router_scenario_file_rejects_direct_and_prefixed_duplicate_identity(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    row = {"label": 0, "metadata": {"simulator": "simpeg", "scenario": "shared"}}
    (scenario_dir / "shared.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")
    (scenario_dir / "mixed.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")

    monkeypatch.setattr(file_catalog, "ROUTER_SCENARIO_DIR", scenario_dir)

    with pytest.raises(ValueError, match="ambiguous router scenario files"):
        file_catalog._resolve_router_scenario_file("shared", simulator="simpeg")
