from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException

from PierNet.shared.storage import portable
from PierNet.shared.storage.portable import PartitionInfo
from PierNet.synth.api.routers import config as config_router
from PierNet.synth.api.routers import files as files_router
from PierNet.synth.api.routers import generation as generation_router
from PierNet.synth.services import file_catalog, file_manager, manifest_store


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


def test_clear_samples_deletes_jsonl_and_parquet_partitions(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str | None, str]] = []
    invalidated: list[bool] = []
    sample_partition = _partition("text2comp", "simpeg", "case_a", tmp_path / "case_a")

    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog.file_manager, "clear_all_samples", lambda: 2)
    monkeypatch.setattr(
        file_catalog.portable,
        "discover_partitions",
        lambda kind: [sample_partition] if kind == "text2comp" else [],
    )

    def delete_partition(kind: str, scenario: str, simulator: str | None = None) -> bool:
        calls.append((kind, simulator, scenario))
        return True

    monkeypatch.setattr(file_catalog.portable, "delete_partition", delete_partition)
    monkeypatch.setattr(file_catalog, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append(True))

    result = file_catalog.clear_group("samples")

    assert result == {"ok": True, "kind": "samples", "deleted": 3}
    assert calls == [("text2comp", "simpeg", "case_a")]
    assert invalidated == [True]


def test_clear_router_deletes_jsonl_train_metadata_and_parquet_partitions(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    scenario_jsonl = scenario_dir / "case_b.jsonl"
    scenario_meta = scenario_dir / "case_b.meta.json"
    train_jsonl = router_dir / "train.jsonl"
    scenario_jsonl.write_text("{}\n", encoding="utf-8")
    scenario_meta.write_text("{}", encoding="utf-8")
    train_jsonl.write_text("{}\n", encoding="utf-8")

    calls: list[tuple[str, str | None, str]] = []
    router_partition = _partition("router", "modflow", "case_b", tmp_path / "case_b")

    monkeypatch.setattr(file_catalog, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(file_catalog, "ROUTER_SCENARIO_DIR", scenario_dir)
    monkeypatch.setenv("PierNet_ROUTER_JSONL_CACHE_DIR", str(router_dir / ".parquet_jsonl_cache"))
    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog, "_assert_no_active_training_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog.manifest_store, "rebuild_router_manifest", lambda: {})
    monkeypatch.setattr(
        file_catalog.portable,
        "discover_partitions",
        lambda kind: [router_partition] if kind == "router" else [],
    )

    def delete_partition(kind: str, scenario: str, simulator: str | None = None) -> bool:
        calls.append((kind, simulator, scenario))
        return True

    monkeypatch.setattr(file_catalog.portable, "delete_partition", delete_partition)

    result = file_catalog.clear_group("router")

    assert result == {"ok": True, "kind": "router", "deleted": 4}
    assert not scenario_jsonl.exists()
    assert not scenario_meta.exists()
    assert not train_jsonl.exists()
    assert calls == [("router", "modflow", "case_b")]



def test_catalog_lists_router_jsonl_cache(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / ".parquet_jsonl_cache"
    cache_dir = cache_root / "modflow"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "case_a.jsonl"
    cache_path.write_text('{"label":1}\n', encoding="utf-8")
    cache_path.with_suffix(".meta.json").write_text(
        json.dumps({"source_path": "/tmp/source", "row_count": 1}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setenv("PierNet_ROUTER_JSONL_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(
        file_catalog,
        "_count_jsonl",
        lambda path: (_ for _ in ()).throw(AssertionError("router cache count must use metadata")),
    )

    assets = file_catalog._router_cache_assets()

    assert len(assets) == 1
    asset = assets[0]
    assert asset["kind"] == "router_cache"
    assert asset["deletable"] is True
    assert asset["title"] == "modflow/case_a"
    assert asset["count"] == 1
    assert asset["details"]["row_count"] == 1


def test_router_cache_count_skips_large_file_without_metadata(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "large.jsonl"
    cache_path.write_text('{"label":1}\n', encoding="utf-8")

    monkeypatch.setattr(file_catalog, "_safe_stat", lambda path: (65 * 1024 * 1024, 0.0))
    monkeypatch.setattr(
        file_catalog,
        "_count_jsonl",
        lambda path: (_ for _ in ()).throw(AssertionError("large router cache must not be line-counted")),
    )

    details: dict[str, object] = {}
    assert file_catalog._router_cache_count(cache_path, details) is None
    assert details["count_source"] == "skipped_large_file"


def test_delete_router_parquet_asset_removes_materialized_cache(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / ".parquet_jsonl_cache"
    cache_dir = cache_root / "modflow"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "case_a.jsonl"
    meta_path = cache_path.with_suffix(".meta.json")
    cache_path.write_text('{"label":1}\n', encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    asset_id = file_catalog.encode_asset_id("router_parquet", "modflow", "case_a")

    monkeypatch.setenv("PierNet_ROUTER_JSONL_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog, "_assert_no_active_training_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog.portable, "delete_partition", lambda kind, scenario, simulator=None: True)

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "router_parquet", "deleted": 3}
    assert not cache_path.exists()
    assert not meta_path.exists()


def test_delete_router_cache_asset_removes_jsonl_and_meta(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / ".parquet_jsonl_cache"
    cache_dir = cache_root / "modflow"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "case_a.jsonl"
    meta_path = cache_path.with_suffix(".meta.json")
    cache_path.write_text('{"label":1}\n', encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    asset_id = file_catalog.encode_asset_id("router_cache", "modflow", "case_a")

    monkeypatch.setenv("PierNet_ROUTER_JSONL_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(file_catalog, "_assert_no_active_training_jobs", lambda *args, **kwargs: None)

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "router_cache", "deleted": 2}
    assert not cache_path.exists()
    assert not meta_path.exists()


def test_delete_encoded_router_cache_asset_preserves_portable_scenario(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / ".parquet_jsonl_cache"
    cache_dir = cache_root / "modflow"
    cache_dir.mkdir(parents=True)
    scenario = "case/a"
    cache_path = cache_dir / f"{portable.safe_partition_value(scenario)}.jsonl"
    meta_path = cache_path.with_suffix(".meta.json")
    cache_path.write_text('{"label":1}\n', encoding="utf-8")
    meta_path.write_text(json.dumps({"simulator": "modflow", "scenario": scenario, "row_count": 1}), encoding="utf-8")

    monkeypatch.setenv("PierNet_ROUTER_JSONL_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(file_catalog, "_assert_no_active_training_jobs", lambda *args, **kwargs: None)

    asset = file_catalog._router_cache_assets()[0]
    assert asset["title"] == "modflow/case/a"
    assert file_catalog.decode_asset_id(asset["id"]) == ["router_cache", "modflow", "case/a"]

    result = file_catalog.delete_asset(asset["id"])
    assert result == {"ok": True, "kind": "router_cache", "deleted": 2}
    assert not cache_path.exists()
    assert not meta_path.exists()

def test_delete_sample_parquet_asset_invalidates_text2comp_cache(monkeypatch, tmp_path: Path) -> None:
    invalidated: list[bool] = []
    deleted: list[tuple[str, str | None, str]] = []
    asset_id = file_catalog.encode_asset_id("sample_parquet", "simpeg", "case_a")

    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)

    def delete_partition(kind: str, scenario: str, simulator: str | None = None) -> bool:
        deleted.append((kind, simulator, scenario))
        return True

    monkeypatch.setattr(file_catalog.portable, "delete_partition", delete_partition)
    monkeypatch.setattr(file_catalog, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append(True))

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "sample_parquet", "deleted": 1}
    assert deleted == [("text2comp", "simpeg", "case_a")]
    assert invalidated == [True]


def test_delete_sample_asset_removes_jsonl_and_matching_parquet_partitions(monkeypatch, tmp_path: Path) -> None:
    invalidated: list[bool] = []
    deleted: list[tuple[str, str | None, str]] = []
    asset_id = file_catalog.encode_asset_id("sample", "case_a")
    sample_partition = _partition("text2comp", "simpeg", "case_a", tmp_path / "case_a")
    other_partition = _partition("text2comp", "simpeg", "case_b", tmp_path / "case_b")

    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog.file_manager, "delete_sample_file", lambda scenario, simulator=None: True)
    monkeypatch.setattr(
        file_catalog.portable,
        "discover_partitions",
        lambda kind: [sample_partition, other_partition] if kind == "text2comp" else [],
    )

    def delete_partition(kind: str, scenario: str, simulator: str | None = None) -> bool:
        deleted.append((kind, simulator, scenario))
        return True

    monkeypatch.setattr(file_catalog.portable, "delete_partition", delete_partition)
    monkeypatch.setattr(file_catalog, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append(True))

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "sample", "deleted": 2}
    assert deleted == [("text2comp", "simpeg", "case_a")]
    assert invalidated == [True]




def test_delete_sample_file_removes_stale_indexes(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    index_root = tmp_path / ".indexes"
    data_dir.mkdir()
    sample_path = data_dir / "case_a.jsonl"
    sample_path.write_text(
        json.dumps({"metadata": {"simulator": "simpeg", "scenario": "case_a", "language": "zh"}}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(file_manager, "DATA_DIR", data_dir)
    monkeypatch.setattr(file_manager.jsonl_index, "INDEX_ROOT", index_root)
    monkeypatch.setattr(file_manager.jsonl_filter_index, "INDEX_ROOT", index_root)
    monkeypatch.setattr(file_manager.manifest_store, "rebuild_sample_manifest", lambda: {})
    monkeypatch.setattr(config_router, "invalidate_text2comp_scenarios_cache", lambda: None)

    file_manager.jsonl_index.rebuild_index(sample_path)
    file_manager.jsonl_filter_index.rebuild_filter_index(sample_path, "sample_language_style")
    index_path = file_manager.jsonl_index.get_index_path(sample_path)
    filter_index_path = file_manager.jsonl_filter_index.get_filter_index_path(sample_path, "sample_language_style")

    assert index_path.exists()
    assert filter_index_path.exists()

    assert file_manager.delete_sample_file("case_a", simulator="simpeg") is True

    assert not index_path.exists()
    assert not filter_index_path.exists()


def test_delete_router_scenario_removes_stale_indexes(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    index_root = tmp_path / ".indexes"
    scenario_dir.mkdir(parents=True)
    scenario_jsonl = scenario_dir / "case_a.jsonl"
    scenario_jsonl.write_text(
        json.dumps({"label": 1, "metadata": {"simulator": "simpeg", "scenario": "case_a"}}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    asset_id = file_catalog.encode_asset_id("router_scenario", "simpeg", "case_a")

    monkeypatch.setattr(file_catalog, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(file_catalog, "ROUTER_SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(file_catalog.jsonl_index, "INDEX_ROOT", index_root)
    monkeypatch.setattr(file_catalog.jsonl_filter_index, "INDEX_ROOT", index_root)
    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog, "_assert_no_active_training_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_catalog.manifest_store, "rebuild_router_manifest", lambda: {})
    monkeypatch.setattr(file_catalog.portable, "discover_partitions", lambda kind: [])

    file_catalog.jsonl_index.rebuild_index(scenario_jsonl)
    file_catalog.jsonl_filter_index.rebuild_filter_index(scenario_jsonl, "router_label")
    index_path = file_catalog.jsonl_index.get_index_path(scenario_jsonl)
    filter_index_path = file_catalog.jsonl_filter_index.get_filter_index_path(scenario_jsonl, "router_label")

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "router_scenario", "deleted": 1, "train_count": 0}
    assert not index_path.exists()
    assert not filter_index_path.exists()


def test_rebuild_indexes_prunes_stale_sample_indexes(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    index_root = tmp_path / ".indexes"
    data_dir.mkdir()

    monkeypatch.setattr(file_catalog, "DATA_DIR", data_dir)
    monkeypatch.setattr(file_catalog.jsonl_index, "INDEX_ROOT", index_root)
    monkeypatch.setattr(file_catalog.jsonl_index, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(file_catalog.jsonl_index, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(file_catalog.jsonl_filter_index, "INDEX_ROOT", index_root)
    monkeypatch.setattr(file_catalog.jsonl_filter_index, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(file_catalog.jsonl_filter_index, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(file_catalog.manifest_store, "rebuild_sample_manifest", lambda: {})

    stale_source = data_dir / "deleted.jsonl"
    stale_index = file_catalog.jsonl_index.get_index_path(stale_source)
    stale_index.parent.mkdir(parents=True)
    stale_index.write_text("{}", encoding="utf-8")

    result = file_catalog.rebuild_indexes("samples")

    assert result["ok"] is True
    assert result["deleted_indexes"] == 1
    assert not stale_index.exists()

def test_delete_sample_file_resolves_metadata_scenario_for_prefixed_jsonl(monkeypatch, tmp_path: Path) -> None:
    invalidated: list[bool] = []
    sample_dir = tmp_path / "text2comp"
    sample_dir.mkdir()
    sample_path = sample_dir / "gcam_shared.jsonl"
    sample_path.write_text(
        json.dumps({"metadata": {"simulator": "gcam", "scenario": "shared"}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(file_manager, "DATA_DIR", sample_dir)
    monkeypatch.setattr(file_manager.manifest_store, "rebuild_sample_manifest", lambda: {})
    monkeypatch.setattr(config_router, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append(True))

    assert file_manager.delete_sample_file("shared", simulator="gcam") is True

    assert not sample_path.exists()
    assert invalidated == [True]


def test_delete_sample_file_rejects_path_component(monkeypatch, tmp_path: Path) -> None:
    sample_dir = tmp_path / "text2comp"
    sample_dir.mkdir()
    outside = tmp_path / "evil.jsonl"
    outside.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(file_manager, "DATA_DIR", sample_dir)

    assert file_manager.delete_sample_file("../evil") is False
    assert outside.exists()


def test_delete_sample_file_invalidates_text2comp_cache(monkeypatch, tmp_path: Path) -> None:
    invalidated: list[bool] = []
    sample_dir = tmp_path / "text2comp"
    sample_dir.mkdir()
    (sample_dir / "case_a.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(file_manager, "DATA_DIR", sample_dir)
    monkeypatch.setattr(file_manager.manifest_store, "rebuild_sample_manifest", lambda: {})
    monkeypatch.setattr(config_router, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append(True))

    assert file_manager.delete_sample_file("case_a") is True

    assert not (sample_dir / "case_a.jsonl").exists()
    assert invalidated == [True]


def test_delete_asset_rejects_path_component_before_template_unlink(monkeypatch, tmp_path: Path) -> None:
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    outside = tmp_path / "evil_templates.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    asset_id = file_catalog.encode_asset_id("template", "../evil")

    monkeypatch.setattr(file_catalog, "_assert_no_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(file_manager, "TEMPLATES_DIR", templates_dir)

    try:
        file_catalog.delete_asset(asset_id)
    except ValueError as exc:
        assert "scenario must be a file name component" in str(exc)
    else:
        raise AssertionError("expected ValueError for template path traversal")

    assert outside.exists()

def test_template_routes_reject_path_component(monkeypatch, tmp_path: Path) -> None:
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    outside = tmp_path / "evil_templates.jsonl"
    payload = '{"scenario":"evil"}\n'
    outside.write_text(payload, encoding="utf-8")

    monkeypatch.setattr(files_router, "TEMPLATES_DIR", templates_dir)

    for action in (
        lambda: files_router.get_template_items("../evil"),
    ):
        try:
            action()
        except HTTPException as exc:
            assert exc.status_code == 400
            assert "file name component" in str(exc.detail)
        else:
            raise AssertionError("expected HTTPException for template path traversal")

    assert outside.read_text(encoding="utf-8") == payload


def test_template_scenario_name_preserves_internal_templates_token(monkeypatch, tmp_path: Path) -> None:
    templates_dir = tmp_path / "data" / "templates"
    templates_dir.mkdir(parents=True)
    scenario = "case_templates_inner"
    template_path = templates_dir / f"{scenario}_templates.jsonl"
    template_path.write_text(
        json.dumps({"language": "en", "style": "technical"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(generation_router, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(manifest_store, "TEMPLATES_DIR", templates_dir)
    monkeypatch.setattr(file_manager, "TEMPLATES_DIR", templates_dir)

    template_info = generation_router.get_templates_status()[0]
    assert template_info.scenario == scenario
    assert template_info.simulator is None
    assert manifest_store._template_manifest_payload([])["items"][0]["scenario"] == scenario
    assert file_manager._legacy_list_template_files()[0].scenario == scenario


def test_template_file_listing_includes_simulator_from_manifest(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "shared_templates.jsonl"
    template_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        file_manager.manifest_store,
        "ensure_template_manifest",
        lambda: {
            "items": [
                {
                    "scenario": "shared",
                    "simulator": "gcam",
                    "template_count": 1,
                    "file_size_bytes": template_path.stat().st_size,
                    "mtime": template_path.stat().st_mtime,
                    "path": str(template_path),
                }
            ]
        },
    )

    item = file_manager.list_template_files()[0]

    assert item.scenario == "shared"
    assert item.simulator == "gcam"


def test_template_manifest_includes_simulator_from_records(monkeypatch, tmp_path: Path) -> None:
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    template_path = templates_dir / "shared_templates.jsonl"
    template_path.write_text(
        json.dumps(
            {
                "simulator": "gcam",
                "scenario": "shared",
                "language": "en",
                "style": "technical",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(manifest_store, "TEMPLATES_DIR", templates_dir)

    item = manifest_store._template_manifest_payload([])["items"][0]

    assert item["scenario"] == "shared"
    assert item["simulator"] == "gcam"


def test_catalog_protects_stage1_hdf5_and_stage2_template_assets(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    h5_path = data_root / "modflow" / "modflow_external.h5"
    h5_path.parent.mkdir(parents=True)
    h5_path.write_bytes(b"placeholder")
    template_path = data_root / "templates" / "external_templates.jsonl"
    template_path.parent.mkdir(parents=True)
    template_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(file_catalog, "DATA_ROOT", data_root)
    monkeypatch.setattr(file_catalog.hdf5_data, "DATA_ROOT", data_root)
    monkeypatch.setattr(
        file_catalog.manifest_store,
        "ensure_template_manifest",
        lambda: {
            "items": [
                {
                    "scenario": "external",
                    "path": str(template_path),
                    "template_count": 1,
                    "file_size_bytes": template_path.stat().st_size,
                    "mtime": template_path.stat().st_mtime,
                }
            ]
        },
    )

    assets = file_catalog._hdf5_assets() + file_catalog._template_assets()
    protected = {asset["kind"]: asset for asset in assets}

    assert protected["template"]["simulator"] is None
    assert protected["hdf5"]["protected"] is True
    assert protected["hdf5"]["deletable"] is False
    assert protected["template"]["protected"] is True
    assert protected["template"]["deletable"] is False


def test_delete_hdf5_asset_is_protected_without_unlink(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    h5_path = data_root / "modflow" / "external.h5"
    h5_path.parent.mkdir(parents=True)
    h5_path.write_bytes(b"placeholder")
    asset_id = file_catalog.encode_asset_id("hdf5", "modflow", "external")

    monkeypatch.setattr(file_catalog, "DATA_ROOT", data_root)

    try:
        file_catalog.delete_asset(asset_id)
    except ValueError as exc:
        assert "Stage 1 HDF5 files are protected" in str(exc)
    else:
        raise AssertionError("expected ValueError for protected HDF5 asset")

    assert h5_path.exists()


def test_delete_template_asset_is_protected_without_unlink(monkeypatch, tmp_path: Path) -> None:
    templates_dir = tmp_path / "templates"
    template_path = templates_dir / "external_templates.jsonl"
    template_path.parent.mkdir(parents=True)
    template_path.write_text("{}\n", encoding="utf-8")
    asset_id = file_catalog.encode_asset_id("template", "external")

    monkeypatch.setattr(file_catalog.file_manager, "TEMPLATES_DIR", templates_dir)

    try:
        file_catalog.delete_asset(asset_id)
    except ValueError as exc:
        assert "Stage 2 template files are protected" in str(exc)
    else:
        raise AssertionError("expected ValueError for protected template asset")

    assert template_path.exists()


def test_delete_sample_merged_asset_is_protected_without_unlink(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    merged_path = data_dir / "all_training_data.jsonl"
    merged_path.parent.mkdir(parents=True)
    merged_path.write_text("", encoding="utf-8")
    asset_id = file_catalog.encode_asset_id("sample_merged", "all_training_data")

    monkeypatch.setattr(file_catalog, "DATA_DIR", data_dir)

    try:
        file_catalog.delete_asset(asset_id)
    except ValueError as exc:
        assert "merged sample file is protected" in str(exc)
    else:
        raise AssertionError("expected ValueError for protected merged sample asset")

    assert merged_path.exists()



def test_training_prepared_cache_asset_is_listed_and_protected_when_active(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "token_router"
    prepared_dir = artifact_root / "modflow" / "prepared" / "modflow-abcd12"
    prepared_dir.mkdir(parents=True)
    (prepared_dir / "meta.json").write_text(
        json.dumps(
            {
                "prepared_format": "router_cached_token_ids_v4",
                "scenarios": ["case_a"],
                "input_representation": "pretrained_embeddings",
                "train_samples": 3,
                "test_samples": 1,
                "source_fingerprint": "abc",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(file_catalog.training_manager, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(
        file_catalog.training_manager,
        "list_jobs",
        lambda refresh=True: [
            {
                "job_id": "train-active",
                "status": "running",
                "simulator": "modflow",
                "prepared_name": "modflow-abcd12",
            }
        ],
    )

    asset = next(asset for asset in file_catalog._training_assets() if asset["kind"] == "training_prepared")

    assert asset["kind"] == "training_prepared"
    assert asset["title"] == "modflow/modflow-abcd12"
    assert asset["count"] == 4
    assert asset["protected"] is True
    assert asset["deletable"] is False



def test_training_prepared_cache_is_protected_for_legacy_active_job_without_prepared_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts" / "token_router"
    prepared_dir = artifact_root / "modflow" / "prepared" / "modflow-abcd12"
    prepared_dir.mkdir(parents=True)
    (prepared_dir / "meta.json").write_text(
        json.dumps({"train_samples": 1, "test_samples": 0}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(file_catalog.training_manager, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(
        file_catalog.training_manager,
        "list_jobs",
        lambda refresh=True: [{"job_id": "train-legacy", "status": "running", "simulator": "modflow"}],
    )

    asset = next(asset for asset in file_catalog._training_assets() if asset["kind"] == "training_prepared")

    assert asset["protected"] is True
    assert asset["deletable"] is False

    try:
        file_catalog.delete_asset(file_catalog.encode_asset_id("training_prepared", "modflow", "modflow-abcd12"))
    except RuntimeError as exc:
        assert "train-legacy" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for legacy active prepared cache reference")


def test_training_prepared_cache_with_bad_sample_counts_is_listed_as_invalid(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts" / "token_router"
    prepared_dir = artifact_root / "modflow" / "prepared" / "bad-counts"
    prepared_dir.mkdir(parents=True)
    (prepared_dir / "meta.json").write_text(
        json.dumps({"train_samples": "bad", "test_samples": -2}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(file_catalog.training_manager, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(file_catalog.training_manager, "list_jobs", lambda refresh=True: [])

    asset = next(asset for asset in file_catalog._training_assets() if asset["kind"] == "training_prepared")

    assert asset["valid"] is False
    assert asset["status"] == "invalid"
    assert asset["count"] == 0
    assert "样本计数字段无效" in asset["errors"][0]

def test_delete_training_prepared_cache_removes_inactive_directory(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "token_router"
    prepared_dir = artifact_root / "modflow" / "prepared" / "modflow-abcd12"
    prepared_dir.mkdir(parents=True)
    (prepared_dir / "meta.json").write_text("{}", encoding="utf-8")
    asset_id = file_catalog.encode_asset_id("training_prepared", "modflow", "modflow-abcd12")

    monkeypatch.setattr(file_catalog.training_manager, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(file_catalog.training_manager, "list_jobs", lambda refresh=True: [])

    result = file_catalog.delete_asset(asset_id)

    assert result == {"ok": True, "kind": "training_prepared", "deleted": 1}
    assert not prepared_dir.exists()


def test_delete_training_prepared_cache_rejects_active_reference(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "token_router"
    prepared_dir = artifact_root / "modflow" / "prepared" / "modflow-abcd12"
    prepared_dir.mkdir(parents=True)
    asset_id = file_catalog.encode_asset_id("training_prepared", "modflow", "modflow-abcd12")

    monkeypatch.setattr(file_catalog.training_manager, "ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(
        file_catalog.training_manager,
        "list_jobs",
        lambda refresh=True: [
            {
                "job_id": "train-active",
                "status": "queued",
                "simulator": "modflow",
                "prepared_name": "modflow-abcd12",
            }
        ],
    )

    try:
        file_catalog.delete_asset(asset_id)
    except RuntimeError as exc:
        assert "正在被任务使用" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for active prepared cache")

    assert prepared_dir.exists()

def test_empty_sample_merged_asset_is_listed_as_protected(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    merged_path = data_dir / "all_training_data.jsonl"
    merged_path.parent.mkdir(parents=True)
    merged_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(file_catalog, "DATA_DIR", data_dir)
    monkeypatch.setattr(file_catalog.manifest_store, "ensure_sample_manifest", lambda: {"items": [], "summary": {}})

    assets = file_catalog._sample_assets()
    merged = next(asset for asset in assets if asset["kind"] == "sample_merged")

    assert merged["protected"] is True
    assert merged["deletable"] is False
    assert "受保护" in merged["warnings"][0]


def test_manifest_index_assets_skip_temporary_index_files(monkeypatch, tmp_path: Path) -> None:
    manifest_dir = tmp_path / ".manifests"
    index_dir = tmp_path / ".indexes"
    manifest_dir.mkdir()
    index_dir.mkdir()
    stable_index = index_dir / "data" / "templates" / "case_templates.jsonl.idx.json"
    tmp_index = index_dir / ".tmp" / "job-1" / "case.jsonl.idx.json"
    stable_index.parent.mkdir(parents=True)
    tmp_index.parent.mkdir(parents=True)
    stable_index.write_text("{}", encoding="utf-8")
    tmp_index.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(file_catalog, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(file_catalog, "INDEX_DIR", index_dir)

    assets = file_catalog._manifest_index_assets()

    assert [asset["title"] for asset in assets] == ["data/templates/case_templates.jsonl.idx.json"]


def test_clear_templates_group_is_protected() -> None:
    try:
        file_catalog.clear_group("templates")
    except ValueError as exc:
        assert "Stage 2 template files are protected" in str(exc)
    else:
        raise AssertionError("expected ValueError for protected templates group")
