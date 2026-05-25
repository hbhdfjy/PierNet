from __future__ import annotations

import json
from pathlib import Path

from piern.shared.storage import portable
from scripts.ci import check_migration_ready
from scripts.storage import build_catalog_db, migrate_jsonl_to_parquet


def test_build_catalog_db_reads_jsonl_identity_from_metadata(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sample_dir = data_root / "text2comp"
    router_dir = data_root / "router" / "by_scenario"
    sample_dir.mkdir(parents=True)
    router_dir.mkdir(parents=True)
    (sample_dir / "gcam_shared.jsonl").write_text(
        json.dumps({"metadata": {"simulator": "gcam", "scenario": "shared"}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (router_dir / "simpeg_shared.jsonl").write_text(
        json.dumps({"metadata": {"simulator": "simpeg", "scenario": "shared"}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(build_catalog_db, "DATA_ROOT", data_root)

    rows = build_catalog_db.legacy_jsonl_assets()

    assert {(row["kind"], row["simulator"], row["scenario"]) for row in rows} == {
        ("text2comp_jsonl", "gcam", "shared"),
        ("router_jsonl", "simpeg", "shared"),
    }
    assert {row["id"] for row in rows} == {
        "jsonl:text2comp_jsonl:gcam:shared",
        "jsonl:router_jsonl:simpeg:shared",
    }


def test_build_catalog_db_strips_simulator_prefix_from_hdf5_scenario(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    h5_dir = data_root / "modflow"
    h5_dir.mkdir(parents=True)
    h5_file = h5_dir / "modflow_coastal.h5"
    h5_file.write_bytes(b"hdf5 placeholder")

    monkeypatch.setattr(build_catalog_db, "DATA_ROOT", data_root)

    rows = build_catalog_db.hdf5_assets()

    assert [(row["simulator"], row["scenario"], row["id"]) for row in rows] == [
        ("modflow", "coastal", "hdf5:modflow:coastal")
    ]


def test_build_catalog_db_discovers_uppercase_hdf5_suffix(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    h5_dir = data_root / "simpeg"
    h5_dir.mkdir(parents=True)
    (h5_dir / "simpeg_external.HDF5").write_bytes(b"hdf5 placeholder")

    monkeypatch.setattr(build_catalog_db, "DATA_ROOT", data_root)

    rows = build_catalog_db.hdf5_assets()

    assert [(row["simulator"], row["scenario"], row["id"]) for row in rows] == [
        ("simpeg", "external", "hdf5:simpeg:external")
    ]


def test_migrate_jsonl_filters_sources_by_metadata_scenario(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    sample_dir = data_root / "text2comp"
    sample_dir.mkdir(parents=True)
    sample_path = sample_dir / "gcam_shared.jsonl"
    sample_path.write_text(
        json.dumps({"metadata": {"simulator": "gcam", "scenario": "shared"}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(migrate_jsonl_to_parquet, "DATA_ROOT", data_root)

    assert migrate_jsonl_to_parquet.iter_sources("text2comp", {"shared"}) == [sample_path]
    assert migrate_jsonl_to_parquet.iter_sources("text2comp", {"gcam/shared"}) == [sample_path]
    assert migrate_jsonl_to_parquet.iter_sources("text2comp", {"gcam_shared"}) == [sample_path]
    assert migrate_jsonl_to_parquet.iter_sources("text2comp", {"simpeg/shared"}) == []



def test_discover_partitions_falls_back_when_manifest_row_count_is_invalid(monkeypatch, tmp_path: Path) -> None:
    parquet_root = tmp_path / "text2comp_parquet"
    monkeypatch.setattr(portable, "TEXT2COMP_PARQUET_DIR", parquet_root)
    result = portable.write_records_partition(
        "text2comp",
        [
            {
                "input": "describe the case",
                "output": "case summary",
                "metadata": {
                    "simulator": "modflow",
                    "scenario": "coastal",
                    "language": "zh",
                    "style": "technical",
                },
            }
        ],
        simulator="modflow",
        scenario="coastal",
    )
    manifest_path = Path(result["path"]) / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = "not-an-int"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    partitions = portable.discover_partitions("text2comp")

    assert len(partitions) == 1
    assert partitions[0].row_count == 1


def test_parquet_partition_values_do_not_collapse_path_separators(tmp_path: Path) -> None:
    root = tmp_path / "text2comp_parquet"

    slash_path = portable.partition_dir_for("text2comp", "sim/peg", "case/a", root)
    underscore_path = portable.partition_dir_for("text2comp", "sim_peg", "case_a", root)

    assert slash_path != underscore_path
    assert slash_path.parts[-2:] == ("simulator=sim%2Fpeg", "scenario=case%2Fa")
    assert underscore_path.parts[-2:] == ("simulator=sim_peg", "scenario=case_a")


def test_migration_ready_accepts_uppercase_tracked_hdf5_suffix() -> None:
    report = check_migration_ready.Report()

    check_migration_ready.check_tracked_data(
        report,
        [
            "data/.gitignore",
            "data/templates/case_templates.jsonl",
            "data/modflow/modflow_case.HDF5",
        ],
    )

    assert report.errors == []
    assert report.info == ["OK: tracked data boundary clean: 1 HDF5, 1 template JSONL"]
