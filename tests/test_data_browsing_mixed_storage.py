from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from PierNet.shared.storage import portable
from PierNet.synth.api.routers import datasets, router_data
from PierNet.synth.services import jsonl_filter_index, jsonl_index, manifest_store


@pytest.fixture(autouse=True)
def _isolate_jsonl_indexes(monkeypatch, tmp_path: Path) -> None:
    index_root = tmp_path / ".indexes"
    monkeypatch.setattr(jsonl_index, "INDEX_ROOT", index_root)
    monkeypatch.setattr(jsonl_filter_index, "INDEX_ROOT", index_root)


def test_samples_fall_back_to_jsonl_when_other_parquet_partitions_exist(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    sample_path = data_dir / "jsonl_only.jsonl"
    sample = {"input": "hello", "metadata": {"language": "zh", "style": "technical"}}
    sample_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.file_manager, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.portable, "partition_for", lambda kind, scenario, simulator=None: None)
    monkeypatch.setattr(
        datasets.portable,
        "read_text2comp_page",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not read parquet")),
    )
    monkeypatch.setattr(datasets, "_sample_total_from_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        datasets.jsonl_index,
        "read_page",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no index")),
    )

    response = datasets.get_samples(scenario="jsonl_only", page=0, page_size=20, language=None, style=None)

    assert response["total"] == 1
    assert response["items"] == [sample]


def test_samples_filter_jsonl_by_simulator_without_language_index(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    sample_path = data_dir / "shared.jsonl"
    gcam_sample = {
        "input": "gcam",
        "metadata": {"simulator": "gcam", "scenario": "shared", "language": "zh", "style": "technical"},
    }
    simpeg_sample = {
        "input": "simpeg",
        "metadata": {"simulator": "simpeg", "scenario": "shared", "language": "zh", "style": "technical"},
    }
    sample_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in [gcam_sample, simpeg_sample]),
        encoding="utf-8",
    )

    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.file_manager, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.portable, "partition_for", lambda kind, scenario, simulator=None: None)
    monkeypatch.setattr(
        datasets.jsonl_filter_index,
        "read_filtered_page",
        lambda *args, **kwargs: (2, [gcam_sample, simpeg_sample]),
    )

    response = datasets.get_samples(
        scenario="shared",
        simulator="gcam",
        language="zh",
        style="technical",
        page=0,
        page_size=20,
    )

    assert response["total"] == 1
    assert response["items"] == [gcam_sample]


def test_samples_resolve_prefixed_jsonl_by_metadata_scenario(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    sample = {"input": "gcam", "metadata": {"simulator": "gcam", "scenario": "shared"}}
    (data_dir / "gcam_shared.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.file_manager, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    response = datasets.get_samples(scenario="shared", simulator="gcam", page=0, page_size=20)

    assert response["total"] == 1
    assert response["items"] == [sample]


def test_samples_scan_mixed_identity_jsonl_when_bare_scenario_is_requested(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    rows = [
        {"input": "gcam", "metadata": {"simulator": "gcam", "scenario": "shared", "language": "en"}},
        {"input": "simpeg", "metadata": {"simulator": "simpeg", "scenario": "shared", "language": "zh"}},
        {"input": "other", "metadata": {"simulator": "gcam", "scenario": "other", "language": "zh"}},
    ]
    sample_path = data_dir / "shared.jsonl"
    sample_path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
    manifest = {
        "items": [
            {"scenario": "shared", "simulator": "gcam", "sample_count": 1, "path": str(sample_path)},
            {"scenario": "shared", "simulator": "simpeg", "sample_count": 1, "path": str(sample_path)},
            {"scenario": "other", "simulator": "gcam", "sample_count": 1, "path": str(sample_path)},
        ]
    }

    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.file_manager, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.portable, "partition_for", lambda kind, scenario, simulator=None: None)
    monkeypatch.setattr(datasets.manifest_store, "ensure_sample_manifest", lambda: manifest)
    monkeypatch.setattr(
        datasets.jsonl_index,
        "read_page",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mixed identity files need record filtering")),
    )
    monkeypatch.setattr(
        datasets.jsonl_filter_index,
        "read_filtered_page",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mixed identity files need scenario filtering")),
    )

    response = datasets.get_samples(scenario="shared", page=0, page_size=20)
    filtered = datasets.get_samples(scenario="shared", language="zh", page=0, page_size=20)

    assert response["total"] == 2
    assert response["items"] == rows[:2]
    assert filtered["total"] == 1
    assert filtered["items"] == [rows[1]]



def test_samples_resolve_non_direct_mixed_identity_jsonl(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    rows = [
        {"input": "gcam", "metadata": {"simulator": "gcam", "scenario": "shared"}},
        {"input": "simpeg", "metadata": {"simulator": "simpeg", "scenario": "shared"}},
        {"input": "fallback"},
    ]
    (data_dir / "mixed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.file_manager, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    response = datasets.get_samples(scenario="shared", simulator="simpeg", page=0, page_size=20)
    bare_response = datasets.get_samples(scenario="shared", page=0, page_size=20)

    assert response["total"] == 1
    assert response["items"] == [rows[1]]
    assert bare_response["total"] == 2
    assert bare_response["items"] == rows[:2]


def test_samples_reject_direct_and_prefixed_duplicate_identity(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    row = {"input": "simpeg", "metadata": {"simulator": "simpeg", "scenario": "shared"}}
    (data_dir / "shared.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")
    (data_dir / "mixed.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")

    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.file_manager, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    with pytest.raises(HTTPException) as exc_info:
        datasets.get_samples(scenario="shared", simulator="simpeg", page=0, page_size=20)

    assert exc_info.value.status_code == 409
    assert "ambiguous sample files" in str(exc_info.value.detail)


def test_legacy_datasets_use_metadata_scenario_for_prefixed_jsonl(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    sample = {"input": "gcam", "metadata": {"simulator": "gcam", "scenario": "shared"}}
    sample_path = data_dir / "gcam_shared.jsonl"
    sample_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)

    result = datasets._legacy_get_datasets()

    assert result == [
        {
            "name": "shared",
            "simulator": "gcam",
            "scenario": "shared",
            "sample_count": 1,
            "file_size_bytes": sample_path.stat().st_size,
            "mtime": sample_path.stat().st_mtime,
        }
    ]


def test_samples_filter_jsonl_by_stripped_simulator(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    sample = {"input": "gcam", "metadata": {"simulator": "gcam", "scenario": "shared"}}
    (data_dir / "shared.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.file_manager, "DATA_DIR", data_dir)
    monkeypatch.setattr(datasets.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    response = datasets.get_samples(scenario="shared", simulator=" gcam ", page=0, page_size=20)

    assert response["total"] == 1
    assert response["items"] == [sample]


def test_router_samples_fall_back_to_jsonl_when_other_parquet_partitions_exist(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    router_sample = {"context": "route me", "label": 1, "metadata": {"scenario": "jsonl_only"}}
    (scenario_dir / "jsonl_only.jsonl").write_text(
        json.dumps(router_sample, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: True)
    monkeypatch.setattr(router_data.portable, "partition_for", lambda kind, scenario, simulator=None: None)
    monkeypatch.setattr(
        router_data.portable,
        "read_router_page",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not read parquet")),
    )
    monkeypatch.setattr(router_data, "_router_total_from_manifest", lambda *_args, **_kwargs: None)

    response = router_data.get_router_samples(scenario="jsonl_only", page=0, page_size=20, label=-1)

    assert response["total"] == 1
    assert response["items"] == [router_sample]



def test_router_samples_resolve_prefixed_jsonl_by_metadata_scenario(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    router_sample = {"context": "gcam", "label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}}
    (scenario_dir / "gcam_shared.jsonl").write_text(
        json.dumps(router_sample, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: False)
    monkeypatch.setattr(router_data.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    response = router_data.get_router_samples(scenario="shared", simulator="gcam", page=0, page_size=20, label=-1)

    assert response["total"] == 1
    assert response["items"] == [router_sample]



def test_router_samples_scan_mixed_identity_jsonl_when_bare_scenario_is_requested(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    rows = [
        {"context": "gcam", "label": "1", "metadata": {"simulator": "gcam", "scenario": "shared"}},
        {"context": "simpeg", "label": 0, "metadata": {"simulator": "simpeg", "scenario": "shared"}},
        {"context": "other", "label": 1, "metadata": {"simulator": "gcam", "scenario": "other"}},
    ]
    path = scenario_dir / "shared.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
    manifest = {
        "kind": "router_manifest",
        "scenarios": [
            {"scenario": "shared", "simulator": "gcam", "router_count": 1, "path": str(path)},
            {"scenario": "shared", "simulator": "simpeg", "router_count": 1, "path": str(path)},
            {"scenario": "other", "simulator": "gcam", "router_count": 1, "path": str(path)},
        ],
        "splits": {"train": {"exists": True, "count": 3}},
    }

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: False)
    monkeypatch.setattr(router_data.portable, "partition_for", lambda kind, scenario, simulator=None: None)
    monkeypatch.setattr(router_data.portable, "router_manifest_like", lambda: {})
    monkeypatch.setattr(router_data.manifest_store, "ensure_router_manifest", lambda: manifest)
    monkeypatch.setattr(
        router_data.jsonl_index,
        "read_page",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mixed identity files need record filtering")),
    )
    monkeypatch.setattr(
        router_data.jsonl_filter_index,
        "read_filtered_page",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mixed identity files need scenario filtering")),
    )

    response = router_data.get_router_samples(scenario="shared", page=0, page_size=20, label=-1)
    label_filtered = router_data.get_router_samples(scenario="shared", page=0, page_size=20, label=1)
    simulator_filtered = router_data.get_router_samples(scenario="shared", simulator="gcam", page=0, page_size=20, label=-1)

    assert response["total"] == 2
    assert response["items"] == rows[:2]
    assert label_filtered["total"] == 1
    assert label_filtered["items"] == [rows[0]]
    assert simulator_filtered["total"] == 1
    assert simulator_filtered["items"] == [rows[0]]



def test_router_samples_resolve_non_direct_mixed_identity_jsonl(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    rows = [
        {"context": "gcam", "label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}},
        {"context": "simpeg", "label": 0, "metadata": {"simulator": "simpeg", "scenario": "shared"}},
        {"context": "fallback", "label": 1},
    ]
    (scenario_dir / "mixed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: False)
    monkeypatch.setattr(router_data.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    response = router_data.get_router_samples(scenario="shared", simulator="simpeg", page=0, page_size=20, label=-1)
    bare_response = router_data.get_router_samples(scenario="shared", page=0, page_size=20, label=-1)

    assert response["total"] == 1
    assert response["items"] == [rows[1]]
    assert bare_response["total"] == 2
    assert bare_response["items"] == rows[:2]


def test_router_samples_reject_direct_and_prefixed_duplicate_identity(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    row = {"context": "simpeg", "label": 0, "metadata": {"simulator": "simpeg", "scenario": "shared"}}
    (scenario_dir / "shared.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")
    (scenario_dir / "mixed.jsonl").write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: False)
    monkeypatch.setattr(router_data.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    with pytest.raises(HTTPException) as exc_info:
        router_data.get_router_samples(scenario="shared", simulator="simpeg", page=0, page_size=20, label=-1)

    assert exc_info.value.status_code == 409
    assert "多个 JSONL 文件" in str(exc_info.value.detail)


def test_router_samples_filter_train_jsonl_by_simulator(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    gcam_sample = {"context": "gcam", "label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}}
    simpeg_sample = {"context": "simpeg", "label": 1, "metadata": {"simulator": "simpeg", "scenario": "shared"}}
    (router_dir / "train.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in [gcam_sample, simpeg_sample]),
        encoding="utf-8",
    )

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: False)
    monkeypatch.setattr(router_data.portable, "partition_for", lambda kind, scenario, simulator=None: None)
    monkeypatch.setattr(router_data, "_combined_router_manifest", lambda: {})

    response = router_data.get_router_samples(simulator="gcam", page=0, page_size=20, label=-1)

    assert response["total"] == 1
    assert response["items"] == [gcam_sample]


def test_router_samples_merge_mixed_storage_when_no_scenario_is_requested(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    jsonl_sample = {"context": "jsonl", "label": 0, "metadata": {"simulator": "gcam", "scenario": "jsonl_only"}}
    parquet_sample = {"context": "parquet", "label": 1, "metadata": {"simulator": "simpeg", "scenario": "parquet_only"}}
    jsonl_path = scenario_dir / "jsonl_only.jsonl"
    jsonl_path.write_text(json.dumps(jsonl_sample, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "kind": "router_manifest",
        "storage": "mixed",
        "scenarios": [
            {
                "scenario": "jsonl_only",
                "simulator": "gcam",
                "router_count": 1,
                "path": str(jsonl_path),
                "storage": "jsonl",
            },
            {
                "scenario": "parquet_only",
                "simulator": "simpeg",
                "router_count": 1,
                "storage": "parquet",
            },
        ],
    }

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data, "_combined_router_manifest", lambda: manifest)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: True)
    monkeypatch.setattr(router_data.portable, "iter_records", lambda kind, filters=(): iter([parquet_sample]))
    monkeypatch.setattr(
        router_data.portable,
        "read_router_page",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("mixed storage must not return parquet-only pages")),
    )

    response = router_data.get_router_samples(page=0, page_size=20, label=-1)

    assert response["storage"] == "mixed"
    assert response["total"] == 2
    assert response["items"] == [parquet_sample, jsonl_sample]


def test_router_samples_filter_mixed_jsonl_string_labels(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    jsonl_sample = {"context": "jsonl", "label": "1", "metadata": {"simulator": "gcam", "scenario": "jsonl_only"}}
    jsonl_path = scenario_dir / "jsonl_only.jsonl"
    jsonl_path.write_text(json.dumps(jsonl_sample, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "kind": "router_manifest",
        "storage": "mixed",
        "scenarios": [
            {
                "scenario": "jsonl_only",
                "simulator": "gcam",
                "router_count": 1,
                "path": str(jsonl_path),
                "storage": "jsonl",
            }
        ],
    }

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data, "_combined_router_manifest", lambda: manifest)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: False)

    response = router_data.get_router_samples(page=0, page_size=20, label=1)

    assert response["storage"] == "mixed"
    assert response["total"] == 1
    assert response["items"] == [jsonl_sample]


def test_router_samples_filter_jsonl_by_simulator(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    gcam_sample = {"context": "gcam", "label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}}
    simpeg_sample = {"context": "simpeg", "label": 1, "metadata": {"simulator": "simpeg", "scenario": "shared"}}
    (scenario_dir / "shared.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in [gcam_sample, simpeg_sample]),
        encoding="utf-8",
    )

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: False)
    monkeypatch.setattr(router_data.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    response = router_data.get_router_samples(scenario="shared", simulator="gcam", page=0, page_size=20, label=1)

    assert response["total"] == 1
    assert response["items"] == [gcam_sample]


def test_router_samples_filter_jsonl_by_stripped_simulator(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    sample = {"context": "gcam", "label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}}
    (scenario_dir / "shared.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: False)
    monkeypatch.setattr(router_data.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    response = router_data.get_router_samples(scenario="shared", simulator=" gcam ", page=0, page_size=20, label=-1)

    assert response["total"] == 1
    assert response["items"] == [sample]


@pytest.mark.parametrize(
    ("kwargs", "expected_field"),
    [
        ({"scenario": "../../outside"}, "scenario"),
        ({"split": "../outside"}, "split"),
        ({"scenario": "shared", "simulator": "../gcam"}, "simulator"),
    ],
)
def test_router_samples_rejects_unsafe_name_components(monkeypatch, tmp_path: Path, kwargs, expected_field) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    scenario_dir.mkdir(parents=True)
    outside_sample = {"context": "outside", "label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}}
    (tmp_path / "outside.jsonl").write_text(
        json.dumps(outside_sample, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "shared.jsonl").write_text(
        json.dumps(outside_sample, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data.portable, "has_partitions", lambda kind: False)
    monkeypatch.setattr(router_data.portable, "partition_for", lambda kind, scenario, simulator=None: None)

    params = {"scenario": "", "split": "train", "simulator": "", "page": 0, "page_size": 20, "label": -1}
    params.update(kwargs)
    with pytest.raises(HTTPException) as exc_info:
        router_data.get_router_samples(**params)

    assert exc_info.value.status_code == 400
    assert expected_field in str(exc_info.value.detail)


def test_jsonl_manifests_use_metadata_scenario_for_prefixed_files(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    data_dir.mkdir()
    scenario_dir.mkdir(parents=True)

    sample = {
        "input": "gcam sample",
        "metadata": {"simulator": "gcam", "scenario": "shared", "language": "en", "style": "technical"},
    }
    router_row = {"context": "gcam route", "label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}}
    (data_dir / "gcam_shared.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    (scenario_dir / "gcam_shared.jsonl").write_text(
        json.dumps(router_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(manifest_store, "DATA_DIR", data_dir)
    monkeypatch.setattr(manifest_store, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(manifest_store, "ROUTER_SCENARIO_DIR", scenario_dir)

    sample_manifest = manifest_store._sample_manifest_payload([])
    router_manifest = manifest_store._router_manifest_payload([])

    assert sample_manifest["items"][0]["scenario"] == "shared"
    assert sample_manifest["summary"]["by_scenario"] == {"shared": 1}
    assert router_manifest["scenarios"][0]["scenario"] == "shared"


def test_jsonl_manifests_split_mixed_identity_files(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    data_dir.mkdir()
    scenario_dir.mkdir(parents=True)

    sample_rows = [
        {
            "input": "gcam shared",
            "metadata": {"simulator": "gcam", "scenario": "shared", "language": "en", "style": "technical"},
        },
        {
            "input": "simpeg shared",
            "metadata": {"simulator": "simpeg", "scenario": "shared", "language": "zh", "style": "casual"},
        },
        {
            "input": "gcam other",
            "metadata": {"simulator": "gcam", "scenario": "other", "language": "en", "style": "technical"},
        },
    ]
    router_rows = [
        {"context": "gcam", "label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}},
        {"context": "simpeg", "label": 0, "metadata": {"simulator": "simpeg", "scenario": "shared"}},
    ]
    (data_dir / "shared.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample_rows),
        encoding="utf-8",
    )
    (scenario_dir / "shared.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in router_rows),
        encoding="utf-8",
    )

    monkeypatch.setattr(manifest_store, "DATA_DIR", data_dir)
    monkeypatch.setattr(manifest_store, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(manifest_store, "ROUTER_SCENARIO_DIR", scenario_dir)

    sample_manifest = manifest_store._sample_manifest_payload([])
    router_manifest = manifest_store._router_manifest_payload([])

    assert [(item["simulator"], item["scenario"], item["sample_count"]) for item in sample_manifest["items"]] == [
        ("gcam", "other", 1),
        ("gcam", "shared", 1),
        ("simpeg", "shared", 1),
    ]
    assert sample_manifest["summary"]["total_samples"] == 3
    assert sample_manifest["summary"]["by_simulator"] == {"gcam": 2, "simpeg": 1}
    assert sample_manifest["summary"]["by_scenario"] == {"other": 1, "gcam/shared": 1, "simpeg/shared": 1}
    assert sample_manifest["summary"]["by_language"] == {"en": 2, "zh": 1}

    assert [(item["simulator"], item["scenario"], item["router_count"]) for item in router_manifest["scenarios"]] == [
        ("gcam", "shared", 1),
        ("simpeg", "shared", 1),
    ]
    assert router_manifest["total"] == 2
    assert router_manifest["label_counts"] == {"0": 1, "1": 1}


def test_legacy_router_status_uses_metadata_scenario_for_prefixed_jsonl(monkeypatch, tmp_path: Path) -> None:
    router_dir = tmp_path / "router"
    scenario_dir = router_dir / "by_scenario"
    text2comp_dir = tmp_path / "text2comp"
    scenario_dir.mkdir(parents=True)
    text2comp_dir.mkdir()

    router_rows = [
        {"context": "gcam positive", "label": 1, "metadata": {"simulator": "gcam", "scenario": "shared"}},
        {"context": "gcam negative", "label": 0, "metadata": {"simulator": "gcam", "scenario": "shared"}},
    ]
    (scenario_dir / "gcam_shared.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in router_rows),
        encoding="utf-8",
    )
    for simulator in ["gcam", "simpeg"]:
        (text2comp_dir / f"{simulator}_shared.jsonl").write_text(
            json.dumps(
                {
                    "input": simulator,
                    "metadata": {"simulator": simulator, "scenario": "shared"},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(router_data, "ROUTER_DIR", router_dir)
    monkeypatch.setattr(router_data, "SCENARIO_DIR", scenario_dir)
    monkeypatch.setattr(router_data, "TEXT2COMP_DIR", text2comp_dir)

    status = router_data._legacy_get_router_status()

    assert status["source_by_scenario"] == {"gcam/shared": 1, "simpeg/shared": 1}
    assert [(item["simulator"], item["scenario"], item["source_count"]) for item in status["scenarios"]] == [
        ("gcam", "shared", 1),
        ("simpeg", "shared", 1),
    ]
    assert status["scenarios"][0]["router_count"] == 2


def test_combined_router_manifest_merges_jsonl_and_parquet_label_counts(monkeypatch, tmp_path: Path) -> None:
    jsonl_manifest = {
        "kind": "router_manifest",
        "storage": "jsonl",
        "generated_at": 1.0,
        "label_counts": {"0": 1, "1": 2},
        "scenarios": [
            {"scenario": "jsonl_case", "simulator": "modflow", "router_count": 3, "file_size_bytes": 30, "mtime": 1.0}
        ],
    }
    parquet_manifest = {
        "kind": "router_manifest",
        "storage": "parquet",
        "generated_at": 2.0,
        "label_counts": {"0": 4, "1": 5},
        "scenarios": [
            {"scenario": "parquet_case", "simulator": "simpeg", "router_count": 9, "file_size_bytes": 90, "mtime": 2.0}
        ],
    }

    monkeypatch.setattr(router_data.manifest_store, "ensure_router_manifest", lambda: jsonl_manifest)
    monkeypatch.setattr(router_data.portable, "router_manifest_like", lambda: parquet_manifest)

    result = router_data._combined_router_manifest()

    assert result["storage"] == "mixed"
    assert result["total"] == 12
    assert result["label_counts"] == {"0": 5, "1": 7}
    assert [item["scenario"] for item in result["scenarios"]] == ["jsonl_case", "parquet_case"]


def test_dataset_combined_sample_manifest_merges_by_simulator_and_scenario(monkeypatch) -> None:
    jsonl_manifest = {
        "kind": "sample_manifest",
        "storage": "jsonl",
        "generated_at": 1.0,
        "items": [
            {"scenario": "shared", "simulator": "simpeg", "sample_count": 2},
            {"scenario": "shared", "simulator": "gcam", "sample_count": 3},
        ],
    }
    parquet_manifest = {
        "kind": "sample_manifest",
        "storage": "parquet",
        "generated_at": 2.0,
        "items": [{"scenario": "shared", "simulator": "simpeg", "sample_count": 20}],
    }

    monkeypatch.setattr(datasets.manifest_store, "ensure_sample_manifest", lambda: jsonl_manifest)
    monkeypatch.setattr(datasets.portable, "text2comp_manifest_like", lambda: parquet_manifest)

    result = datasets._combined_sample_manifest()

    assert result["summary"]["total_samples"] == 23
    assert result["summary"]["by_scenario"] == {"gcam/shared": 3, "simpeg/shared": 20}
    assert [(item["simulator"], item["scenario"], item["sample_count"]) for item in result["items"]] == [
        ("gcam", "shared", 3),
        ("simpeg", "shared", 20),
    ]


def test_router_combined_manifests_merge_by_simulator_and_scenario(monkeypatch) -> None:
    jsonl_sample_manifest = {
        "kind": "sample_manifest",
        "storage": "jsonl",
        "items": [
            {"scenario": "shared", "simulator": "simpeg", "sample_count": 2},
            {"scenario": "shared", "simulator": "gcam", "sample_count": 3},
        ],
    }
    parquet_sample_manifest = {
        "kind": "sample_manifest",
        "storage": "parquet",
        "items": [{"scenario": "shared", "simulator": "simpeg", "sample_count": 20}],
    }
    jsonl_router_manifest = {
        "kind": "router_manifest",
        "storage": "jsonl",
        "label_counts": {"0": 1},
        "scenarios": [
            {"scenario": "shared", "simulator": "simpeg", "router_count": 2},
            {"scenario": "shared", "simulator": "gcam", "router_count": 3},
        ],
    }
    parquet_router_manifest = {
        "kind": "router_manifest",
        "storage": "parquet",
        "label_counts": {"1": 20},
        "scenarios": [{"scenario": "shared", "simulator": "simpeg", "router_count": 20}],
    }

    monkeypatch.setattr(router_data.manifest_store, "ensure_sample_manifest", lambda: jsonl_sample_manifest)
    monkeypatch.setattr(router_data.portable, "text2comp_manifest_like", lambda: parquet_sample_manifest)
    monkeypatch.setattr(router_data.manifest_store, "ensure_router_manifest", lambda: jsonl_router_manifest)
    monkeypatch.setattr(router_data.portable, "router_manifest_like", lambda: parquet_router_manifest)

    sample_manifest = router_data._combined_sample_manifest()
    router_manifest = router_data._combined_router_manifest()
    status = router_data._build_router_status_from_manifests(router_manifest, sample_manifest)

    assert [(item["simulator"], item["scenario"], item["router_count"]) for item in router_manifest["scenarios"]] == [
        ("gcam", "shared", 3),
        ("simpeg", "shared", 20),
    ]
    assert [(item["simulator"], item["scenario"], item["source_count"]) for item in status["scenarios"]] == [
        ("gcam", "shared", 3),
        ("simpeg", "shared", 20),
    ]
    assert status["source_by_scenario"] == {"gcam/shared": 3, "simpeg/shared": 20}


def test_jsonl_sample_manifest_disambiguates_duplicate_scenarios(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    for simulator, count in [("gcam", 1), ("simpeg", 2)]:
        rows = [
            {"input": f"{simulator}-{index}", "metadata": {"simulator": simulator, "scenario": "shared"}}
            for index in range(count)
        ]
        (data_dir / f"{simulator}_shared.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows),
            encoding="utf-8",
        )

    monkeypatch.setattr(manifest_store, "DATA_DIR", data_dir)

    result = manifest_store._sample_manifest_payload([])

    assert result["summary"]["by_scenario"] == {"gcam/shared": 1, "simpeg/shared": 2}


def test_legacy_stats_disambiguates_duplicate_jsonl_scenarios(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "text2comp"
    data_dir.mkdir()
    for simulator, count in [("gcam", 1), ("simpeg", 2)]:
        rows = [
            {"input": f"{simulator}-{index}", "metadata": {"simulator": simulator, "scenario": "shared"}}
            for index in range(count)
        ]
        (data_dir / f"{simulator}_shared.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows),
            encoding="utf-8",
        )

    monkeypatch.setattr(datasets, "DATA_DIR", data_dir)

    result = datasets._compute_stats_from_individual()

    assert result["total_samples"] == 3
    assert result["by_simulator"] == {"gcam": 1, "simpeg": 2}
    assert result["by_scenario"] == {"gcam/shared": 1, "simpeg/shared": 2}


def test_parquet_text2comp_stats_disambiguates_duplicate_scenarios(monkeypatch, tmp_path: Path) -> None:
    partitions = [
        portable.PartitionInfo(
            kind="text2comp",
            simulator="gcam",
            scenario="shared",
            path=tmp_path / "gcam.parquet",
            row_count=3,
            file_size_bytes=30,
            mtime=1.0,
            metadata={},
        ),
        portable.PartitionInfo(
            kind="text2comp",
            simulator="simpeg",
            scenario="shared",
            path=tmp_path / "simpeg.parquet",
            row_count=4,
            file_size_bytes=40,
            mtime=2.0,
            metadata={},
        ),
    ]

    monkeypatch.setattr(portable, "discover_partitions", lambda kind: partitions if kind == "text2comp" else [])
    monkeypatch.setattr(portable, "duckdb_available", lambda: False)

    result = portable.text2comp_stats()

    assert result["total_samples"] == 7
    assert result["by_scenario"] == {"gcam/shared": 3, "simpeg/shared": 4}

def test_datasets_endpoint_does_not_expose_raw_hdf5_fallback_as_samples(monkeypatch) -> None:
    monkeypatch.setattr(datasets, "_combined_sample_manifest", lambda: {"items": []})

    def fail_raw_fallback():
        raise AssertionError("/api/datasets must not expose raw HDF5/template fallback rows")

    monkeypatch.setattr(datasets, "_raw_data_manifest_like", fail_raw_fallback)

    assert datasets.get_datasets() == []


def test_dashboard_summary_still_uses_raw_hdf5_fallback_when_samples_are_absent(monkeypatch) -> None:
    raw_manifest = {
        "items": [
            {
                "scenario": "coastal",
                "simulator": "modflow",
                "sample_count": 5,
                "file_size_bytes": 12,
                "mtime": 3.0,
                "storage": "hdf5",
            }
        ],
        "summary": {
            "total_samples": 5,
            "by_simulator": {"modflow": 5},
            "by_scenario": {"coastal": 5},
            "by_language": {},
            "by_style": {},
            "by_time_mode": {},
            "timeseries_shapes": {"modflow": [2, 4]},
        },
    }
    monkeypatch.setattr(datasets, "_combined_sample_manifest", lambda: {"items": []})
    monkeypatch.setattr(datasets, "_raw_data_manifest_like", lambda: raw_manifest)
    monkeypatch.setattr(router_data, "_combined_router_manifest", lambda: {})
    monkeypatch.setattr(router_data, "_build_router_status_from_manifests", lambda *_args, **_kwargs: {"total": 0})

    result = datasets.get_dashboard_summary()

    assert result["datasets"][0]["storage"] == "hdf5"
    assert result["datasets"][0]["scenario"] == "coastal"
    assert result["stats"]["total_samples"] == 5


def test_raw_hdf5_scan_includes_hdf5_suffix_and_ignores_runtime_dirs(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    h5_path = data_root / "modflow" / "modflow_coastal.h5"
    uppercase_path = data_root / "power_flow" / "power_flow_case.HDF5"
    hdf5_path = data_root / "simpeg" / "simpeg_external.hdf5"
    ignored_path = data_root / "text2comp" / "ignored.HDF5"
    for path in [h5_path, uppercase_path, hdf5_path, ignored_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    monkeypatch.setattr(datasets, "DATA_ROOT", data_root)

    assert datasets._iter_raw_hdf5_files() == [h5_path, uppercase_path, hdf5_path]


def test_raw_data_manifest_matches_templates_by_simulator_and_scenario(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    hdf5_paths = [
        data_root / "gcam" / "gcam_shared.hdf5",
        data_root / "simpeg" / "simpeg_shared.hdf5",
    ]
    for path in hdf5_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    template_manifest = {
        "items": [
            {"simulator": "gcam", "scenario": "shared", "by_language": {"en": 2}},
            {"simulator": "simpeg", "scenario": "shared", "by_language": {"zh": 3}},
        ]
    }

    monkeypatch.setattr(datasets, "DATA_ROOT", data_root)
    monkeypatch.setattr(datasets.manifest_store, "ensure_template_manifest", lambda: template_manifest)
    monkeypatch.setattr(datasets, "_read_hdf5_stats", lambda _path: (5, [2, 4]))

    result = datasets._raw_data_manifest_like()

    by_identity = {(item["simulator"], item["scenario"]): item for item in result["items"]}
    assert by_identity[("gcam", "shared")]["by_language"] == {"en": 2}
    assert by_identity[("simpeg", "shared")]["by_language"] == {"zh": 3}
    assert result["summary"]["by_simulator"] == {"gcam": 5, "simpeg": 5}

def test_router_build_normalizes_unique_bare_scenario(monkeypatch) -> None:
    sample_manifest = {
        "kind": "sample_manifest",
        "items": [{"scenario": "shared", "simulator": "gcam", "sample_count": 3}],
    }
    created: dict[str, object] = {}

    def fake_create_job(job_type: str, request: dict, lock_keys: list[str], status: str):
        created.update(job_type=job_type, request=request, lock_keys=lock_keys, status=status)
        return SimpleNamespace(job_id="router-test", status=status)

    monkeypatch.setattr(router_data, "_running_job_ids", lambda _job_types: [])
    monkeypatch.setattr(router_data, "_use_worker_queue", lambda: True)
    monkeypatch.setattr(router_data, "_combined_sample_manifest", lambda: sample_manifest)
    monkeypatch.setattr(router_data.job_manager, "create_job", fake_create_job)
    monkeypatch.setattr(router_data, "publish", lambda *_args, **_kwargs: None)

    response = asyncio.run(
        router_data.build_router_data(seed=7, neg_ratio=2, max_workers=3, scenarios="shared,gcam/shared")
    )

    assert response == {"job_id": "router-test", "status": "queued"}
    assert created["job_type"] == "router"
    assert created["request"] == {
        "seed": 7,
        "neg_ratio": 2,
        "max_workers": 3,
        "scenarios": ["gcam/shared"],
    }
    assert created["lock_keys"] == ["router:gcam/shared", "dataset:gcam/shared"]


def test_router_build_accepts_repeated_scenario_query_values(monkeypatch) -> None:
    sample_manifest = {
        "kind": "sample_manifest",
        "items": [
            {"scenario": "shared", "simulator": "gcam", "sample_count": 3},
            {"scenario": "other", "simulator": "simpeg", "sample_count": 4},
        ],
    }
    created: dict[str, object] = {}

    def fake_create_job(job_type: str, request: dict, lock_keys: list[str], status: str):
        created.update(job_type=job_type, request=request, lock_keys=lock_keys, status=status)
        return SimpleNamespace(job_id="router-test", status=status)

    monkeypatch.setattr(router_data, "_running_job_ids", lambda _job_types: [])
    monkeypatch.setattr(router_data, "_use_worker_queue", lambda: True)
    monkeypatch.setattr(router_data, "_combined_sample_manifest", lambda: sample_manifest)
    monkeypatch.setattr(router_data.job_manager, "create_job", fake_create_job)
    monkeypatch.setattr(router_data, "publish", lambda *_args, **_kwargs: None)

    response = asyncio.run(
        router_data.build_router_data(seed=7, neg_ratio=2, max_workers=3, scenarios=["gcam/shared", "simpeg/other"])
    )

    assert response == {"job_id": "router-test", "status": "queued"}
    assert created["request"] == {
        "seed": 7,
        "neg_ratio": 2,
        "max_workers": 3,
        "scenarios": ["gcam/shared", "simpeg/other"],
    }
    assert created["lock_keys"] == [
        "router:gcam/shared",
        "router:simpeg/other",
        "dataset:gcam/shared",
        "dataset:simpeg/other",
    ]


def test_router_build_rejects_ambiguous_bare_scenario(monkeypatch) -> None:
    sample_manifest = {
        "kind": "sample_manifest",
        "items": [
            {"scenario": "shared", "simulator": "gcam", "sample_count": 3},
            {"scenario": "shared", "simulator": "simpeg", "sample_count": 2},
        ],
    }
    created: list[dict] = []

    monkeypatch.setattr(router_data, "_running_job_ids", lambda _job_types: [])
    monkeypatch.setattr(router_data, "_use_worker_queue", lambda: True)
    monkeypatch.setattr(router_data, "_combined_sample_manifest", lambda: sample_manifest)
    monkeypatch.setattr(router_data.job_manager, "create_job", lambda *args, **kwargs: created.append(kwargs))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(router_data.build_router_data(scenarios="shared"))

    assert exc_info.value.status_code == 400
    assert "完整选择器" in str(exc_info.value.detail)
    assert created == []


def test_router_build_rejects_unknown_selected_scenario(monkeypatch) -> None:
    sample_manifest = {
        "kind": "sample_manifest",
        "items": [{"scenario": "shared", "simulator": "gcam", "sample_count": 3}],
    }
    created: list[dict] = []

    monkeypatch.setattr(router_data, "_running_job_ids", lambda _job_types: [])
    monkeypatch.setattr(router_data, "_use_worker_queue", lambda: True)
    monkeypatch.setattr(router_data, "_combined_sample_manifest", lambda: sample_manifest)
    monkeypatch.setattr(router_data.job_manager, "create_job", lambda *args, **kwargs: created.append(kwargs))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(router_data.build_router_data(scenarios="simpeg/shared"))

    assert exc_info.value.status_code == 400
    assert "未找到场景" in str(exc_info.value.detail)
    assert created == []
