from __future__ import annotations

import json
from pathlib import Path

from piern.synth.services import jsonl_filter_index, jsonl_index, manifest_store


def test_jsonl_indexes_support_data_root_outside_project_root(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "external-data"
    source_path = data_root / "text2comp" / "case_a.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        json.dumps({"metadata": {"language": "zh", "style": "technical"}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(jsonl_index, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(jsonl_index, "DATA_ROOT", data_root)
    monkeypatch.setattr(jsonl_index, "INDEX_ROOT", data_root / ".indexes")
    monkeypatch.setattr(jsonl_filter_index, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(jsonl_filter_index, "DATA_ROOT", data_root)
    monkeypatch.setattr(jsonl_filter_index, "INDEX_ROOT", data_root / ".indexes")

    total, items = jsonl_index.read_page(source_path, page=0, page_size=10)
    filtered_total, filtered_items = jsonl_filter_index.read_filtered_page(
        source_path,
        "sample_language_style",
        "language=zh|style=technical",
        page=0,
        page_size=10,
    )

    assert total == 1
    assert items == [{"metadata": {"language": "zh", "style": "technical"}}]
    assert filtered_total == 1
    assert filtered_items == items
    assert (data_root / ".indexes" / "text2comp" / "case_a.jsonl.idx.json").exists()
    assert (data_root / ".indexes" / "text2comp" / "case_a.jsonl.sample_language_style.idx.json").exists()


def test_manifest_snapshot_supports_data_root_outside_project_root(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "external-data"
    source_path = data_root / "templates" / "case_a_templates.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(manifest_store, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(manifest_store, "DATA_ROOT", data_root)

    assert manifest_store._snapshot([source_path])[0]["relative_path"] == "templates/case_a_templates.jsonl"
