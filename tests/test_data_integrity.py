
from scripts.storage.verify_data_integrity import build_manifest, verify_manifest


def test_source_integrity_manifest_detects_source_changes(tmp_path, monkeypatch):
    project = tmp_path / "project"
    data = project / "data"
    templates = data / "templates"
    raw = data / "modflow"
    templates.mkdir(parents=True)
    raw.mkdir(parents=True)
    (templates / "coastal_templates.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    (raw / "coastal.h5").write_bytes(b"hdf5")
    (raw / "external.hdf5").write_bytes(b"hdf5-new")

    monkeypatch.setattr("scripts.storage.verify_data_integrity.PROJECT_ROOT", project)
    manifest = build_manifest(data)
    assert len(manifest["entries"]) == 3
    assert verify_manifest(manifest) == []

    (templates / "coastal_templates.jsonl").write_text("changed\n", encoding="utf-8")
    assert verify_manifest(manifest)


def test_source_integrity_manifest_handles_missing_data_root(tmp_path):
    missing = tmp_path / "missing-data"

    manifest = build_manifest(missing)

    assert manifest["entries"] == []
