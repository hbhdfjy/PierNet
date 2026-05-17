"""Source data integrity manifests for templates and raw scientific data."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from piern.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT

DEFAULT_MANIFEST = DATA_ROOT / ".manifests" / "source_integrity.json"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, *, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def source_files(data_root: Path = DATA_ROOT) -> list[Path]:
    files = []
    files.extend(sorted((data_root / "templates").glob("*_templates.jsonl")))
    files.extend(sorted(data_root.glob("*/*.h5")))
    return [path for path in files if path.is_file()]


def build_manifest(data_root: Path = DATA_ROOT, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    entries = []
    for path in source_files(data_root):
        stat = path.stat()
        entries.append(
            {
                "path": _rel(path, project_root=project_root),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "project_root": str(project_root),
        "data_root": str(data_root),
        "entries": entries,
    }


def verify_manifest(manifest: dict[str, Any], *, project_root: Path = PROJECT_ROOT) -> list[str]:
    errors = []
    for entry in manifest.get("entries", []):
        path = project_root / str(entry.get("path", ""))
        if not path.exists():
            errors.append(f"missing: {entry.get('path')}")
            continue
        actual_size = path.stat().st_size
        if actual_size != int(entry.get("size_bytes", -1)):
            errors.append(f"size mismatch: {entry.get('path')} expected={entry.get('size_bytes')} actual={actual_size}")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != entry.get("sha256"):
            errors.append(f"sha256 mismatch: {entry.get('path')}")
    return errors


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def write_manifest(
    path: Path = DEFAULT_MANIFEST,
    *,
    data_root: Path = DATA_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    manifest = build_manifest(data_root, project_root=project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def status(
    path: Path = DEFAULT_MANIFEST,
    *,
    data_root: Path = DATA_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    manifest = load_manifest(path)
    if manifest is None:
        scanned = build_manifest(data_root, project_root=project_root)
        return {
            "ok": True,
            "manifest_exists": False,
            "manifest_path": str(path),
            "checked_entries": 0,
            "scanned_entries": len(scanned.get("entries", [])),
            "errors": [],
            "generated_at": None,
        }
    errors = verify_manifest(manifest, project_root=project_root)
    return {
        "ok": not errors,
        "manifest_exists": True,
        "manifest_path": str(path),
        "checked_entries": len(manifest.get("entries", [])),
        "scanned_entries": len(source_files(data_root)),
        "errors": errors,
        "generated_at": manifest.get("generated_at"),
    }
