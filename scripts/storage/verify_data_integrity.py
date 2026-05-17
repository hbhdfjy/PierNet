#!/usr/bin/env python3
"""Build or verify checksums for source templates and raw scientific data."""

from __future__ import annotations

import argparse
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


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def source_files(data_root: Path = DATA_ROOT) -> list[Path]:
    files = []
    files.extend(sorted((data_root / "templates").glob("*_templates.jsonl")))
    files.extend(sorted(data_root.glob("*/*.h5")))
    return [path for path in files if path.is_file()]


def build_manifest(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    entries = []
    for path in source_files(data_root):
        stat = path.stat()
        entries.append(
            {
                "path": _rel(path),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(data_root),
        "entries": entries,
    }


def verify_manifest(manifest: dict[str, Any]) -> list[str]:
    errors = []
    for entry in manifest.get("entries", []):
        path = PROJECT_ROOT / str(entry.get("path", ""))
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args(argv)

    if args.write_manifest:
        manifest = build_manifest(args.data_root)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {len(manifest['entries'])} source checksums to {args.manifest}")
        return 0

    if not args.manifest.exists():
        manifest = build_manifest(args.data_root)
        print(f"source files scanned: {len(manifest['entries'])}; no manifest found at {args.manifest}")
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = verify_manifest(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"FAILED: {len(errors)} integrity issue(s)")
        return 1
    print(f"PASSED: {len(manifest.get('entries', []))} source file checksum(s) verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
