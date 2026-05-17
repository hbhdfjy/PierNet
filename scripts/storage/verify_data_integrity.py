#!/usr/bin/env python3
"""Build or verify checksums for source templates and raw scientific data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from piern.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT as DEFAULT_PROJECT_ROOT
from piern.shared.storage import integrity

PROJECT_ROOT = DEFAULT_PROJECT_ROOT
DEFAULT_MANIFEST = integrity.DEFAULT_MANIFEST


def build_manifest(data_root: Path = DATA_ROOT) -> dict:
    return integrity.build_manifest(data_root, project_root=PROJECT_ROOT)


def verify_manifest(manifest: dict) -> list[str]:
    return integrity.verify_manifest(manifest, project_root=PROJECT_ROOT)


def write_manifest(path: Path = DEFAULT_MANIFEST, *, data_root: Path = DATA_ROOT) -> dict:
    return integrity.write_manifest(path, data_root=data_root, project_root=PROJECT_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args(argv)

    if args.write_manifest:
        manifest = write_manifest(args.manifest, data_root=args.data_root)
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
