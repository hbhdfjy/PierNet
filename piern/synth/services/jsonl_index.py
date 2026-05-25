"""Sparse byte-offset indexes for paginating large JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from piern.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT
from piern.shared.storage.path_ids import source_relative_path

INDEX_ROOT = DATA_ROOT / ".indexes"
INDEX_VERSION = 1
DEFAULT_STRIDE = 1000


def ensure_index(source_path: Path, stride: int = DEFAULT_STRIDE) -> dict:
    fingerprint = _fingerprint(source_path)
    index_path = get_index_path(source_path)
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if payload and _matches(payload, fingerprint, stride):
            return payload
    return rebuild_index(source_path, stride=stride)


def rebuild_index(source_path: Path, stride: int = DEFAULT_STRIDE) -> dict:
    fingerprint = _fingerprint(source_path)
    offsets: list[int] = []
    total_rows = 0

    with open(source_path, "rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            if total_rows % stride == 0:
                offsets.append(offset)
            total_rows += 1

    payload = {
        "version": INDEX_VERSION,
        "source_relative_path": str(source_relative_path(source_path, roots=(PROJECT_ROOT, DATA_ROOT))),
        "file_size_bytes": fingerprint["file_size_bytes"],
        "mtime_ns": fingerprint["mtime_ns"],
        "stride": stride,
        "total_rows": total_rows,
        "offsets": offsets,
    }

    index_path = get_index_path(source_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(index_path)
    return payload


def read_page(source_path: Path, page: int, page_size: int, total_rows: int | None = None) -> tuple[int, list[dict]]:
    index = ensure_index(source_path)
    total = index.get("total_rows", 0) if total_rows is None else total_rows
    start = page * page_size
    end = start + page_size

    if start >= total or total == 0:
        return total, []

    stride = max(int(index.get("stride", DEFAULT_STRIDE)), 1)
    anchor_row = (start // stride) * stride
    anchor_slot = anchor_row // stride
    offsets = index.get("offsets", [])
    anchor_offset = offsets[anchor_slot] if anchor_slot < len(offsets) else 0

    items: list[dict] = []
    row_index = anchor_row

    with open(source_path, "rb") as handle:
        handle.seek(anchor_offset)
        while row_index < end:
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            if row_index >= start:
                try:
                    items.append(json.loads(raw.decode("utf-8")))
                except Exception:
                    pass
            row_index += 1

    return total, items


def get_index_path(source_path: Path) -> Path:
    relative = source_relative_path(source_path, roots=(PROJECT_ROOT, DATA_ROOT))
    return INDEX_ROOT / relative.parent / f"{relative.name}.idx.json"


def _fingerprint(source_path: Path) -> dict:
    stat = source_path.stat()
    return {
        "file_size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _matches(payload: dict, fingerprint: dict, stride: int) -> bool:
    return (
        payload.get("version") == INDEX_VERSION
        and int(payload.get("file_size_bytes", -1)) == fingerprint["file_size_bytes"]
        and int(payload.get("mtime_ns", -1)) == fingerprint["mtime_ns"]
        and int(payload.get("stride", -1)) == stride
    )
