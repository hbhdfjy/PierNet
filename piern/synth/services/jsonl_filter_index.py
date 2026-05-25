"""Sparse filtered indexes for common JSONL pagination filters."""

from __future__ import annotations

import json
from pathlib import Path

from piern.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT
from piern.shared.storage.path_ids import source_relative_path

INDEX_ROOT = DATA_ROOT / ".indexes"
INDEX_VERSION = 1
DEFAULT_STRIDE = 1000

SUPPORTED_PROFILES = {
    "sample_language_style",
    "template_language_style",
    "router_label",
}


def ensure_filter_index(source_path: Path, profile: str, stride: int = DEFAULT_STRIDE) -> dict:
    _validate_profile(profile)
    fingerprint = _fingerprint(source_path)
    index_path = get_filter_index_path(source_path, profile)
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if payload and _matches(payload, fingerprint, profile, stride):
            return payload
    return rebuild_filter_index(source_path, profile, stride=stride)


def rebuild_filter_index(source_path: Path, profile: str, stride: int = DEFAULT_STRIDE) -> dict:
    _validate_profile(profile)
    fingerprint = _fingerprint(source_path)
    entries: dict[str, dict] = {}

    with open(source_path, "rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            for key in _profile_keys(profile, record):
                entry = entries.setdefault(key, {"count": 0, "offsets": []})
                if entry["count"] % stride == 0:
                    entry["offsets"].append(offset)
                entry["count"] += 1

    payload = {
        "version": INDEX_VERSION,
        "profile": profile,
        "source_relative_path": str(source_relative_path(source_path, roots=(PROJECT_ROOT, DATA_ROOT))),
        "file_size_bytes": fingerprint["file_size_bytes"],
        "mtime_ns": fingerprint["mtime_ns"],
        "stride": stride,
        "entries": entries,
    }

    index_path = get_filter_index_path(source_path, profile)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(index_path)
    return payload


def read_filtered_page(source_path: Path, profile: str, key: str, page: int, page_size: int) -> tuple[int, list[dict]]:
    index = ensure_filter_index(source_path, profile)
    entry = index.get("entries", {}).get(key)
    if not entry:
        return 0, []

    total = int(entry.get("count", 0))
    start = page * page_size
    end = start + page_size
    if start >= total or total == 0:
        return total, []

    stride = max(int(index.get("stride", DEFAULT_STRIDE)), 1)
    anchor_match = (start // stride) * stride
    anchor_slot = anchor_match // stride
    offsets = entry.get("offsets", [])
    anchor_offset = offsets[anchor_slot] if anchor_slot < len(offsets) else 0

    items: list[dict] = []
    match_count = anchor_match

    with open(source_path, "rb") as handle:
        handle.seek(anchor_offset)
        while match_count < end:
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if key not in _profile_keys(profile, record):
                continue
            if match_count >= start:
                items.append(record)
            match_count += 1

    return total, items


def get_filter_index_path(source_path: Path, profile: str) -> Path:
    relative = source_relative_path(source_path, roots=(PROJECT_ROOT, DATA_ROOT))
    return INDEX_ROOT / relative.parent / f"{relative.name}.{profile}.idx.json"


def _profile_keys(profile: str, record: dict) -> list[str]:
    if profile == "sample_language_style":
        metadata = record.get("metadata", {})
        language = metadata.get("language")
        style = metadata.get("style")
        return _language_style_keys(language, style)
    if profile == "template_language_style":
        return _language_style_keys(record.get("language"), record.get("style"))
    if profile == "router_label":
        label = record.get("label")
        if label in (0, 1, "0", "1"):
            return [f"label={label}"]
        return []
    raise ValueError(f"Unsupported filter index profile: {profile}")


def _language_style_keys(language, style) -> list[str]:
    keys: list[str] = []
    if language not in (None, ""):
        keys.append(f"language={language}")
    if style not in (None, ""):
        keys.append(f"style={style}")
    if language not in (None, "") and style not in (None, ""):
        keys.append(f"language={language}|style={style}")
    return keys


def _fingerprint(source_path: Path) -> dict:
    stat = source_path.stat()
    return {
        "file_size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _matches(payload: dict, fingerprint: dict, profile: str, stride: int) -> bool:
    return (
        payload.get("version") == INDEX_VERSION
        and payload.get("profile") == profile
        and int(payload.get("file_size_bytes", -1)) == fingerprint["file_size_bytes"]
        and int(payload.get("mtime_ns", -1)) == fingerprint["mtime_ns"]
        and int(payload.get("stride", -1)) == stride
    )


def _validate_profile(profile: str) -> None:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported filter index profile: {profile}")
