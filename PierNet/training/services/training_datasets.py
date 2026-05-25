"""Dataset manifest helpers for token-router training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PierNet.shared.storage import portable

LOGGER = logging.getLogger(__name__)


def read_router_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to read router manifest at %s: %s", path, exc)
        return None


def manifest_item_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("simulator") or "unknown"), str(item.get("scenario") or ""))


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_router_scenario(item: dict[str, Any]) -> dict[str, Any] | None:
    scenario = str(item.get("scenario") or "").strip()
    if not scenario:
        return None
    simulator = str(item.get("simulator") or "unknown").strip() or "unknown"
    return {
        **item,
        "scenario": scenario,
        "simulator": simulator,
        "router_count": _coerce_int(item.get("router_count")),
        "file_size_bytes": _coerce_int(item.get("file_size_bytes")),
        "mtime": _coerce_float(item.get("mtime")),
        "path": str(item.get("path") or ""),
    }


def _normalized_router_scenarios(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for item in manifest.get("scenarios", []):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_router_scenario(item)
        if normalized is not None:
            scenarios.append(normalized)
    return scenarios


def merge_router_manifests(primary: dict[str, Any], fallback: dict[str, Any] | None) -> dict[str, Any]:
    primary_scenarios = _normalized_router_scenarios(primary)
    if not fallback:
        return {**primary, "scenarios": sorted(primary_scenarios, key=manifest_item_key)}
    primary_keys = {manifest_item_key(item) for item in primary_scenarios}
    fallback_scenarios = [
        item for item in _normalized_router_scenarios(fallback) if manifest_item_key(item) not in primary_keys
    ]
    scenarios = [*primary_scenarios, *fallback_scenarios]
    total = sum(int(item.get("router_count") or 0) for item in scenarios)
    return {
        **primary,
        "storage": "mixed" if fallback_scenarios else primary.get("storage", "parquet"),
        "total": total,
        "scenarios": sorted(scenarios, key=manifest_item_key),
    }


def list_datasets(*, router_manifest_path: Path, default_router_manifest_path: Path) -> list[dict[str, Any]]:
    parquet_manifest = portable.router_manifest_like() if router_manifest_path == default_router_manifest_path else None
    jsonl_manifest = read_router_manifest(router_manifest_path)
    manifest = merge_router_manifests(parquet_manifest, jsonl_manifest) if parquet_manifest else jsonl_manifest
    if not manifest:
        return []
    grouped: dict[str, dict[str, Any]] = {}
    for scenario in _normalized_router_scenarios(manifest):
        simulator = scenario["simulator"]
        bucket = grouped.setdefault(simulator, {"simulator": simulator, "total_count": 0, "scenarios": []})
        bucket["total_count"] += scenario["router_count"]
        bucket["scenarios"].append(scenario)
    for bucket in grouped.values():
        bucket["scenarios"].sort(key=lambda item: item["scenario"])
    return sorted(grouped.values(), key=lambda item: item["simulator"])
