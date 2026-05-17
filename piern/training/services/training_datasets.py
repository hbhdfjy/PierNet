"""Dataset manifest helpers for token-router training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from piern.shared.storage import portable

LOGGER = logging.getLogger(__name__)


def read_router_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to read router manifest at %s: %s", path, exc)
        return None


def merge_router_manifests(primary: dict[str, Any], fallback: dict[str, Any] | None) -> dict[str, Any]:
    if not fallback:
        return primary
    primary_scenarios = {item.get("scenario") for item in primary.get("scenarios", [])}
    scenarios = [*primary.get("scenarios", [])]
    scenarios.extend(item for item in fallback.get("scenarios", []) if item.get("scenario") not in primary_scenarios)
    total = sum(int(item.get("router_count") or 0) for item in scenarios)
    return {
        **primary,
        "storage": "mixed"
        if len(scenarios) != len(primary.get("scenarios", []))
        else primary.get("storage", "parquet"),
        "total": total,
        "scenarios": sorted(scenarios, key=lambda item: item.get("scenario", "")),
    }


def list_datasets(*, router_manifest_path: Path, default_router_manifest_path: Path) -> list[dict[str, Any]]:
    parquet_manifest = portable.router_manifest_like() if router_manifest_path == default_router_manifest_path else None
    jsonl_manifest = read_router_manifest(router_manifest_path)
    manifest = merge_router_manifests(parquet_manifest, jsonl_manifest) if parquet_manifest else jsonl_manifest
    if not manifest:
        return []
    grouped: dict[str, dict[str, Any]] = {}
    for scenario in manifest.get("scenarios", []):
        simulator = scenario["simulator"]
        bucket = grouped.setdefault(simulator, {"simulator": simulator, "total_count": 0, "scenarios": []})
        bucket["total_count"] += int(scenario["router_count"])
        bucket["scenarios"].append(scenario)
    for bucket in grouped.values():
        bucket["scenarios"].sort(key=lambda item: item["scenario"])
    return sorted(grouped.values(), key=lambda item: item["simulator"])
