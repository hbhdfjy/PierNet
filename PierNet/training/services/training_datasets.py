"""Dataset manifest helpers for token-router training."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from PierNet.new_synth.training_bridge import list_router_datasets
from PierNet.shared.storage import portable

LOGGER = logging.getLogger(__name__)
LEGACY_MANIFEST_CACHE_TTL_SECONDS = 15.0
_legacy_manifest_cache_lock = threading.Lock()
_legacy_manifest_cache_ready = False
_legacy_manifest_cache_at = 0.0
_legacy_manifest_cache: dict[str, Any] | None = None

_SIMPLE_DATASET_NAMES = {
    "gcam": "GCAM",
    "modflow": "MODFLOW 地下水",
    "power_flow": "电力潮流",
    "simpeg": "SimPEG 地球物理",
    "transient": "电力系统暂态",
    "mechanics_column_buckling": "结构力学 · 柱屈曲",
}
_INTERNAL_LEGACY_SIMULATORS = {"expert_model", "uploaded_expert"}


def _cached_legacy_router_manifest() -> dict[str, Any] | None:
    global _legacy_manifest_cache_at, _legacy_manifest_cache, _legacy_manifest_cache_ready
    now = time.monotonic()
    with _legacy_manifest_cache_lock:
        if _legacy_manifest_cache_ready and now - _legacy_manifest_cache_at < LEGACY_MANIFEST_CACHE_TTL_SECONDS:
            return _legacy_manifest_cache
        _legacy_manifest_cache = portable.router_manifest_like(include_label_counts=False)
        _legacy_manifest_cache_at = now
        _legacy_manifest_cache_ready = True
        return _legacy_manifest_cache


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
    parquet_manifest = _cached_legacy_router_manifest() if router_manifest_path == default_router_manifest_path else None
    jsonl_manifest = read_router_manifest(router_manifest_path)
    manifest = merge_router_manifests(parquet_manifest, jsonl_manifest) if parquet_manifest else jsonl_manifest
    new_synth_datasets = list_router_datasets()
    if not manifest:
        return new_synth_datasets
    grouped: dict[str, dict[str, Any]] = {}
    for scenario in _normalized_router_scenarios(manifest):
        simulator = scenario["simulator"]
        bucket = grouped.setdefault(simulator, {"simulator": simulator, "total_count": 0, "scenarios": []})
        bucket["total_count"] += scenario["router_count"]
        bucket["scenarios"].append(scenario)
    for bucket in grouped.values():
        bucket["scenarios"].sort(key=lambda item: item["scenario"])
    legacy = sorted(grouped.values(), key=lambda item: item["simulator"])
    return [*new_synth_datasets, *legacy]


def _without_internal_stage(name: str) -> str:
    cleaned = re.sub(r"\s*(?:·|-)\s*(?:router|text2comp)\s*$", "", name.strip(), flags=re.IGNORECASE)
    parts = []
    for part in (cleaned or name.strip()).split("·"):
        value = part.strip()
        if not value:
            continue
        parts.append(_SIMPLE_DATASET_NAMES.get(value.lower(), value.replace("_", " ")))
    return " · ".join(parts)


def _business_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _new_synth_legacy_key(dataset: dict[str, Any]) -> tuple[str, str] | None:
    parts = [part.strip() for part in _without_internal_stage(str(dataset.get("display_name") or "")).split("·")]
    if len(parts) < 2:
        return None
    simulator_aliases = {
        "gcam": "gcam",
        "modflow": "modflow",
        "powerflow": "power_flow",
        "simpeg": "simpeg",
        "transient": "transient",
    }
    simulator = simulator_aliases.get(_business_token(parts[0]))
    return (simulator, _business_token(parts[1])) if simulator else None


def list_simple_datasets(
    datasets: list[dict[str, Any]],
    *,
    text2comp_datasets: list[dict[str, Any]],
    data_root: Path,
    min_router_samples: int,
    min_text2comp_samples: int,
) -> list[dict[str, Any]]:
    """Return only data sources that can run the complete simple-training pipeline."""

    legacy_scenario_keys = {
        (str(dataset.get("simulator") or ""), _business_token(str(scenario.get("scenario") or "")))
        for dataset in datasets
        if str(dataset.get("source") or "legacy") != "new_synth"
        for scenario in dataset.get("scenarios") or []
        if str(scenario.get("scenario") or "").strip()
    }
    paired_text2comp = {
        str(item.get("dataset_id")): item
        for item in text2comp_datasets
        if str(item.get("dataset_id") or "").strip()
    }
    ready: list[dict[str, Any]] = []
    for dataset in datasets:
        source = str(dataset.get("source") or "legacy")
        simulator = str(dataset.get("simulator") or "").strip()
        if source == "new_synth":
            if _new_synth_legacy_key(dataset) in legacy_scenario_keys:
                continue
            paired_id = str(dataset.get("text2comp_dataset_id") or "").strip()
            paired = paired_text2comp.get(paired_id)
            paired_path = Path(str((paired or {}).get("path") or ""))
            if (
                not paired
                or int(dataset.get("total_count") or 0) < min_router_samples
                or int(paired.get("sample_count") or 0) < min_text2comp_samples
                or not paired_path.is_file()
            ):
                continue
            item = dict(dataset)
            item["display_name"] = _without_internal_stage(
                str(dataset.get("display_name") or dataset.get("simulator") or "训练数据")
            )
            ready.append(item)
            continue

        if simulator in _INTERNAL_LEGACY_SIMULATORS:
            continue
        scenarios: list[dict[str, Any]] = []
        for scenario in dataset.get("scenarios") or []:
            scenario_name = str(scenario.get("scenario") or "").strip()
            if not scenario_name or int(scenario.get("router_count") or 0) < min_router_samples:
                continue
            h5_path = data_root / simulator / f"{simulator}_{scenario_name}.h5"
            template_path = data_root / "templates" / f"{scenario_name}_templates.jsonl"
            if not h5_path.is_file() or not template_path.is_file():
                continue
            scenarios.append(dict(scenario))
        if not scenarios:
            continue
        item = dict(dataset)
        item["display_name"] = _SIMPLE_DATASET_NAMES.get(
            simulator,
            simulator.replace("_", " ").strip().upper(),
        )
        item["scenarios"] = scenarios
        item["total_count"] = sum(int(scenario.get("router_count") or 0) for scenario in scenarios)
        ready.append(item)
    return ready
