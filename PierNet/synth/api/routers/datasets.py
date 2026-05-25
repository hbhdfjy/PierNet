"""数据集相关路由：/api/datasets, /api/samples, /api/dashboard/summary。"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from PierNet.shared.runtime.paths import DATA_DIR, DATA_ROOT
from PierNet.shared.storage import portable
from PierNet.shared.storage.hdf5_files import iter_hdf5_files_in_child_dirs
from PierNet.shared.storage.scenario_summary import duplicate_scenario_names, scenario_summary_key
from PierNet.synth.api.routers import router_data as router_data_router
from PierNet.synth.services import file_manager, jsonl_filter_index, jsonl_index, manifest_store

router = APIRouter()
LOGGER = logging.getLogger(__name__)


def _record_matches_sample_identity(
    record: dict,
    fallback_scenario: str,
    scenario: str,
    simulator: str | None,
) -> bool:
    metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    record_scenario = str(metadata.get("scenario") or fallback_scenario).strip()
    record_simulator = str(metadata.get("simulator") or "").strip() or None
    return record_scenario == scenario and (simulator is None or record_simulator in {None, simulator})


@router.get("/datasets")
def get_datasets():
    """返回真实 Stage 3 样本数据集列表，优先走 manifest。"""
    try:
        manifest = _combined_sample_manifest()
        return [
            {
                "name": item["scenario"],
                "simulator": item.get("simulator", "unknown"),
                "scenario": item["scenario"],
                "sample_count": item.get("sample_count", 0),
                "file_size_bytes": item.get("file_size_bytes", 0),
                "mtime": item.get("mtime", 0),
                "storage": item.get("storage", "jsonl"),
            }
            for item in manifest.get("items", [])
        ]
    except Exception:
        LOGGER.exception("Falling back to legacy /api/datasets scan")
        return _legacy_get_datasets()


@router.get("/samples")
def get_samples(
    scenario: str = Query(...),
    page: int = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
    language: Optional[str] = Query(None),
    style: Optional[str] = Query(None),
    simulator: Optional[str] = Query(None),
):
    """分页读取指定场景样本；Parquet 存在时优先使用 Parquet。"""
    language_filter = language.strip() if isinstance(language, str) and language.strip() else None
    style_filter = style.strip() if isinstance(style, str) and style.strip() else None
    simulator_filter = simulator.strip() if isinstance(simulator, str) and simulator.strip() else None
    if portable.partition_for("text2comp", scenario, simulator=simulator_filter) is not None:
        try:
            parquet_page = portable.read_text2comp_page(
                scenario=scenario,
                page=page,
                page_size=page_size,
                language=language_filter,
                style=style_filter,
                simulator=simulator_filter,
            )
            if parquet_page is not None:
                total, items = parquet_page
                return {"total": total, "page": page, "page_size": page_size, "items": items, "storage": "parquet"}
        except Exception as exc:
            raise HTTPException(500, f"读取 Parquet 失败: {exc}")

    try:
        jsonl_path = file_manager.resolve_sample_file(scenario, simulator=simulator_filter)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if jsonl_path is None or not jsonl_path.exists():
        raise HTTPException(404, f"场景 {scenario} 的数据文件不存在")

    sample_manifest_items = _sample_manifest_items_for_path(jsonl_path)
    requires_record_filter = _sample_requires_record_filter(sample_manifest_items, scenario, simulator_filter)
    requires_record_filter = requires_record_filter or jsonl_path.stem != scenario
    has_filter = bool(language_filter or style_filter or simulator_filter or requires_record_filter)
    start = page * page_size
    end = start + page_size

    try:
        if not has_filter:
            total_hint = _sample_total_from_manifest(scenario, simulator_filter, source_path=jsonl_path)
            try:
                total, items = jsonl_index.read_page(
                    jsonl_path,
                    page=page,
                    page_size=page_size,
                    total_rows=total_hint,
                )
                return {"total": total, "page": page, "page_size": page_size, "items": items}
            except Exception:
                pass

        filter_key = _sample_filter_key(language=language_filter, style=style_filter)
        if filter_key and not simulator_filter and not requires_record_filter:
            try:
                total, items_page = jsonl_filter_index.read_filtered_page(
                    jsonl_path,
                    profile="sample_language_style",
                    key=filter_key,
                    page=page,
                    page_size=page_size,
                )
                return {"total": total, "page": page, "page_size": page_size, "items": items_page}
            except Exception:
                pass

        total = 0
        items_page: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
                if not _record_matches_sample_identity(sample, jsonl_path.stem, scenario, simulator_filter):
                    continue
                if language_filter and metadata.get("language") != language_filter:
                    continue
                if style_filter and metadata.get("style") != style_filter:
                    continue
                if start <= total < end:
                    items_page.append(sample)
                total += 1
        return {"total": total, "page": page, "page_size": page_size, "items": items_page}
    except Exception as exc:
        raise HTTPException(500, f"读取 JSONL 失败: {exc}")


def _get_sample_stats():
    """返回 Stage 3 聚合统计，优先走 manifest。"""
    try:
        manifest = _combined_sample_manifest()
        if not manifest.get("items"):
            manifest = _raw_data_manifest_like()
        return manifest.get(
            "summary",
            {
                "total_samples": 0,
                "by_simulator": {},
                "by_scenario": {},
                "by_language": {},
                "by_style": {},
                "by_time_mode": {},
                "timeseries_shapes": {},
            },
        )
    except Exception:
        LOGGER.exception("Falling back to legacy stats scan")
        return _compute_stats_from_individual()


@router.get("/dashboard/summary")
def get_dashboard_summary():
    """返回统计页所需的聚合摘要，优先复用 manifest。"""
    try:
        sample_manifest = _combined_sample_manifest()
        if not sample_manifest.get("items"):
            sample_manifest = _raw_data_manifest_like()
        router_manifest = router_data_router._combined_router_manifest()
        datasets = [
            {
                "name": item["scenario"],
                "simulator": item.get("simulator", "unknown"),
                "scenario": item["scenario"],
                "sample_count": item.get("sample_count", 0),
                "file_size_bytes": item.get("file_size_bytes", 0),
                "mtime": item.get("mtime", 0),
                "storage": item.get("storage", "jsonl"),
            }
            for item in sample_manifest.get("items", [])
        ]
        stats = sample_manifest.get(
            "summary",
            {
                "total_samples": 0,
                "by_simulator": {},
                "by_scenario": {},
                "by_language": {},
                "by_style": {},
                "by_time_mode": {},
                "timeseries_shapes": {},
            },
        )
        router = router_data_router._build_router_status_from_manifests(
            router_manifest,
            sample_manifest,
        )
        return {
            "stats": stats,
            "datasets": datasets,
            "router": router,
        }
    except Exception:
        LOGGER.exception("Falling back to composed dashboard summary reads")
        return {
            "stats": _get_sample_stats(),
            "datasets": get_datasets(),
            "router": router_data_router.get_router_status(),
        }


def _manifest_item_key(item: dict) -> tuple[str, str]:
    return (str(item.get("simulator") or "unknown"), str(item.get("scenario") or ""))


def _combined_sample_manifest() -> dict:
    jsonl_manifest = manifest_store.ensure_sample_manifest()
    parquet_manifest = portable.text2comp_manifest_like()
    if not parquet_manifest:
        return jsonl_manifest

    parquet_items = list(parquet_manifest.get("items", []))
    parquet_scenarios = {_manifest_item_key(item) for item in parquet_items}
    jsonl_items = [
        {**item, "storage": "jsonl"}
        for item in jsonl_manifest.get("items", [])
        if _manifest_item_key(item) not in parquet_scenarios
    ]
    items = sorted([*parquet_items, *jsonl_items], key=_manifest_item_key)
    return {
        "version": 2,
        "kind": "sample_manifest",
        "storage": "mixed" if jsonl_items else "parquet",
        "generated_at": max(float(jsonl_manifest.get("generated_at") or 0), float(parquet_manifest.get("generated_at") or 0)),
        "items": items,
        "summary": _summary_from_sample_items(items),
    }


def _raw_data_manifest_like() -> dict:
    """Build a dashboard-compatible manifest from Stage 1 HDF5 and Stage 2 templates.

    A portable deployment may intentionally contain only raw scientific HDF5
    files plus language templates. The dashboard should still describe that
    source corpus instead of showing an empty Stage 3 view.
    """
    template_manifest = manifest_store.ensure_template_manifest()
    templates_by_identity: dict[tuple[str, str], dict] = {}
    templates_by_scenario: dict[str, dict] = {}
    for item in template_manifest.get("items", []):
        scenario = str(item.get("scenario") or "")
        if not scenario:
            continue
        simulator = str(item.get("simulator") or "")
        if simulator:
            templates_by_identity[(simulator, scenario)] = item
        templates_by_scenario.setdefault(scenario, item)

    items = [
        _raw_hdf5_item(path, templates_by_identity, templates_by_scenario)
        for path in _iter_raw_hdf5_files()
    ]
    if not items:
        items = [_template_item_like(item) for item in template_manifest.get("items", [])]

    items = sorted(items, key=lambda item: (item.get("simulator", ""), item.get("scenario", "")))
    return {
        "version": 2,
        "kind": "raw_data_manifest",
        "storage": "hdf5" if any(item.get("storage") == "hdf5" for item in items) else "template",
        "generated_at": time.time(),
        "items": items,
        "summary": _summary_from_sample_items(items),
    }


def _iter_raw_hdf5_files() -> list[Path]:
    ignored_dirs = {
        ".manifests",
        ".parquet_jsonl_cache",
        "router",
        "templates",
        "text2comp",
    }
    return iter_hdf5_files_in_child_dirs(DATA_ROOT, skip_dirs=ignored_dirs)


def _raw_hdf5_item(
    path: Path,
    templates_by_identity: dict[tuple[str, str], dict],
    templates_by_scenario: dict[str, dict],
) -> dict:
    stat = path.stat()
    simulator = path.parent.name
    scenario = path.stem
    prefix = f"{simulator}_"
    if scenario.startswith(prefix):
        scenario = scenario[len(prefix):]

    template_item = templates_by_identity.get((simulator, scenario)) or templates_by_scenario.get(scenario, {})
    sample_count, timeseries_shape = _read_hdf5_stats(path)
    return {
        "scenario": scenario,
        "simulator": simulator,
        "sample_count": sample_count,
        "file_size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "path": str(path),
        "storage": "hdf5",
        "by_language": template_item.get("by_language", {}),
        "by_style": template_item.get("by_style", {}),
        "by_time_mode": {},
        "timeseries_shape_obs": timeseries_shape,
    }


def _read_hdf5_stats(path: Path) -> tuple[int, list[int] | None]:
    try:
        import h5py
    except Exception:
        return 0, None

    try:
        with h5py.File(path, "r") as h5_file:
            sample_count = _coerce_int(h5_file.attrs.get("n_samples"))
            timeseries_shape = None
            if "timeseries" in h5_file:
                shape = tuple(int(dim) for dim in h5_file["timeseries"].shape)
                if sample_count is None and shape:
                    sample_count = shape[0]
                if len(shape) >= 3:
                    timeseries_shape = list(shape[1:])

            if timeseries_shape is None:
                channels = _coerce_int(h5_file.attrs.get("n_channels"))
                timesteps = _coerce_int(h5_file.attrs.get("n_timesteps"))
                if channels is not None and timesteps is not None:
                    timeseries_shape = [channels, timesteps]

            return int(sample_count or 0), timeseries_shape
    except Exception:
        LOGGER.exception("Failed to inspect raw HDF5 file %s", path)
        return 0, None


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value.item() if hasattr(value, "item") else value)
    except Exception:
        return None


def _template_item_like(item: dict) -> dict:
    scenario = str(item.get("scenario") or "unknown")
    return {
        "scenario": scenario,
        "simulator": "template",
        "sample_count": int(item.get("template_count") or 0),
        "file_size_bytes": int(item.get("file_size_bytes") or 0),
        "mtime": float(item.get("mtime") or 0),
        "path": str(item.get("path") or ""),
        "storage": "template",
        "by_language": item.get("by_language", {}),
        "by_style": item.get("by_style", {}),
        "by_time_mode": {},
        "timeseries_shape_obs": None,
    }


def _summary_from_sample_items(items: list[dict]) -> dict:
    by_simulator: Counter = Counter()
    by_scenario: Counter = Counter()
    by_language: Counter = Counter()
    by_style: Counter = Counter()
    by_time_mode: Counter = Counter()
    timeseries_shapes: dict = {}
    duplicate_scenarios = duplicate_scenario_names(items)
    total = 0
    for item in items:
        count = int(item.get("sample_count") or 0)
        simulator = str(item.get("simulator") or "unknown")
        scenario = str(item.get("scenario") or "unknown")
        total += count
        by_simulator[simulator] += count
        by_scenario[scenario_summary_key(simulator, scenario, duplicate_scenarios)] += count
        by_language.update({str(k): int(v) for k, v in (item.get("by_language") or {}).items()})
        by_style.update({str(k): int(v) for k, v in (item.get("by_style") or {}).items()})
        by_time_mode.update({str(k): int(v) for k, v in (item.get("by_time_mode") or {}).items()})
        shape = item.get("timeseries_shape_obs")
        if shape is not None:
            timeseries_shapes.setdefault(simulator, shape)
    return {
        "total_samples": total,
        "by_simulator": dict(by_simulator),
        "by_scenario": dict(by_scenario),
        "by_language": dict(by_language),
        "by_style": dict(by_style),
        "by_time_mode": dict(by_time_mode),
        "timeseries_shapes": timeseries_shapes,
    }


def _legacy_get_datasets():
    if not DATA_DIR.exists():
        return []

    results = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        if path.name == "all_training_data.jsonl":
            continue
        scenario = path.stem
        stat = path.stat()

        sample_count = 0
        simulator = "unknown"
        try:
            with open(path, "rb") as handle:
                content = handle.read()
                sample_count = content.count(b"\n")
                if content and not content.endswith(b"\n"):
                    sample_count += 1
            with open(path, "r", encoding="utf-8") as handle:
                first = handle.readline().strip()
                if first:
                    metadata = json.loads(first).get("metadata", {})
                    if isinstance(metadata, dict):
                        simulator = metadata.get("simulator", "unknown")
                        scenario = str(metadata.get("scenario") or scenario)
        except Exception:
            pass

        results.append(
            {
                "name": scenario,
                "simulator": simulator,
                "scenario": scenario,
                "sample_count": sample_count,
                "file_size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )

    return results


def _sample_manifest_items_for_path(source_path: Path) -> list[dict]:
    try:
        manifest = manifest_store.ensure_sample_manifest()
    except Exception:
        return []

    source = str(source_path)
    items: list[dict] = []
    for item in manifest.get("items", []):
        item_path = item.get("path")
        if not item_path:
            continue
        try:
            if str(Path(str(item_path))) == source:
                items.append(item)
        except Exception:
            continue
    return items


def _sample_requires_record_filter(manifest_items: list[dict], scenario: str, simulator: str | None) -> bool:
    if not manifest_items:
        return False

    target_scenario = str(scenario)
    target_simulator = str(simulator) if simulator else None
    matching = [
        item
        for item in manifest_items
        if str(item.get("scenario") or "") == target_scenario
        and (target_simulator is None or str(item.get("simulator") or "unknown") == target_simulator)
    ]
    return len(matching) != len(manifest_items)


def _sample_total_from_manifest(scenario: str, simulator: str | None = None, source_path: Path | None = None) -> int | None:
    try:
        items = _sample_manifest_items_for_path(source_path) if source_path is not None else manifest_store.ensure_sample_manifest().get("items", [])
    except Exception:
        return None

    total = 0
    matched = False
    for item in items:
        if str(item.get("scenario") or "") != scenario:
            continue
        if simulator is not None and str(item.get("simulator") or "unknown") != simulator:
            continue
        total += int(item.get("sample_count", 0))
        matched = True
    return total if matched else None


def _sample_filter_key(language: Optional[str], style: Optional[str]) -> str | None:
    if language and style:
        return f"language={language}|style={style}"
    if language:
        return f"language={language}"
    if style:
        return f"style={style}"
    return None


def _compute_stats_from_individual() -> dict:
    by_simulator: Counter = Counter()
    by_identity: Counter = Counter()
    by_language: Counter = Counter()
    by_style: Counter = Counter()
    by_time_mode: Counter = Counter()
    timeseries_shapes: dict = {}
    total = 0

    if not DATA_DIR.exists():
        return {
            "total_samples": 0,
            "by_simulator": {},
            "by_scenario": {},
            "by_language": {},
            "by_style": {},
            "by_time_mode": {},
            "timeseries_shapes": {},
        }

    for path in DATA_DIR.glob("*.jsonl"):
        if path.name == "all_training_data.jsonl":
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        metadata = sample.get("metadata", {})
                        if not isinstance(metadata, dict):
                            metadata = {}
                    except Exception:
                        continue

                    total += 1
                    simulator = str(metadata.get("simulator") or "unknown")
                    scenario = str(metadata.get("scenario") or "unknown")
                    by_simulator[simulator] += 1
                    by_identity[(simulator, scenario)] += 1
                    by_language[str(metadata.get("language") or "unknown")] += 1
                    by_style[str(metadata.get("style") or "unknown")] += 1
                    observation = metadata.get("observation", {})
                    if not isinstance(observation, dict):
                        observation = {}
                    by_time_mode[str(observation.get("time_mode") or "unknown")] += 1
                    shape_obs = metadata.get("timeseries_shape_obs")
                    if shape_obs and simulator not in timeseries_shapes:
                        timeseries_shapes[simulator] = shape_obs
        except Exception:
            continue

    summary_items = [
        {"simulator": simulator, "scenario": scenario, "sample_count": count}
        for (simulator, scenario), count in by_identity.items()
    ]
    duplicate_scenarios = duplicate_scenario_names(summary_items)
    by_scenario = {
        scenario_summary_key(simulator, scenario, duplicate_scenarios): count
        for (simulator, scenario), count in by_identity.items()
    }

    return {
        "total_samples": total,
        "by_simulator": dict(by_simulator),
        "by_scenario": by_scenario,
        "by_language": dict(by_language),
        "by_style": dict(by_style),
        "by_time_mode": dict(by_time_mode),
        "timeseries_shapes": timeseries_shapes,
    }
