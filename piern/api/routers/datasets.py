"""数据集相关路由：/api/datasets, /api/samples, /api/stats。"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from piern.api.deps import DATA_DIR
from piern.api.routers import router_data as router_data_router
from piern.api.services import jsonl_filter_index, jsonl_index, manifest_store

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.get("/datasets")
def get_datasets():
    """返回 Stage 3 数据集列表，优先走 manifest。"""
    try:
        manifest = manifest_store.ensure_sample_manifest()
        return [
            {
                "name": item["scenario"],
                "simulator": item.get("simulator", "unknown"),
                "scenario": item["scenario"],
                "sample_count": item.get("sample_count", 0),
                "file_size_bytes": item.get("file_size_bytes", 0),
                "mtime": item.get("mtime", 0),
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
):
    """分页读取指定场景的 JSONL 样本（当前仍使用流式扫描）。"""
    jsonl_path = DATA_DIR / f"{scenario}.jsonl"
    if not jsonl_path.exists():
        raise HTTPException(404, f"场景 {scenario} 的 JSONL 文件不存在")

    has_filter = bool(language or style)
    start = page * page_size
    end = start + page_size

    try:
        if not has_filter:
            total_hint = _sample_total_from_manifest(scenario)
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

        filter_key = _sample_filter_key(language=language, style=style)
        if filter_key:
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

                metadata = sample.get("metadata", {})
                if language and metadata.get("language") != language:
                    continue
                if style and metadata.get("style") != style:
                    continue
                if start <= total < end:
                    items_page.append(sample)
                total += 1
        return {"total": total, "page": page, "page_size": page_size, "items": items_page}
    except Exception as exc:
        raise HTTPException(500, f"读取 JSONL 失败: {exc}")


@router.get("/stats")
def get_stats():
    """返回 Stage 3 聚合统计，优先走 manifest。"""
    try:
        manifest = manifest_store.ensure_sample_manifest()
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
        LOGGER.exception("Falling back to legacy /api/stats scan")
        return _compute_stats_from_individual()


@router.get("/dashboard/summary")
def get_dashboard_summary():
    """返回统计页所需的聚合摘要，优先复用 manifest。"""
    try:
        sample_manifest = manifest_store.ensure_sample_manifest()
        router_manifest = manifest_store.ensure_router_manifest()
        datasets = [
            {
                "name": item["scenario"],
                "simulator": item.get("simulator", "unknown"),
                "scenario": item["scenario"],
                "sample_count": item.get("sample_count", 0),
                "file_size_bytes": item.get("file_size_bytes", 0),
                "mtime": item.get("mtime", 0),
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
            "stats": get_stats(),
            "datasets": get_datasets(),
            "router": router_data_router.get_router_status(),
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
                    simulator = metadata.get("simulator", "unknown")
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


def _sample_total_from_manifest(scenario: str) -> int | None:
    try:
        manifest = manifest_store.ensure_sample_manifest()
    except Exception:
        return None

    for item in manifest.get("items", []):
        if item.get("scenario") == scenario:
            return int(item.get("sample_count", 0))
    return None


def _sample_filter_key(language: Optional[str], style: Optional[str]) -> str | None:
    if language and style:
        return f"language={language}|style={style}"
    if language:
        return f"language={language}"
    if style:
        return f"style={style}"
    return None


def _compute_stats(jsonl_path: Path) -> dict:
    by_simulator: Counter = Counter()
    by_scenario: Counter = Counter()
    by_language: Counter = Counter()
    by_style: Counter = Counter()
    by_time_mode: Counter = Counter()
    timeseries_shapes: dict = {}
    total = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    metadata = sample.get("metadata", {})
                except Exception:
                    continue

                total += 1
                simulator = metadata.get("simulator", "unknown")
                by_simulator[simulator] += 1
                scenario = metadata.get("scenario", "unknown")
                by_scenario[scenario] += 1
                by_language[metadata.get("language", "?")] += 1
                by_style[metadata.get("style", "?")] += 1
                observation = metadata.get("observation", {})
                by_time_mode[observation.get("time_mode", "?")] += 1
                shape_obs = metadata.get("timeseries_shape_obs")
                if shape_obs and simulator not in timeseries_shapes:
                    timeseries_shapes[simulator] = shape_obs
    except Exception:
        pass

    return {
        "total_samples": total,
        "by_simulator": dict(by_simulator),
        "by_scenario": dict(by_scenario),
        "by_language": dict(by_language),
        "by_style": dict(by_style),
        "by_time_mode": dict(by_time_mode),
        "timeseries_shapes": timeseries_shapes,
    }


def _compute_stats_from_individual() -> dict:
    by_simulator: Counter = Counter()
    by_scenario: Counter = Counter()
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
        stats = _compute_stats(path)
        total += stats["total_samples"]
        by_simulator.update(stats["by_simulator"])
        by_scenario.update(stats["by_scenario"])
        by_language.update(stats["by_language"])
        by_style.update(stats["by_style"])
        by_time_mode.update(stats["by_time_mode"])
        for simulator, shape in stats["timeseries_shapes"].items():
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
