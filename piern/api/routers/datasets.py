"""数据集相关路由：/api/datasets, /api/samples, /api/stats。"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from piern.api.deps import DATA_DIR

router = APIRouter()

_stats_cache: Optional[dict] = None
_stats_cache_key: tuple = ()


@router.get("/datasets")
def get_datasets():
    """扫描 data/text2comp/*.jsonl，返回数据集列表（排除 all_training_data.jsonl）。"""
    if not DATA_DIR.exists():
        return []

    results = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        if f.name == "all_training_data.jsonl":
            continue
        scenario = f.stem
        stat = f.stat()

        sample_count = 0
        simulator = "unknown"
        try:
            with open(f, "rb") as fh:
                content = fh.read()
                # count newlines; if file doesn't end with \n, last line still counts
                sample_count = content.count(b"\n")
                if content and not content.endswith(b"\n"):
                    sample_count += 1
            with open(f, "r", encoding="utf-8") as fh:
                first = fh.readline().strip()
                if first:
                    meta = json.loads(first).get("metadata", {})
                    simulator = meta.get("simulator", "unknown")
        except Exception:
            pass

        results.append({
            "name": scenario,
            "simulator": simulator,
            "scenario": scenario,
            "sample_count": sample_count,
            "file_size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
        })

    return results


@router.get("/samples")
def get_samples(
    scenario: str = Query(...),
    page: int = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
    language: Optional[str] = Query(None),
    style: Optional[str] = Query(None),
):
    """分页读取指定场景的 JSONL 样本（流式，不全量加载）。"""
    jsonl_path = DATA_DIR / f"{scenario}.jsonl"
    if not jsonl_path.exists():
        raise HTTPException(404, f"场景 {scenario} 的 JSONL 文件不存在")

    has_filter = bool(language or style)
    start = page * page_size
    end = start + page_size

    try:
        if not has_filter:
            # 无筛选：两次扫描，第一次只数行数，第二次取目标行
            total = 0
            with open(jsonl_path, "rb") as fb:
                content = fb.read()
                total = content.count(b"\n")
                if content and not content.endswith(b"\n"):
                    total += 1

            items = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= end:
                        break
                    if idx < start:
                        continue
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return {"total": total, "page": page, "page_size": page_size, "items": items}
        else:
            # 有筛选：必须全量扫描以统计 total，但不在内存中保留全部对象
            total = 0
            items_page: list = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        meta = sample.get("metadata", {})
                        if language and meta.get("language") != language:
                            continue
                        if style and meta.get("style") != style:
                            continue
                        if start <= total < end:
                            items_page.append(sample)
                        total += 1
                    except json.JSONDecodeError:
                        continue
            return {"total": total, "page": page, "page_size": page_size, "items": items_page}
    except Exception as e:
        raise HTTPException(500, f"读取 JSONL 失败: {e}")


def _stats_snapshot_key() -> tuple:
    if not DATA_DIR.exists():
        return ()

    snapshot = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        if f.name == "all_training_data.jsonl":
            continue
        stat = f.stat()
        snapshot.append((f.name, stat.st_size, stat.st_mtime_ns))
    return tuple(snapshot)


@router.get("/stats")
def get_stats():
    """????? JSONL ???????????????????? all_training_data.jsonl??"""
    global _stats_cache, _stats_cache_key

    snapshot_key = _stats_snapshot_key()
    if _stats_cache is not None and snapshot_key == _stats_cache_key:
        return _stats_cache

    stats = _compute_stats_from_individual()
    _stats_cache = stats
    _stats_cache_key = snapshot_key
    return stats


def _compute_stats(jsonl_path: Path) -> dict:
    by_simulator: Counter = Counter()
    by_scenario: Counter = Counter()
    by_language: Counter = Counter()
    by_style: Counter = Counter()
    by_time_mode: Counter = Counter()
    timeseries_shapes: dict = {}
    total = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    meta = sample.get("metadata", {})
                    total += 1
                    by_simulator[meta.get("simulator", "unknown")] += 1
                    sc = meta.get("scenario", "unknown")
                    by_scenario[sc] += 1
                    by_language[meta.get("language", "?")] += 1
                    by_style[meta.get("style", "?")] += 1
                    obs = meta.get("observation", {})
                    by_time_mode[obs.get("time_mode", "?")] += 1
                    shape_obs = meta.get("timeseries_shape_obs")
                    sim = meta.get("simulator", "unknown")
                    if shape_obs and sim not in timeseries_shapes:
                        timeseries_shapes[sim] = shape_obs
                except Exception:
                    continue
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
            "by_simulator": {}, "by_scenario": {},
            "by_language": {}, "by_style": {},
            "by_time_mode": {}, "timeseries_shapes": {},
        }

    for f in DATA_DIR.glob("*.jsonl"):
        if f.name == "all_training_data.jsonl":
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        meta = sample.get("metadata", {})
                        total += 1
                        by_simulator[meta.get("simulator", "unknown")] += 1
                        sc = meta.get("scenario", "unknown")
                        by_scenario[sc] += 1
                        by_language[meta.get("language", "?")] += 1
                        by_style[meta.get("style", "?")] += 1
                        obs = meta.get("observation", {})
                        by_time_mode[obs.get("time_mode", "?")] += 1
                        shape_obs = meta.get("timeseries_shape_obs")
                        sim = meta.get("simulator", "unknown")
                        if shape_obs and sim not in timeseries_shapes:
                            timeseries_shapes[sim] = shape_obs
                    except Exception:
                        continue
        except Exception:
            continue

    return {
        "total_samples": total,
        "by_simulator": dict(by_simulator),
        "by_scenario": dict(by_scenario),
        "by_language": dict(by_language),
        "by_style": dict(by_style),
        "by_time_mode": dict(by_time_mode),
        "timeseries_shapes": timeseries_shapes,
    }
