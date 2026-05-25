"""Stage 4 Token Router routes under /api/router/*."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from PierNet.shared.runtime.paths import DATA_ROOT
from PierNet.shared.storage import portable
from PierNet.synth.services import job_manager, jsonl_filter_index, jsonl_index, manifest_store, router_executor, worker_queue
from PierNet.synth.services.job_manager import publish

router = APIRouter()

ROUTER_DIR = DATA_ROOT / "router"
SCENARIO_DIR = ROUTER_DIR / "by_scenario"
TEXT2COMP_DIR = DATA_ROOT / "text2comp"
DEFAULT_LOCAL_QWEN_DIR = str(Path.home() / "Qwen" / "Qwen2.5-0.5B-Instruct")
DEFAULT_QWEN_EMBEDDING_MODEL = os.getenv("PierNet_QWEN_EMBEDDING_MODEL", DEFAULT_LOCAL_QWEN_DIR)
DEFAULT_QWEN_EMBEDDING_TOKENIZER = os.getenv("PierNet_QWEN_EMBEDDING_TOKENIZER", DEFAULT_QWEN_EMBEDDING_MODEL)


def _running_job_ids(job_types: set[str]) -> list[str]:
    return [job.job_id for job in job_manager.running_jobs(job_types)]


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return 0


def _manifest_item_key(item: dict) -> tuple[str, str]:
    return (str(item.get("simulator") or "unknown"), str(item.get("scenario") or ""))


def _scenario_selector(simulator: str, scenario: str) -> str:
    return f"{simulator or 'unknown'}/{scenario}"


def _parse_scenario_selector(selector: str) -> tuple[str | None, str]:
    value = str(selector or "").strip()
    if not value:
        return None, ""
    if "::" in value:
        simulator, scenario = value.split("::", 1)
        return simulator.strip() or "unknown", scenario.strip()
    if "/" in value:
        simulator, scenario = value.split("/", 1)
        return simulator.strip() or "unknown", scenario.strip()
    return None, value


def _sample_manifest_selector_index(sample_manifest: dict) -> tuple[set[str], dict[str, set[str]]]:
    selectors: set[str] = set()
    simulators_by_scenario: dict[str, set[str]] = {}
    for item in sample_manifest.get("items", []):
        scenario = str(item.get("scenario") or "").strip()
        if not scenario:
            continue
        simulator = str(item.get("simulator") or "unknown").strip() or "unknown"
        selectors.add(_scenario_selector(simulator, scenario))
        simulators_by_scenario.setdefault(scenario, set()).add(simulator)
    return selectors, simulators_by_scenario


def _split_router_build_scenario_query(scenarios: str | list[str] | None) -> list[str]:
    if not scenarios:
        return []
    raw_values = [scenarios] if isinstance(scenarios, str) else scenarios
    requested: list[str] = []
    for value in raw_values:
        requested.extend(item.strip() for item in str(value).split(",") if item.strip())
    return requested


def _normalize_router_build_scenarios(scenarios: list[str]) -> list[str]:
    if not scenarios:
        return []

    sample_manifest = _combined_sample_manifest()
    selectors, simulators_by_scenario = _sample_manifest_selector_index(sample_manifest)
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_selector in scenarios:
        simulator, scenario = _parse_scenario_selector(raw_selector)
        if not scenario:
            continue
        if simulator is not None:
            selector = _scenario_selector(simulator, scenario)
            if selector not in selectors:
                raise HTTPException(status_code=400, detail=f"阶段 3 样本中未找到场景: {selector}")
        else:
            simulators = sorted(simulators_by_scenario.get(scenario, set()))
            if not simulators:
                raise HTTPException(status_code=400, detail=f"阶段 3 样本中未找到场景: {scenario}")
            if len(simulators) > 1:
                choices = ", ".join(_scenario_selector(candidate, scenario) for candidate in simulators)
                raise HTTPException(status_code=400, detail=f"场景 {scenario} 同时存在于多个模拟器，请使用完整选择器: {choices}")
            selector = _scenario_selector(simulators[0], scenario)
        if selector in seen:
            continue
        seen.add(selector)
        normalized.append(selector)

    return normalized


def _is_safe_name_component(value: str | None) -> bool:
    component = str(value or "")
    return (
        bool(component)
        and component not in {".", ".."}
        and "\x00" not in component
        and Path(component).name == component
        and "\\" not in component
    )


def _router_record_matches_identity(
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


def _router_record_matches_query(
    record: dict,
    fallback_scenario: str,
    scenario: str,
    simulator: str | None,
) -> bool:
    if scenario:
        return _router_record_matches_identity(record, fallback_scenario, scenario, simulator)
    if simulator is None:
        return True
    metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
    if not isinstance(metadata, dict):
        return False
    return str(metadata.get("simulator") or "").strip() == simulator


def _router_label_value(record: dict) -> int | None:
    label = record.get("label")
    if label in (0, 1, "0", "1"):
        return int(label)
    return None


def _router_label_matches(record: dict, label: int) -> bool:
    if label not in (0, 1):
        return True
    return _router_label_value(record) == label


def _router_jsonl_contains_identity(path: Path, scenario: str, simulator: str | None = None) -> bool:
    saw_record = False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                saw_record = True
                if _router_record_matches_identity(record, path.stem, scenario, simulator):
                    return True
    except OSError:
        return False
    return not saw_record and path.stem == scenario


def _resolve_router_jsonl_file(scenario: str, simulator: str | None = None) -> Path | None:
    if not _is_safe_name_component(scenario) or (simulator is not None and not _is_safe_name_component(simulator)):
        return None
    direct = SCENARIO_DIR / f"{scenario}.jsonl"
    matches: list[Path] = []
    if direct.exists() and _router_jsonl_contains_identity(direct, scenario, simulator=simulator):
        matches.append(direct)

    if SCENARIO_DIR.exists():
        for path in sorted(SCENARIO_DIR.glob("*.jsonl")):
            if path == direct:
                continue
            if _router_jsonl_contains_identity(path, scenario, simulator=simulator):
                matches.append(path)

    if len(matches) > 1:
        target = f"{simulator}/{scenario}" if simulator else scenario
        raise HTTPException(status_code=409, detail=f"路由场景匹配到多个 JSONL 文件: {target}")
    return matches[0] if matches else None


def _build_router_status_from_manifests(router_manifest: dict, sample_manifest: dict) -> dict:
    sample_items = {_manifest_item_key(item): item for item in sample_manifest.get("items", [])}
    source_by_scenario: dict[str, int] = {}
    for item in sample_items.values():
        scenario = str(item.get("scenario") or "unknown")
        simulator = str(item.get("simulator") or "unknown")
        source_by_scenario[_scenario_selector(simulator, scenario)] = int(item.get("sample_count") or 0)

    scenario_map: dict[tuple[str, str], dict] = {
        scenario: {
            "scenario": item.get("scenario", "unknown"),
            "simulator": item.get("simulator", "unknown"),
            "source_count": item.get("sample_count", 0),
        }
        for scenario, item in sample_items.items()
    }

    for item in router_manifest.get("scenarios", []):
        key = _manifest_item_key(item)
        entry = scenario_map.setdefault(
            key,
            {
                "scenario": item["scenario"],
                "simulator": item.get("simulator", "unknown"),
                "source_count": 0,
            },
        )
        if entry.get("simulator", "unknown") == "unknown":
            entry["simulator"] = item.get("simulator", "unknown")
        entry["router_count"] = item.get("router_count", 0)
        entry["file_size_bytes"] = item.get("file_size_bytes", 0)
        entry["mtime"] = item.get("mtime", 0)
        entry["storage"] = item.get("storage", "jsonl")

    return {
        "splits": router_manifest.get(
            "splits",
            {
                "train": {
                    "exists": False,
                    "count": 0,
                    "file_size_bytes": 0,
                    "mtime": 0,
                }
            },
        ),
        "total": router_manifest.get("total", 0),
        "label_counts": router_manifest.get("label_counts", {"0": 0, "1": 0}),
        "scenarios": sorted(scenario_map.values(), key=_manifest_item_key),
        "source_count": sample_manifest.get("summary", {}).get("total_samples", 0),
        "source_by_scenario": source_by_scenario,
        "router_dir": str(portable.ROUTER_PARQUET_DIR if router_manifest.get("storage") == "parquet" else ROUTER_DIR),
        "storage": router_manifest.get("storage", "jsonl"),
    }


def _legacy_get_router_status() -> dict:
    splits: dict[str, dict] = {}
    total = 0
    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        stat = train_path.stat()
        count = _count_lines(train_path)
        splits["train"] = {
            "exists": True,
            "count": count,
            "file_size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
        }
        total = count
    else:
        splits["train"] = {"exists": False, "count": 0, "file_size_bytes": 0, "mtime": 0}

    label_counts = {"0": 0, "1": 0}
    if train_path.exists():
        try:
            with train_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    label = _router_label_value(json.loads(line))
                    if label is not None:
                        label_counts[str(label)] += 1
        except Exception:
            pass

    scenarios: list[dict] = []
    if SCENARIO_DIR.exists():
        for path in sorted(SCENARIO_DIR.glob("*.jsonl")):
            stat = path.stat()
            simulator = "unknown"
            scenario = path.stem
            count = 0
            try:
                with path.open("r", encoding="utf-8") as handle:
                    first = None
                    for raw in handle:
                        line = raw.strip()
                        if not line:
                            continue
                        count += 1
                        if first is None:
                            first = line
                    if first:
                        metadata = json.loads(first).get("metadata", {})
                        simulator = metadata.get("simulator", "unknown")
                        scenario = str(metadata.get("scenario") or path.stem)
            except Exception:
                pass
            scenarios.append(
                {
                    "scenario": scenario,
                    "simulator": simulator,
                    "count": count,
                    "file_size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )

    source_by_scenario: dict[str, int] = {}
    source_scenarios: list[dict] = []
    if TEXT2COMP_DIR.exists():
        for path in sorted(TEXT2COMP_DIR.glob("*.jsonl")):
            if path.name == "all_training_data.jsonl":
                continue
            simulator = "unknown"
            scenario = path.stem
            count = 0
            try:
                with path.open("r", encoding="utf-8") as handle:
                    first = None
                    for raw in handle:
                        line = raw.strip()
                        if not line:
                            continue
                        count += 1
                        if first is None:
                            first = line
                    if first:
                        metadata = json.loads(first).get("metadata", {})
                        simulator = metadata.get("simulator", "unknown")
                        scenario = str(metadata.get("scenario") or path.stem)
            except Exception:
                pass
            source_by_scenario[_scenario_selector(simulator, scenario)] = count
            source_scenarios.append(
                {
                    "scenario": scenario,
                    "simulator": simulator,
                    "source_count": count,
                }
            )

    scenario_map = {_manifest_item_key(item): item for item in source_scenarios}
    for item in scenarios:
        key = _manifest_item_key(item)
        entry = scenario_map.setdefault(
            key,
            {
                "scenario": item["scenario"],
                "simulator": item.get("simulator", "unknown"),
                "source_count": 0,
            },
        )
        if entry.get("simulator", "unknown") == "unknown":
            entry["simulator"] = item.get("simulator", "unknown")
        entry["router_count"] = item["count"]
        entry["file_size_bytes"] = item["file_size_bytes"]
        entry["mtime"] = item["mtime"]

    return {
        "splits": splits,
        "total": total,
        "label_counts": label_counts,
        "scenarios": sorted(scenario_map.values(), key=_manifest_item_key),
        "source_count": sum(source_by_scenario.values()),
        "source_by_scenario": source_by_scenario,
        "router_dir": str(ROUTER_DIR),
    }


def _combined_sample_manifest() -> dict:
    try:
        jsonl_manifest = manifest_store.ensure_sample_manifest()
    except Exception:
        jsonl_manifest = {}
    parquet_manifest = portable.text2comp_manifest_like()
    if not parquet_manifest:
        return jsonl_manifest
    if not jsonl_manifest:
        return parquet_manifest

    parquet_items = list(parquet_manifest.get("items", []))
    parquet_scenarios = {_manifest_item_key(item) for item in parquet_items}
    items = [*parquet_items]
    items.extend(
        {**item, "storage": "jsonl"}
        for item in jsonl_manifest.get("items", [])
        if _manifest_item_key(item) not in parquet_scenarios
    )
    total = sum(int(item.get("sample_count") or 0) for item in items)
    return {
        "version": 2,
        "kind": "sample_manifest",
        "storage": "mixed" if len(items) != len(parquet_items) else "parquet",
        "items": sorted(items, key=_manifest_item_key),
        "summary": {"total_samples": total},
    }


def _merge_label_counts(*sources: dict) -> dict[str, int]:
    counts: dict[str, int] = {"0": 0, "1": 0}
    for source in sources:
        for key, value in (source.get("label_counts") or {}).items():
            try:
                counts[str(key)] = counts.get(str(key), 0) + int(value or 0)
            except (TypeError, ValueError):
                continue
    return counts


def _combined_router_manifest() -> dict:
    try:
        jsonl_manifest = manifest_store.ensure_router_manifest()
    except Exception:
        jsonl_manifest = {}
    parquet_manifest = portable.router_manifest_like()
    if not parquet_manifest:
        return jsonl_manifest
    if not jsonl_manifest:
        return parquet_manifest

    parquet_scenarios = {_manifest_item_key(item) for item in parquet_manifest.get("scenarios", [])}
    scenarios = [*parquet_manifest.get("scenarios", [])]
    scenarios.extend(
        {**item, "storage": "jsonl"}
        for item in jsonl_manifest.get("scenarios", [])
        if _manifest_item_key(item) not in parquet_scenarios
    )
    total = sum(int(item.get("router_count") or 0) for item in scenarios)
    size = sum(int(item.get("file_size_bytes") or 0) for item in scenarios)
    mtime = max((float(item.get("mtime") or 0) for item in scenarios), default=0.0)
    label_counts = _merge_label_counts(parquet_manifest, jsonl_manifest)
    return {
        "version": 2,
        "kind": "router_manifest",
        "storage": "mixed" if len(scenarios) != len(parquet_manifest.get("scenarios", [])) else "parquet",
        "generated_at": max(float(jsonl_manifest.get("generated_at") or 0), float(parquet_manifest.get("generated_at") or 0)),
        "splits": {"train": {"exists": bool(scenarios), "count": total, "file_size_bytes": size, "mtime": mtime}},
        "total": total,
        "label_counts": label_counts,
        "scenarios": sorted(scenarios, key=_manifest_item_key),
    }


def _router_manifest_items_for_path(source_path: Path) -> list[dict]:
    try:
        manifest = _combined_router_manifest()
    except Exception:
        return []

    source = str(source_path)
    items: list[dict] = []
    for item in manifest.get("scenarios", []):
        item_path = item.get("path")
        if not item_path:
            continue
        try:
            if str(Path(str(item_path))) == source:
                items.append(item)
        except Exception:
            continue
    return items


def _router_jsonl_paths_from_manifest(manifest: dict) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in manifest.get("scenarios", []):
        if item.get("storage") == "parquet":
            continue
        raw_path = item.get("path")
        if not raw_path:
            continue
        try:
            path = Path(str(raw_path))
        except Exception:
            continue
        if path in seen or not path.exists():
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _iter_router_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _router_requires_record_filter(manifest_items: list[dict], scenario: str, simulator: str | None) -> bool:
    if not scenario or not manifest_items:
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


def _read_mixed_router_page(
    *,
    page: int,
    page_size: int,
    label: int,
    simulator: str | None,
) -> dict | None:
    try:
        manifest = _combined_router_manifest()
    except Exception:
        return None
    if manifest.get("storage") != "mixed":
        return None

    filters: list[tuple[str, object]] = []
    if simulator:
        filters.append(("simulator", simulator))
    if label in (0, 1):
        filters.append(("label", label))

    start = page * page_size
    end = start + page_size
    items: list[dict] = []
    total = 0
    try:
        if portable.has_partitions("router"):
            for obj in portable.iter_records("router", filters=filters):
                if start <= total < end:
                    items.append(obj)
                total += 1

        for path in _router_jsonl_paths_from_manifest(manifest):
            for obj in _iter_router_jsonl_records(path):
                if not _router_record_matches_query(obj, path.stem, "", simulator):
                    continue
                if not _router_label_matches(obj, label):
                    continue
                if start <= total < end:
                    items.append(obj)
                total += 1
    except Exception as exc:
        return {"total": 0, "page": page, "page_size": page_size, "items": [], "error": f"读取 mixed 路由数据失败: {exc}"}

    return {"total": total, "page": page, "page_size": page_size, "items": items, "storage": "mixed"}


def _router_total_from_manifest(
    split: str,
    scenario: str,
    simulator: str | None = None,
    source_path: Path | None = None,
) -> int | None:
    try:
        manifest = _combined_router_manifest()
    except Exception:
        return None

    if scenario:
        total = 0
        matched = False
        source = str(source_path) if source_path is not None else None
        for item in manifest.get("scenarios", []):
            if source is not None and str(Path(str(item.get("path") or ""))) != source:
                continue
            if str(item.get("scenario") or "") != scenario:
                continue
            if simulator is not None and str(item.get("simulator") or "unknown") != simulator:
                continue
            total += int(item.get("router_count", 0))
            matched = True
        return total if matched else 0

    splits = manifest.get("splits", {})
    split_info = splits.get(split)
    if not split_info:
        return 0
    return int(split_info.get("count", 0))


def _use_worker_queue() -> bool:
    return worker_queue.queue_enabled() and os.getenv("PierNet_WORKER_QUEUE_SYNTH", "1").strip().lower() not in {"0", "false", "no", "off"}


@router.get("/router/status")
def get_router_status():
    """Return router dataset status and per-scenario stats."""
    try:
        router_manifest = _combined_router_manifest()
        sample_manifest = _combined_sample_manifest()
        return _build_router_status_from_manifests(router_manifest, sample_manifest)
    except Exception:
        return _legacy_get_router_status()


@router.post("/router/build")
async def build_router_data(
    seed: int = Query(42, ge=0, le=2_147_483_647),
    neg_ratio: int = Query(1, ge=1, le=10),
    max_workers: int = Query(8, ge=1, le=64),
    scenarios: list[str] = Query(default_factory=list),
):
    """Start Stage 4 router build and return a job id for SSE."""
    active_jobs = _running_job_ids({"fill_samples", "router"})
    if active_jobs:
        raise HTTPException(
            status_code=409,
            detail=(
                "样本填充或路由构建任务仍在运行（"
                + ", ".join(active_jobs)
                + "）。请先终止或等待完成后再构建路由，避免从正在写入的样本数据生成不完整路由集。"
            ),
        )
    requested_scenarios = _split_router_build_scenario_query(scenarios)
    scenario_list = _normalize_router_build_scenarios(requested_scenarios)
    router_lock_scenarios = scenario_list or ["all"]
    queued = _use_worker_queue()
    payload = {"seed": seed, "neg_ratio": neg_ratio, "max_workers": max_workers, "scenarios": scenario_list}
    try:
        record = job_manager.create_job(
            "router",
            request=payload,
            lock_keys=[
                *(f"router:{scenario}" for scenario in router_lock_scenarios),
                *(f"dataset:{scenario}" for scenario in router_lock_scenarios),
            ],
            status="queued" if queued else "running",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if queued:
        publish(record, {"type": "queued", "ts": time.time(), "message": "任务已进入 worker 队列"})
    else:
        threading.Thread(target=router_executor.run_router_build_job, args=(record, payload), daemon=True).start()
    return {"job_id": record.job_id, "status": record.status}


@router.get("/router/samples")
def get_router_samples(
    split: str = Query("train"),
    scenario: str = Query(""),
    simulator: str = Query(""),
    page: int = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
    label: int = Query(-1, ge=-1, le=1),
):
    """Read router samples page-wise, optionally filtered by split, scenario and label."""
    split_filter = split if isinstance(split, str) and split else "train"
    scenario_filter = scenario.strip() if isinstance(scenario, str) and scenario.strip() else ""
    label_filter = label if isinstance(label, int) else -1
    simulator_filter = simulator.strip() if isinstance(simulator, str) and simulator.strip() else None
    if not _is_safe_name_component(split_filter):
        raise HTTPException(status_code=400, detail="split 只能是单个文件名组件")
    if scenario_filter and not _is_safe_name_component(scenario_filter):
        raise HTTPException(status_code=400, detail="scenario 只能是单个文件名组件")
    if simulator_filter is not None and not _is_safe_name_component(simulator_filter):
        raise HTTPException(status_code=400, detail="simulator 只能是单个文件名组件")
    if not scenario_filter:
        mixed_page = _read_mixed_router_page(
            page=page,
            page_size=page_size,
            label=label_filter,
            simulator=simulator_filter,
        )
        if mixed_page is not None:
            return mixed_page
    has_matching_parquet = (
        portable.has_partitions("router")
        if not scenario_filter
        else portable.partition_for("router", scenario_filter, simulator=simulator_filter) is not None
    )
    if has_matching_parquet:
        try:
            parquet_page = portable.read_router_page(
                scenario=scenario_filter,
                page=page,
                page_size=page_size,
                label=label_filter,
                simulator=simulator_filter,
            )
            if parquet_page is not None:
                total, items = parquet_page
                return {"total": total, "page": page, "page_size": page_size, "items": items, "storage": "parquet"}
        except Exception as exc:
            return {"total": 0, "page": page, "page_size": page_size, "items": [], "error": f"读取 Parquet 失败: {exc}"}

    path = (
        _resolve_router_jsonl_file(scenario_filter, simulator=simulator_filter)
        if scenario_filter
        else ROUTER_DIR / f"{split_filter}.jsonl"
    )
    if path is None or not path.exists():
        return {"total": 0, "page": page, "page_size": page_size, "items": []}

    router_manifest_items = _router_manifest_items_for_path(path) if scenario_filter else []
    requires_record_filter = _router_requires_record_filter(router_manifest_items, scenario_filter, simulator_filter)
    requires_record_filter = requires_record_filter or bool(scenario_filter and path.stem != scenario_filter)
    has_filter = label_filter in (0, 1) or bool(simulator_filter) or requires_record_filter
    start = page * page_size
    end = start + page_size

    try:
        if not has_filter:
            total_hint = _router_total_from_manifest(
                split=split_filter,
                scenario=scenario_filter,
                simulator=simulator_filter,
                source_path=path if scenario_filter else None,
            )
            try:
                total, items = jsonl_index.read_page(
                    path,
                    page=page,
                    page_size=page_size,
                    total_rows=total_hint,
                )
                return {"total": total, "page": page, "page_size": page_size, "items": items}
            except Exception:
                pass

        if label_filter in (0, 1) and not simulator_filter and not requires_record_filter:
            try:
                total, items = jsonl_filter_index.read_filtered_page(
                    path,
                    profile="router_label",
                    key=f"label={label_filter}",
                    page=page,
                    page_size=page_size,
                )
                return {"total": total, "page": page, "page_size": page_size, "items": items}
            except Exception:
                pass

        items: list[dict] = []
        total = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not _router_record_matches_query(obj, path.stem, scenario_filter, simulator_filter):
                    continue
                if not _router_label_matches(obj, label_filter):
                    continue
                if start <= total < end:
                    items.append(obj)
                total += 1
    except Exception as exc:
        return {"total": 0, "page": page, "page_size": page_size, "items": [], "error": str(exc)}

    return {"total": total, "page": page, "page_size": page_size, "items": items}
