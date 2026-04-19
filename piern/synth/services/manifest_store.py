"""Sidecar manifest builders for template, sample, and router JSONL artifacts."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from piern.shared.runtime.paths import DATA_DIR, PROJECT_ROOT, TEMPLATES_DIR

ROUTER_DIR = PROJECT_ROOT / "data" / "router"
ROUTER_SCENARIO_DIR = ROUTER_DIR / "by_scenario"
MANIFEST_DIR = PROJECT_ROOT / "data" / ".manifests"

TEMPLATE_MANIFEST_PATH = MANIFEST_DIR / "templates.json"
SAMPLE_MANIFEST_PATH = MANIFEST_DIR / "samples.json"
ROUTER_MANIFEST_PATH = MANIFEST_DIR / "router.json"

_LOCK = threading.Lock()
_VERSION = 1


def _iter_template_files() -> list[Path]:
    if not TEMPLATES_DIR.exists():
        return []
    return sorted(TEMPLATES_DIR.glob("*_templates.jsonl"))


def _iter_sample_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        path
        for path in DATA_DIR.glob("*.jsonl")
        if path.name != "all_training_data.jsonl"
    )


def _iter_router_scenario_files() -> list[Path]:
    if not ROUTER_SCENARIO_DIR.exists():
        return []
    return sorted(ROUTER_SCENARIO_DIR.glob("*.jsonl"))


def _snapshot(paths: Iterable[Path]) -> list[dict]:
    snapshot: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        stat = path.stat()
        snapshot.append(
            {
                "name": path.name,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return snapshot


def _template_snapshot() -> list[dict]:
    return _snapshot(_iter_template_files())


def _sample_snapshot() -> list[dict]:
    return _snapshot(_iter_sample_files())


def _router_snapshot() -> list[dict]:
    files = _iter_router_scenario_files()
    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        files = [train_path, *files]
    return _snapshot(files)


def _read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_manifest(path: Path, payload: dict) -> dict:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return payload


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _template_manifest_payload(snapshot: list[dict]) -> dict:
    items: list[dict] = []
    total_templates = 0
    by_language: Counter = Counter()
    by_style: Counter = Counter()

    for path in _iter_template_files():
        stat = path.stat()
        template_count = 0
        item_language: Counter = Counter()
        item_style: Counter = Counter()
        scenario = path.stem.replace("_templates", "")

        for record in _iter_jsonl(path):
            template_count += 1
            language = str(record.get("language") or "?")
            style = str(record.get("style") or "?")
            item_language[language] += 1
            item_style[style] += 1

        total_templates += template_count
        by_language.update(item_language)
        by_style.update(item_style)
        items.append(
            {
                "scenario": scenario,
                "template_count": template_count,
                "file_size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "path": str(path),
                "by_language": dict(item_language),
                "by_style": dict(item_style),
            }
        )

    return {
        "version": _VERSION,
        "kind": "template_manifest",
        "generated_at": time.time(),
        "snapshot": snapshot,
        "items": items,
        "summary": {
            "total_templates": total_templates,
            "by_language": dict(by_language),
            "by_style": dict(by_style),
        },
    }


def _sample_manifest_payload(snapshot: list[dict]) -> dict:
    items: list[dict] = []
    total_samples = 0
    by_simulator: Counter = Counter()
    by_scenario: Counter = Counter()
    by_language: Counter = Counter()
    by_style: Counter = Counter()
    by_time_mode: Counter = Counter()
    timeseries_shapes: dict[str, list[int]] = {}

    for path in _iter_sample_files():
        stat = path.stat()
        scenario = path.stem
        sample_count = 0
        simulator = "unknown"
        item_language: Counter = Counter()
        item_style: Counter = Counter()
        item_time_mode: Counter = Counter()
        timeseries_shape_obs = None

        for record in _iter_jsonl(path):
            sample_count += 1
            metadata = record.get("metadata", {})
            simulator = metadata.get("simulator", simulator) or simulator
            item_language[str(metadata.get("language") or "?")] += 1
            item_style[str(metadata.get("style") or "?")] += 1
            observation = metadata.get("observation", {})
            item_time_mode[str(observation.get("time_mode") or "?")] += 1
            if timeseries_shape_obs is None:
                shape_obs = metadata.get("timeseries_shape_obs")
                if isinstance(shape_obs, list):
                    timeseries_shape_obs = shape_obs
                elif isinstance(shape_obs, tuple):
                    timeseries_shape_obs = list(shape_obs)

        total_samples += sample_count
        by_simulator[simulator] += sample_count
        by_scenario[scenario] += sample_count
        by_language.update(item_language)
        by_style.update(item_style)
        by_time_mode.update(item_time_mode)
        if simulator not in timeseries_shapes and timeseries_shape_obs is not None:
            timeseries_shapes[simulator] = timeseries_shape_obs

        items.append(
            {
                "scenario": scenario,
                "simulator": simulator,
                "sample_count": sample_count,
                "file_size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "path": str(path),
                "by_language": dict(item_language),
                "by_style": dict(item_style),
                "by_time_mode": dict(item_time_mode),
                "timeseries_shape_obs": timeseries_shape_obs,
            }
        )

    return {
        "version": _VERSION,
        "kind": "sample_manifest",
        "generated_at": time.time(),
        "snapshot": snapshot,
        "items": items,
        "summary": {
            "total_samples": total_samples,
            "by_simulator": dict(by_simulator),
            "by_scenario": dict(by_scenario),
            "by_language": dict(by_language),
            "by_style": dict(by_style),
            "by_time_mode": dict(by_time_mode),
            "timeseries_shapes": timeseries_shapes,
        },
    }


def _router_manifest_payload(snapshot: list[dict]) -> dict:
    items: list[dict] = []
    total = 0
    label_counts: Counter = Counter({"0": 0, "1": 0})

    scenario_files = _iter_router_scenario_files()
    for path in scenario_files:
        stat = path.stat()
        scenario = path.stem
        simulator = "unknown"
        router_count = 0

        for record in _iter_jsonl(path):
            router_count += 1
            metadata = record.get("metadata", {})
            simulator = metadata.get("simulator", simulator) or simulator
            label = str(record.get("label"))
            if label in ("0", "1"):
                label_counts[label] += 1

        total += router_count
        items.append(
            {
                "scenario": scenario,
                "simulator": simulator,
                "router_count": router_count,
                "file_size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "path": str(path),
            }
        )

    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        stat = train_path.stat()
        splits = {
            "train": {
                "exists": True,
                "count": total if scenario_files else _count_jsonl_rows(train_path),
                "file_size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        }
        if not scenario_files:
            label_counts = Counter({"0": 0, "1": 0})
            for record in _iter_jsonl(train_path):
                label = str(record.get("label"))
                if label in ("0", "1"):
                    label_counts[label] += 1
            total = splits["train"]["count"]
    else:
        splits = {
            "train": {
                "exists": False,
                "count": 0,
                "file_size_bytes": 0,
                "mtime": 0,
            }
        }

    return {
        "version": _VERSION,
        "kind": "router_manifest",
        "generated_at": time.time(),
        "snapshot": snapshot,
        "splits": splits,
        "total": total,
        "label_counts": {"0": int(label_counts.get("0", 0)), "1": int(label_counts.get("1", 0))},
        "scenarios": items,
    }


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                count += 1
    return count


def ensure_template_manifest(refresh: bool = False) -> dict:
    snapshot = _template_snapshot()
    if not refresh:
        manifest = _read_manifest(TEMPLATE_MANIFEST_PATH)
        if manifest and manifest.get("snapshot") == snapshot:
            return manifest
    return rebuild_template_manifest(snapshot=snapshot)


def ensure_sample_manifest(refresh: bool = False) -> dict:
    snapshot = _sample_snapshot()
    if not refresh:
        manifest = _read_manifest(SAMPLE_MANIFEST_PATH)
        if manifest and manifest.get("snapshot") == snapshot:
            return manifest
    return rebuild_sample_manifest(snapshot=snapshot)


def ensure_router_manifest(refresh: bool = False) -> dict:
    snapshot = _router_snapshot()
    if not refresh:
        manifest = _read_manifest(ROUTER_MANIFEST_PATH)
        if manifest and manifest.get("snapshot") == snapshot:
            return manifest
    return rebuild_router_manifest(snapshot=snapshot)


def rebuild_template_manifest(snapshot: list[dict] | None = None) -> dict:
    with _LOCK:
        current_snapshot = snapshot if snapshot is not None else _template_snapshot()
        payload = _template_manifest_payload(current_snapshot)
        return _write_manifest(TEMPLATE_MANIFEST_PATH, payload)


def rebuild_sample_manifest(snapshot: list[dict] | None = None) -> dict:
    with _LOCK:
        current_snapshot = snapshot if snapshot is not None else _sample_snapshot()
        payload = _sample_manifest_payload(current_snapshot)
        return _write_manifest(SAMPLE_MANIFEST_PATH, payload)


def rebuild_router_manifest(snapshot: list[dict] | None = None) -> dict:
    with _LOCK:
        current_snapshot = snapshot if snapshot is not None else _router_snapshot()
        payload = _router_manifest_payload(current_snapshot)
        return _write_manifest(ROUTER_MANIFEST_PATH, payload)
