"""Rebuild sparse and filtered indexes for Stage 2 and legacy Stage 3/4 JSONL artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from piern.shared.runtime.paths import DATA_DIR, DATA_ROOT, TEMPLATES_DIR  # noqa: E402
from piern.synth.services import jsonl_filter_index, jsonl_index, manifest_store  # noqa: E402

ROUTER_DIR = DATA_ROOT / "router"
ROUTER_SCENARIO_DIR = ROUTER_DIR / "by_scenario"


def _iter_paths() -> list[Path]:
    paths: list[Path] = []
    if TEMPLATES_DIR.exists():
        paths.extend(sorted(TEMPLATES_DIR.glob("*_templates.jsonl")))
    if DATA_DIR.exists():
        paths.extend(
            sorted(path for path in DATA_DIR.glob("*.jsonl") if path.name != "all_training_data.jsonl")
        )
    if ROUTER_SCENARIO_DIR.exists():
        paths.extend(sorted(ROUTER_SCENARIO_DIR.glob("*.jsonl")))
    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        paths.append(train_path)
    return paths


def _filter_profile(path: Path) -> str | None:
    if path.parent == TEMPLATES_DIR:
        return "template_language_style"
    if path.parent == DATA_DIR and path.name != "all_training_data.jsonl":
        return "sample_language_style"
    if path == ROUTER_DIR / "train.jsonl" or path.parent == ROUTER_SCENARIO_DIR:
        return "router_label"
    return None


def _filter_summary(path: Path, profile: str, payload: dict) -> dict:
    counts = {
        key: value.get("count", 0)
        for key, value in sorted(payload.get("entries", {}).items())
    }
    return {
        "profile": profile,
        "index": str(jsonl_filter_index.get_filter_index_path(path, profile).relative_to(PROJECT_ROOT)),
        "keys": counts,
    }


def main() -> None:
    manifest_store.ensure_template_manifest()
    manifest_store.ensure_sample_manifest()
    manifest_store.ensure_router_manifest()

    summary = []
    for path in _iter_paths():
        payload = jsonl_index.rebuild_index(path)
        item = {
            "source": str(path.relative_to(PROJECT_ROOT)),
            "index": str(jsonl_index.get_index_path(path).relative_to(PROJECT_ROOT)),
            "rows": payload.get("total_rows", 0),
            "stride": payload.get("stride", jsonl_index.DEFAULT_STRIDE),
            "filters": [],
        }
        profile = _filter_profile(path)
        if profile:
            filter_payload = jsonl_filter_index.rebuild_filter_index(path, profile)
            item["filters"].append(_filter_summary(path, profile, filter_payload))
        summary.append(item)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
