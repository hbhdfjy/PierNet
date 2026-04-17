"""Rebuild sparse filtered indexes for common JSONL query filters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from piern.api.deps import DATA_DIR, TEMPLATES_DIR
from piern.api.services import jsonl_filter_index

ROUTER_DIR = PROJECT_ROOT / "data" / "router"
ROUTER_SCENARIO_DIR = ROUTER_DIR / "by_scenario"


def main() -> None:
    summary = []

    if TEMPLATES_DIR.exists():
        for path in sorted(TEMPLATES_DIR.glob("*_templates.jsonl")):
            payload = jsonl_filter_index.rebuild_filter_index(path, "template_language_style")
            summary.append(_summary(path, "template_language_style", payload))

    if DATA_DIR.exists():
        for path in sorted(path for path in DATA_DIR.glob("*.jsonl") if path.name != "all_training_data.jsonl"):
            payload = jsonl_filter_index.rebuild_filter_index(path, "sample_language_style")
            summary.append(_summary(path, "sample_language_style", payload))

    if ROUTER_SCENARIO_DIR.exists():
        for path in sorted(ROUTER_SCENARIO_DIR.glob("*.jsonl")):
            payload = jsonl_filter_index.rebuild_filter_index(path, "router_label")
            summary.append(_summary(path, "router_label", payload))

    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        payload = jsonl_filter_index.rebuild_filter_index(train_path, "router_label")
        summary.append(_summary(train_path, "router_label", payload))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _summary(path: Path, profile: str, payload: dict) -> dict:
    counts = {
        key: value.get("count", 0)
        for key, value in payload.get("entries", {}).items()
    }
    return {
        "source": str(path.relative_to(PROJECT_ROOT)),
        "profile": profile,
        "index": str(jsonl_filter_index.get_filter_index_path(path, profile).relative_to(PROJECT_ROOT)),
        "keys": counts,
    }


if __name__ == "__main__":
    main()
