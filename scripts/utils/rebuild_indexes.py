"""Rebuild sparse JSONL pagination indexes for Stage 2-4 artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from piern.api.deps import DATA_DIR, TEMPLATES_DIR
from piern.api.services import jsonl_index, manifest_store

ROUTER_DIR = PROJECT_ROOT / "data" / "router"
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


def main() -> None:
    manifest_store.ensure_template_manifest()
    manifest_store.ensure_sample_manifest()
    manifest_store.ensure_router_manifest()

    summary = []
    for path in _iter_paths():
        payload = jsonl_index.rebuild_index(path)
        summary.append(
            {
                "source": str(path.relative_to(PROJECT_ROOT)),
                "index": str(jsonl_index.get_index_path(path).relative_to(PROJECT_ROOT)),
                "rows": payload.get("total_rows", 0),
                "stride": payload.get("stride", jsonl_index.DEFAULT_STRIDE),
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
