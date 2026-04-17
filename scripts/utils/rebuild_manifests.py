"""Rebuild sidecar manifests for templates, samples, and router data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from piern.api.services import manifest_store


def main() -> None:
    template_manifest = manifest_store.rebuild_template_manifest()
    sample_manifest = manifest_store.rebuild_sample_manifest()
    router_manifest = manifest_store.rebuild_router_manifest()

    summary = {
        "templates": {
            "path": str(manifest_store.TEMPLATE_MANIFEST_PATH),
            "count": len(template_manifest.get("items", [])),
            "total_templates": template_manifest.get("summary", {}).get("total_templates", 0),
        },
        "samples": {
            "path": str(manifest_store.SAMPLE_MANIFEST_PATH),
            "count": len(sample_manifest.get("items", [])),
            "total_samples": sample_manifest.get("summary", {}).get("total_samples", 0),
        },
        "router": {
            "path": str(manifest_store.ROUTER_MANIFEST_PATH),
            "scenario_count": len(router_manifest.get("scenarios", [])),
            "total": router_manifest.get("total", 0),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
