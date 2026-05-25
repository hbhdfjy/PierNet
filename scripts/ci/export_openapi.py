"""Export the FastAPI OpenAPI schema without starting a server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piern.api.main import app


def export_schema(output: Path) -> dict[str, Any]:
    schema = app.openapi()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="frontend/src/lib/generated/openapi.json",
        help="Path to write the OpenAPI JSON schema.",
    )
    args = parser.parse_args()
    output = Path(args.output)
    schema = export_schema(output)
    print(
        "exported OpenAPI schema "
        f"title={schema.get('info', {}).get('title')} "
        f"version={schema.get('info', {}).get('version')} "
        f"paths={len(schema.get('paths', {}))} "
        f"to={output}"
    )


if __name__ == "__main__":
    main()
