#!/usr/bin/env python3
"""Clean expired PierNet training caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PierNet.shared.runtime.config import load_runtime_config
from PierNet.training.services.cache_cleanup import (
    ROUTER_JSONL_CACHE_KIND,
    TRAINING_PREPARED_CACHE_KIND,
    cleanup_training_cache,
    result_as_json,
)


def _gb(value: float) -> int:
    return int(max(0.0, value) * 1024**3)


def _fmt_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def main(argv: list[str] | None = None) -> int:
    config = load_runtime_config()
    parser = argparse.ArgumentParser(description="Clean expired Router JSONL and Token Router prepared caches.")
    parser.add_argument("--execute", action="store_true", help="Delete expired caches. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Report candidates without deleting.")
    parser.add_argument(
        "--kind",
        choices=("all", ROUTER_JSONL_CACHE_KIND, TRAINING_PREPARED_CACHE_KIND),
        default="all",
        help="Cache kind to clean.",
    )
    parser.add_argument("--router-jsonl-ttl-days", type=float, default=config.router_jsonl_cache_ttl_days)
    parser.add_argument("--training-prepared-ttl-days", type=float, default=config.training_prepared_cache_ttl_days)
    parser.add_argument("--max-delete-gb", type=float, default=config.cache_cleanup_max_delete_gb)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    dry_run = not args.execute or args.dry_run
    kinds = None if args.kind == "all" else [args.kind]
    result = cleanup_training_cache(
        router_jsonl_cache_dir=config.router_jsonl_cache_dir,
        training_artifact_root=config.artifact_root / "token_router",
        router_jsonl_ttl_days=args.router_jsonl_ttl_days,
        training_prepared_ttl_days=args.training_prepared_ttl_days,
        max_delete_bytes=_gb(args.max_delete_gb),
        dry_run=dry_run,
        kinds=kinds,
    )

    if args.json:
        print(result_as_json(result))
        return 1 if result.errors else 0

    mode = "DRY-RUN" if result.dry_run else "EXECUTE"
    print(f"Mode: {mode}")
    print(f"Reclaimable: {_fmt_bytes(result.reclaimable_bytes)} across {len(result.candidates)} candidate(s)")
    print(f"Deleted: {_fmt_bytes(result.deleted_bytes)} across {len(result.deleted)} item(s)")
    print(f"Skipped: {len(result.skipped)} item(s); max delete budget {_fmt_bytes(result.max_delete_bytes)}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"- {error['kind']} {error['path']}: {error['error']}")
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
