#!/usr/bin/env python3
"""Lightweight migration readiness checks for PiERN."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from piern.shared.storage.hdf5_files import is_hdf5_path, iter_hdf5_files_in_child_dirs  # noqa: E402

REQUIRED_ENV_EXAMPLE_KEYS = [
    "PIERN_ROOT",
    "PIERN_DATA_ROOT",
    "PIERN_ARTIFACT_ROOT",
    "PIERN_RUNLOG_ROOT",
    "PIERN_BACKEND_PORT",
    "PIERN_FRONTEND_PORT",
    "PIERN_CONDA_BASE",
    "PIERN_CONDA_ENV",
    "PIERN_PYTHON",
    "PIERN_NODE_BIN",
    "PIERN_QWEN_EMBEDDING_MODEL",
    "PIERN_QWEN_EMBEDDING_TOKENIZER",
    "PIERN_MODFLOW_EXE",
    "PIERN_JOB_STORE_PATH",
    "PIERN_TRAINING_JOB_STORE_PATH",
    "PIERN_ROUTER_JSONL_CACHE_DIR",
    "PIERN_WORKER_QUEUE_SYNTH",
    "PIERN_WORKER_QUEUE_TRAINING",
    "PIERN_LOCK_TTL_SECONDS",
    "PIERN_SYNTH_LOCK_TTL_SECONDS",
    "PIERN_GPU_LOCK_TTL_SECONDS",
    "PIERN_TRAINING_QUEUE_DEFAULT_PRIORITY",
    "PIERN_SERVICE_WORKER",
    "PIERN_SYSTEMD_USER_DIR",
    "PIERN_INSTALL_WORKER",
    "PIERN_ROUTER_BUILD_WORKERS",
]
DERIVED_PREFIXES = (
    "data/text2comp/",
    "data/text2comp_parquet/",
    "data/router/",
    "data/router_parquet/",
    "data/.manifests/",
    "data/.indexes/",
    "artifacts/",
    ".runlogs/",
)


def load_local_env() -> None:
    """Load local .env defaults without overriding explicit environment values."""
    env_file = Path(os.getenv("PIERN_ENV_FILE", ROOT / ".env")).expanduser()
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


class Report:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def ok(self, msg: str) -> None:
        self.info.append(f"OK: {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(f"WARN: {msg}")

    def error(self, msg: str) -> None:
        self.errors.append(f"ERROR: {msg}")

    def finish(self) -> None:
        for line in self.info:
            print(line)
        for line in self.warnings:
            print(line)
        for line in self.errors:
            print(line)
        if self.errors:
            print(f"FAILED: {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
            sys.exit(1)
        print(f"PASSED: migration ready with {len(self.warnings)} warning(s)")


def git_ls_files() -> List[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [item.decode() for item in out.split(b"\0") if item]


def load_env_example_keys() -> Set[str]:
    path = ROOT / ".env.example"
    keys: Set[str] = set()
    if not path.exists():
        return keys
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.split("=", 1)[0].strip())
    return keys


def check_tracked_data(report: Report, tracked: List[str]) -> None:
    bad: List[str] = []
    h5_count = 0
    template_count = 0
    for path in tracked:
        if not path.startswith("data/"):
            continue
        if path == "data/.gitignore":
            continue
        if path.startswith("data/templates/") and path.endswith("_templates.jsonl"):
            template_count += 1
            continue
        if is_hdf5_path(path):
            h5_count += 1
            continue
        bad.append(path)
    if bad:
        for path in bad:
            report.error(f"tracked data outside allowed boundary: {path}")
    else:
        report.ok(f"tracked data boundary clean: {h5_count} HDF5, {template_count} template JSONL")


def check_required_files(report: Report) -> None:
    required = [
        "configs/text2comp/default.yaml",
        "configs/text2comp/registry.yaml",
        "docs/MIGRATION.md",
        "scripts/services/start.sh",
        "scripts/services/status.sh",
    ]
    for rel in required:
        if (ROOT / rel).exists():
            report.ok(rel)
        else:
            report.error(f"missing required migration file: {rel}")


def check_env_example(report: Report) -> None:
    keys = load_env_example_keys()
    if not keys:
        report.error("missing or empty .env.example")
        return
    missing = [key for key in REQUIRED_ENV_EXAMPLE_KEYS if key not in keys]
    if missing:
        report.error(".env.example missing keys: " + ", ".join(missing))
    else:
        report.ok(f".env.example covers {len(REQUIRED_ENV_EXAMPLE_KEYS)} required keys")


def check_local_source_data(report: Report) -> None:
    data_root = Path(os.getenv("PIERN_DATA_ROOT", ROOT / "data")).expanduser()
    templates = sorted((data_root / "templates").glob("*_templates.jsonl"))
    hdf5 = iter_hdf5_files_in_child_dirs(data_root)
    if templates:
        report.ok(f"template files present: {len(templates)}")
    else:
        report.error(f"no template JSONL files under {data_root / 'templates'}")
    if hdf5:
        report.ok(f"raw HDF5 files present: {len(hdf5)}")
    else:
        report.error(f"no raw HDF5 files under {data_root}")


def check_derived_not_tracked(report: Report, tracked: List[str]) -> None:
    bad = [path for path in tracked if path.startswith(DERIVED_PREFIXES)]
    if bad:
        for path in bad:
            report.error(f"derived/runtime path is tracked: {path}")
    else:
        report.ok("derived/runtime directories are not tracked")


def check_model_paths(report: Report) -> None:
    default_model = Path.home() / "Qwen" / "Qwen2.5-0.5B-Instruct"
    model = Path(os.getenv("PIERN_QWEN_EMBEDDING_MODEL", str(default_model))).expanduser()
    tokenizer = Path(os.getenv("PIERN_QWEN_EMBEDDING_TOKENIZER", str(model))).expanduser()
    missing = [str(path) for path in [model, tokenizer] if not path.exists()]
    if missing:
        report.warn(
            "embedding model/tokenizer path not found: "
            + ", ".join(missing)
            + "; download Qwen/Qwen2.5-0.5B-Instruct or set PIERN_QWEN_EMBEDDING_MODEL/PIERN_QWEN_EMBEDDING_TOKENIZER in .env"
        )
    else:
        report.ok("embedding model/tokenizer paths exist")


def main() -> None:
    load_local_env()
    report = Report()
    tracked = git_ls_files()
    check_tracked_data(report, tracked)
    check_required_files(report)
    check_env_example(report)
    check_local_source_data(report)
    check_derived_not_tracked(report, tracked)
    check_model_paths(report)
    report.finish()


if __name__ == "__main__":
    main()
