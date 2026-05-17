#!/usr/bin/env python3
"""Prevent runtime artifacts, derived data, private paths, and secrets from entering Git."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]

BANNED_PREFIXES = (
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".runlogs/",
    "artifacts/",
    "data/text2comp/",
    "data/text2comp_parquet/",
    "data/router/",
    "data/router_parquet/",
    "data/.manifests/",
    "data/.indexes/",
    "frontend/dist/",
    "frontend/test-results/",
    "frontend/playwright-report/",
    "node_modules/",
)
BANNED_PARTS = {"__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
BANNED_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite-shm",
    ".sqlite-wal",
    ".parquet",
    ".duckdb",
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".log",
    ".env",
)
SECRET_PATTERNS = [
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
]
GENERIC_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|token)\b\s*[:=]\s*['\"]([A-Za-z0-9][A-Za-z0-9_:+/-]{23,})['\"]"
)
PRIVATE_PATH_PATTERNS = [
    re.compile(r"/home/(?:tpx|fjy)(?:/|\b)"),
    re.compile(r"/Users/fanjingyuan(?:/|\b)"),
    re.compile(r"/data/fjy(?:/|\b)"),
]
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".sh", ".md", ".yml", ".yaml", ".json", ".toml", ".txt", ".css", ".html", ".example", ""}


def git_ls_files() -> List[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [item.decode() for item in out.split(b"\0") if item]


def is_banned_path(path: str) -> Optional[str]:
    for prefix in BANNED_PREFIXES:
        if path.startswith(prefix):
            return f"banned runtime/derived prefix: {prefix}"
    parts = set(Path(path).parts)
    overlap = parts & BANNED_PARTS
    if overlap:
        return f"banned cache path part: {sorted(overlap)[0]}"
    if path.endswith(BANNED_SUFFIXES):
        return "banned runtime/derived file suffix"
    if path == ".env" or path.endswith("/.env"):
        return "real .env files must not be tracked"
    return None


def is_probably_text(path: Path) -> bool:
    if path.suffix in TEXT_SUFFIXES:
        return True
    return path.name in {"Dockerfile", "Makefile"}


def is_placeholder_secret(value: str) -> bool:
    compact = value.strip().strip("'\"").strip()
    lower = compact.lower()
    if not compact:
        return True
    if any(marker in lower for marker in ("placeholder", "replace-me", "your-key", "your_api_key", "example")):
        return True
    if re.fullmatch(r"sk-(?:[a-z]+-)?[x*]+", lower):
        return True
    if compact.startswith("sk-"):
        body = compact[3:].replace("-", "").replace("_", "")
        return bool(body) and set(body) <= {"x", "X", "*"}
    return set(compact) <= {"x", "X", "*"}


def has_secret_literal(text: str) -> bool:
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if not is_placeholder_secret(match.group(0)):
                return True
    for match in GENERIC_SECRET_ASSIGNMENT.finditer(text):
        if not is_placeholder_secret(match.group(1)):
            return True
    return False


def scan_text(path: str) -> List[str]:
    full = ROOT / path
    if not full.exists() or not is_probably_text(full):
        return []
    try:
        if full.stat().st_size > 1_000_000:
            return []
        text = full.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    issues: List[str] = []
    if has_secret_literal(text):
        issues.append("possible secret/token literal")
    for pattern in PRIVATE_PATH_PATTERNS:
        if pattern.search(text):
            issues.append("server-private absolute path")
            break
    return issues


def check_gitignore() -> List[str]:
    required = [
        "data/text2comp/",
        "data/router/",
        "data/**/*.parquet",
        "artifacts/",
        ".runlogs/",
        ".env",
        "frontend/test-results/",
    ]
    text = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
    return [item for item in required if item not in text]


def main() -> None:
    errors: List[str] = []
    tracked = git_ls_files()
    for path in tracked:
        reason = is_banned_path(path)
        if reason:
            errors.append(f"{path}: {reason}")
        for issue in scan_text(path):
            errors.append(f"{path}: {issue}")

    missing_ignores = check_gitignore()
    for item in missing_ignores:
        errors.append(f".gitignore missing required rule: {item}")

    if errors:
        for item in errors:
            print(f"ERROR: {item}")
        print(f"FAILED: {len(errors)} repository hygiene issue(s)")
        sys.exit(1)
    print(f"PASSED: repository hygiene clean ({len(tracked)} tracked files checked)")


if __name__ == "__main__":
    main()
