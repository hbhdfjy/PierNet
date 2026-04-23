from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".sh",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    ".runlogs",
    ".venv",
    "artifacts",
    "build",
    "data",
    "node_modules",
}
IGNORE_FILES = {
    Path("scripts/utils/check_garbled_text.py"),
    Path("tests/test_check_garbled_text.py"),
}
QUESTION_RUN = re.compile(r"(?<!\?)\?{3,}(?!\?)")
REPLACEMENT_CHAR = "\ufffd"
PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")
MOJIBAKE_MARKERS = (
    "\u93c2\u677f\u7f13",
    "\u923a",
    "\u923d",
    "\u934a",
    "\u934f",
)


def should_scan(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_SUFFIXES


def iter_files(repo_root: Path, base_paths: Iterable[Path]) -> Iterable[Path]:
    for base in base_paths:
        if base.is_file():
            if should_scan(base) and base.relative_to(repo_root) not in IGNORE_FILES:
                yield base
            continue
        for path in base.rglob("*"):
            relative_path = path.relative_to(repo_root)
            if relative_path in IGNORE_FILES:
                continue
            if any(part in SKIP_DIRS for part in relative_path.parts):
                continue
            if should_scan(path):
                yield path


def find_garbled_text(path: Path) -> list[str]:
    findings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        findings.append(f"{path}: decode-error: {exc}")
        return findings

    for line_number, line in enumerate(text.splitlines(), start=1):
        if REPLACEMENT_CHAR in line:
            findings.append(f"{path}:{line_number}: replacement-char")
        if QUESTION_RUN.search(line):
            findings.append(f"{path}:{line_number}: suspicious-question-run")
        if PRIVATE_USE_RE.search(line):
            findings.append(f"{path}:{line_number}: suspicious-private-use-char")
        for marker in MOJIBAKE_MARKERS:
            if marker in line:
                findings.append(f"{path}:{line_number}: suspicious-mojibake:{marker}")
                break
    return findings


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan the repository for likely text encoding corruption.")
    parser.add_argument("paths", nargs="*", help="Optional repository-relative paths to scan.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or [])
    repo_root = Path(__file__).resolve().parents[2]
    base_paths = [repo_root / path for path in args.paths] if args.paths else [
        repo_root / "frontend",
        repo_root / "piern",
        repo_root / "scripts",
        repo_root / "tests",
    ]

    findings: list[str] = []
    for path in iter_files(repo_root, base_paths):
        findings.extend(find_garbled_text(path))

    if findings:
        print("\n".join(findings))
        return 1

    print("No garbled text markers found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
