from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.check_garbled_text import find_garbled_text  # noqa: E402


def test_find_garbled_text_flags_question_runs(tmp_path: Path):
    path = tmp_path / "garbled.tsx"
    path.write_text('const title = "????";\n', encoding="utf-8")

    findings = find_garbled_text(path)

    assert findings
    assert findings[0].endswith("suspicious-question-run")


def test_find_garbled_text_flags_replacement_char(tmp_path: Path):
    path = tmp_path / "garbled.md"
    path.write_text("标题：\ufffd\n", encoding="utf-8")

    findings = find_garbled_text(path)

    assert findings
    assert findings[0].endswith("replacement-char")


def test_find_garbled_text_flags_known_mojibake_markers(tmp_path: Path):
    path = tmp_path / "garbled.tsx"
    path.write_text('const title = "鏂板缓璁粌";\n', encoding="utf-8")

    findings = find_garbled_text(path)

    assert findings
    assert any("suspicious-private-use-char" in finding or "suspicious-mojibake" in finding for finding in findings)


def test_find_garbled_text_ignores_normal_nullish_operator(tmp_path: Path):
    path = tmp_path / "safe.tsx"
    path.write_text("const value = input ?? fallback\n", encoding="utf-8")

    findings = find_garbled_text(path)

    assert findings == []
