from __future__ import annotations

from scripts.ci import check_repo_hygiene


def test_egg_info_paths_are_banned() -> None:
    reason = check_repo_hygiene.is_banned_path("PierNet_data_synthesis.egg-info/PKG-INFO")

    assert reason is not None
    assert "egg-info" in reason


def test_nested_egg_info_paths_are_banned() -> None:
    reason = check_repo_hygiene.is_banned_path("build/output/pkg.egg-info/SOURCES.txt")

    assert reason is not None
    assert "egg-info" in reason


def test_local_runtime_roots_are_banned() -> None:
    banned_paths = [
        ".cache/tool/state.json",
        ".conda/env/bin/python",
        ".flopy_bin/modflow",
        ".node/bin/node",
        ".tmp/work/file.json",
        "models/checkpoint.pt",
        "frontend/node_modules/.bin/vite",
    ]

    for path in banned_paths:
        reason = check_repo_hygiene.is_banned_path(path)
        assert reason is not None, path


def test_nested_local_runtime_dirs_are_banned() -> None:
    banned_paths = [
        "frontend/.cache/vite/index.json",
        "tools/.conda/env/bin/python",
        "sandbox/.tmp/output.json",
    ]

    for path in banned_paths:
        reason = check_repo_hygiene.is_banned_path(path)
        assert reason is not None, path
