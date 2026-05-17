"""Filesystem cleanup helpers for training jobs."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from threading import Thread
from typing import Any

from piern.training.services.checkpoint_store import clear_checkpoint_metadata_cache

LOGGER = logging.getLogger(__name__)


def stage_job_artifacts_for_delete(entry: dict[str, Any]) -> list[tuple[Path, Path]]:
    staged: list[tuple[Path, Path]] = []
    run_dir_value = entry.get("run_dir")
    if not run_dir_value:
        return staged

    run_dir = Path(run_dir_value)
    if not run_dir.exists():
        return staged

    timestamp_ms = int(time.time() * 1000)
    target = run_dir.with_name(f".deleting-{run_dir.name}-{timestamp_ms}")
    suffix = 1
    while target.exists():
        target = run_dir.with_name(f".deleting-{run_dir.name}-{timestamp_ms}-{suffix}")
        suffix += 1

    run_dir.rename(target)
    staged.append((run_dir, target))
    clear_checkpoint_metadata_cache()
    return staged


def restore_staged_job_artifacts(staged: list[tuple[Path, Path]]) -> None:
    for original, target in reversed(staged):
        if target.exists() and not original.exists():
            try:
                target.rename(original)
            except OSError:
                LOGGER.exception("Failed to restore staged training artifact %s to %s", target, original)


def delete_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        LOGGER.exception("Failed to delete training artifact directory %s", path)
    finally:
        clear_checkpoint_metadata_cache()


def delete_job_artifacts_in_background(paths: list[Path]) -> None:
    for path in paths:
        thread = Thread(target=delete_tree, args=(path,), name=f"delete-{path.name}", daemon=True)
        thread.start()


def remove_job_log(entry: dict[str, Any]) -> None:
    log_path_value = entry.get("log_path")
    if not log_path_value:
        return

    log_path = Path(log_path_value)
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        LOGGER.exception("Failed to delete training log %s", log_path)


def remove_job_stop_file(entry: dict[str, Any]) -> None:
    stop_file_value = entry.get("stop_file")
    if not stop_file_value:
        return
    try:
        Path(stop_file_value).unlink(missing_ok=True)
    except OSError:
        LOGGER.exception("Failed to delete training stop file %s", stop_file_value)
