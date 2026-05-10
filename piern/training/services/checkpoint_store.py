from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from piern.training.router.data import PRETRAINED_EMBEDDINGS


@lru_cache(maxsize=256)
def load_checkpoint_metadata(path_str: str, mtime_ns: int, size_bytes: int) -> dict[str, Any] | None:
    del mtime_ns, size_bytes
    try:
        import torch

        checkpoint = torch.load(path_str, map_location="cpu", weights_only=True)
    except Exception:
        return None
    prepared_summary = checkpoint.get("prepared_summary") or {}
    config = checkpoint.get("config") or {}
    epoch = checkpoint.get("epoch")
    try:
        parsed_epoch = int(epoch) if epoch is not None else None
    except (TypeError, ValueError):
        parsed_epoch = None

    return {
        "epoch": parsed_epoch,
        "simulator": prepared_summary.get("simulator") or config.get("simulator"),
        "scenarios": sorted(prepared_summary.get("scenarios") or config.get("scenarios") or []),
        "test_ratio": prepared_summary.get("test_ratio")
        if prepared_summary.get("test_ratio") is not None
        else config.get("test_ratio"),
        "input_representation": prepared_summary.get("input_representation") or config.get("input_representation"),
        "embedding_model": prepared_summary.get("embedding_model") or config.get("embedding_model"),
        "embedding_tokenizer": prepared_summary.get("embedding_tokenizer") or config.get("embedding_tokenizer"),
    }


def clear_checkpoint_metadata_cache() -> None:
    load_checkpoint_metadata.cache_clear()


def _checkpoint_epoch_from_file(path_str: str, mtime_ns: int, size_bytes: int) -> int | None:
    metadata = load_checkpoint_metadata(path_str, mtime_ns, size_bytes)
    if metadata is None:
        return None
    return metadata.get("epoch")


def checkpoint_entries(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not run_dir.exists():
        return entries
    for path in sorted(run_dir.glob("router*.pt")):
        stat = path.stat()
        epoch = None
        stem = path.stem
        if stem.startswith("router_epoch_"):
            try:
                epoch = int(stem.split("_")[-1])
            except ValueError:
                epoch = None
        else:
            epoch = _checkpoint_epoch_from_file(str(path), stat.st_mtime_ns, stat.st_size)
        entries.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "epoch": epoch,
            }
        )
    return sorted(entries, key=lambda item: item["mtime"], reverse=True)


def validate_resume_checkpoint(
    *,
    resume_from: str,
    simulator: str,
    scenarios: list[str],
    test_ratio: float,
    input_representation: str,
    embedding_model: str = "",
    embedding_tokenizer: str = "",
) -> None:
    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise ValueError(f"resume checkpoint not found: {resume_from}")

    stat = checkpoint_path.stat()
    metadata = load_checkpoint_metadata(str(checkpoint_path), stat.st_mtime_ns, stat.st_size)
    if metadata is None:
        raise ValueError(f"resume checkpoint is unreadable: {resume_from}")

    checkpoint_simulator = metadata.get("simulator")
    checkpoint_scenarios = sorted(metadata.get("scenarios") or [])
    checkpoint_ratio = metadata.get("test_ratio")
    checkpoint_input_representation = metadata.get("input_representation")
    checkpoint_embedding_model = str(metadata.get("embedding_model") or "")
    checkpoint_embedding_tokenizer = str(metadata.get("embedding_tokenizer") or "")

    if checkpoint_simulator and checkpoint_simulator != simulator:
        raise ValueError(f"resume checkpoint simulator mismatch: expected {simulator}, got {checkpoint_simulator}")
    if checkpoint_scenarios and checkpoint_scenarios != sorted(scenarios):
        raise ValueError(
            f"resume checkpoint scenario mismatch: expected {sorted(scenarios)}, got {checkpoint_scenarios}"
        )
    if checkpoint_ratio is not None and not math.isclose(
        float(checkpoint_ratio), float(test_ratio), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(f"resume checkpoint test_ratio mismatch: expected {test_ratio}, got {checkpoint_ratio}")
    if checkpoint_input_representation and checkpoint_input_representation != input_representation:
        raise ValueError(
            "resume checkpoint input_representation mismatch: "
            f"expected {input_representation}, got {checkpoint_input_representation}"
        )
    if input_representation == PRETRAINED_EMBEDDINGS:
        if checkpoint_embedding_model and checkpoint_embedding_model != embedding_model:
            raise ValueError(
                "resume checkpoint embedding_model mismatch: "
                f"expected {embedding_model}, got {checkpoint_embedding_model}"
            )
        if checkpoint_embedding_tokenizer and checkpoint_embedding_tokenizer != embedding_tokenizer:
            raise ValueError(
                "resume checkpoint embedding_tokenizer mismatch: "
                f"expected {embedding_tokenizer}, got {checkpoint_embedding_tokenizer}"
            )
