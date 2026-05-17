"""Training progress and curve parsing helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from piern.training.services.checkpoint_store import checkpoint_entries
from piern.training.services.training_runtime import (
    coerce_float,
    coerce_int,
    coerce_training_curve_point,
    normalize_per_scenario_metrics,
    read_json,
    tail_lines,
)


def latest_training_point(run_dir: Path) -> dict[str, Any] | None:
    log_path = run_dir / "train_log.jsonl"
    for line in reversed(tail_lines(log_path, 20)):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        point = coerce_training_curve_point(payload, include_steps_per_epoch=True)
        if point is not None:
            return point
    return None


def latest_test_metrics(run_dir: Path) -> dict[str, Any] | None:
    return read_json(run_dir / "test_metrics_latest.json")


def downsample(points: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    sampled = [points[idx] for idx in range(0, len(points), stride)]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def build_curves(*, job_id: str, run_dir: Path, max_points: int = 2000) -> dict[str, Any]:
    training_points: list[dict[str, Any]] = []
    train_log_path = run_dir / "train_log.jsonl"
    if train_log_path.exists():
        try:
            with train_log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    point = coerce_training_curve_point(payload)
                    if point is not None:
                        training_points.append(point)
        except OSError:
            pass
    epoch_points_map: dict[int, dict[str, Any]] = {}
    for point in training_points:
        epoch_points_map[int(point["epoch"])] = point
    training_epoch_points = [epoch_points_map[epoch] for epoch in sorted(epoch_points_map)]

    test_points: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("test_metrics_epoch_*.json")):
        payload = read_json(path)
        if not payload:
            continue
        epoch = coerce_int(payload.get("epoch"))
        if epoch is None:
            continue
        overall = payload.get("overall", {}) if isinstance(payload.get("overall"), dict) else {}
        test_points.append(
            {
                "epoch": epoch,
                "accuracy": coerce_float(overall.get("accuracy"), 0.0) or 0.0,
                "precision": coerce_float(overall.get("precision"), 0.0) or 0.0,
                "recall": coerce_float(overall.get("recall"), 0.0) or 0.0,
                "f1": coerce_float(overall.get("f1"), 0.0) or 0.0,
                "pr_auc": coerce_float(overall.get("pr_auc"), 0.0) or 0.0,
                "per_scenario": normalize_per_scenario_metrics(payload.get("per_scenario")),
            }
        )

    return {
        "job_id": job_id,
        "training_points": downsample(training_points, max_points=max_points),
        "training_epoch_points": downsample(training_epoch_points, max_points=max_points),
        "test_points": downsample(test_points, max_points=max_points),
        "checkpoints": checkpoint_entries(run_dir),
    }
