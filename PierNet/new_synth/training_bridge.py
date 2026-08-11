from __future__ import annotations

from pathlib import Path
from typing import Any

from PierNet.new_synth import store


def _available(item: dict[str, Any]) -> bool:
    path = Path(str(item.get("path") or ""))
    root = Path(str(item.get("root_path") or ""))
    return path.is_file() and root.is_dir()


def list_router_datasets() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for item in store.list_datasets(kind="router"):
        if not _available(item):
            continue
        scenario = {
            "dataset_id": item["dataset_id"],
            "scenario": item["scenario"],
            "simulator": item["simulator"],
            "router_count": int(item["sample_count"]),
            "file_size_bytes": int(item["size_bytes"]),
            "mtime": float(item["created_at"]),
            "path": item["path"],
        }
        datasets.append(
            {
                "dataset_id": item["dataset_id"],
                "display_name": item["name"],
                "source": "new_synth",
                "workflow_id": item["workflow_id"],
                "simulator": item["simulator"],
                "total_count": int(item["sample_count"]),
                "text2comp_dataset_id": item.get("paired_dataset_id"),
                "scenarios": [scenario],
            }
        )
    return datasets


def list_text2comp_datasets() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for item in store.list_datasets(kind="text2comp"):
        if not _available(item):
            continue
        datasets.append(
            {
                "dataset_id": item["dataset_id"],
                "display_name": item["name"],
                "source": "new_synth",
                "workflow_id": item["workflow_id"],
                "path": item["path"],
                "name": item["name"],
                "simulator": item["simulator"],
                "scenario": item["scenario"],
                "n_samples": int(item["sample_count"]),
                "sample_count": int(item["sample_count"]),
                "size_bytes": int(item["size_bytes"]),
                "file_size_bytes": int(item["size_bytes"]),
                "mtime": float(item["created_at"]),
                "output_dim": int((item.get("metadata") or {}).get("input_dim") or 0),
                "label_semantics": (item.get("metadata") or {}).get("label_semantics"),
                "router_dataset_id": item.get("paired_dataset_id"),
            }
        )
    return datasets


def resolve_router_dataset(dataset_id: str) -> dict[str, Any]:
    item = store.get_dataset(dataset_id)
    if item.get("kind") != "router" or not _available(item):
        raise ValueError(f"Router dataset is unavailable: {dataset_id}")
    return item


def resolve_text2comp_dataset(dataset_id: str) -> dict[str, Any]:
    item = store.get_dataset(dataset_id)
    if item.get("kind") != "text2comp" or not _available(item):
        raise ValueError(f"Text2Comp dataset is unavailable: {dataset_id}")
    return item


def resolve_paired_text2comp(router_dataset: dict[str, Any]) -> dict[str, Any] | None:
    paired_id = str(router_dataset.get("paired_dataset_id") or "")
    if not paired_id:
        return None
    return resolve_text2comp_dataset(paired_id)
