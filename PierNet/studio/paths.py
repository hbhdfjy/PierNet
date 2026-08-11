from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PierNet.shared.runtime.paths import ARTIFACT_ROOT, DATA_ROOT, RUNLOG_ROOT

STUDIO_DATA_ROOT = Path(os.getenv("PierNet_STUDIO_DATA_ROOT", DATA_ROOT / "studio")).resolve()
STUDIO_ARTIFACT_ROOT = Path(
    os.getenv("PierNet_STUDIO_ARTIFACT_ROOT", ARTIFACT_ROOT / "studio")
).resolve()
STUDIO_RUNLOG_ROOT = Path(
    os.getenv("PierNet_STUDIO_RUNLOG_ROOT", RUNLOG_ROOT / "studio")
).resolve()
STUDIO_DB_PATH = Path(
    os.getenv("PierNet_STUDIO_DB_PATH", STUDIO_RUNLOG_ROOT / "projects.sqlite")
).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    project_id: str
    data_root: Path
    source: Path
    canonical: Path
    training_data: Path
    artifact_root: Path
    expert: Path
    router: Path
    text2comp: Path
    assembly: Path
    logs: Path


def project_paths(project_id: str, *, create: bool = True) -> ProjectPaths:
    data_root = STUDIO_DATA_ROOT / project_id
    artifact_root = STUDIO_ARTIFACT_ROOT / project_id
    paths = ProjectPaths(
        project_id=project_id,
        data_root=data_root,
        source=data_root / "source",
        canonical=data_root / "canonical",
        training_data=data_root / "training",
        artifact_root=artifact_root,
        expert=artifact_root / "expert",
        router=artifact_root / "router",
        text2comp=artifact_root / "text2comp",
        assembly=artifact_root / "assembly",
        logs=STUDIO_RUNLOG_ROOT / project_id,
    )
    if create:
        for path in (
            paths.source,
            paths.canonical,
            paths.training_data,
            paths.expert,
            paths.router,
            paths.text2comp,
            paths.assembly,
            paths.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_studio_roots() -> None:
    for path in (STUDIO_DATA_ROOT, STUDIO_ARTIFACT_ROOT, STUDIO_RUNLOG_ROOT):
        path.mkdir(parents=True, exist_ok=True)
