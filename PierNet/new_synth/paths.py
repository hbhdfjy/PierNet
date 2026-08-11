from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from PierNet.shared.runtime.paths import DATA_ROOT, RUNLOG_ROOT

NEW_SYNTH_DATA_ROOT = Path(
    os.getenv("PIERN_NEW_SYNTH_DATA_ROOT", DATA_ROOT / "new_synth")
).resolve()
NEW_SYNTH_RUNLOG_ROOT = Path(
    os.getenv("PIERN_NEW_SYNTH_RUNLOG_ROOT", RUNLOG_ROOT / "new_synth")
).resolve()
NEW_SYNTH_DB_PATH = Path(
    os.getenv("PIERN_NEW_SYNTH_DB_PATH", NEW_SYNTH_RUNLOG_ROOT / "state.sqlite")
).resolve()
NEW_SYNTH_CACHE_ROOT = Path(
    os.getenv("PIERN_NEW_SYNTH_CACHE_ROOT", DATA_ROOT / ".cache" / "new_synth")
).resolve()

_WORKFLOW_ID_RE = re.compile(r"^new-synth-[a-f0-9]{12}$")


@dataclass(frozen=True)
class WorkflowPaths:
    workflow_id: str
    root: Path
    source: Path
    canonical: Path
    definitions: Path
    artifacts: Path
    templates: Path
    text2comp: Path
    router: Path
    evaluation: Path
    logs: Path


def workflow_paths(workflow_id: str, *, create: bool = True) -> WorkflowPaths:
    if not _WORKFLOW_ID_RE.fullmatch(str(workflow_id)):
        raise ValueError(f"invalid new-synth workflow id: {workflow_id}")
    root = NEW_SYNTH_DATA_ROOT / workflow_id
    artifacts = root / "artifacts"
    paths = WorkflowPaths(
        workflow_id=workflow_id,
        root=root,
        source=root / "source",
        canonical=root / "canonical",
        definitions=root / "definitions",
        artifacts=artifacts,
        templates=artifacts / "templates",
        text2comp=artifacts / "text2comp",
        router=artifacts / "router",
        evaluation=artifacts / "evaluation",
        logs=NEW_SYNTH_RUNLOG_ROOT / workflow_id,
    )
    if create:
        for path in (
            paths.source,
            paths.canonical,
            paths.definitions,
            paths.templates,
            paths.text2comp,
            paths.router,
            paths.evaluation,
            paths.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_roots() -> None:
    for path in (NEW_SYNTH_DATA_ROOT, NEW_SYNTH_RUNLOG_ROOT, NEW_SYNTH_CACHE_ROOT):
        path.mkdir(parents=True, exist_ok=True)
