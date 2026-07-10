from __future__ import annotations

import sqlite3
from pathlib import Path
from types import ModuleType

import pytest

from PierNet.shared.audit import store as audit_store
from PierNet.shared.tasks import locks, workers
from PierNet.synth.services import job_store as synth_job_store
from PierNet.training.services import job_store as training_job_store


@pytest.mark.parametrize(
    ("store", "path_attr"),
    [
        (audit_store, "AUDIT_STORE_PATH"),
        (locks, "LOCK_STORE_PATH"),
        (workers, "WORKER_STORE_PATH"),
        (synth_job_store, "JOB_STORE_PATH"),
        (training_job_store, "TRAINING_JOB_STORE_PATH"),
    ],
)
def test_store_connection_context_closes_file_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    store: ModuleType,
    path_attr: str,
) -> None:
    monkeypatch.setattr(store, path_attr, tmp_path / f"{store.__name__.replace('.', '-')}.sqlite")
    with store._connection() as conn:
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
