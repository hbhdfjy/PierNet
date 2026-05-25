from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from PierNet.synth.api.routers import interview


def _set_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    project_root.mkdir()
    data_root.mkdir()
    monkeypatch.setattr(interview, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(interview, "DATA_ROOT", data_root)
    return project_root, data_root


def test_resolve_hdf5_path_maps_data_prefix_to_runtime_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _project_root, data_root = _set_roots(monkeypatch, tmp_path)
    h5_path = data_root / "modflow" / "case.h5"
    h5_path.parent.mkdir()
    h5_path.write_bytes(b"not a real hdf5")

    assert interview._resolve_hdf5_path("data/modflow/case.h5") == str(h5_path)


def test_resolve_hdf5_path_rejects_data_prefix_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_roots(monkeypatch, tmp_path)

    with pytest.raises(HTTPException) as exc:
        interview._resolve_hdf5_path("data/../../outside/case.h5")

    assert exc.value.status_code == 400


def test_resolve_hdf5_path_rejects_absolute_path_outside_allowed_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _set_roots(monkeypatch, tmp_path)
    outside = tmp_path / "outside" / "case.h5"
    outside.parent.mkdir()
    outside.write_bytes(b"not a real hdf5")

    with pytest.raises(HTTPException) as exc:
        interview._resolve_hdf5_path(str(outside))

    assert exc.value.status_code == 400


def test_resolve_hdf5_path_ignores_missing_or_non_hdf5_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _project_root, data_root = _set_roots(monkeypatch, tmp_path)
    text_path = data_root / "notes.txt"
    text_path.write_text("metadata", encoding="utf-8")

    assert interview._resolve_hdf5_path("data/missing.h5") is None
    assert interview._resolve_hdf5_path("data/notes.txt") is None


def test_validate_start_request_trims_valid_names():
    req = interview.InterviewStartRequest(simulator=" modflow ", scenario=" coastal ", mode="scenario")

    assert interview._validate_start_request(req) == ("modflow", "coastal", "scenario")


def test_validate_start_request_rejects_invalid_registry_names():
    req = interview.InterviewStartRequest(simulator="../modflow", scenario="coastal", mode="simulator")

    with pytest.raises(HTTPException) as exc:
        interview._validate_start_request(req)

    assert exc.value.status_code == 400


def test_validate_start_request_rejects_unknown_mode():
    req = interview.InterviewStartRequest(simulator="modflow", scenario="coastal", mode="legacy")

    with pytest.raises(HTTPException) as exc:
        interview._validate_start_request(req)

    assert exc.value.status_code == 400


def test_start_interview_passes_validated_values_to_agent(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class Response:
        def to_dict(self) -> dict[str, object]:
            return {
                "step": 7,
                "step_label": "场景描述",
                "total_steps": 7,
                "question": "describe",
                "extracted": None,
                "needs_confirmation": False,
                "extraction_uncertain": False,
                "done": False,
                "saved": False,
                "registry_key": None,
                "error": None,
                "hdf5_loaded": False,
                "github_prefilled": None,
            }

    def fake_create(**kwargs: object):
        captured.update(kwargs)
        return "iv-test", Response()

    monkeypatch.setattr(interview, "_load_llm_cfg", lambda: {})
    monkeypatch.setattr(interview, "_resolve_hdf5_path", lambda _value: None)
    monkeypatch.setattr(interview, "_iv_create", fake_create)

    result = asyncio.run(
        interview.start_interview(
            interview.InterviewStartRequest(simulator=" modflow ", scenario=" coastal ", mode="scenario")
        )
    )

    assert result["session_id"] == "iv-test"
    assert captured["simulator"] == "modflow"
    assert captured["scenario"] == "coastal"
    assert captured["mode"] == "scenario"
