from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
import yaml

from piern.synth.api.routers import simulation


def test_cleanup_stale_tmp_configs_removes_only_yaml(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "tmp_configs"
    tmp_dir.mkdir()
    stale = tmp_dir / "stale.yaml"
    keep = tmp_dir / "keep.txt"
    stale.write_text("seed: 1\n", encoding="utf-8")
    keep.write_text("keep\n", encoding="utf-8")

    deleted = simulation.cleanup_stale_tmp_configs(tmp_dir)

    assert deleted == 1
    assert not stale.exists()
    assert keep.exists()


def test_api_lifespan_cleans_stale_simulation_configs(monkeypatch) -> None:
    from piern.api import main as api_main

    calls: list[str] = []
    monkeypatch.setattr(api_main.simulation, "cleanup_stale_tmp_configs", lambda: calls.append("cleanup"))
    monkeypatch.setattr(api_main, "log_runtime_config", lambda: calls.append("log"))

    async def run_lifespan() -> None:
        async with api_main._lifespan(None):
            calls.append("running")

    asyncio.run(run_lifespan())

    assert calls == ["cleanup", "log", "running"]


def test_simulator_pipeline_default_config_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for pipeline in sorted((root / "piern" / "simulators").glob("*/pipeline.py")):
        text = pipeline.read_text(encoding="utf-8")
        for match in re.finditer(r"default=[\"'](configs/[^\"']+\.ya?ml)[\"']", text):
            rel = match.group(1)
            assert (root / rel).exists(), f"{pipeline.relative_to(root)} references missing default config {rel}"


def test_simulation_scan_resolves_data_output_dir_to_runtime_data_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "runtime-data"
    variants_dir = project_root / "configs" / "modflow" / "variants"
    variants_dir.mkdir(parents=True)
    cfg_path = variants_dir / "external.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"output_dir": "data/modflow", "output_file": "modflow_external.h5"}),
        encoding="utf-8",
    )
    h5_path = data_root / "modflow" / "modflow_external.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"placeholder")

    monkeypatch.setattr(simulation, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(simulation, "DATA_ROOT", data_root)
    monkeypatch.setattr(simulation, "SIMULATORS", ["modflow"])
    monkeypatch.setattr(simulation, "_read_h5_info", lambda path: (7, [2, 4], 123))
    simulation._invalidate_cache()

    try:
        scenarios = simulation._scan_scenarios()
        resolved = simulation._resolve_output_h5_path(
            "configs/modflow/variants/external.yaml",
            "modflow",
        )
    finally:
        simulation._invalidate_cache()

    assert len(scenarios) == 1
    assert scenarios[0].scenario == "external"
    assert scenarios[0].h5_path == "modflow/modflow_external.h5"
    assert scenarios[0].sample_count == 7
    assert resolved == h5_path


def _sim_scenario(simulator: str, scenario: str, config_path: str) -> simulation.SimulationScenario:
    return simulation.SimulationScenario(
        simulator=simulator,
        scenario=scenario,
        config_path=config_path,
        h5_path=None,
        sample_count=0,
        output_shape=None,
        file_size_bytes=0,
    )


class _SubmitRecorder:
    def __init__(self) -> None:
        self.submitted: list[tuple[object, object, list[simulation.SimulateRequest]]] = []

    def submit(self, func, record, reqs):
        self.submitted.append((func, record, reqs))
        return None


def test_start_batch_simulate_accepts_composite_scenario_selectors(monkeypatch) -> None:
    scenarios = [
        _sim_scenario("simpeg", "shared", "cfg-simpeg"),
        _sim_scenario("modflow", "shared", "cfg-modflow"),
    ]
    created: list[tuple[str, dict[str, int]]] = []
    recorder = _SubmitRecorder()

    def create_job(job_type: str, scenario_totals: dict[str, int], **kwargs):
        created.append((job_type, scenario_totals, kwargs.get("lock_keys")))
        return SimpleNamespace(job_id="job-1", status="running")

    monkeypatch.setattr(simulation, "_get_scenarios_cached", lambda: scenarios)
    monkeypatch.setattr(simulation.job_manager, "create_job", create_job)
    monkeypatch.setattr(simulation, "_executor", recorder)

    result = asyncio.run(
        simulation.start_batch_simulate(
            simulation.BatchSimulateRequest(
                scenarios=["simpeg/shared", "modflow/shared"],
                n_samples=5,
                seed=11,
                skip_existing=True,
                parallel=False,
                max_workers=2,
            )
        )
    )

    assert result.scenario_totals == {"simpeg/shared": 5, "modflow/shared": 5}
    assert created == [
        (
            "simulate",
            {"simpeg/shared": 5, "modflow/shared": 5},
            ["raw:simpeg/shared", "raw:modflow/shared"],
        )
    ]
    submitted_reqs = recorder.submitted[0][2]
    assert [(req.simulator, req.scenario, req.config_path) for req in submitted_reqs] == [
        ("simpeg", "shared", "cfg-simpeg"),
        ("modflow", "shared", "cfg-modflow"),
    ]


def test_start_batch_simulate_rejects_ambiguous_legacy_scenario_selector(monkeypatch) -> None:
    monkeypatch.setattr(
        simulation,
        "_get_scenarios_cached",
        lambda: [
            _sim_scenario("simpeg", "shared", "cfg-simpeg"),
            _sim_scenario("modflow", "shared", "cfg-modflow"),
        ],
    )

    try:
        asyncio.run(
            simulation.start_batch_simulate(
                simulation.BatchSimulateRequest(
                    scenarios=["shared"],
                    n_samples=1,
                    seed=1,
                    skip_existing=False,
                    parallel=False,
                    max_workers=1,
                )
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "歧义" in str(exc.detail)
    else:
        raise AssertionError("expected ambiguous scenario selection to fail")


def test_start_batch_simulate_rejects_partial_missing_selection(monkeypatch) -> None:
    created: list[bool] = []
    monkeypatch.setattr(
        simulation,
        "_get_scenarios_cached",
        lambda: [_sim_scenario("modflow", "coastal", "cfg-coastal")],
    )
    monkeypatch.setattr(simulation.job_manager, "create_job", lambda *_args, **_kwargs: created.append(True))

    try:
        asyncio.run(
            simulation.start_batch_simulate(
                simulation.BatchSimulateRequest(
                    scenarios=["modflow/coastal", "modflow/missing"],
                    n_samples=1,
                    seed=1,
                    skip_existing=False,
                    parallel=False,
                    max_workers=1,
                )
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "未找到场景" in str(exc.detail)
    else:
        raise AssertionError("expected partial missing scenario selection to fail")

    assert created == []


def test_start_batch_simulate_rejects_duplicate_selection(monkeypatch) -> None:
    created: list[bool] = []
    monkeypatch.setattr(
        simulation,
        "_get_scenarios_cached",
        lambda: [_sim_scenario("modflow", "coastal", "cfg-coastal")],
    )
    monkeypatch.setattr(simulation.job_manager, "create_job", lambda *_args, **_kwargs: created.append(True))

    try:
        asyncio.run(
            simulation.start_batch_simulate(
                simulation.BatchSimulateRequest(
                    scenarios=["modflow/coastal", "coastal"],
                    n_samples=1,
                    seed=1,
                    skip_existing=False,
                    parallel=False,
                    max_workers=1,
                )
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "重复场景" in str(exc.detail)
    else:
        raise AssertionError("expected duplicate scenario selection to fail")

    assert created == []


def _simulate_request(
    *,
    n_samples: int = 10,
    skip_existing: bool = True,
) -> simulation.SimulateRequest:
    return simulation.SimulateRequest(
        simulator="modflow",
        scenario="coastal",
        n_samples=n_samples,
        seed=1,
        config_path="configs/modflow/variants/current.yaml",
        skip_existing=skip_existing,
    )


def _capture_publish(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(simulation, "publish", lambda _record, event: events.append(event))
    return events


def test_start_simulate_uses_server_catalog_config_path(monkeypatch) -> None:
    recorder = _SubmitRecorder()
    created: list[tuple[str, dict[str, int]]] = []

    monkeypatch.setattr(
        simulation,
        "_get_scenarios_cached",
        lambda: [_sim_scenario("modflow", "coastal", "configs/modflow/variants/current.yaml")],
    )

    def create_job(job_type: str, scenario_totals: dict[str, int], **kwargs):
        created.append((job_type, scenario_totals, kwargs.get("lock_keys")))
        return SimpleNamespace(job_id="job-1", status="running")

    monkeypatch.setattr(simulation.job_manager, "create_job", create_job)
    monkeypatch.setattr(simulation, "_executor", recorder)

    result = asyncio.run(
        simulation.start_simulate(
            simulation.SimulateRequest(
                simulator="modflow",
                scenario="coastal",
                n_samples=5,
                seed=9,
                config_path="configs/modflow/variants/stale.yaml",
                skip_existing=True,
                parallel=False,
                max_workers=2,
            )
        )
    )

    assert result.scenario_totals == {"modflow/coastal": 5}
    assert created == [("simulate", {"modflow/coastal": 5}, ["raw:modflow/coastal"])]
    submitted_req = recorder.submitted[0][2]
    assert submitted_req.config_path == "configs/modflow/variants/current.yaml"
    assert submitted_req.seed == 9
    assert submitted_req.skip_existing is True


class _UploadRequest:
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def stream(self):
        for chunk in self._chunks:
            yield chunk


def _upload_validation(path: Path, *, valid: bool) -> dict:
    return {
        "valid": valid,
        "path": str(path),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "sample_count": 1 if valid else 0,
        "output_shape": [1, 1] if valid else None,
        "params_shape": [1, 1] if valid else None,
        "n_params": 1 if valid else 0,
        "param_names_preview": ["p0"] if valid else [],
        "attrs": {},
        "errors": [] if valid else ["invalid hdf5"],
        "warnings": [],
    }


def test_upload_invalid_hdf5_does_not_replace_existing_target(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "data" / "modflow" / "modflow_case.h5"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original")
    invalidated: list[str] = []

    monkeypatch.setattr(simulation, "RUNLOG_ROOT", tmp_path / "runlogs")
    monkeypatch.setattr(simulation, "canonical_hdf5_path", lambda _simulator, _scenario: target)
    monkeypatch.setattr(simulation, "validate_hdf5_file", lambda path: _upload_validation(path, valid=False))
    monkeypatch.setattr(simulation, "_invalidate_cache", lambda: invalidated.append("simulation"))
    monkeypatch.setattr(simulation, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append("text2comp"))

    response = asyncio.run(
        simulation.upload_simulation_data(
            _UploadRequest(b"not hdf5"),
            simulator="modflow",
            scenario="case",
            overwrite=True,
        )
    )

    assert response["ok"] is False
    assert response["saved_path"] == ""
    assert response["validation"]["valid"] is False
    assert target.read_bytes() == b"original"
    assert invalidated == []
    assert list((tmp_path / "runlogs" / "uploads").glob("*")) == []


def test_upload_valid_hdf5_replaces_target_after_validation(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "data" / "modflow" / "modflow_case.h5"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original")
    invalidated: list[str] = []

    monkeypatch.setattr(simulation, "RUNLOG_ROOT", tmp_path / "runlogs")
    monkeypatch.setattr(simulation, "canonical_hdf5_path", lambda _simulator, _scenario: target)
    monkeypatch.setattr(simulation, "validate_hdf5_file", lambda path: _upload_validation(path, valid=True))
    monkeypatch.setattr(simulation, "_invalidate_cache", lambda: invalidated.append("simulation"))
    monkeypatch.setattr(simulation, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append("text2comp"))

    response = asyncio.run(
        simulation.upload_simulation_data(
            _UploadRequest(b"valid hdf5"),
            simulator="modflow",
            scenario="case",
            overwrite=True,
        )
    )

    assert response["ok"] is True
    assert response["validation"]["valid"] is True
    assert target.read_bytes() == b"valid hdf5"
    assert invalidated == ["simulation", "text2comp"]


def test_start_simulate_rejects_unknown_server_catalog_scenario(monkeypatch) -> None:
    created: list[bool] = []

    monkeypatch.setattr(simulation, "_get_scenarios_cached", lambda: [])
    monkeypatch.setattr(simulation.job_manager, "create_job", lambda *_args, **_kwargs: created.append(True))

    try:
        asyncio.run(
            simulation.start_simulate(
                simulation.SimulateRequest(
                    simulator="modflow",
                    scenario="missing",
                    n_samples=1,
                    seed=1,
                    config_path="configs/modflow/missing.yaml",
                )
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "未找到仿真场景" in str(exc.detail)
    else:
        raise AssertionError("expected unknown single-scenario simulate request to fail")

    assert created == []

def test_start_simulate_reports_conflicting_output_lock(monkeypatch) -> None:
    recorder = _SubmitRecorder()

    monkeypatch.setattr(
        simulation,
        "_get_scenarios_cached",
        lambda: [_sim_scenario("modflow", "coastal", "configs/modflow/variants/current.yaml")],
    )

    def create_job(_job_type: str, _scenario_totals: dict[str, int], **_kwargs):
        raise RuntimeError("资源已被其他任务占用: raw:modflow/coastal")

    monkeypatch.setattr(simulation.job_manager, "create_job", create_job)
    monkeypatch.setattr(simulation, "_executor", recorder)

    try:
        asyncio.run(
            simulation.start_simulate(
                simulation.SimulateRequest(
                    simulator="modflow",
                    scenario="coastal",
                    n_samples=5,
                    seed=9,
                    config_path="configs/modflow/variants/current.yaml",
                )
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 409
        assert "资源已被其他任务占用" in str(exc.detail)
    else:
        raise AssertionError("expected locked simulation output to fail")

    assert recorder.submitted == []


def test_start_batch_simulate_rejects_duplicate_output_file(monkeypatch) -> None:
    created: list[dict] = []
    recorder = _SubmitRecorder()
    shared_output = Path("/tmp/piern-shared-output.h5")

    monkeypatch.setattr(
        simulation,
        "_get_scenarios_cached",
        lambda: [
            _sim_scenario("modflow", "one", "cfg-one"),
            _sim_scenario("simpeg", "two", "cfg-two"),
        ],
    )
    monkeypatch.setattr(simulation, "_resolve_output_h5_path", lambda _config, _simulator: shared_output)
    monkeypatch.setattr(simulation.job_manager, "create_job", lambda *args, **kwargs: created.append(kwargs))
    monkeypatch.setattr(simulation, "_executor", recorder)

    try:
        asyncio.run(
            simulation.start_batch_simulate(
                simulation.BatchSimulateRequest(
                    scenarios=["modflow/one", "simpeg/two"],
                    n_samples=1,
                    seed=1,
                )
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "同一 HDF5" in str(exc.detail)
    else:
        raise AssertionError("expected duplicate output file selection to fail")

    assert created == []
    assert recorder.submitted == []


def test_run_one_scenario_skips_existing_when_target_reached(monkeypatch, tmp_path: Path) -> None:
    h5_path = tmp_path / "modflow_coastal.h5"
    h5_path.write_bytes(b"placeholder")
    events = _capture_publish(monkeypatch)
    called: list[str] = []

    monkeypatch.setattr(simulation, "_resolve_output_h5_path", lambda _config, _simulator: h5_path)
    monkeypatch.setattr(
        simulation,
        "validate_hdf5_file",
        lambda _path: {"valid": True, "sample_count": 12, "errors": []},
    )
    monkeypatch.setattr(simulation, "_run_via_subprocess", lambda *_args: called.append("run") or True)

    run_state: dict = {}
    ok = simulation._run_one_scenario(
        SimpleNamespace(status="running"),
        _simulate_request(n_samples=10),
        run_state,
    )

    assert ok is True
    assert called == []
    assert run_state["final_sample_count"] == 12
    assert any("已达到目标 12/10" in event.get("line", "") for event in events)


def test_run_one_scenario_reruns_existing_when_target_not_reached(monkeypatch, tmp_path: Path) -> None:
    h5_path = tmp_path / "modflow_coastal.h5"
    h5_path.write_bytes(b"placeholder")
    events = _capture_publish(monkeypatch)
    called: list[simulation.SimulateRequest] = []

    monkeypatch.setattr(simulation, "_resolve_output_h5_path", lambda _config, _simulator: h5_path)
    monkeypatch.setattr(
        simulation,
        "validate_hdf5_file",
        lambda _path: {"valid": True, "sample_count": 4, "errors": []},
    )
    monkeypatch.setattr(
        simulation,
        "_run_via_subprocess",
        lambda _record, req, _run_state: called.append(req) or True,
    )

    ok = simulation._run_one_scenario(
        SimpleNamespace(status="running"),
        _simulate_request(n_samples=10),
        {},
    )

    assert ok is True
    assert [req.scenario for req in called] == ["coastal"]
    assert any("已有 4/10" in event.get("line", "") for event in events)


def test_finalize_simulation_result_rejects_missing_hdf5(monkeypatch, tmp_path: Path) -> None:
    events = _capture_publish(monkeypatch)
    monkeypatch.setattr(simulation, "_resolve_output_h5_path", lambda _config, _simulator: tmp_path / "missing.h5")
    monkeypatch.setattr(simulation, "_invalidate_cache", lambda: None)

    ok = simulation._finalize_simulation_result(
        SimpleNamespace(status="running"),
        _simulate_request(n_samples=10, skip_existing=False),
        {},
        True,
    )

    assert ok is False
    assert any("未生成 HDF5" in event.get("line", "") for event in events)


def test_finalize_simulation_result_rejects_invalid_hdf5(monkeypatch, tmp_path: Path) -> None:
    h5_path = tmp_path / "bad.h5"
    h5_path.write_bytes(b"bad")
    events = _capture_publish(monkeypatch)
    invalidated: list[str] = []
    monkeypatch.setattr(simulation, "_resolve_output_h5_path", lambda _config, _simulator: h5_path)
    monkeypatch.setattr(simulation, "_invalidate_cache", lambda: invalidated.append("simulation"))
    monkeypatch.setattr(simulation, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append("text2comp"))
    monkeypatch.setattr(
        simulation,
        "validate_hdf5_file",
        lambda _path: {"valid": False, "sample_count": 0, "errors": ["缺少 timeseries"]},
    )

    ok = simulation._finalize_simulation_result(
        SimpleNamespace(status="running"),
        _simulate_request(n_samples=10, skip_existing=False),
        {},
        True,
    )

    assert ok is False
    assert invalidated == ["simulation", "text2comp"]
    assert any("HDF5 校验失败" in event.get("line", "") for event in events)


def test_finalize_simulation_result_records_valid_hdf5_count(monkeypatch, tmp_path: Path) -> None:
    h5_path = tmp_path / "good.h5"
    h5_path.write_bytes(b"good")
    invalidated: list[str] = []
    monkeypatch.setattr(simulation, "_resolve_output_h5_path", lambda _config, _simulator: h5_path)
    monkeypatch.setattr(simulation, "_invalidate_cache", lambda: invalidated.append("simulation"))
    monkeypatch.setattr(simulation, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append("text2comp"))
    monkeypatch.setattr(
        simulation,
        "validate_hdf5_file",
        lambda _path: {"valid": True, "sample_count": 8, "errors": []},
    )
    run_state: dict = {}

    ok = simulation._finalize_simulation_result(
        SimpleNamespace(status="running"),
        _simulate_request(n_samples=10, skip_existing=False),
        run_state,
        True,
    )

    assert ok is True
    assert run_state["final_sample_count"] == 8
    assert invalidated == ["simulation", "text2comp"]
