from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from PierNet.shared.storage import portable
from PierNet.synth.api.routers import config as config_router
from PierNet.synth.api.routers import generation
from PierNet.synth.services import generation_executor
from PierNet.synth.services.job_manager import JobRecord
from scripts.text2comp import fill_samples as fill_samples_script
from scripts.text2comp import generate_templates as generate_templates_script


def test_fill_samples_job_invalidates_text2comp_scenario_cache(monkeypatch) -> None:
    invalidated: list[bool] = []
    record = JobRecord(
        job_id="fill-test",
        job_type="fill_samples",
        status="running",
        loop=None,
        scenario_totals={},
        started_at=0.0,
    )

    def fake_run_fill_samples(**kwargs) -> None:
        assert kwargs["scenarios"] == ["case_a"]

    monkeypatch.setattr(generation_executor, "run_fill_samples", fake_run_fill_samples)
    monkeypatch.setattr(generation_executor, "publish", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generation_executor.job_manager, "should_stop", lambda _record: False)
    monkeypatch.setattr(config_router, "invalidate_text2comp_scenarios_cache", lambda: invalidated.append(True))

    generation_executor.run_fill_samples_job(record, {"scenarios": ["case_a"], "n_samples": 1})

    assert record.status == "done"
    assert invalidated == [True]


def test_generation_request_schemas_normalize_scenarios() -> None:
    template_req = generation.GenerateTemplatesRequest(scenarios=[" shared ", "", "shared", "other"])
    fill_req = generation.FillSamplesRequest(scenarios=" shared, ,other,shared ")

    assert template_req.scenarios == ["shared", "other"]
    assert fill_req.scenarios == ["shared", "other"]


def test_fill_samples_request_bounds_seed_to_backend_supported_range() -> None:
    generation.FillSamplesRequest(scenarios=["case_a"], seed=2_147_483_647)

    with pytest.raises(ValidationError):
        generation.FillSamplesRequest(scenarios=["case_a"], seed=-1)
    with pytest.raises(ValidationError):
        generation.FillSamplesRequest(scenarios=["case_a"], seed=2_147_483_648)


def test_generation_routes_reject_duplicate_stage_scenarios(monkeypatch) -> None:
    paths = [
        (Path("/data/gcam/gcam_shared.h5"), "gcam", None),
        (Path("/data/simpeg/simpeg_shared.h5"), "simpeg", None),
    ]

    monkeypatch.setattr(generation, "load_config", lambda _path: {"data_root": "data"})
    monkeypatch.setattr(generation, "_scan_h5_files", lambda _cfg, _base_dir: paths)
    monkeypatch.setattr(generation, "_scenario_name_from_path", lambda path, _suffix: path.stem.split("_", 1)[1])

    with pytest.raises(HTTPException) as exc_info:
        generation._assert_unique_stage_scenarios("configs/text2comp/default.yaml", ["shared"])

    assert exc_info.value.status_code == 400
    assert "同名场景" in str(exc_info.value.detail)
    assert "gcam" in str(exc_info.value.detail)
    assert "simpeg" in str(exc_info.value.detail)


def test_generation_routes_allow_unique_selected_stage_scenarios(monkeypatch) -> None:
    paths = [
        (Path("/data/gcam/gcam_shared.h5"), "gcam", None),
        (Path("/data/simpeg/simpeg_other.h5"), "simpeg", None),
    ]

    monkeypatch.setattr(generation, "load_config", lambda _path: {"data_root": "data"})
    monkeypatch.setattr(generation, "_scan_h5_files", lambda _cfg, _base_dir: paths)
    monkeypatch.setattr(generation, "_scenario_name_from_path", lambda path, _suffix: path.stem.split("_", 1)[1])

    generation._assert_unique_stage_scenarios("configs/text2comp/default.yaml", ["shared"])


def test_generation_routes_apply_duplicate_scenario_guard_before_starting_jobs(monkeypatch) -> None:
    paths = [
        (Path("/data/gcam/gcam_shared.h5"), "gcam", None),
        (Path("/data/simpeg/simpeg_shared.h5"), "simpeg", None),
    ]
    created: list[bool] = []

    monkeypatch.setattr(generation, "_reject_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(generation, "load_config", lambda _path: {"data_root": "data"})
    monkeypatch.setattr(generation, "_scan_h5_files", lambda _cfg, _base_dir: paths)
    monkeypatch.setattr(generation, "_scenario_name_from_path", lambda path, _suffix: path.stem.split("_", 1)[1])

    def fail_create_job(*_args, **_kwargs):
        created.append(True)
        raise AssertionError("duplicate scenario request must not create a job")

    monkeypatch.setattr(generation.job_manager, "create_job", fail_create_job)

    with pytest.raises(HTTPException) as template_exc:
        asyncio.run(
            generation.start_generate_templates(
                generation.GenerateTemplatesRequest(scenarios=["shared"], n_templates=1)
            )
        )
    with pytest.raises(HTTPException) as fill_exc:
        asyncio.run(generation.start_fill_samples(generation.FillSamplesRequest(scenarios=["shared"], n_samples=1)))

    assert template_exc.value.status_code == 400
    assert fill_exc.value.status_code == 400
    assert "同名场景" in str(template_exc.value.detail)
    assert "同名场景" in str(fill_exc.value.detail)
    assert created == []


def test_generation_routes_use_shared_stage_collision_guard(monkeypatch) -> None:
    paths = [
        (Path("/data/gcam/gcam_shared.h5"), "gcam", None),
        (Path("/data/simpeg/simpeg_shared.h5"), "simpeg", None),
    ]

    monkeypatch.setattr(generation, "load_config", lambda _path: {"data_root": "data"})
    monkeypatch.setattr(generation, "_scan_h5_files", lambda _cfg, _base_dir: paths)
    monkeypatch.setattr(generation, "_scenario_name_from_path", lambda path, _suffix: path.stem.split("_", 1)[1])
    monkeypatch.setattr(generation, "duplicate_stage_scenarios", lambda h5_files: ["shared (gcam, simpeg)"])

    with pytest.raises(HTTPException) as exc_info:
        generation._assert_unique_stage_scenarios("configs/text2comp/default.yaml", [])

    assert exc_info.value.status_code == 400
    assert "shared (gcam, simpeg)" in str(exc_info.value.detail)


def test_generation_routes_start_jobs_with_normalized_scenarios(monkeypatch) -> None:
    created: dict[str, object] = {}

    def fake_start_job(**kwargs):
        created.update(kwargs)
        return JobRecord(
            job_id="tmpl-normalized",
            job_type=kwargs["job_type"],
            status="queued",
            loop=None,
            scenario_totals=kwargs["scenario_totals"],
            started_at=0.0,
        )

    monkeypatch.setattr(generation, "_reject_active_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(generation, "_assert_unique_stage_scenarios", lambda *args, **kwargs: None)
    monkeypatch.setattr(generation, "_start_job", fake_start_job)

    response = asyncio.run(
        generation.start_generate_templates(
            generation.GenerateTemplatesRequest(scenarios=[" shared ", "", "shared"], n_templates=3)
        )
    )

    assert response.scenario_totals == {"shared": 3}
    assert created["scenario_totals"] == {"shared": 3}
    assert created["payload"]["scenarios"] == ["shared"]
    assert created["lock_keys"] == ["template:shared"]



def test_generate_templates_reports_per_scenario_metadata_failures(monkeypatch, tmp_path: Path) -> None:
    h5_path = tmp_path / "data" / "unknown_sim" / "unknown_sim_case_a.h5"
    logs: list[str] = []

    class DummyLLMClient:
        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(generate_templates_script, "LLMClient", DummyLLMClient)
    monkeypatch.setattr(
        generate_templates_script,
        "load_config",
        lambda _path: {"generation": {"max_workers": 1}, "llm": {}, "registry": "registry.yaml"},
    )
    monkeypatch.setattr(generate_templates_script, "_load_registry", lambda _path: {})
    monkeypatch.setattr(generate_templates_script, "_resolve_data_path", lambda *_args, **_kwargs: tmp_path / "templates")
    monkeypatch.setattr(generate_templates_script, "_scan_h5_files", lambda _cfg, _base_dir: [(h5_path, "unknown_sim", None)])
    monkeypatch.setattr(generate_templates_script, "_scenario_name_from_path", lambda _path, _suffix: "case_a")
    monkeypatch.setattr(generate_templates_script, "assert_unique_stage_scenarios", lambda _paths: None)

    with pytest.raises(RuntimeError, match="模板生成失败"):
        generate_templates_script.run_generate_templates(
            str(tmp_path / "configs" / "text2comp" / "default.yaml"),
            n_templates=1,
            scenarios=["case_a"],
            on_log=logs.append,
        )

    assert any("未找到 simulator 'unknown_sim' 的元数据" in line for line in logs)


def test_fill_samples_reports_missing_template_as_job_failure(monkeypatch, tmp_path: Path) -> None:
    h5_path = tmp_path / "data" / "simpeg" / "simpeg_case_a.h5"
    logs: list[str] = []

    monkeypatch.setattr(
        fill_samples_script,
        "load_config",
        lambda _path: {"generation": {"max_workers": 1}, "registry": "registry.yaml"},
    )
    monkeypatch.setattr(fill_samples_script, "_resolve_data_path", lambda *_args, **_kwargs: tmp_path / "templates")
    monkeypatch.setattr(fill_samples_script, "_scan_h5_files", lambda _cfg, _base_dir: [(h5_path, "simpeg", None)])
    monkeypatch.setattr(fill_samples_script, "_scenario_name_from_path", lambda _path, _suffix: "case_a")
    monkeypatch.setattr(fill_samples_script, "assert_unique_stage_scenarios", lambda _paths: None)

    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="样本填充失败"):
        fill_samples_script.run_fill_samples(
            str(tmp_path / "configs" / "text2comp" / "default.yaml"),
            n_samples=1,
            scenarios=["case_a"],
            output_format="jsonl",
            output_dir=str(out_dir),
            on_log=logs.append,
        )

    assert any("模板文件不存在" in line for line in logs)
    assert not (out_dir / "all_training_data.jsonl").exists()


def _fill_skip_base(monkeypatch, tmp_path: Path) -> Path:
    h5_path = tmp_path / "data" / "simpeg" / "simpeg_case_a.h5"
    monkeypatch.setattr(
        fill_samples_script,
        "load_config",
        lambda _path: {"generation": {"max_workers": 1}, "registry": "registry.yaml"},
    )
    monkeypatch.setattr(fill_samples_script, "_scan_h5_files", lambda _cfg, _base_dir: [(h5_path, "simpeg", None)])
    monkeypatch.setattr(fill_samples_script, "_scenario_name_from_path", lambda _path, _suffix: "case_a")
    monkeypatch.setattr(fill_samples_script, "assert_unique_stage_scenarios", lambda _paths: None)
    return h5_path


def test_fill_samples_skip_existing_jsonl_requires_target_count(monkeypatch, tmp_path: Path) -> None:
    _fill_skip_base(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "case_a.jsonl").write_text('{"ok": 1}\n' * 5, encoding="utf-8")
    logs: list[str] = []

    fill_samples_script.run_fill_samples(
        str(tmp_path / "configs" / "text2comp" / "default.yaml"),
        n_samples=5,
        scenarios=["case_a"],
        output_format="jsonl",
        output_dir=str(out_dir),
        templates_dir=str(tmp_path / "templates"),
        skip_existing=True,
        on_log=logs.append,
    )

    assert any("已达到目标" in line and "JSONL=5/5" in line for line in logs)


def test_fill_samples_reruns_incomplete_jsonl_when_skip_existing(monkeypatch, tmp_path: Path) -> None:
    _fill_skip_base(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "case_a.jsonl").write_text('{"ok": 1}\n' * 2, encoding="utf-8")
    logs: list[str] = []

    with pytest.raises(RuntimeError, match="样本填充失败"):
        fill_samples_script.run_fill_samples(
            str(tmp_path / "configs" / "text2comp" / "default.yaml"),
            n_samples=5,
            scenarios=["case_a"],
            output_format="jsonl",
            output_dir=str(out_dir),
            templates_dir=str(tmp_path / "templates"),
            skip_existing=True,
            on_log=logs.append,
        )

    assert any("已有输出未达到目标" in line and "JSONL=2/5" in line for line in logs)
    assert any("模板文件不存在" in line for line in logs)


def test_fill_samples_skip_existing_parquet_requires_target_count(monkeypatch, tmp_path: Path) -> None:
    _fill_skip_base(monkeypatch, tmp_path)
    parquet_root = tmp_path / "parquet"
    part_dir = portable.partition_dir_for("text2comp", "simpeg", "case_a", parquet_root)
    part_dir.mkdir(parents=True)
    (part_dir / "_manifest.json").write_text('{"row_count": 6}', encoding="utf-8")
    logs: list[str] = []

    fill_samples_script.run_fill_samples(
        str(tmp_path / "configs" / "text2comp" / "default.yaml"),
        n_samples=5,
        scenarios=["case_a"],
        output_format="parquet",
        output_dir=str(parquet_root),
        templates_dir=str(tmp_path / "templates"),
        skip_existing=True,
        on_log=logs.append,
    )

    assert any("已达到目标" in line and "Parquet=6/5" in line for line in logs)


def test_fill_samples_reruns_incomplete_parquet_when_skip_existing(monkeypatch, tmp_path: Path) -> None:
    _fill_skip_base(monkeypatch, tmp_path)
    parquet_root = tmp_path / "parquet"
    part_dir = portable.partition_dir_for("text2comp", "simpeg", "case_a", parquet_root)
    part_dir.mkdir(parents=True)
    (part_dir / "_manifest.json").write_text('{"row_count": 2}', encoding="utf-8")
    logs: list[str] = []

    with pytest.raises(RuntimeError, match="样本填充失败"):
        fill_samples_script.run_fill_samples(
            str(tmp_path / "configs" / "text2comp" / "default.yaml"),
            n_samples=5,
            scenarios=["case_a"],
            output_format="parquet",
            output_dir=str(parquet_root),
            templates_dir=str(tmp_path / "templates"),
            skip_existing=True,
            on_log=logs.append,
        )

    assert any("已有输出未达到目标" in line and "Parquet=2/5" in line for line in logs)
    assert any("模板文件不存在" in line for line in logs)


def test_fill_samples_template_index_filter_rejects_negative_indices() -> None:
    assert fill_samples_script._valid_time_indices([-1, 0, 1, 2], 2).tolist() == [0, 1]
    channels = fill_samples_script._valid_channel_indices([-2, 0, 3, 4], 4)
    assert channels is not None
    assert channels.tolist() == [0, 3]
