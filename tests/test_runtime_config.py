from pathlib import Path

from piern.shared.runtime.config import load_runtime_config, validate_runtime_config


def _set_minimal_env(monkeypatch, tmp_path: Path):
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    runlogs = tmp_path / "runlogs"
    model = tmp_path / "Qwen" / "Qwen2.5-0.5B-Instruct"
    for path in [data, artifacts, runlogs, model]:
        path.mkdir(parents=True)
    monkeypatch.setenv("PIERN_ROOT", str(tmp_path))
    monkeypatch.setenv("PIERN_DATA_ROOT", str(data))
    monkeypatch.setenv("PIERN_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("PIERN_RUNLOG_ROOT", str(runlogs))
    monkeypatch.setenv("PIERN_JOB_STORE_PATH", str(runlogs / "jobs.sqlite"))
    monkeypatch.setenv("PIERN_TRAINING_JOB_STORE_PATH", str(runlogs / "training_jobs.sqlite"))
    monkeypatch.setenv("PIERN_ROUTER_JSONL_CACHE_DIR", str(data / "router" / ".parquet_jsonl_cache"))
    monkeypatch.setenv("PIERN_SERVICE_RUN_DIR", str(runlogs / "services"))
    monkeypatch.setenv("PIERN_QWEN_EMBEDDING_MODEL", str(model))
    monkeypatch.setenv("PIERN_QWEN_EMBEDDING_TOKENIZER", str(model))


def test_runtime_config_validates_minimal_portable_environment(monkeypatch, tmp_path):
    _set_minimal_env(monkeypatch, tmp_path)

    validation = validate_runtime_config()

    assert validation.ok
    assert validation.config.project_root == tmp_path.resolve()
    assert validation.config.safe_summary()["project_root"] == str(tmp_path.resolve())


def test_runtime_config_reports_missing_data_root(monkeypatch, tmp_path):
    _set_minimal_env(monkeypatch, tmp_path)
    missing = tmp_path / "missing-data"
    monkeypatch.setenv("PIERN_DATA_ROOT", str(missing))

    validation = validate_runtime_config(load_runtime_config())

    assert not validation.ok
    assert any("PIERN_DATA_ROOT" in item for item in validation.errors)
