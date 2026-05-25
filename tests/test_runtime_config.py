from pathlib import Path

from PierNet.shared.runtime.config import load_runtime_config, validate_runtime_config


def _set_minimal_env(monkeypatch, tmp_path: Path):
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    runlogs = tmp_path / "runlogs"
    model = tmp_path / "Qwen" / "Qwen2.5-0.5B-Instruct"
    for path in [data, artifacts, runlogs, model]:
        path.mkdir(parents=True)
    monkeypatch.setenv("PierNet_ROOT", str(tmp_path))
    monkeypatch.setenv("PierNet_DATA_ROOT", str(data))
    monkeypatch.setenv("PierNet_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("PierNet_RUNLOG_ROOT", str(runlogs))
    monkeypatch.setenv("PierNet_JOB_STORE_PATH", str(runlogs / "jobs.sqlite"))
    monkeypatch.setenv("PierNet_TRAINING_JOB_STORE_PATH", str(runlogs / "training_jobs.sqlite"))
    monkeypatch.setenv("PierNet_ROUTER_JSONL_CACHE_DIR", str(data / "router" / ".parquet_jsonl_cache"))
    monkeypatch.setenv("PierNet_SERVICE_RUN_DIR", str(runlogs / "services"))
    monkeypatch.setenv("PierNet_QWEN_EMBEDDING_MODEL", str(model))
    monkeypatch.setenv("PierNet_QWEN_EMBEDDING_TOKENIZER", str(model))


def test_runtime_config_validates_minimal_portable_environment(monkeypatch, tmp_path):
    _set_minimal_env(monkeypatch, tmp_path)

    validation = validate_runtime_config()

    assert validation.ok
    assert validation.config.project_root == tmp_path.resolve()
    assert validation.config.safe_summary()["project_root"] == str(tmp_path.resolve())


def test_runtime_config_reports_missing_data_root(monkeypatch, tmp_path):
    _set_minimal_env(monkeypatch, tmp_path)
    missing = tmp_path / "missing-data"
    monkeypatch.setenv("PierNet_DATA_ROOT", str(missing))

    validation = validate_runtime_config(load_runtime_config())

    assert not validation.ok
    assert any("PierNet_DATA_ROOT" in item for item in validation.errors)


def test_runtime_config_prefers_repo_local_python_and_node_defaults(monkeypatch, tmp_path):
    repo_python = tmp_path / ".conda" / "env" / "bin" / "python"
    repo_node = tmp_path / ".node" / "current" / "bin" / "node"
    repo_python.parent.mkdir(parents=True)
    repo_node.parent.mkdir(parents=True)
    repo_python.write_text("#!/bin/sh\n", encoding="utf-8")
    repo_node.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("PierNet_ROOT", str(tmp_path))
    for key in (
        "PierNet_CONDA_ENV",
        "PierNet_PYTHON",
        "PierNet_TRAINING_PYTHON",
        "PierNet_NODE_BIN",
        "PierNet_NODE",
    ):
        monkeypatch.delenv(key, raising=False)

    config = load_runtime_config()

    assert config.conda_env == (tmp_path / ".conda" / "env").resolve()
    assert config.python == repo_python.resolve()
    assert config.node_bin == repo_node.resolve()
