from PierNet.training.services import training_cleanup


def test_stage_and_restore_training_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(training_cleanup, "ARTIFACT_ROOT", tmp_path)
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "router_epoch_0001.pt").write_text("weights", encoding="utf-8")

    staged = training_cleanup.stage_job_artifacts_for_delete({"run_dir": str(run_dir)})

    assert len(staged) == 1
    original, target = staged[0]
    assert original == run_dir
    assert not run_dir.exists()
    assert target.exists()

    training_cleanup.restore_staged_job_artifacts(staged)

    assert run_dir.exists()
    assert (run_dir / "router_epoch_0001.pt").exists()


def test_stage_training_artifacts_rejects_run_dir_outside_artifact_root(monkeypatch, tmp_path):
    artifact_root = tmp_path / "artifacts"
    outside_dir = tmp_path / "outside-run"
    artifact_root.mkdir()
    outside_dir.mkdir()
    (outside_dir / "router_epoch_0001.pt").write_text("weights", encoding="utf-8")
    monkeypatch.setattr(training_cleanup, "ARTIFACT_ROOT", artifact_root)

    try:
        training_cleanup.stage_job_artifacts_for_delete({"run_dir": str(outside_dir)})
    except ValueError as exc:
        assert "artifact root" in str(exc)
    else:
        raise AssertionError("expected ValueError for run directory outside artifact root")

    assert outside_dir.exists()
    assert (outside_dir / "router_epoch_0001.pt").exists()


def test_stage_training_artifacts_rejects_artifact_root(monkeypatch, tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    marker = artifact_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(training_cleanup, "ARTIFACT_ROOT", artifact_root)

    try:
        training_cleanup.stage_job_artifacts_for_delete({"run_dir": str(artifact_root)})
    except ValueError as exc:
        assert "artifact root" in str(exc)
    else:
        raise AssertionError("expected ValueError for artifact root run directory")

    assert marker.exists()


def test_remove_training_log_and_stop_file_skip_paths_outside_runlog_root(monkeypatch, tmp_path):
    runlog_root = tmp_path / "runlogs"
    outside_dir = tmp_path / "outside"
    runlog_root.mkdir()
    outside_dir.mkdir()
    log_path = outside_dir / "train.log"
    stop_path = outside_dir / "stop.json"
    log_path.write_text("log", encoding="utf-8")
    stop_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(training_cleanup, "RUNLOG_ROOT", runlog_root)

    training_cleanup.remove_job_log({"log_path": str(log_path)})
    training_cleanup.remove_job_stop_file({"stop_file": str(stop_path)})

    assert log_path.exists()
    assert stop_path.exists()


def test_remove_training_log_and_stop_file_are_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(training_cleanup, "RUNLOG_ROOT", tmp_path)
    log_path = tmp_path / "train.log"
    stop_path = tmp_path / "stop.json"
    log_path.write_text("log", encoding="utf-8")
    stop_path.write_text("{}", encoding="utf-8")

    entry = {"log_path": str(log_path), "stop_file": str(stop_path)}
    training_cleanup.remove_job_log(entry)
    training_cleanup.remove_job_stop_file(entry)
    training_cleanup.remove_job_log(entry)
    training_cleanup.remove_job_stop_file(entry)

    assert not log_path.exists()
    assert not stop_path.exists()
