from piern.training.services import training_cleanup


def test_stage_and_restore_training_artifacts(tmp_path):
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


def test_remove_training_log_and_stop_file_are_idempotent(tmp_path):
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
