from PierNet.training.services import training_progress


def test_training_progress_handles_corrupt_logs_and_metrics(tmp_path):
    run_dir = tmp_path
    valid_point = {"epoch": 1, "step": 2, "global_step": 2, "avg_loss": 0.4, "steps_per_sec": 3.0, "eta_seconds": 4.0}
    (run_dir / "train_log.jsonl").write_text('{"epoch": 1, "step": 2, "global_step": 2, "avg_loss": 0.4, "steps_per_sec": 3.0, "eta_seconds": 4.0}\n{bad\n', encoding="utf-8")
    (run_dir / "test_metrics_epoch_0001.json").write_text(
        '{"epoch": 1, "overall": {"accuracy": 1, "precision": 0.5, "recall": 0.25, "f1": 0.33, "pr_auc": "nan"}, "per_scenario": {"a": {"f1": 0.8}}}',
        encoding="utf-8",
    )
    (run_dir / "test_metrics_epoch_0002.json").write_text('{"epoch": 2,', encoding="utf-8")

    assert training_progress.latest_training_point(run_dir)["epoch"] == 1
    curves = training_progress.build_curves(job_id="train-x", run_dir=run_dir)

    assert curves["training_points"] == [valid_point]
    assert curves["training_epoch_points"] == [valid_point]
    assert curves["test_points"] == [
        {
            "epoch": 1,
            "accuracy": 1.0,
            "precision": 0.5,
            "recall": 0.25,
            "f1": 0.33,
            "pr_auc": 0.0,
            "per_scenario": {"a": {"f1": 0.8}},
        }
    ]
