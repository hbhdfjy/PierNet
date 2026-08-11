from __future__ import annotations

import json

from PierNet.training.api.routers import assembly


def test_text2comp_metadata_recovers_gcam_from_legacy_simple_training_path(tmp_path):
    run_dir = tmp_path / "text2comp-expert_model_gcam_train_3b5aa638_20260729_193344"
    run_dir.mkdir()
    model_path = run_dir / "final_model.pt"
    model_path.write_bytes(b"model")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "task_name": "expert_model_train",
                "simulator": "expert_model",
                "output_dim": 18,
            }
        ),
        encoding="utf-8",
    )

    simulator, output_dim, name = assembly._text2comp_metadata(str(model_path))

    assert simulator == "gcam"
    assert output_dim == 18
    assert name == "gcam_train-3b5aa638"


def test_platform_router_registry_uses_binary_target_labels(monkeypatch, tmp_path):
    artifact_root = tmp_path / "token_router"
    run_dir = artifact_root / "gcam" / "runs" / "train-12345678"
    run_dir.mkdir(parents=True)
    (run_dir / "router_final.pt").write_bytes(b"final")
    (run_dir / "router_latest.pt").write_bytes(b"latest")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "training": {
                    "simulator": "gcam",
                    "scenarios": ["carbon_pricing", "energy_transition"],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(assembly, "_ROUTER_ARTIFACTS_ROOT", artifact_root)
    monkeypatch.setattr(assembly, "_load_config", lambda: {"router_dir": str(tmp_path / "legacy")})

    models = assembly._scan_router()

    by_name = {model["name"]: model for model in models}
    assert by_name["train-12345678"]["class_names"] == ["not_target", "target"]
    assert by_name["train-12345678_latest"]["class_names"] == ["not_target", "target"]
    assert "GCAM" in by_name["train-12345678"]["description"]
