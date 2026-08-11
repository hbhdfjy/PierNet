from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from PierNet.training.api.routers import assembly
from PierNet.training.services import training_manager
from PierNet.training.text2comp.data import PromptNumbersDataset
from PierNet.training.text2comp import train
from PierNet.training.text2comp import text2comp_manager


def test_group_split_keeps_prompt_variants_together() -> None:
    group_ids = ["sample-a", "sample-a", "sample-b", "sample-b", "sample-c", "sample-c"]

    class Dataset:
        def __len__(self) -> int:
            return 6

    Dataset.group_ids = group_ids

    train_indices, test_indices = train._split_indices_by_group(Dataset(), test_ratio=0.34, seed=42)
    train_groups = {Dataset.group_ids[index] for index in train_indices}
    test_groups = {Dataset.group_ids[index] for index in test_indices}

    assert train_groups
    assert test_groups
    assert train_groups.isdisjoint(test_groups)


def test_simple_dataset_uses_rendered_parameters_instead_of_row_number(monkeypatch, tmp_path) -> None:
    data_root = tmp_path / "data"
    h5_path = data_root / "gcam" / "gcam_carbon_pricing.h5"
    h5_path.parent.mkdir(parents=True)
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("params", data=np.asarray([[12.5, 3.0], [18.0, 4.0]], dtype=np.float32))
        handle.create_dataset("timeseries", data=np.zeros((2, 1, 1), dtype=np.float32))
    template_path = data_root / "templates" / "carbon_pricing_templates.jsonl"
    template_path.parent.mkdir(parents=True)
    template_path.write_text("{}\n", encoding="utf-8")

    template = SimpleNamespace(scenario="carbon_pricing", channel_indices=[0], time_indices=[0])
    monkeypatch.setattr(training_manager, "DATA_ROOT", data_root)
    monkeypatch.setattr(training_manager, "load_templates", lambda path: [template])
    monkeypatch.setattr(
        training_manager,
        "fill_sample",
        lambda template, params, timeseries, sample_index: {
            "input": f"carbon_tax={params[0]:.1f}; growth={params[1]:.1f}",
            "params_transformed": params.tolist(),
        },
    )
    entry = {
        "job_id": "train-formal-data",
        "simulator": "gcam",
        "scenarios": ["carbon_pricing"],
        "config": {"simple_text2comp_max_samples": 10},
    }

    result = training_manager._prepare_simple_text2comp_dataset(entry)
    with open(result["path"], encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    assert result["generated"] == 2
    assert result["target_source"] == "params_transformed"
    assert rows[0]["prompt"] == "carbon_tax=12.5; growth=3.0"
    assert rows[0]["label"] == [12.5, 3.0]
    assert rows[0]["metadata"]["sample_index"] == 0


def test_label_statistics_are_scale_aware() -> None:
    dataset = SimpleNamespace(
        samples=[
            ("a", torch.tensor([10.0, 1000.0])),
            ("b", torch.tensor([20.0, 3000.0])),
        ]
    )

    statistics = train._label_statistics(dataset, [0, 1], enabled=True)

    assert statistics["enabled"] is True
    assert torch.allclose(statistics["mean"], torch.tensor([15.0, 2000.0]))
    assert torch.allclose(statistics["scale"], torch.tensor([5.0, 1000.0]))


def test_text2comp_quality_failure_is_not_marked_done(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_bytes(b"best")
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "quality_passed": False,
                "best_metrics": {"normalized_rmse": 0.8},
                "quality_gate": {"required_max": 0.25},
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "train.log"
    log_path.write_text("quality failed\n", encoding="utf-8")
    monkeypatch.setattr(text2comp_manager, "_pid_alive", lambda pid: False)
    entry = {
        "job_id": "text2comp-quality-failed",
        "name": "quality failed",
        "status": "running",
        "simulator": "uploaded_expert",
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "pid": 123,
        "config": {},
    }

    refreshed = text2comp_manager._refresh_entry(entry)

    assert refreshed["status"] == "error"
    assert refreshed["quality_passed"] is False
    assert "质量未达到" in refreshed["error_message"]


def test_router_quality_gate_blocks_text2comp_launch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        training_manager,
        "_start_simple_text2comp_stage",
        lambda entry: (_ for _ in ()).throw(AssertionError("Text2Comp must not start")),
    )
    entry = {
        "job_id": "train-low-router-quality",
        "name": "low quality",
        "status": "done",
        "log_path": str(tmp_path / "router.log"),
        "config": {
            "simple_pipeline_enabled": True,
            "simple_quality_gate_enabled": True,
            "simple_router_min_f1": 0.95,
        },
        "latest_metrics": {"f1": 0.7},
        "simple_pipeline": {"stage": "router"},
    }

    refreshed = training_manager._sync_simple_pipeline(entry)

    assert refreshed["status"] == "error"
    assert refreshed["simple_pipeline"]["router_metrics"] == {"f1": 0.7}
    assert "Router 训练已结束" in refreshed["error_message"]


def test_assembly_retokenizes_raw_text_with_text2comp_contract(monkeypatch) -> None:
    class Tokenizer:
        prompt = ""

        def __call__(self, prompt, **kwargs):
            self.prompt = prompt
            return {
                "input_ids": torch.full((1, 4), 7, dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.text2comp_tokenizer = Tokenizer()
            self.text2comp_max_length = 512
            self.seen_input_ids = None

        def forward(self, input_ids, attention_mask):
            self.seen_input_ids = input_ids.detach().cpu()
            return torch.zeros((1, 18), device=input_ids.device)

    model = Model()
    monkeypatch.setattr(assembly, "_select_text2comp_for_simulator", lambda simulator: (simulator, model))
    monkeypatch.setitem(assembly._LOADED_MODELS, "text2comp_device", torch.device("cpu"))
    monkeypatch.setitem(assembly._LOADED_MODELS, "llm_device", torch.device("cpu"))
    monkeypatch.setitem(assembly._LOADED_MODELS, "expert_executor", "fno")
    monkeypatch.setitem(assembly._LOADED_MODELS, "fno", {})

    assembly.expert_generate_response(
        "gcam",
        torch.full((1, 3), 999, dtype=torch.long),
        torch.ones((1, 3), dtype=torch.long),
        raw_text="carbon_tax=96",
    )

    assert torch.equal(model.seen_input_ids, torch.full((1, 4), 7, dtype=torch.long))
    assert model.text2comp_tokenizer.prompt == PromptNumbersDataset.wrap_prompt_text("carbon_tax=96")
