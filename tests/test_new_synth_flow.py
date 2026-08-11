from __future__ import annotations

import json
import time

import h5py
import numpy as np
from PierNet.new_synth import paths, service, store, training_bridge
from PierNet.training.services import training_manager


def _configure_roots(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    runlog_root = tmp_path / "runlogs"
    cache_root = tmp_path / "cache"
    database = runlog_root / "state.sqlite"
    monkeypatch.setattr(paths, "NEW_SYNTH_DATA_ROOT", data_root)
    monkeypatch.setattr(paths, "NEW_SYNTH_RUNLOG_ROOT", runlog_root)
    monkeypatch.setattr(paths, "NEW_SYNTH_CACHE_ROOT", cache_root)
    monkeypatch.setattr(paths, "NEW_SYNTH_DB_PATH", database)
    monkeypatch.setattr(store, "NEW_SYNTH_DB_PATH", database)
    monkeypatch.setattr(service, "NEW_SYNTH_CACHE_ROOT", cache_root)
    monkeypatch.setattr(service, "MIN_FREE_DISK_BYTES", 0)
    monkeypatch.setattr(store, "_INITIALIZED", False)
    return data_root


def _write_hdf5(path):
    params = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ],
        dtype=np.float32,
    )
    timeseries = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("params", data=params)
        handle.create_dataset("timeseries", data=timeseries)
        handle.create_dataset("param_names", data=np.asarray([b"alpha", b"beta"]))
    return params, timeseries


def _wait_for_terminal(owner_id: str, workflow_id: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = service.get_workflow(owner_id, workflow_id)
        if snapshot["status"] in {"succeeded", "failed", "cancelled"}:
            return snapshot
        time.sleep(0.05)
    raise AssertionError("new-synth generation did not finish")


def test_new_synth_generates_training_contracts_and_registry(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    source_path = tmp_path / "source.h5"
    params, timeseries = _write_hdf5(source_path)
    owner_id = "a" * 32

    workflow = service.create_workflow(owner_id, "结构预测数据")
    workflow = service.attach_uploaded_hdf5(owner_id, workflow["workflow_id"], source_path, "source.h5")

    assert workflow["source"]["ready"] is True
    assert workflow["source"]["input_dim"] == 2
    assert workflow["source"]["output_shape"] == [2, 4]
    with h5py.File(paths.workflow_paths(workflow["workflow_id"]).canonical / "data.h5", "r") as handle:
        assert int(handle.attrs["n_samples"]) == 6
        assert int(handle.attrs["n_params"]) == 2

    definition = dict(workflow["definition"])
    definition.update(
        {
            "simulator": "mechanics",
            "scenario": "column_buckling",
            "task_description": "预测柱屈曲响应",
            "sampling": {"channels": [0, 1], "time_stride": 2, "max_time_points": 2},
        }
    )
    workflow = service.save_definition(owner_id, workflow["workflow_id"], definition)
    assert workflow["can_generate"] is True

    service.start_generation(
        owner_id,
        workflow["workflow_id"],
        {"max_samples": 4, "variants_per_sample": 2, "negative_ratio": 1, "seed": 7},
    )
    finished = _wait_for_terminal(owner_id, workflow["workflow_id"])

    assert finished["status"] == "succeeded", finished.get("error")
    assert finished["artifacts"]["text2comp"]["sample_count"] == 8
    assert finished["artifacts"]["router"]["sample_count"] == 16
    assert finished["artifacts"]["text2comp"]["label_semantics"] == "expert_input"

    datasets = store.list_datasets(workflow_id=workflow["workflow_id"])
    assert {item["kind"] for item in datasets} == {"text2comp", "router", "evaluation"}
    text_dataset = next(item for item in datasets if item["kind"] == "text2comp")
    router_dataset = next(item for item in datasets if item["kind"] == "router")
    evaluation_dataset = next(item for item in datasets if item["kind"] == "evaluation")
    assert text_dataset["paired_dataset_id"] == router_dataset["dataset_id"]
    assert router_dataset["paired_dataset_id"] == text_dataset["dataset_id"]

    text_rows = [json.loads(line) for line in open(text_dataset["path"], encoding="utf-8")]
    assert all(row["schema_name"] == "piernet.text2comp" for row in text_rows)
    assert all(row["metadata"]["label_semantics"] == "expert_input" for row in text_rows)
    assert all(row["label"] == row["expert_input"] for row in text_rows)
    assert all(not np.array_equal(row["label"], timeseries.reshape(6, -1)[0]) for row in text_rows)
    assert all(any(np.allclose(row["label"], candidate) for candidate in params) for row in text_rows)

    router_rows = [json.loads(line) for line in open(router_dataset["path"], encoding="utf-8")]
    assert {row["label"] for row in router_rows} == {0, 1}
    assert all(row["metadata"]["class_names"] == ["not_target", "target"] for row in router_rows)

    evaluation_rows = [json.loads(line) for line in open(evaluation_dataset["path"], encoding="utf-8")]
    assert evaluation_rows[0]["schema_name"] == "piernet.expert-evaluation"
    assert "expected_expert_output" in evaluation_rows[0]
    assert "expected_expert_output" not in text_rows[0]

    listed_router = training_bridge.list_router_datasets()
    listed_text2comp = training_bridge.list_text2comp_datasets()
    assert [item["dataset_id"] for item in listed_router] == [router_dataset["dataset_id"]]
    assert [item["dataset_id"] for item in listed_text2comp] == [text_dataset["dataset_id"]]

    dataset_id, simulator, scenarios, router_root = training_manager._resolve_router_training_source(
        {"dataset_id": router_dataset["dataset_id"]}
    )
    assert dataset_id == router_dataset["dataset_id"]
    assert simulator == "mechanics"
    assert scenarios == ["column_buckling"]
    assert router_root == paths.workflow_paths(workflow["workflow_id"]).router

    prepared = training_manager._prepare_simple_text2comp_dataset(
        {
            "job_id": "test-simple-pipeline",
            "dataset_id": router_dataset["dataset_id"],
            "config": {"dataset_id": router_dataset["dataset_id"]},
        }
    )
    assert prepared["path"] == text_dataset["path"]
    assert prepared["dataset_id"] == text_dataset["dataset_id"]
    assert prepared["output_dim"] == 2
    assert prepared["target_source"] == "expert_input"


def test_definition_suggestion_only_fills_human_readable_metadata(monkeypatch, tmp_path):
    _configure_roots(monkeypatch, tmp_path)
    source_path = tmp_path / "source.h5"
    _write_hdf5(source_path)
    owner_id = "b" * 32
    workflow = service.create_workflow(owner_id, "智能识别数据")
    workflow = service.attach_uploaded_hdf5(owner_id, workflow["workflow_id"], source_path, "source.h5")
    draft = json.loads(json.dumps(workflow["definition"]))
    draft["parameters"][1]["display_name"] = "用户自定义参数"
    draft["parameters"][1]["description"] = "用户已经确认的含义"

    monkeypatch.setattr(
        service,
        "_request_definition_suggestion",
        lambda _definition, _source: {
            "task_description": "预测结构在给定载荷下的响应",
            "parameters": [
                {
                    "index": 0,
                    "name": "alpha",
                    "display_name": "材料系数",
                    "description": "控制材料响应的输入系数",
                    "unit": "",
                },
                {
                    "index": 1,
                    "name": "beta",
                    "display_name": "不应覆盖",
                    "description": "不应覆盖",
                    "unit": "MPa",
                },
            ],
            "outputs": [
                {
                    "index": 0,
                    "name": "output_1",
                    "display_name": "位移响应",
                    "description": "第一个物理输出通道",
                    "unit": "mm",
                }
            ],
        },
    )

    suggested = service.suggest_definition(owner_id, workflow["workflow_id"], draft)

    assert suggested["task_description"] == "预测结构在给定载荷下的响应"
    assert [(item["index"], item["name"]) for item in suggested["parameters"]] == [
        (0, "alpha"),
        (1, "beta"),
    ]
    assert suggested["parameters"][0]["display_name"] == "材料系数"
    assert suggested["parameters"][0]["description"] == "控制材料响应的输入系数"
    assert suggested["parameters"][1]["display_name"] == "用户自定义参数"
    assert suggested["parameters"][1]["description"] == "用户已经确认的含义"
    assert suggested["parameters"][1]["unit"] == "MPa"
    assert suggested["outputs"][0]["display_name"] == "位移响应"
    assert suggested["sampling"] == draft["sampling"]


def test_cache_cleanup_never_touches_canonical_data(monkeypatch, tmp_path):
    data_root = _configure_roots(monkeypatch, tmp_path)
    canonical = data_root / "new-synth-deadbeefcafe" / "canonical" / "data.h5"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical")
    cache_file = service.NEW_SYNTH_CACHE_ROOT / "prepared" / "cache.bin"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"cache")
    old = time.time() - service.CACHE_TTL_SECONDS - 10
    import os

    os.utime(cache_file, (old, old))

    result = service.cleanup_expired_caches(now=time.time())

    assert result["deleted_files"] == 1
    assert not cache_file.exists()
    assert canonical.read_bytes() == b"canonical"
