from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from piern.simulators.gcam import pipeline as gcam_pipeline
from piern.simulators.power_flow import pipeline as power_flow_pipeline
from piern.simulators.simpeg import pipeline as simpeg_pipeline
from piern.simulators.transient import pipeline as transient_pipeline


class _DummyConverter:
    param_names = [f"u{i}" for i in range(18)]

    def convert(self, _scenario: str, _params: dict[str, float]) -> list[float]:
        return [float(i) for i in range(18)]


@pytest.mark.parametrize(
    ("module", "converter_name", "filter_name", "output_channels"),
    [
        (simpeg_pipeline, "SimPEGParamConverter", "filter_simpeg_dataset", 1),
        (gcam_pipeline, "GCAMParamConverter", "filter_gcam_dataset", 5),
        (power_flow_pipeline, "PowerFlowParamConverter", "filter_dataset", 3),
        (transient_pipeline, "TransientParamConverter", "filter_dataset", 2),
    ],
)
def test_seed_ratio_one_tops_up_after_filtering(
    monkeypatch,
    tmp_path: Path,
    module,
    converter_name: str,
    filter_name: str,
    output_channels: int,
) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "data" / "sim"
    config = {
        "n_samples": 3,
        "seed": 7,
        "output_dir": str(output_dir),
        "output_file": "sim_case.h5",
        "augmentation": {"seed_ratio": 1.0},
        "validation": {},
        "params": {},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    batch_sizes: list[int] = []
    saved: dict[str, object] = {}

    def fake_generate_batch(_cfg, n_samples, **_kwargs):
        batch_sizes.append(int(n_samples))
        if len(batch_sizes) == 1:
            n = 1
            start = 0
        else:
            n = 2
            start = 1
        timeseries = np.arange(start, start + n * output_channels * 4, dtype=np.float32).reshape(
            n,
            output_channels,
            4,
        )
        params = np.arange(start, start + n, dtype=np.float32).reshape(n, 1)
        return timeseries, params, ["p0"]

    def fake_filter(timeseries, params, *_args, **_kwargs):
        return timeseries, params, {"n_original": len(timeseries), "n_valid": len(timeseries)}

    def fake_save_dataset(path, timeseries, params, param_names, metadata):
        saved["path"] = path
        saved["timeseries_shape"] = tuple(timeseries.shape)
        saved["params_shape"] = tuple(params.shape)
        saved["param_names"] = list(param_names)
        saved["metadata_json"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)

    monkeypatch.setattr(module, "generate_batch", fake_generate_batch)
    monkeypatch.setattr(module, filter_name, fake_filter)
    monkeypatch.setattr(module, converter_name, _DummyConverter)
    monkeypatch.setattr(module, "save_dataset", fake_save_dataset)

    result = module.run_pipeline(str(config_path), n_samples=3)

    assert result == str(output_dir / "sim_case.h5")
    assert batch_sizes == [3, 2]
    assert saved["timeseries_shape"] == (3, output_channels, 4)
    assert saved["params_shape"] == (3, 18)
    assert saved["param_names"] == _DummyConverter.param_names
    assert '"scenario_name": "case"' in str(saved["metadata_json"])
