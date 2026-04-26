from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from piern.synth.api.routers.registry import _validate_registry_entry
from piern.synth.text2comp.generator import LLMTextGenerator
from piern.synth.text2comp.template_store import OutputSlot, TemplateRecord, fill_sample


def _output_info():
    return [
        {"name": "bus_voltage", "description": "voltage", "unit": "p.u.", "slice": [0, 1]},
        {"name": "line_power", "description": "line power", "unit": "MW", "slice": [1, 2]},
        {"name": "angle", "description": "angle", "unit": "rad", "slice": [2, 3]},
    ]


def _power_output_info():
    return [
        {"name": "bus_voltages", "description": "bus voltage magnitudes", "unit": "p.u.", "slice": [0, 14]},
        {"name": "voltage_angles", "description": "bus voltage angles", "unit": "rad", "slice": [14, 28]},
        {"name": "line_power_flows", "description": "line active power flows", "unit": "MW", "slice": [28, 43]},
    ]


def _domain(channel_level="row", fixed_channels=None):
    return {
        "observation_config": {
            "fixed_time_mode": "full",
            "fixed_channels": fixed_channels,
            "channel_level": channel_level,
            "time_modes": [
                {
                    "name": "full",
                    "indices": "full",
                    "desc_en": "full time series, all time points",
                    "desc_zh": "full time series",
                }
            ],
            "channel_name_template": "channel {i}",
            "channel_name_template_zh": "channel {i}",
        }
    }


def test_registry_rejects_empty_fixed_channels():
    body = {
        "output_info": _output_info(),
        "observation_config": {"channel_level": "row", "fixed_channels": []},
    }

    with pytest.raises(HTTPException) as exc:
        _validate_registry_entry("demo", body)

    assert exc.value.status_code == 400


def test_registry_accepts_output_info_name_selection_and_normalizes_level():
    body = {
        "output_info": _output_info(),
        "observation_config": {"channel_level": "output", "fixed_channels": ["line_power"]},
    }

    _validate_registry_entry("demo", body)

    assert body["observation_config"]["channel_level"] == "output_info"


def test_registry_rejects_unknown_output_info_name():
    body = {
        "output_info": _output_info(),
        "observation_config": {"channel_level": "output_info", "fixed_channels": ["missing"]},
    }

    with pytest.raises(HTTPException) as exc:
        _validate_registry_entry("demo", body)

    assert exc.value.status_code == 400


def test_sample_observation_row_level_keeps_full_output_schema():
    generator = LLMTextGenerator(llm_client=object())
    output_info = [_output_info()[0]]

    spec = generator._sample_observation(
        domain=_domain(channel_level="row", fixed_channels=[0, 2]),
        timeseries_shape=(5, 8),
        output_info=output_info,
    )

    assert spec.selected_output_info == output_info
    assert np.array_equal(spec.channel_indices, np.array([0, 2]))


def test_sample_observation_output_info_level_selects_named_dimension():
    generator = LLMTextGenerator(llm_client=object())

    spec = generator._sample_observation(
        domain=_domain(channel_level="output_info", fixed_channels=["line_power"]),
        timeseries_shape=(3, 8),
        output_info=_output_info(),
    )

    assert [item["name"] for item in spec.selected_output_info] == ["line_power"]
    assert spec.selected_output_info[0]["slice"] == [0, 1]
    assert np.array_equal(spec.channel_indices, np.array([1]))


def test_sample_observation_output_info_level_expands_output_slices():
    generator = LLMTextGenerator(llm_client=object())

    spec = generator._sample_observation(
        domain=_domain(channel_level="output_info", fixed_channels=["bus_voltages", "line_power_flows"]),
        timeseries_shape=(43, 8),
        output_info=_power_output_info(),
    )

    assert [item["name"] for item in spec.selected_output_info] == ["bus_voltages", "line_power_flows"]
    assert spec.selected_output_info[0]["slice"] == [0, 14]
    assert spec.selected_output_info[1]["slice"] == [14, 29]
    assert np.array_equal(
        spec.channel_indices,
        np.array(list(range(0, 14)) + list(range(28, 43))),
    )


def test_fill_sample_metadata_uses_compact_selected_output_slices():
    template = TemplateRecord(
        input_template="predict",
        target_template="result {output_0}{output_1}",
        placeholder_schema=[],
        output_schema=[
            OutputSlot(index=0, name="bus_voltages"),
            OutputSlot(index=1, name="line_power_flows"),
        ],
        transform_descs=[],
        simulator="power_flow",
        scenario="demo",
        language="en",
        style="technical",
        time_mode="full",
        n_time_points=2,
        time_indices=[0, 1],
        channel_indices=list(range(0, 14)) + list(range(28, 43)),
        selected_output_names=["bus_voltages", "line_power_flows"],
        timeseries_shape_orig=[43, 2],
        timeseries_shape_obs=[29, 2],
        param_names=[],
    )

    sample = fill_sample(
        template,
        params=np.array([], dtype=float),
        timeseries_obs=np.zeros((29, 2), dtype=float),
        sample_idx=0,
        output_info=_power_output_info(),
    )

    meta_output = sample["metadata"]["output_info"]
    assert [item["name"] for item in meta_output] == ["bus_voltages", "line_power_flows"]
    assert meta_output[0]["slice"] == [0, 14]
    assert meta_output[1]["slice"] == [14, 29]
