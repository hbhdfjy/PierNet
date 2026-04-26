from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from piern.synth.api.routers.registry import _validate_registry_entry
from piern.text2comp.generator import LLMTextGenerator


def _output_info():
    return [
        {"name": "bus_voltage", "description": "voltage", "unit": "p.u.", "slice": [0, 1]},
        {"name": "line_power", "description": "line power", "unit": "MW", "slice": [1, 2]},
        {"name": "angle", "description": "angle", "unit": "rad", "slice": [2, 3]},
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
    assert np.array_equal(spec.channel_indices, np.array([1]))
