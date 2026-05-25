from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from piern.synth.api.routers import registry as registry_router
from piern.synth.api.routers.registry import _registry_key_parts, _validate_registry_entry
from piern.synth.text2comp.generator import LLMTextGenerator, _get_time_indices
from piern.synth.text2comp.template_store import OutputSlot, TemplateRecord, fill_sample


def test_registry_key_parts_rejects_nested_or_invalid_components():
    with pytest.raises(HTTPException) as nested:
        _registry_key_parts("modflow/case/extra")
    assert nested.value.status_code == 400

    with pytest.raises(HTTPException) as unsafe:
        _registry_key_parts("../modflow")
    assert unsafe.value.status_code == 400


def test_registry_update_rejects_invalid_key_without_writing(monkeypatch, tmp_path):
    registry_path = tmp_path / "registry.yaml"
    monkeypatch.setattr(registry_router, "REGISTRY_PATH", registry_path)

    with pytest.raises(HTTPException) as exc:
        registry_router.update_registry_entry("modflow/case/extra", {"scenario_description": "bad"})

    assert exc.value.status_code == 400
    assert not registry_path.exists()


def test_registry_update_and_delete_normalize_valid_scenario_key(monkeypatch, tmp_path):
    registry_path = tmp_path / "registry.yaml"
    invalidated: list[str] = []
    monkeypatch.setattr(registry_router, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(
        registry_router,
        "invalidate_text2comp_scenarios_cache",
        lambda: invalidated.append("text2comp"),
    )

    saved = registry_router.update_registry_entry(" modflow / coastal ", {"scenario_description": "coast"})
    assert saved == {"ok": True, "key": "modflow/coastal"}

    data = registry_router._load_registry_raw()
    assert data["modflow"]["scenarios"] == {"coastal": "coast"}
    assert invalidated == ["text2comp"]

    deleted = registry_router.delete_registry_entry("modflow/coastal")
    assert deleted == {"ok": True, "key": "modflow/coastal"}
    assert registry_router._load_registry_raw()["modflow"] == {}
    assert invalidated == ["text2comp", "text2comp"]


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


@pytest.mark.parametrize(
    "output_info",
    [
        [{"description": "bad", "unit": "-", "slice": [0, 1]}],
        [{"name": " ", "description": "bad", "unit": "-", "slice": [0, 1]}],
        [
            {"name": "dup", "description": "first", "unit": "-", "slice": [0, 1]},
            {"name": "dup", "description": "second", "unit": "-", "slice": [1, 2]},
        ],
    ],
)
def test_registry_rejects_blank_or_duplicate_output_info_names(output_info):
    body = {
        "output_info": output_info,
        "observation_config": {"channel_level": "row", "fixed_channels": None},
    }

    with pytest.raises(HTTPException) as exc:
        _validate_registry_entry("demo", body)

    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "slice_value",
    [
        [0],
        [-1, 2],
        [1, 0],
        [True, 2],
        [0, False],
        ["0", 1],
    ],
)
def test_registry_rejects_invalid_output_info_slice(slice_value):
    body = {
        "output_info": [{"name": "bad", "description": "bad", "unit": "-", "slice": slice_value}],
        "observation_config": {"channel_level": "row", "fixed_channels": None},
    }

    with pytest.raises(HTTPException) as exc:
        _validate_registry_entry("demo", body)

    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "observation_config",
    [
        {"channel_min": 0},
        {"channel_max": 0},
        {"channel_min": True},
        {"channel_min": 3, "channel_max": 2},
    ],
)
def test_registry_rejects_invalid_channel_range(observation_config):
    body = {
        "output_info": _output_info(),
        "observation_config": {"channel_level": "row", "fixed_channels": None, **observation_config},
    }

    with pytest.raises(HTTPException) as exc:
        _validate_registry_entry("demo", body)

    assert exc.value.status_code == 400


@pytest.mark.parametrize("n_timesteps", [1, 8, 20, 38])
def test_monthly_time_indices_stay_valid_for_short_series(n_timesteps):
    indices = _get_time_indices("monthly", n_timesteps)

    assert len(indices) == min(12, n_timesteps)
    assert indices.tolist() == sorted(set(indices.tolist()))
    assert all(0 <= int(idx) < n_timesteps for idx in indices)


def test_sample_observation_rejects_empty_time_selection():
    generator = LLMTextGenerator(llm_client=object())

    with pytest.raises(ValueError, match="empty time selection"):
        generator._sample_observation(
            domain=_domain(channel_level="row", fixed_channels=None),
            timeseries_shape=(3, 0),
            output_info=_output_info(),
        )


def test_sample_observation_monthly_short_series_uses_valid_indices():
    generator = LLMTextGenerator(llm_client=object())
    domain = _domain(channel_level="row", fixed_channels=None)
    domain["observation_config"]["fixed_time_mode"] = "monthly"
    domain["observation_config"]["time_modes"][0] = {
        "name": "monthly",
        "indices": "monthly",
        "desc_en": "monthly, 12 time points",
        "desc_zh": "月度，共12个时间点",
    }

    spec = generator._sample_observation(domain=domain, timeseries_shape=(3, 20), output_info=_output_info())

    assert len(spec.time_indices) == 12
    assert all(0 <= int(idx) < 20 for idx in spec.time_indices)


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
