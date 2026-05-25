from __future__ import annotations

import numpy as np
import pytest

from PierNet.synth.text2comp.template_store import (
    OutputSlot,
    PlaceholderSlot,
    TemplateRecord,
    TransformDesc,
    fill_sample,
)


def _template(**overrides) -> TemplateRecord:
    values = {
        "input_template": "value {value_0}",
        "target_template": "out {output_0}",
        "placeholder_schema": [
            PlaceholderSlot(
                index=0,
                param_name="alpha",
                param_index=0,
                use_transformed=False,
                fmt="scalar",
            )
        ],
        "output_schema": [OutputSlot(index=0, name="out")],
        "transform_descs": [
            TransformDesc(
                param_name="alpha",
                param_index=0,
                transform_type=None,
                factor=None,
                note_en="",
                note_zh="",
            )
        ],
        "simulator": "simpeg",
        "scenario": "case",
        "language": "en",
        "style": "technical",
        "time_mode": "full",
        "n_time_points": 2,
        "time_indices": [0, 1],
        "channel_indices": [0],
        "selected_output_names": ["out"],
        "timeseries_shape_orig": [1, 2],
        "timeseries_shape_obs": [1, 2],
        "param_names": ["alpha", "beta"],
    }
    values.update(overrides)
    return TemplateRecord(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"transform_descs": [TransformDesc("beta", -1, None, None, "", "")]},
        {
            "placeholder_schema": [
                PlaceholderSlot(0, "beta", -1, use_transformed=False, fmt="scalar")
            ]
        },
        {
            "placeholder_schema": [
                PlaceholderSlot(0, "gamma", 2, use_transformed=False, fmt="scalar")
            ]
        },
    ],
)
def test_fill_sample_rejects_invalid_param_indices(overrides: dict) -> None:
    with pytest.raises(ValueError, match="无效参数索引"):
        fill_sample(_template(**overrides), np.array([1.0, 2.0]), np.array([[3.0, 4.0]]), 0)
