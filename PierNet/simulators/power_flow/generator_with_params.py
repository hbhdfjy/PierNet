"""Generate pandapower samples from explicit parameter rows."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from PierNet.simulators.power_flow.generator import _run_powerflow_365d

logger = logging.getLogger(__name__)


def generate_batch_from_params(
    params_array: np.ndarray,
    param_names: List[str],
    cfg: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    ts_list = []
    params_list = []
    rng = np.random.default_rng()
    scenario = cfg.get("scenario_name", "ieee14_baseload")

    for param_row in params_array:
        params_dict = {name: float(param_row[j]) for j, name in enumerate(param_names)}
        try:
            result = _run_powerflow_365d(scenario, params_dict, rng)
            if result is None:
                continue
            ts = np.concatenate(
                [result["V_bus"], result["theta_bus"], result["P_line"]],
                axis=0,
            )
            if ts is not None:
                ts_list.append(ts)
                params_list.append(param_row)
        except Exception as exc:
            logger.debug("潮流增强样本生成失败: %s", exc)

    return ts_list, params_list
