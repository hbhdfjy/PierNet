"""Power-flow specific conversion to the persisted 18-D parameter vector."""

from __future__ import annotations

from typing import Dict

import numpy as np


class PowerFlowParamConverter:
    """Convert pandapower scenario parameters into the stable 18-D schema."""

    def __init__(self) -> None:
        self.param_names = [
            "V_base_kv",
            "load_scale",
            "P_load_mean",
            "P_load_std",
            "Q_load_ratio",
            "P_gen_total",
            "renewable_ratio",
            "fault_duration",
            "grid_voltage_pu",
            "n_buses",
            "inertia_mean",
            "damping_mean",
            "line_loading",
            "voltage_dev",
            "freq_dev",
            "scenario_type",
            "output_type",
            "complexity",
        ]

        self._scenario_complexity = {
            "ieee14_baseload": 1,
            "ieee14_renewable": 1,
            "ieee14_peak": 1,
            "ieee14_light": 1,
            "ieee14_voltage_stress": 1,
            "ieee30_contingency": 2,
            "distribution_33bus": 2,
            "ieee118_dispatch": 4,
        }
        self._bus_count = {
            "ieee14_baseload": 14,
            "ieee14_renewable": 14,
            "ieee14_peak": 14,
            "ieee14_light": 14,
            "ieee14_voltage_stress": 14,
            "ieee30_contingency": 30,
            "distribution_33bus": 33,
            "ieee118_dispatch": 118,
        }
        self._base_voltage = {
            "ieee14_baseload": 138.0,
            "ieee14_renewable": 138.0,
            "ieee14_peak": 138.0,
            "ieee14_light": 138.0,
            "ieee14_voltage_stress": 138.0,
            "ieee30_contingency": 135.0,
            "distribution_33bus": 12.66,
            "ieee118_dispatch": 138.0,
        }
        self._base_load = {
            "ieee14_baseload": 259.0,
            "ieee14_renewable": 259.0,
            "ieee14_peak": 259.0,
            "ieee14_light": 259.0,
            "ieee14_voltage_stress": 259.0,
            "ieee30_contingency": 283.4,
            "distribution_33bus": 3.715,
            "ieee118_dispatch": 4242.0,
        }

    def convert(self, scenario_name: str, original_params: Dict) -> np.ndarray:
        n_buses = float(original_params.get("_n_bus", self._bus_count.get(scenario_name, 14)))
        v_base = self._base_voltage.get(scenario_name, 138.0)
        p_base = self._base_load.get(scenario_name, 259.0)
        load_scale = float(original_params.get("load_scale", 1.0))
        p_load_mean = p_base * load_scale
        p_load_std = float(original_params.get("P_load_std", 50.0)) * load_scale
        q_load_ratio = float(original_params.get("Q_load_ratio", 0.3))
        p_gen_total = p_load_mean * 1.05
        renewable_ratio = float(original_params.get("renewable_ratio", 0.0))
        fault_duration = float(original_params.get("fault_duration", 0.0))
        grid_voltage_pu = float(original_params.get("grid_voltage_pu", 1.0))
        inertia_mean = float(original_params.get("inertia_mean", 5.0))
        damping_mean = float(original_params.get("damping_mean", 2.0))
        line_loading = float(original_params.get("line_loading", 0.6))
        voltage_dev = abs(grid_voltage_pu - 1.0)
        freq_dev = float(original_params.get("freq_dev", 0.0))
        complexity = float(self._scenario_complexity.get(scenario_name, 1))

        return np.array(
            [
                v_base,
                load_scale,
                p_load_mean,
                p_load_std,
                q_load_ratio,
                p_gen_total,
                renewable_ratio,
                fault_duration,
                grid_voltage_pu,
                n_buses,
                inertia_mean,
                damping_mean,
                line_loading,
                voltage_dev,
                freq_dev,
                0.0,  # scenario_type: steady-state power flow
                0.0,  # output_type: voltage/angle/line-flow
                complexity,
            ],
            dtype=np.float32,
        )

    def get_param_ranges(self) -> Dict[str, tuple]:
        return {
            "V_base_kv": (10.0, 500.0),
            "load_scale": (0.5, 1.5),
            "P_load_mean": (1.0, 10000.0),
            "P_load_std": (0.0, 1000.0),
            "Q_load_ratio": (0.1, 0.5),
            "P_gen_total": (1.0, 10000.0),
            "renewable_ratio": (0.0, 1.0),
            "fault_duration": (0.0, 0.3),
            "grid_voltage_pu": (0.9, 1.1),
            "n_buses": (14.0, 118.0),
            "inertia_mean": (2.0, 10.0),
            "damping_mean": (0.0, 5.0),
            "line_loading": (0.3, 0.95),
            "voltage_dev": (0.0, 0.1),
            "freq_dev": (0.0, 0.5),
            "scenario_type": (0.0, 0.0),
            "output_type": (0.0, 0.0),
            "complexity": (1.0, 4.0),
        }
