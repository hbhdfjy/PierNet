"""
电力系统统一参数转换器。

将各场景的特定参数转换为统一的18维参数向量。

统一参数结构（电力系统版）：
  核心参数（10维）：
    p1: V_base_kv        基准电压 (kV)
    p2: load_scale       负荷整体缩放因子
    p3: P_load_mean      平均有功负荷 (MW)
    p4: P_load_std       负荷波动标准差 (MW)
    p5: Q_load_ratio     功率因数（Q/P比）
    p6: P_gen_total      总发电量 (MW)
    p7: renewable_ratio  可再生能源占比
    p8: fault_duration   故障持续时间 (s)
    p9: grid_voltage_pu  电网电压水平 (pu)
    p10: n_buses         节点数

  扩展参数（5维）：
    p11: inertia_mean    平均惯性常数 (s)
    p12: damping_mean    平均阻尼系数
    p13: line_loading    线路负载率
    p14: voltage_dev     电压偏差 (pu)
    p15: freq_dev        频率偏差 (Hz)

  元数据（3维）：
    m1: scenario_type    0=稳态潮流, 1=暂态稳定, 2=OPF
    m2: output_type      0=电压/相角, 1=转子角/频率
    m3: complexity       1=14节点, 2=30节点, 3=39节点, 4=118节点
"""

import numpy as np
from typing import Dict


class PowerSystemParamConverter:
    """电力系统统一参数转换器"""

    def __init__(self):
        self.param_names = [
            # 核心参数（10个）
            'V_base_kv',        # p1: 基准电压 (kV)
            'load_scale',       # p2: 负荷缩放因子
            'P_load_mean',      # p3: 平均有功负荷 (MW)
            'P_load_std',       # p4: 负荷波动标准差 (MW)
            'Q_load_ratio',     # p5: Q/P比
            'P_gen_total',      # p6: 总发电量 (MW)
            'renewable_ratio',  # p7: 可再生能源占比
            'fault_duration',   # p8: 故障持续时间 (s)
            'grid_voltage_pu',  # p9: 电网电压 (pu)
            'n_buses',          # p10: 节点数

            # 扩展参数（5个）
            'inertia_mean',     # p11: 平均惯性常数 (s)
            'damping_mean',     # p12: 平均阻尼系数
            'line_loading',     # p13: 线路负载率
            'voltage_dev',      # p14: 电压偏差 (pu)
            'freq_dev',         # p15: 频率偏差 (Hz)

            # 元数据（3个）
            'scenario_type',    # m1: 场景类型
            'output_type',      # m2: 输出类型
            'complexity',       # m3: 复杂度
        ]

        # 场景元数据映射
        self._scenario_meta = {
            # 稳态潮流场景
            'ieee14_baseload':      (0, 0, 1),
            'ieee14_renewable':     (0, 0, 1),
            'ieee30_contingency':   (0, 0, 2),
            'ieee118_dispatch':     (0, 0, 4),
            'distribution_33bus':   (0, 0, 2),
            # 暂态稳定场景
            'ieee39_fault':         (1, 1, 3),
            'ieee39_trip':          (1, 1, 3),
            'ieee14_load_step':     (1, 1, 1),
        }

        # 网络规模映射（节点数）
        self._bus_count = {
            'ieee14_baseload':    14,
            'ieee14_renewable':   14,
            'ieee30_contingency': 30,
            'ieee118_dispatch':   118,
            'distribution_33bus': 33,
            'ieee39_fault':       39,
            'ieee39_trip':        39,
            'ieee14_load_step':   14,
        }

        # 典型基准电压（kV）
        self._base_voltage = {
            'ieee14_baseload':    138.0,
            'ieee14_renewable':   138.0,
            'ieee30_contingency': 135.0,
            'ieee118_dispatch':   138.0,
            'distribution_33bus': 12.66,
            'ieee39_fault':       345.0,
            'ieee39_trip':        345.0,
            'ieee14_load_step':   138.0,
        }

        # 典型有功负荷（MW）
        self._base_load = {
            'ieee14_baseload':    259.0,
            'ieee14_renewable':   259.0,
            'ieee30_contingency': 283.4,
            'ieee118_dispatch':   4242.0,
            'distribution_33bus': 3.715,
            'ieee39_fault':       6254.2,
            'ieee39_trip':        6254.2,
            'ieee14_load_step':   259.0,
        }

    def convert(self, scenario_name: str, original_params: Dict) -> np.ndarray:
        """
        将场景特定参数转换为统一18维参数向量。

        Args:
            scenario_name: 场景名称
            original_params: 原始参数字典

        Returns:
            18维统一参数向量
        """
        meta = self._scenario_meta.get(scenario_name, (0, 0, 1))
        scenario_type, output_type, complexity = meta

        n_buses = float(self._bus_count.get(scenario_name, 14))
        V_base = self._base_voltage.get(scenario_name, 138.0)
        P_base = self._base_load.get(scenario_name, 259.0)

        load_scale = original_params.get('load_scale', 1.0)
        P_load_mean = P_base * load_scale
        P_load_std = P_load_mean * original_params.get('P_load_std', 50.0) / P_base
        Q_load_ratio = original_params.get('Q_load_ratio', 0.3)
        P_gen_total = P_load_mean * 1.05  # 发电略大于负荷
        renewable_ratio = original_params.get('renewable_ratio', 0.0)
        fault_duration = original_params.get('fault_duration', 0.0)
        grid_voltage_pu = original_params.get('grid_voltage_pu', 1.0)
        inertia_mean = original_params.get('inertia_mean', 5.0)
        damping_mean = original_params.get('damping_mean', 2.0)
        line_loading = original_params.get('line_loading', 0.6)
        voltage_dev = abs(grid_voltage_pu - 1.0)
        freq_dev = original_params.get('freq_dev', 0.0)

        return np.array([
            V_base,
            load_scale,
            P_load_mean,
            P_load_std,
            Q_load_ratio,
            P_gen_total,
            renewable_ratio,
            fault_duration,
            grid_voltage_pu,
            n_buses,
            inertia_mean,
            damping_mean,
            line_loading,
            voltage_dev,
            freq_dev,
            float(scenario_type),
            float(output_type),
            float(complexity),
        ], dtype=np.float32)

    def get_param_ranges(self) -> Dict[str, tuple]:
        """返回统一参数的合理范围（用于归一化）。"""
        return {
            'V_base_kv':       (10.0, 500.0),
            'load_scale':      (0.5, 1.5),
            'P_load_mean':     (1.0, 10000.0),
            'P_load_std':      (0.0, 1000.0),
            'Q_load_ratio':    (0.1, 0.5),
            'P_gen_total':     (1.0, 10000.0),
            'renewable_ratio': (0.0, 1.0),
            'fault_duration':  (0.0, 0.3),
            'grid_voltage_pu': (0.9, 1.1),
            'n_buses':         (14.0, 118.0),
            'inertia_mean':    (2.0, 10.0),
            'damping_mean':    (0.0, 5.0),
            'line_loading':    (0.3, 0.95),
            'voltage_dev':     (0.0, 0.1),
            'freq_dev':        (0.0, 0.5),
            'scenario_type':   (0.0, 2.0),
            'output_type':     (0.0, 1.0),
            'complexity':      (1.0, 4.0),
        }
