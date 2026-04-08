"""
GCAM 统一参数转换器。

将各场景的特定参数转换为统一的18维参数向量。

统一参数结构（GCAM版）：
  核心参数（10维）：
    p1:  solar_cost_init     太阳能初始成本 ($/MWh)
    p2:  cost_learning_rate  学习曲线斜率
    p3:  carbon_tax          碳税 ($/tCO2)
    p4:  gdp_growth          GDP增长率 (%/年)
    p5:  population_growth   人口增长率 (%/年)
    p6:  energy_intensity    能源强度 (GJ/GDP)
    p7:  fossil_reserve      化石燃料储量指数
    p8:  climate_sensitivity 气候敏感度 (K/CO2 doubling)
    p9:  discount_rate       折现率
    p10: n_regions           地区数（固定=3）

  扩展参数（5维）：
    p11: nuclear_cost        核电成本 ($/MWh)
    p12: ccs_cost            碳捕获成本 ($/tCO2)
    p13: ev_penetration      电动车渗透率
    p14: industrial_eff      工业效率改进
    p15: land_use_change     土地利用变化排放 (GtCO2/yr)

  元数据（3维）：
    m1:  scenario_type       0=能源转型, 1=碳定价, 2=气候反馈
    m2:  output_type         0=能源结构, 1=排放+温度
    m3:  complexity          1=简单, 2=中等, 3=复杂
"""

import numpy as np
from typing import Dict


class GCAMParamConverter:
    """GCAM 统一参数转换器"""

    def __init__(self):
        self.param_names = [
            # 核心参数（10个）
            'solar_cost_init',     # p1
            'cost_learning_rate',  # p2
            'carbon_tax',          # p3
            'gdp_growth',          # p4
            'population_growth',   # p5
            'energy_intensity',    # p6
            'fossil_reserve',      # p7
            'climate_sensitivity', # p8
            'discount_rate',       # p9
            'n_regions',           # p10

            # 扩展参数（5个）
            'nuclear_cost',        # p11
            'ccs_cost',            # p12
            'ev_penetration',      # p13
            'industrial_eff',      # p14
            'land_use_change',     # p15

            # 元数据（3个）
            'scenario_type',       # m1
            'output_type',         # m2
            'complexity',          # m3
        ]

        self._scenario_meta = {
            'energy_transition': (0, 0, 2),
            'carbon_pricing':    (1, 1, 2),
            'climate_feedback':  (2, 1, 3),
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

        return np.array([
            # 核心参数
            original_params.get('solar_cost_init', 50.0),
            original_params.get('cost_learning_rate', 0.2),
            original_params.get('carbon_tax', 0.0),
            original_params.get('gdp_growth', 2.5),
            original_params.get('population_growth', 0.8),
            original_params.get('energy_intensity', 1.0),
            original_params.get('fossil_reserve', 1.0),
            original_params.get('climate_sensitivity', 3.0),
            original_params.get('discount_rate', 0.05),
            3.0,  # n_regions 固定为3

            # 扩展参数
            original_params.get('nuclear_cost', 100.0),
            original_params.get('ccs_cost', 100.0),
            original_params.get('ev_penetration', 0.0),
            original_params.get('industrial_eff', 0.0),
            original_params.get('land_use_change', 1.0),

            # 元数据
            float(scenario_type),
            float(output_type),
            float(complexity),
        ], dtype=np.float32)

    def get_param_ranges(self) -> Dict[str, tuple]:
        """返回统一参数的合理范围（用于归一化）。"""
        return {
            'solar_cost_init':     (30.0, 200.0),
            'cost_learning_rate':  (0.1, 0.3),
            'carbon_tax':          (0.0, 200.0),
            'gdp_growth':          (1.0, 5.0),
            'population_growth':   (0.0, 2.0),
            'energy_intensity':    (0.5, 2.0),
            'fossil_reserve':      (0.5, 2.0),
            'climate_sensitivity': (1.5, 4.5),
            'discount_rate':       (0.03, 0.10),
            'n_regions':           (1.0, 5.0),
            'nuclear_cost':        (50.0, 300.0),
            'ccs_cost':            (50.0, 300.0),
            'ev_penetration':      (0.0, 1.0),
            'industrial_eff':      (0.0, 0.5),
            'land_use_change':     (-2.0, 5.0),
            'scenario_type':       (0.0, 2.0),
            'output_type':         (0.0, 1.0),
            'complexity':          (1.0, 3.0),
        }
