"""
基于 PyPSA 的能源转型仿真器。

使用 PyPSA（https://github.com/PyPSA/PyPSA）进行多期能源系统规划，
替代自实现的 SimpleGCAM 模型。

数学基础：
  PyPSA 求解线性规划（LP）：
    min  Σ_t Σ_g (c_g(t) · p_g(t) + C_g · p̄_g)
    s.t. Σ_g p_g(t) = D(t)  ∀t        [供需平衡]
         p_g(t) ≤ p̄_g       ∀g,t      [容量约束]
         p_g(t) ≥ p_g_min   ∀g,t      [最小出力约束]

  其中：
    c_g(t) = c_g0 · (1 - lr_g)^(t/5)  [学习曲线]
    D(t) = D0 · (1 + γ)^(t/5)          [需求增长]

多样性保证：
  通过参数化 p_nom_min（最小容量约束）强制多技术共存，
  不同 fossil_reserve、solar_cost_init、carbon_tax 参数
  产生不同的技术组合和排放路径。

输出变量（5个 × 16时间步，2025-2100年，5年间隔）：
  1. coal_share(t)       煤炭发电份额
  2. renewable_share(t)  可再生能源份额（风+太阳能）
  3. co2_emission(t)     CO2排放 (GtCO2/yr，近似值)
  4. energy_price(t)     加权平均能源价格 ($/MWh)
  5. temperature(t)      温度异常 (°C)，基于简化气候模型
"""

import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 技术排放因子 (tCO2/MWh)
EMISSION_FACTOR = {
    'coal':    0.820,
    'gas':     0.490,
    'nuclear': 0.012,
    'wind':    0.011,
    'solar':   0.020,
}

# 大气 CO2 基准值
CO2_PREINDUSTRIAL = 280.0
CO2_INITIAL = 420.0       # 2025年 (ppm)
TEMP_INITIAL = 1.1        # 2025年温度异常 (°C)
THERMAL_INERTIA = 50.0    # 气候热惯性 (年)
N_YEARS = 16              # 时间步数 (2025-2100, 5年间隔)


def run_pypsa_energy_transition(
    params: Dict[str, float],
    scenario: str = 'energy_transition',
) -> Optional[np.ndarray]:
    """
    使用 PyPSA 运行多期能源系统规划。

    Args:
        params: 参数字典（见 unified_params.py 中的参数定义）
        scenario: 场景名称（影响约束设置）

    Returns:
        输出数组 (5, 16)；失败返回 None
    """
    try:
        import pypsa
        import pandas as pd
        import warnings
        import logging as _logging
        warnings.filterwarnings('ignore')
        _logging.getLogger('pypsa').setLevel(_logging.ERROR)
        _logging.getLogger('linopy').setLevel(_logging.ERROR)
    except ImportError:
        raise ImportError("请安装 PyPSA：pip install pypsa")

    # 提取参数
    solar_cost_init   = params.get('solar_cost_init', 50.0)
    cost_lr           = params.get('cost_learning_rate', 0.20)
    carbon_tax        = params.get('carbon_tax', 0.0)
    gdp_growth        = params.get('gdp_growth', 2.5) / 100.0
    population_growth = params.get('population_growth', 0.8) / 100.0
    energy_intensity  = params.get('energy_intensity', 1.0)
    fossil_reserve    = params.get('fossil_reserve', 1.0)
    climate_sens      = params.get('climate_sensitivity', 3.0)
    nuclear_cost      = params.get('nuclear_cost', 100.0)
    ccs_cost          = params.get('ccs_cost', 100.0)
    ev_penetration    = params.get('ev_penetration', 0.0)
    industrial_eff    = params.get('industrial_eff', 0.0)
    land_use_change   = params.get('land_use_change', 1.0)

    years = list(range(2025, 2025 + N_YEARS * 5, 5))
    demand_base = 600.0  # EJ/yr 等效

    # 计算各年需求（GDP + 人口 + 效率改进 + EV）
    demand = []
    for i, y in enumerate(years):
        dt = (y - 2025) / 5  # 以5年为单位的时间步数
        d = (demand_base
             * ((1 + gdp_growth) ** (dt * 5))
             * ((1 + population_growth) ** (dt * 5))
             * ((1 - industrial_eff) ** dt)
             * energy_intensity
             * (1 + ev_penetration * 0.1 * dt / N_YEARS))
        demand.append(np.clip(d, 100.0, 5000.0))

    max_demand = max(demand)

    # 技术参数
    techs = {
        'coal': {
            'cost0': 60.0,
            'lr': 0.02,
            'cap_cost': 500,
            'p_nom_min': demand_base * 0.05 * fossil_reserve,
            'p_nom_max': demand_base * 0.8 * fossil_reserve,
        },
        'gas': {
            'cost0': 55.0,
            'lr': 0.03,
            'cap_cost': 600,
            'p_nom_min': demand_base * 0.05 * fossil_reserve,
            'p_nom_max': demand_base * 0.7 * fossil_reserve,
        },
        'nuclear': {
            'cost0': nuclear_cost,
            'lr': 0.05,
            'cap_cost': 2000,
            'p_nom_min': demand_base * 0.02,
            'p_nom_max': demand_base * 0.5,
        },
        'wind': {
            'cost0': 45.0,
            'lr': 0.15,
            'cap_cost': 300,
            'p_nom_min': 0.0,
            'p_nom_max': max_demand * 0.6,
        },
        'solar': {
            'cost0': solar_cost_init,
            'lr': cost_lr,
            'cap_cost': 200,
            'p_nom_min': 0.0,
            'p_nom_max': max_demand * 0.8,
        },
    }

    # CCS 效果：降低煤炭有效排放成本（但增加运营成本）
    ccs_reduction = max(0.0, (200.0 - ccs_cost) / 200.0) if ccs_cost < 200 else 0.0

    # 构建 PyPSA 网络
    n = pypsa.Network()
    n.set_snapshots(pd.Index(years, name='year'))
    n.add('Bus', 'global', carrier='AC')

    for name, d in techs.items():
        # 学习曲线：成本随时间递减
        costs = []
        for i, y in enumerate(years):
            dt = (y - 2025) / 5
            c = d['cost0'] * ((1.0 - d['lr']) ** dt)
            # 碳税
            if name in ('coal', 'gas'):
                ef = EMISSION_FACTOR[name]
                tax_cost = carbon_tax * ef
                # CCS 降低煤炭碳税负担
                if name == 'coal' and ccs_reduction > 0:
                    tax_cost *= (1.0 - ccs_reduction * 0.9)
                    c += ccs_cost * ccs_reduction * 0.3  # CCS 运营成本
                c += tax_cost
            costs.append(max(c, 1.0))  # 成本下限

        n.add('Generator', name,
              bus='global',
              p_nom_extendable=True,
              p_nom_min=d['p_nom_min'],
              p_nom_max=d['p_nom_max'],
              marginal_cost=pd.Series(costs, index=years),
              capital_cost=d['cap_cost'])

    n.add('Load', 'demand', bus='global',
          p_set=pd.Series(demand, index=years))

    # 求解（静默 HiGHS 输出，避免大量日志污染 SSE 流）
    try:
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status, cond = n.optimize(solver_name='highs', solver_options={'output_flag': False})
        if cond != 'optimal':
            logger.debug(f"PyPSA 优化未收敛: {cond}")
            return None
    except Exception as e:
        logger.debug(f"PyPSA 求解失败: {e}")
        return None

    # 提取结果
    dispatch = n.generators_t.p  # (16, 5)
    total = dispatch.sum(axis=1).replace(0, np.nan)

    coal_share    = (dispatch['coal'] / total).fillna(0).values
    renew_share   = ((dispatch['wind'] + dispatch['solar']) / total).fillna(0).values

    # 加权平均能源价格
    mc_df = pd.DataFrame({
        name: [
            d['cost0'] * ((1.0 - d['lr']) ** ((y - 2025) / 5))
            + (carbon_tax * EMISSION_FACTOR[name] if name in ('coal', 'gas') else 0.0)
            for y in years
        ]
        for name, d in techs.items()
    }, index=years)
    price = ((dispatch * mc_df).sum(axis=1) / total).fillna(0).values

    # CO2 排放（GtCO2/yr，近似）
    co2_gen = np.array([
        sum(dispatch[t].iloc[i] * EMISSION_FACTOR[t] for t in EMISSION_FACTOR) / 1000.0
        for i in range(N_YEARS)
    ])
    co2_total = co2_gen + land_use_change  # 加土地利用变化排放
    co2_total = np.maximum(0.0, co2_total)

    # 简化气候模型（与 GCAM 一致）
    co2_atm = CO2_INITIAL
    temp = TEMP_INITIAL
    temps = []
    for e in co2_total:
        co2_atm += e * 5.0 * 0.45  # 5年步长，50%大气保留率
        forcing = 3.7 * np.log2(co2_atm / CO2_PREINDUSTRIAL)
        eq_temp = climate_sens * forcing / 3.7
        temp += (eq_temp - temp) * 5.0 / THERMAL_INERTIA
        temps.append(temp)

    output = np.stack([
        coal_share,
        renew_share,
        co2_total,
        price,
        np.array(temps),
    ], axis=0).astype(np.float32)

    return output


def validate_output(output: np.ndarray) -> bool:
    """验证 PyPSA 输出的物理合理性。"""
    if output is None:
        return False
    if np.any(np.isnan(output)) or np.any(np.isinf(output)):
        return False
    if np.any(output[0] < 0) or np.any(output[0] > 1):   # coal_share
        return False
    if np.any(output[1] < 0) or np.any(output[1] > 1):   # renewable_share
        return False
    if np.any(output[2] < 0):                              # co2_emission
        return False
    if np.any(output[4] < -1) or np.any(output[4] > 8):  # temperature
        return False
    return True
