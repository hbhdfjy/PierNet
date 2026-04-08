"""
MODFLOW 统一参数表示转换器。

将25个不同场景的特定参数转换为统一的18维参数向量。
"""

import numpy as np
from typing import Dict, List


class UnifiedParamConverter:
    """统一参数转换器"""

    def __init__(self):
        """初始化参数名称列表"""
        self.param_names = [
            # 核心参数（10个）
            'K_mean',           # p1: 平均水力传导系数 (m/day)
            'K_std',            # p2: 水力传导系数标准差 (m/day)
            'K_anisotropy',     # p3: 水平/垂向各向异性比 (-)
            'S_storage',        # p4: 储水系数 (-)
            'Q_pumping',        # p5: 抽水量（负）/注水量（正）(m³/day)
            'H_initial',        # p6: 初始水头 (m)
            'R_recharge',       # p7: 平均补给量 (m/day)
            'R_variation',      # p8: 补给量变化幅度 (-)
            'BC_strength',      # p9: 边界条件强度 (m)
            'n_layers',         # p10: 含水层层数 (-)

            # 扩展参数（5个）
            'C_concentration',  # p11: 污染物浓度/温度 (mg/L or °C)
            'D_dispersion',     # p12: 弥散系数/热扩散系数 (m or W/m/K)
            'lambda_decay',     # p13: 衰减/冷却系数 (/day or 1/J/kg/K)
            'phi_porosity',     # p14: 孔隙度 (-)
            'rho_density',      # p15: 密度 (kg/m³)

            # 元数据（3个）
            'scenario_type',    # m1: 场景类型 (0-5)
            'output_type',      # m2: 输出变量类型 (0-2)
            'complexity',       # m3: 复杂度等级 (1-5)
        ]

        # 场景类型编码
        self.scenario_types = {
            'basic': 0,
            'multilayer': 1,
            'heterogeneous': 2,
            'boundary': 3,
            'seasonal': 4,
            'special': 5,
        }

        # 输出类型编码
        self.output_types = {
            'head': 0,
            'concentration': 1,
            'temperature': 2,
        }

    def get_scenario_category(self, scenario_name: str) -> tuple:
        """
        根据场景名称返回场景类别。

        Returns:
            (scenario_type, output_type, complexity)
        """
        # 基础场景（1个）
        basic_scenarios = ['unified_aquifer']

        # 多层场景（1个）
        multilayer_scenarios = ['multilayer_confined']

        # 非均质场景（1个）
        heterogeneous_scenarios = ['heterogeneous_unconfined']

        # 边界场景（2个）
        boundary_scenarios = ['river_recharge', 'lake_interaction']

        # 季节场景（1个）
        seasonal_scenarios = ['monsoon_seasonal']

        # 特殊场景（4个）
        special_scenarios = ['coastal_seawater', 'land_subsidence', 'contaminant_transport', 'geothermal_reservoir']

        if scenario_name in basic_scenarios:
            return (0, 0, 1)  # 基础，水头，简单
        elif scenario_name in multilayer_scenarios:
            return (1, 0, 3)  # 多层，水头，中等复杂
        elif scenario_name in heterogeneous_scenarios:
            return (2, 0, 3)  # 非均质，水头，中等
        elif scenario_name in boundary_scenarios:
            return (3, 0, 2)  # 边界，水头，中等
        elif scenario_name in seasonal_scenarios:
            return (4, 0, 2)  # 季节，水头，中等
        elif scenario_name == 'coastal_seawater':
            return (5, 0, 4)  # 特殊，水头，复杂
        elif scenario_name == 'land_subsidence':
            return (5, 0, 5)  # 特殊，水头，很复杂
        elif scenario_name == 'contaminant_transport':
            return (5, 1, 4)  # 特殊，浓度，复杂
        elif scenario_name == 'geothermal_reservoir':
            return (5, 2, 5)  # 特殊，温度，很复杂
        else:
            return (0, 0, 1)  # 默认为基础场景

    def convert(self, scenario_name: str, original_params: Dict) -> np.ndarray:
        """
        将场景特定参数转换为统一参数表示。

        Args:
            scenario_name: 场景名称
            original_params: 原始参数字典

        Returns:
            18维统一参数向量
        """
        # 获取场景类别
        scenario_type, output_type, complexity = self.get_scenario_category(scenario_name)

        # 根据场景类型选择转换方法
        if scenario_type == 0:  # 基础场景
            return self._convert_basic(original_params, complexity)
        elif scenario_type == 1:  # 多层场景
            return self._convert_multilayer(original_params, scenario_name, complexity)
        elif scenario_type == 2:  # 非均质场景
            return self._convert_heterogeneous(original_params, complexity)
        elif scenario_type == 3:  # 边界场景
            return self._convert_boundary(original_params, scenario_name, complexity)
        elif scenario_type == 4:  # 季节场景
            return self._convert_seasonal(original_params, complexity)
        elif scenario_type == 5:  # 特殊场景
            return self._convert_special(original_params, scenario_name, output_type, complexity)
        else:
            return self._convert_basic(original_params, complexity)

    def _convert_basic(self, params: Dict, complexity: int) -> np.ndarray:
        """转换基础场景参数"""
        return np.array([
            # 核心参数
            params.get('hk', 10.0),
            0.0,  # K_std (均质)
            1.0,  # K_anisotropy (各向同性)
            params.get('sy', 0.15),
            params.get('pumping', -200.0),
            params.get('strt', 7.0),
            params.get('rch', 0.001),
            0.0,  # R_variation (恒定补给)
            0.0,  # BC_strength (无特殊边界)
            1.0,  # n_layers (单层)

            # 扩展参数（默认值）
            0.0,      # C_concentration
            0.0,      # D_dispersion
            0.0,      # lambda_decay
            0.30,     # phi_porosity (默认)
            1000.0,   # rho_density (水密度)

            # 元数据
            0.0,      # scenario_type (基础)
            0.0,      # output_type (水头)
            float(complexity),
        ], dtype=np.float32)

    def _convert_multilayer(self, params: Dict, scenario_name: str, complexity: int) -> np.ndarray:
        """转换多层场景参数"""
        # 提取各层参数
        n_layers = 3

        hk_layers = []
        strt_layers = []
        for i in range(1, n_layers + 1):
            hk_layers.append(params.get(f'hk_layer{i}', 10.0))
            strt_layers.append(params.get(f'strt_layer{i}', 7.0))

        # 计算统计量
        K_mean = np.mean(hk_layers)
        K_std = np.std(hk_layers)
        vka = params.get('vka', 1.0)
        K_anisotropy = K_mean / vka if vka > 0 else 1.0

        return np.array([
            # 核心参数
            K_mean,
            K_std,
            K_anisotropy,
            params.get('sy', 0.15),
            params.get('pumping', -300.0),
            np.mean(strt_layers),
            params.get('rch', 0.001),
            0.0,  # R_variation
            0.0,  # BC_strength
            float(n_layers),

            # 扩展参数
            0.0,
            0.0,
            0.0,
            0.30,
            1000.0,

            # 元数据
            1.0,  # scenario_type (多层)
            0.0,  # output_type (水头)
            float(complexity),
        ], dtype=np.float32)

    def _convert_heterogeneous(self, params: Dict, complexity: int) -> np.ndarray:
        """转换非均质场景参数"""
        # 对数正态分布参数转换
        hk_mean_log = params.get('hk_mean_log', 1.0)
        hk_std_log = params.get('hk_std_log', 0.5)

        K_mean = 10 ** hk_mean_log
        K_std = K_mean * (10 ** hk_std_log - 1)  # 近似标准差

        correlation_length = params.get('hk_correlation_length', 500.0)
        anisotropy_ratio = params.get('anisotropy_ratio', 2.0)

        return np.array([
            # 核心参数
            K_mean,
            K_std,
            anisotropy_ratio,
            params.get('sy', 0.15),
            params.get('pumping', -200.0),
            params.get('strt', 7.0),
            params.get('rch', 0.001),
            0.0,  # R_variation
            0.0,  # BC_strength
            1.0,  # n_layers

            # 扩展参数
            0.0,
            correlation_length / 100.0,  # 归一化相关长度
            0.0,
            0.30,
            1000.0,

            # 元数据
            2.0,  # scenario_type (非均质)
            0.0,  # output_type (水头)
            float(complexity),
        ], dtype=np.float32)

    def _convert_boundary(self, params: Dict, scenario_name: str, complexity: int) -> np.ndarray:
        """转换边界场景参数"""
        # 提取边界参数
        if 'river' in scenario_name:
            BC_strength = params.get('river_stage', 9.0)
            D_dispersion = params.get('river_cond', 100.0) / 100.0  # 归一化
        elif 'lake' in scenario_name:
            BC_strength = params.get('lake_stage', 9.0)
            D_dispersion = params.get('lake_cond', 100.0) / 100.0
        else:
            BC_strength = 0.0
            D_dispersion = 0.0

        return np.array([
            # 核心参数
            params.get('hk', 10.0),
            0.0,  # K_std
            1.0,  # K_anisotropy
            params.get('sy', 0.15),
            params.get('pumping', -200.0),
            params.get('strt', 7.0),
            params.get('rch', 0.001),
            0.0,  # R_variation
            BC_strength,
            1.0,  # n_layers

            # 扩展参数
            0.0,
            D_dispersion,
            0.0,
            0.30,
            1000.0,

            # 元数据
            3.0,  # scenario_type (边界)
            0.0,  # output_type (水头)
            float(complexity),
        ], dtype=np.float32)

    def _convert_seasonal(self, params: Dict, complexity: int) -> np.ndarray:
        """转换季节场景参数"""
        # 提取季节性参数
        rch_wet = params.get('rch_wet_season', 0.005)
        rch_dry = params.get('rch_dry_season', 0.0002)
        wet_duration = params.get('wet_season_duration', 180)

        R_recharge = (rch_wet + rch_dry) / 2.0
        R_variation = (rch_wet - rch_dry) / R_recharge if R_recharge > 0 else 0.0

        return np.array([
            # 核心参数
            params.get('hk', 10.0),
            0.0,
            1.0,
            params.get('sy', 0.15),
            params.get('pumping', -200.0),
            params.get('strt', 7.0),
            R_recharge,
            R_variation,
            0.0,  # BC_strength
            1.0,  # n_layers

            # 扩展参数
            0.0,
            0.0,
            365.0 / wet_duration,  # 周期频率
            0.30,
            1000.0,

            # 元数据
            4.0,  # scenario_type (季节)
            0.0,  # output_type (水头)
            float(complexity),
        ], dtype=np.float32)

    def _convert_special(self, params: Dict, scenario_name: str, output_type: int, complexity: int) -> np.ndarray:
        """转换特殊场景参数"""
        if scenario_name in ('seawater_intrusion', 'coastal_seawater'):
            return self._convert_seawater(params, complexity)
        elif scenario_name == 'land_subsidence':
            return self._convert_subsidence(params, complexity)
        elif scenario_name == 'contaminant_transport':
            return self._convert_contaminant(params, complexity)
        elif scenario_name == 'geothermal_reservoir':
            return self._convert_geothermal(params, complexity)
        else:
            return self._convert_basic(params, complexity)

    def _convert_seawater(self, params: Dict, complexity: int) -> np.ndarray:
        """转换海水入侵场景"""
        return np.array([
            params.get('hk', 10.0),
            0.0,
            1.0,
            params.get('sy', 0.15),
            params.get('pumping', -200.0),
            params.get('strt', 7.0),
            params.get('rch', 0.001),
            0.0,
            params.get('coastal_boundary_head', 0.0),
            1.0,

            # 扩展参数
            0.0,
            0.0,
            0.0,
            0.30,
            params.get('saltwater_density', 1025.0),

            # 元数据
            5.0,  # scenario_type (特殊)
            0.0,  # output_type (水头)
            float(complexity),
        ], dtype=np.float32)

    def _convert_subsidence(self, params: Dict, complexity: int) -> np.ndarray:
        """转换地面沉降场景"""
        # 多层系统
        hk_layers = [params.get(f'hk_layer{i}', 10.0) for i in range(1, 4)]
        strt_layers = [params.get(f'strt_layer{i}', 7.0) for i in range(1, 4)]

        elastic_storage = params.get('elastic_storage', 0.001)
        inelastic_storage = params.get('inelastic_storage', 0.01)
        S_storage = (elastic_storage + inelastic_storage) / 2.0

        return np.array([
            np.mean(hk_layers),
            np.std(hk_layers),
            np.mean(hk_layers) / params.get('vka', 1.0),
            S_storage,
            params.get('pumping', -500.0),
            np.mean(strt_layers),
            params.get('rch', 0.001),
            0.0,
            params.get('preconsolidation_head', 5.0),
            3.0,  # n_layers

            # 扩展参数
            0.0,
            0.0,
            0.0,
            0.30,
            1000.0,

            # 元数据
            5.0,
            0.0,
            float(complexity),
        ], dtype=np.float32)

    def _convert_contaminant(self, params: Dict, complexity: int) -> np.ndarray:
        """转换污染物运移场景"""
        return np.array([
            params.get('hk', 10.0),
            0.0,
            1.0,
            params.get('sy', 0.15),
            params.get('pumping', -200.0),
            params.get('strt', 7.0),
            params.get('rch', 0.001),
            0.0,
            0.0,
            1.0,

            # 扩展参数
            params.get('contaminant_source_conc', 500.0),
            params.get('dispersivity_long', 20.0),
            params.get('decay_rate', 0.001),
            params.get('porosity', 0.30),
            1000.0,

            # 元数据
            5.0,
            1.0,  # output_type (浓度)
            float(complexity),
        ], dtype=np.float32)

    def _convert_geothermal(self, params: Dict, complexity: int) -> np.ndarray:
        """转换地热储层场景"""
        thermal_conductivity = params.get('thermal_conductivity', 2.0)
        heat_capacity = params.get('heat_capacity', 4200.0)

        return np.array([
            params.get('hk', 10.0),
            0.0,
            1.0,
            params.get('sy', 0.15),
            params.get('pumping', -300.0),
            params.get('strt', 7.0),
            params.get('rch', 0.001),
            0.0,
            0.0,
            2.0,  # n_layers

            # 扩展参数
            params.get('temperature_initial', 80.0),
            thermal_conductivity,
            1.0 / heat_capacity if heat_capacity > 0 else 0.0,
            params.get('porosity', 0.30),
            params.get('fluid_density', 1000.0),

            # 元数据
            5.0,
            2.0,  # output_type (温度)
            float(complexity),
        ], dtype=np.float32)

    def get_param_ranges(self) -> Dict[str, tuple]:
        """
        返回每个统一参数的合理范围（用于归一化）。

        Returns:
            参数名 -> (min, max) 字典
        """
        return {
            'K_mean': (0.1, 100.0),
            'K_std': (0.0, 50.0),
            'K_anisotropy': (1.0, 100.0),
            'S_storage': (0.01, 0.40),
            'Q_pumping': (-1500.0, 500.0),
            'H_initial': (0.5, 48.0),
            'R_recharge': (0.00001, 0.010),
            'R_variation': (0.0, 1.0),
            'BC_strength': (0.0, 20.0),
            'n_layers': (1.0, 5.0),
            'C_concentration': (0.0, 1000.0),
            'D_dispersion': (0.0, 50.0),
            'lambda_decay': (0.0, 0.1),
            'phi_porosity': (0.15, 0.45),
            'rho_density': (1000.0, 1100.0),
            'scenario_type': (0.0, 5.0),
            'output_type': (0.0, 2.0),
            'complexity': (1.0, 5.0),
        }
