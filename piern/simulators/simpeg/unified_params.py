"""
SimPEG 统一参数转换器。

将4个场景的特定参数转换为统一的18维参数向量。

统一参数结构（SimPEG版）：
  核心参数（10维）：
    p1:  sigma_bg         背景电导率 (S/m)
    p2:  sigma_anomaly    异常体电导率 (S/m)
    p3:  depth_top        异常体顶部深度 (m)
    p4:  depth_bottom     异常体底部深度 (m)
    p5:  width_x          异常体水平宽度 (m)
    p6:  width_z          异常体垂向厚度 (m)
    p7:  source_spacing   收发距/回线半径 (m)
    p8:  noise_level      噪声水平（相对）
    p9:  n_layers         分层数
    p10: survey_length    测量范围 (m)

  扩展参数（5维）：
    p11: chargeability    充电率（IP场景）
    p12: time_constant    时间常数 (s)
    p13: freq_min         最低频率 (Hz，MT场景)
    p14: freq_max         最高频率 (Hz，MT场景)
    p15: mu_r             相对磁导率

  元数据（3维）：
    m1:  scenario_type    0=DC, 1=MT, 2=TEM, 3=IP
    m2:  output_type      0=视电阻率, 1=相位, 2=EMF, 3=充电率
    m3:  complexity       1-5
"""

import numpy as np
from typing import Dict


class SimPEGParamConverter:
    """SimPEG 统一参数转换器"""

    def __init__(self):
        self.param_names = [
            # 核心参数（10个）
            'sigma_bg',         # p1: 背景电导率 (S/m)
            'sigma_anomaly',    # p2: 异常体电导率 (S/m)
            'depth_top',        # p3: 异常体顶部深度 (m)
            'depth_bottom',     # p4: 异常体底部深度 (m)
            'width_x',          # p5: 水平宽度 (m)
            'width_z',          # p6: 垂向厚度 (m)
            'source_spacing',   # p7: 收发距/回线半径 (m)
            'noise_level',      # p8: 噪声水平
            'n_layers',         # p9: 分层数
            'survey_length',    # p10: 测量范围 (m)

            # 扩展参数（5个）
            'chargeability',    # p11: 充电率
            'time_constant',    # p12: 时间常数 (s)
            'freq_min',         # p13: 最低频率 (Hz)
            'freq_max',         # p14: 最高频率 (Hz)
            'mu_r',             # p15: 相对磁导率

            # 元数据（3个）
            'scenario_type',    # m1: 场景类型
            'output_type',      # m2: 输出变量类型
            'complexity',       # m3: 复杂度
        ]

        self._scenario_meta = {
            'dc_resistivity':   (0, 0, 2),
            'mt_sounding':      (1, 1, 3),
            'tem_decay':        (2, 2, 3),
            'ip_chargeability': (3, 3, 3),
        }

    def convert(self, scenario_name: str, original_params: Dict) -> np.ndarray:
        """
        将场景特定参数转换为统一18维参数向量。

        Args:
            scenario_name:   场景名称
            original_params: 原始参数字典

        Returns:
            18维统一参数向量
        """
        meta = self._scenario_meta.get(scenario_name, (0, 0, 1))
        scenario_type, output_type, complexity = meta

        # 计算 width_x/width_z（从 depth_top/depth_bottom 派生）
        depth_top = original_params.get('depth_top', 10.0)
        depth_bottom = original_params.get('depth_bottom', 50.0)
        width_z = depth_bottom - depth_top
        width_x = width_z * 2  # 假设水平宽度是垂向厚度的2倍

        return np.array([
            # 核心参数
            original_params.get('sigma_bg', 0.01),
            original_params.get('sigma_anomaly', 0.1),
            depth_top,
            depth_bottom,
            width_x,
            width_z,
            original_params.get('source_spacing', 50.0),
            original_params.get('noise_level', 0.02),
            3.0,  # n_layers（固定3层：表层+异常+半空间）
            original_params.get('survey_length', 1000.0),

            # 扩展参数
            original_params.get('chargeability', 0.0),
            original_params.get('time_constant', 0.0),
            original_params.get('freq_min', 1e-3),
            original_params.get('freq_max', 1e3),
            1.0,  # mu_r（非磁性介质）

            # 元数据
            float(scenario_type),
            float(output_type),
            float(complexity),
        ], dtype=np.float32)

    def get_param_ranges(self) -> Dict[str, tuple]:
        """返回统一参数的合理范围（用于归一化）。"""
        return {
            'sigma_bg':       (1e-4, 1.0),
            'sigma_anomaly':  (1e-4, 10.0),
            'depth_top':      (1.0, 2000.0),
            'depth_bottom':   (5.0, 10000.0),
            'width_x':        (2.0, 20000.0),
            'width_z':        (1.0, 10000.0),
            'source_spacing': (1.0, 500.0),
            'noise_level':    (0.0, 0.1),
            'n_layers':       (2.0, 5.0),
            'survey_length':  (10.0, 10000.0),
            'chargeability':  (0.0, 0.5),
            'time_constant':  (0.0, 1.0),
            'freq_min':       (1e-4, 1.0),
            'freq_max':       (1.0, 1e4),
            'mu_r':           (1.0, 10.0),
            'scenario_type':  (0.0, 3.0),
            'output_type':    (0.0, 3.0),
            'complexity':     (1.0, 5.0),
        }
