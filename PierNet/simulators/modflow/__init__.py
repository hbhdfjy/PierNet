"""
MODFLOW 地下水模拟器。

提供 MODFLOW 数据生成、参数空间采样增强和 Stage 1 pipeline。
"""

from PierNet.simulators.modflow.generator import generate_sample, generate_batch
from PierNet.simulators.modflow.generator_with_params import (
    generate_sample_from_params,
    generate_batch_from_params,
)
from PierNet.simulators.modflow.pipeline import (
    run_pipeline,
    perturb_params,
    augment_with_parameter_sampling,
)

__all__ = [
    "generate_sample",
    "generate_batch",
    "generate_sample_from_params",
    "generate_batch_from_params",
    "perturb_params",
    "augment_with_parameter_sampling",
    "run_pipeline",
]
