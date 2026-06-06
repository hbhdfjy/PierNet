"""
文生计算模块 (Text-to-Computation Module)

架构说明：
- Text2Comp：文本 → 数值预测（生成专家模型输入参数）
- 专家模型：接收Text2Comp输出 → 计算最终物理结果

流程：
  用户文本 → Text2Comp(输出N维) → 专家模型(输入N维) → 最终物理结果

功能：
- 支持任意专家模型（FNO, MODFLOW, PowerFlow, Custom等）
- 自动生成训练数据
- 小参数LLM + MLP回归头
- 一键训练流程
"""

from .config import (
    Text2CompTrainingConfig,
    ExpertModelInfo,
    get_expert_info,
    create_custom_expert_info,
    PDEBENCH_EXPERTS,
    PIERN_EXPERTS,
    GENERIC_EXPERT_TEMPLATE,
)
from .data import PromptNumbersDataset, HDF5DataGenerator, generate_text2comp_data
from .model import Text2CompModel, RegressionHead, create_text2comp_model
from .train import run_training, prepare_data, evaluate
from .text2comp_manager import (
    EXPERT_MODEL_LIBRARY,
    DEFAULT_TRAINING_CONFIG,
    list_jobs,
    get_job,
    get_gpu_inventory,
    list_simulators,
    list_datasets,
    get_overview,
    validate_training_data,
    create_job,
    stop_job,
    delete_job,
    get_job_logs,
    get_curves,
    register_custom_expert,
    get_all_experts,
)
from .data_converter import (
    convert_jsonl,
    convert_sample,
    extract_numbers_from_target,
    compute_output_dim_from_h5,
)
from .run_pipeline import (
    run_full_pipeline,
    step_auto_register,
    step_generate_templates,
    step_fill_samples,
    step_convert_data,
    step_train,
)

__all__ = [
    # 配置
    "Text2CompTrainingConfig",
    "ExpertModelInfo",
    "get_expert_info",
    "create_custom_expert_info",
    "PDEBENCH_EXPERTS",
    "PIERN_EXPERTS",
    "GENERIC_EXPERT_TEMPLATE",
    # 数据
    "PromptNumbersDataset",
    "HDF5DataGenerator",
    "generate_text2comp_data",
    # 模型
    "Text2CompModel",
    "RegressionHead",
    "create_text2comp_model",
    # 训练
    "run_training",
    "prepare_data",
    "evaluate",
    # 管理
    "EXPERT_MODEL_LIBRARY",
    "DEFAULT_TRAINING_CONFIG",
    "list_jobs",
    "get_job",
    "get_gpu_inventory",
    "list_simulators",
    "list_datasets",
    "get_overview",
    "validate_training_data",
    "create_job",
    "stop_job",
    "delete_job",
    "get_job_logs",
    "get_curves",
    "register_custom_expert",
    "get_all_experts",
    # 数据转换
    "convert_jsonl",
    "convert_sample",
    "extract_numbers_from_target",
    "compute_output_dim_from_h5",
    # 一键流程
    "run_full_pipeline",
    "step_auto_register",
    "step_generate_templates",
    "step_fill_samples",
    "step_convert_data",
    "step_train",
]