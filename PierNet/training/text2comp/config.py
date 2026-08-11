"""
文生计算模块训练配置

架构说明：
- Text2Comp模型：文本 → 数值预测（生成专家模型的输入参数）
- 专家模型：接收Text2Comp输出 → 计算最终物理结果

关键概念：
- expert_input_dim：Text2Comp输出维度 = 专家模型输入维度
- expert_output_dim：专家模型最终输出维度

示例流程（FNO专家）：
  用户文本 → Text2Comp(输出128维) → FNO(输入128维,输出64维) → 最终结果

支持：
- 任意专家模型类型（FNO, MODFLOW, Custom等）
- 不同base LLM（Qwen2.5-0.5B等）
- 可配置的MLP回归头结构
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpertModelInfo:
    """
    专家模型信息配置

    用于定义：
    1. Text2Comp的输出维度（专家模型需要的输入参数维度）
    2. 专家模型的类型和配置
    3. Prompt模板生成
    """

    # === 基本信息 ===
    name: str                          # 专家模型名称 (唯一标识)
    domain: str = ""                   # 物理领域描述，如 "1D Diffusion-Sorption PDE"

    # === 专家模型类型 ===
    expert_type: str = "generic"       # 专家模型类型: FNO, MODFLOW, PowerFlow, Custom等
    expert_class: str = ""             # 专家模型类名（用于动态加载）
    expert_weights_path: str = ""      # 专家模型预训练权重路径

    # === 维度配置（核心）===
    # Text2Comp输出维度 = 专家模型输入维度
    expert_input_dim: int = 0          # 专家模型需要的输入参数维度（Text2Comp预测目标）
    expert_output_dim: int = 0         # 专家模型最终输出维度

    # 维度计算参数（用于从HDF5自动推断）
    spatial_points: int = 0            # 空间网格点数
    time_steps: int = 0                # 输入时间帧数
    channels: int = 1                  # 通道数
    # expert_input_dim = spatial_points × time_steps × channels

    # === Prompt生成配置 ===
    input_description: str = ""        # 输入数据描述，如 "过去2帧的64个网格点浓度数据"
    output_description: str = ""       # 输出描述，如 "下一帧的状态预测"
    param_names: list[str] = field(default_factory=list)  # 物理参数名称列表

    # 自定义模板
    prompt_template: str = ""          # 用户输入的prompt模板
    system_prompt: str = ""            # system prompt

    # === 专家模型特有配置 ===
    expert_config: dict[str, Any] = field(default_factory=dict)
    # FNO示例: {"modes": 16, "width": 64, "initial_step": 2}
    # MODFLOW示例: {"grid_size": 100, "n_layers": 3}

    # === 数据路径 ===
    data_path: str = ""                # HDF5数据路径
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_input_dim(self) -> int:
        """
        根据空间/时间/通道参数计算专家输入维度

        公式: expert_input_dim = spatial_points × time_steps × channels
        """
        if self.expert_input_dim > 0:
            return self.expert_input_dim
        if self.spatial_points > 0 and self.time_steps > 0:
            return self.spatial_points * self.time_steps * self.channels
        return 0

    def validate(self) -> list[str]:
        """验证配置完整性，返回错误列表"""
        errors = []
        if not self.name:
            errors.append("name is required")
        if self.expert_input_dim <= 0 and self.compute_input_dim() <= 0:
            errors.append("expert_input_dim or (spatial_points × time_steps × channels) must be > 0")
        return errors


@dataclass
class Text2CompTrainingConfig:
    """
    Text2Comp训练配置

    Text2Comp模型：LLM + MLP回归头
    输入：文本描述
    输出：专家模型需要的数值参数（维度 = expert_input_dim）
    """

    # === 任务配置 ===
    task_name: str = "text2comp_train"
    simulator: str = ""                 # 模拟器/专家模型名称
    expert_info: ExpertModelInfo | None = None

    # === 模型配置 ===
    base_model_path: str = ""           # 预训练LLM路径 (如Qwen2.5-0.5B-Instruct)
    freeze_base_model: bool = False     # xhb reference fine-tunes the base model together with the head
    trainable_base_layers: int = 0      # >0时只训练LLM最后N层；0表示按freeze_base_model决定全冻/全训
    hidden_size: int = 0                # LLM hidden size (自动推断)

    # 输出维度 = 专家模型输入维度
    output_dim: int = 0                 # 同expert_info.expert_input_dim

    # MLP回归头结构
    head_layers: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    head_activation: str = "relu"       # 激活函数: relu, gelu
    head_dropout: float = 0.0           # xhb reference head has no dropout

    # === 训练配置 ===
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.0
    loss_fn: str = "mse"                # 损失函数: mse, mae, huber
    max_length: int = 2048              # 最大序列长度
    normalize_labels: bool = False      # 按训练集统计量标准化回归目标
    min_samples: int = 2                # 正式训练所需最少有效样本
    min_epochs: int = 1                 # 达到目标前至少训练的轮数
    early_stop_patience: int = 0        # 0表示关闭基于验证集停滞的早停
    target_normalized_rmse: float = 0.15
    max_normalized_rmse: float = 0.25
    require_quality: bool = False       # 未达到max_normalized_rmse时任务失败

    # === 数据配置 ===
    train_data_path: str = ""           # 训练数据路径 (JSONL格式: {prompt, label})
    test_data_path: str = ""            # 测试数据路径
    test_ratio: float = 0.1             # 测试集比例（若无单独测试数据）
    skip_invalid: bool = True           # 跳过无效样本

    # === 运行配置 ===
    device: str = "cuda:0"
    num_workers: int = 4
    seed: int = 42
    use_ddp: bool = False               # 是否使用分布式训练

    # === 输出配置 ===
    output_dir: str = "artifacts/text2comp"
    run_name: str = ""
    save_best_only: bool = True
    eval_interval: int = 10             # 评估间隔 (epochs)
    log_interval: int = 20              # 日志间隔 (steps)

    # === 恢复训练 ===
    resume_from: str | None = None

    def __post_init__(self):
        if not self.run_name:
            self.run_name = self.task_name
        # 自动同步output_dim
        if self.expert_info and self.output_dim == 0:
            self.output_dim = self.expert_info.compute_input_dim()


# === 预定义的专家模型配置 ===
# 注意：expert_input_dim是Text2Comp输出维度（即专家模型输入维度）

PDEBENCH_EXPERTS = {
    "diff-sorp": ExpertModelInfo(
        name="diff-sorp",
        domain="1D Diffusion-Sorption PDE",
        expert_type="FNO",
        expert_class="FNO1d",
        # FNO输入：64点 × 2帧 = 128维
        expert_input_dim=128,           # Text2Comp输出128维给FNO
        expert_output_dim=64,           # FNO最终输出64维
        spatial_points=64,
        time_steps=2,
        channels=1,
        input_description="过去2帧的64个网格点浓度数据",
        output_description="下一帧的64个网格点浓度预测",
        expert_config={"modes": 16, "width": 64, "initial_step": 2},
        prompt_template="这是1d_diff-sorp任务，请根据以下经过处理的过去{time_steps}帧{spatial_points}个网格点数据，预测下一帧的状态。\n数据如下：\n{data_str}",
    ),
    "burgers": ExpertModelInfo(
        name="burgers",
        domain="1D Burgers Equation",
        expert_type="FNO",
        expert_class="FNO1d",
        expert_input_dim=32,            # Text2Comp输出32维
        expert_output_dim=32,
        spatial_points=32,
        time_steps=1,
        channels=1,
        input_description="过去帧的速度场数据",
        output_description="下一帧的速度场预测",
        expert_config={"modes": 16, "width": 64, "initial_step": 1},
    ),
    "diff-reaction": ExpertModelInfo(
        name="diff-reaction",
        domain="1D Diffusion-Reaction PDE",
        expert_type="FNO",
        expert_input_dim=32,
        expert_output_dim=32,
        spatial_points=32,
        time_steps=1,
        input_description="过去帧的反应浓度数据",
        output_description="下一帧的反应浓度预测",
    ),
}

PIERN_EXPERTS = {
    "modflow": ExpertModelInfo(
        name="modflow",
        domain="Groundwater flow simulation (MODFLOW)",
        expert_type="MODFLOW",
        expert_class="MODFLOWModel",
        # MODFLOW输入：5观测井 × 12时间点 = 60维（示例）
        expert_input_dim=60,
        expert_output_dim=60,           # 输出同样维度的水头预测
        spatial_points=5,
        time_steps=12,
        channels=1,
        input_description="含水层参数（渗透系数、补给率等）",
        output_description="观测井的水力水头时序预测",
        param_names=["K_mean", "R_recharge", "Q_pumping"],
        expert_config={"grid_size": 100, "n_layers": 3},
    ),
    "power_flow": ExpertModelInfo(
        name="power_flow",
        domain="Power system steady-state flow",
        expert_type="PowerFlow",
        expert_class="PowerFlowModel",
        expert_input_dim=43,            # IEEE 14节点参数
        expert_output_dim=43,           # 节点电压和线路功率
        input_description="IEEE 14-bus系统负荷参数",
        output_description="节点电压和线路功率",
    ),
}

# 通用专家模型模板（用于自定义专家）
GENERIC_EXPERT_TEMPLATE = ExpertModelInfo(
    name="generic",
    domain="Custom Physics Simulation",
    expert_type="Custom",
    expert_input_dim=0,                # 需用户指定
    expert_output_dim=0,
    input_description="物理参数数据",
    output_description="物理预测结果",
)


def get_expert_info(simulator: str) -> ExpertModelInfo:
    """获取预定义的专家模型信息"""
    all_experts = {**PDEBENCH_EXPERTS, **PIERN_EXPERTS}
    if simulator in all_experts:
        return all_experts[simulator]
    # 返回通用模板，让用户自定义
    if simulator == "generic" or simulator == "custom":
        return GENERIC_EXPERT_TEMPLATE
    raise ValueError(
        f"Unknown simulator: {simulator}. "
        f"Available: {list(all_experts.keys())}, or use 'generic' for custom experts"
    )


def create_custom_expert_info(
    name: str,
    expert_input_dim: int,
    domain: str = "",
    expert_type: str = "Custom",
    **kwargs
) -> ExpertModelInfo:
    """
    创建自定义专家模型配置

    Args:
        name: 专家模型名称
        expert_input_dim: Text2Comp输出维度（专家模型输入维度）
        domain: 物理领域描述
        expert_type: 专家模型类型
        **kwargs: 其他ExpertModelInfo字段

    Returns:
        ExpertModelInfo实例
    """
    return ExpertModelInfo(
        name=name,
        domain=domain,
        expert_type=expert_type,
        expert_input_dim=expert_input_dim,
        **kwargs
    )
