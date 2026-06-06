# 文生计算模块自动化训练

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PiERN完整推理流程                                   │
└─────────────────────────────────────────────────────────────────────────┘

                    用户输入
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Router (可选)               │
        │   文本 → 选择专家模型           │
        │   输出: scenario_id            │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Text2Comp (文生计算模块)     │
        │   文本 → 数值向量              │
        │   输出: N维 (专家输入参数)      │◄── 训练目标
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   专家模型 (FNO等)             │
        │   数值向量 → 物理计算           │
        │   输入: N维                    │
        │   输出: M维 (最终结果)          │
        └───────────────────────────────┘
                        │
                        ▼
                    最终输出
```

**关键概念**：Text2Comp不替代专家模型，而是**生成专家模型需要的输入参数**。

---

## 二、数据流

### 2.1 训练数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      训练数据生成流程                                     │
└─────────────────────────────────────────────────────────────────────────┘

Step 0: 原始数据
─────────────────────────────────────────────────────────────────────────
输入: HDF5物理数据文件
    路径: data/{simulator}/{simulator}_{scenario}.h5

    HDF5结构 (PDEBench格式):
    ├── 0000/
    │   ├── data: [T, S, C]   # 时间步×空间点×通道
    │   └── grid
    ├── 0001/
    │   ├── data: [T, S, C]
    └── ... (N个样本)

    示例 (diff-sorp):
    └── 0000/
        └── data: [101, 1024, 1]  # 101时间步, 1024空间点, 1通道


Step 1: auto_register (LLM自动推断)
─────────────────────────────────────────────────────────────────────────
输入: HDF5文件路径
输出: registry.yaml (元数据配置)

    内容:
    simulator: diff-sorp
    domain: "1D Diffusion-Sorption PDE"
    output_info:
      - name: concentration
        unit: mol/L
    observation:
      time_mode: monthly
      channel_indices: [0]


Step 2: generate_templates (LLM生成语言模板)
─────────────────────────────────────────────────────────────────────────
输入: registry.yaml
输出: templates.jsonl (语言模板库)

    每条模板:
    {
        "input_template": "这是{domain}任务，请根据{input_desc}预测...",
        "target_template": "预测结果为{output_0}",
        "placeholder_schema": [...],   # 参数占位符
        "output_schema": [...],        # 输出结构
        "time_indices": [0, 1],        # 选择的时间步
        "channel_indices": [0]         # 选择的通道
    }


Step 3: fill_samples (数值填充)
─────────────────────────────────────────────────────────────────────────
输入:
    - templates.jsonl (模板)
    - HDF5数据 (数值)

处理:
    1. 读取HDF5的timeseries数据
    2. 按模板的time_indices/channel_indices切片
    3. 将数值填入模板

输出: samples.jsonl (5字段格式)

    {
        "input": "这是1d_diff-sorp任务，请根据以下经过处理的过去2帧64个网格点数据，预测下一帧的状态。\n数据如下：\n[0.1098, 0.9826, ...]",     # 填充后的prompt

        "number": [原始参数值],                          # 原始HDF5参数

        "params_transformed": [变换后参数],              # 经过变换的参数

        "target": "好的，科学计算预测结果为：[[[数值矩阵]]]",  # 填充后的target (包含专家输入数值)

        "metadata": {
            "simulator": "diff-sorp",
            "timeseries_shape_obs": [2, 64],    # 2时间帧 × 64空间点
            ...
        }
    }


Step 4: convert (格式转换) ★关键★
─────────────────────────────────────────────────────────────────────────
输入: samples.jsonl (5字段)
输出: train.jsonl (2字段 - Text2Comp训练格式)

    转换逻辑:
    - input → prompt
    - target中的数值矩阵 → label (展平为1D列表)

    {
        "prompt": "这是1d_diff-sorp任务，请根据...",    # 同input

        "label": [128个数值]                           # 专家模型输入参数
                                                        # 64点 × 2帧 = 128维
    }

    ★label的含义★:
    - label不是最终物理结果
    - label是**专家模型(FNO)的输入参数**
    - Text2Comp训练目标: 学习预测这些参数


Step 5: train (训练Text2Comp)
─────────────────────────────────────────────────────────────────────────
输入:
    - train.jsonl (训练数据)
    - output_dim: 128 (专家输入维度)
    - base_model: Qwen3-0.6B (LLM backbone)

模型结构:
    Text2Comp = LLM (冻结) + MLP回归头

    输入: prompt文本
    输出: 128维数值向量 (专家输入参数)

输出:
    - best_model.pt
    - training_logs.jsonl
```
















---

### 2.2 推理数据流（使用训练好的模型）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      推理数据流                                           │
└─────────────────────────────────────────────────────────────────────────┘

用户输入:
    "这是1d_diff-sorp任务，请根据以下数据预测。
     数据如下：[0.1098, 0.9826, ...]"

        │
        ▼
┌─────────────────────────────────────┐
│   Step 1: Tokenizer                  │
│   文本 → input_ids                   │
│   输出: [batch, seq_len]             │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│   Step 2: Text2Comp模型              │
│   LLM embedding → MLP回归            │
│                                      │
│   输入: input_ids                    │
│   输出: [batch, 128]                 │◄── 128维 = 64点×2帧
│         (专家模型输入参数)            │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│   Step 3: reshape                    │
│   [batch, 128] → [batch, 64, 2]      │◄── 64空间点, 2时间帧
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│   Step 4: FNO专家模型                │
│   神经算子计算                        │
│                                      │
│   输入: [batch, 64, 2] + grid        │
│   输出: [batch, 64]                  │◄── 最终64维物理预测
└─────────────────────────────────────┘
        │
        ▼
最终输出:
    [0.9875, 0.7897, ...]  # 64个物理数值预测
```

---

## 三、关键维度说明

### 3.1 维度关系表

| 概念 | 维度 | 说明 |
|------|------|------|
| `expert_input_dim` | 128 | Text2Comp输出维度 = FNO输入维度 |
| `expert_output_dim` | 64 | FNO最终输出维度 |
| `output_dim` | 128 | 同expert_input_dim（训练时使用） |
| `spatial_points` | 64 | 空间网格点数 |
| `time_steps` | 2 | 输入时间帧数 |
| `channels` | 1 | 物理量通道数 |

**计算公式**:
```
expert_input_dim = spatial_points × time_steps × channels
                 = 64 × 2 × 1
                 = 128
```

### 3.2 不同专家模型配置

| 专家模型 | expert_input_dim | expert_output_dim | 说明 |
|----------|------------------|-------------------|------|
| FNO (diff-sorp) | 128 | 64 | 64点×2帧输入 |
| FNO (burgers) | 32 | 32 | 32点×1帧输入 |
| MODFLOW | 60 | 60 | 5井×12时间点 |
| PowerFlow | 43 | 43 | IEEE 14节点 |
| **自定义** | 用户指定 | 用户指定 | register_custom_expert() |

---

## 四、文件格式详解

### 4.1 训练数据格式 (train.jsonl)

```json
{
    "prompt": "这是1d_diff-sorp任务，请根据以下经过处理的过去2帧64个网格点数据，预测下一帧的状态。\n数据如下：\n[0.1098, 0.9826, 0.1098, ...]",

    "label": [0.1098, 0.9826, ..., 0.00702],  // 128个数值

    "completion": [0.99702, 0.94176, ...],    // 64个数值 (可选，用于验证FNO输出)

    "meta": {
        "type": "normal",
        "template_id": 1318,
        "mode": "norm"
    }
}
```

**字段说明**:
- `prompt`: 输入文本（包含物理参数描述和部分数值）
- `label`: Text2Comp预测目标 = 专家模型输入参数（128维）
- `completion`: 专家模型(FNO)输出的真值（64维），用于验证

### 4.2 专家模型权重格式

```
FNO权重文件: 1D_diff-sorp_NA_NA_FNO_2_1.pt

结构:
    fc0.weight: [64, 3]      # initial_step=2, num_channels=1 → 2×1+1=3
    fc1.weight: [128, 64]
    fc2.weight: [1, 128]     # 输出1通道

输入要求: [batch, 64, 2]     # 64空间点, 2时间帧
输出维度: [batch, 64]        # 64空间点预测
```

---

## 五、自动化训练API

### 5.1 一键训练流程

```python
from PierNet.training.text2comp import run_full_pipeline

# 预定义专家
result = run_full_pipeline(
    h5_path="data/diff-sorp/data.h5",
    simulator="diff-sorp",       # 使用预定义配置
    n_samples=1000,
    epochs=50,
    gpu_id=0,
)

# 自定义专家
result = run_full_pipeline(
    h5_path="data/my_expert/data.h5",
    simulator="my_expert",       # 自定义名称
    output_dim=256,              # 必须指定：专家输入维度
    n_samples=1000,
    epochs=50,
    gpu_id=0,
)
```

### 5.2 注册自定义专家

```python
from PierNet.training.text2comp import register_custom_expert

# 注册新专家
register_custom_expert(
    name="thermal_simulation",
    output_dim=100,              # Text2Comp输出100维给专家
    domain="Thermal Conductivity",
    expert_type="Custom",
    expert_output_dim=50,        # 专家最终输出50维
    description="热传导模拟专家",
)

# 之后可直接使用
result = run_full_pipeline(
    h5_path="...",
    simulator="thermal_simulation",  # 使用注册的专家
    ...
)
```

### 5.3 创建训练任务

```python
from PierNet.training.text2comp import create_job

# 预定义专家
job = create_job({
    "simulator": "diff-sorp",
    "train_data_path": "/path/to/train.jsonl",
    "gpu_id": 0,
    "epochs": 100,
})

# 自定义专家（动态创建）
job = create_job({
    "simulator": "new_expert",
    "output_dim": 128,           # 自动注册临时专家
    "train_data_path": "/path/to/train.jsonl",
    "gpu_id": 0,
})
```

---

## 六、服务器资源路径

### 6.1 数据路径

| 数据类型 | 路径 |
|----------|------|
| 训练数据 | `/mnt/disk1/PiERN-Webshop/1d_diff-sorp/stage2_*.jsonl` |
| HDF5数据 | `/home/research/data/PDEBench_new/1d_diff-sorp/*.h5` |

### 6.2 模型路径

| 模型类型 | 路径 |
|----------|------|
| Base LLM | `/mnt/disk1/models/Qwen/Qwen3-0___6B` |
| Text2Comp | `/home/research/data/xhb/PiERN/text2computation_model/*.pt` |
| FNO专家 | `/home/research/data/xhb/example_piern/Expert_model/*.pt` |

### 6.3 自动化模块部署路径

```
/home/research/data/zyx/piern-auto-training/
├── piern/
│   └── training/
│       └── text2comp/
│           ├── config.py           # ExpertModelInfo定义
│           ├── text2comp_manager.py # 任务管理API
│           ├── data_converter.py    # 格式转换
│           ├── run_pipeline.py      # 一键流程
│           ├── model.py             # Text2Comp模型
│           ├── train.py             # 训练逻辑
│           └── ARCHITECTURE.md      # 架构文档
├── demo_auto_training.py           # API演示
└── quick_demo.py                   # 完整推理演示
```

---

## 七、运行演示命令

```bash
# 连接服务器
ssh research@115.191.57.217

# 进入目录
cd /home/research/data/zyx/piern-auto-training

# 设置GPU
export CUDA_VISIBLE_DEVICES=4

# 运行演示
/home/research/.conda/envs/HiPER/bin/python quick_demo.py --quick
```

---

## 八、常见问题

### Q1: output_dim应该设置多少？

**答**: `output_dim = 专家模型输入维度`

例如FNO专家需要 `[batch, 64, 2]` 的输入，则:
- `output_dim = 64 × 2 = 128`

### Q2: label和completion的区别？

| 字段 | 维度 | 含义 |
|------|------|------|
| `label` | 128 | Text2Comp预测目标 = FNO输入 |
| `completion` | 64 | FNO输出真值 = 最终物理结果 |

### Q3: 如何为新专家模型训练？

```python
# 1. 确定专家输入维度
expert_input_dim = 计算或测量专家需要的输入维度

# 2. 准备训练数据
# label维度必须 = expert_input_dim

# 3. 训练
register_custom_expert(
    name="new_expert",
    output_dim=expert_input_dim,
    ...
)
create_job({
    "simulator": "new_expert",
    "train_data_path": "...",
    "gpu_id": 0,
})
```

---

*文档版本: v1.0*
*最后更新: 2026-05-10*
*作者: Claude Code*