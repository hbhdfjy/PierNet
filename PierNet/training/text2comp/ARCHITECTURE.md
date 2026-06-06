# 文生计算模块架构说明

## 一、核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    完整推理流程                               │
└─────────────────────────────────────────────────────────────┘

用户输入: "这是1d_diff-sorp任务，数据如下[0.1, 0.2, ...]"
    │
    ▼
┌─────────────────────────────────────────┐
│  Router (可选)                          │
│  判断任务类型 → 选择对应专家模型          │
│  输出: scenario_id                       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Text2Comp (文生计算模块)                │
│  输入: 文本描述                          │
│  输出: N维数值向量                       │
│  作用: 生成专家模型需要的输入参数         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  专家模型 (FNO/MODFLOW/Custom等)        │
│  输入: N维数值向量 (来自Text2Comp)       │
│  输出: M维物理结果                       │
│  作用: 执行实际物理计算                  │
└─────────────────────────────────────────┘
    │
    ▼
最终输出: 物理预测结果
```

## 二、关键概念

| 概念 | 说明 |
|------|------|
| **expert_input_dim** | Text2Comp输出维度 = 专家模型输入维度 |
| **expert_output_dim** | 专家模型最终输出维度 |
| **output_dim** | 同expert_input_dim（训练时使用） |

**示例（FNO专家）**：
- Text2Comp输出: 128维 (64点×2帧)
- FNO输入: 128维
- FNO输出: 64维 (下一帧预测)

## 三、Text2Comp不是替代专家模型

Text2Comp的作用是**生成专家模型的输入参数**，而不是替代专家模型进行物理计算。

| 错误理解 | 正确理解 |
|---------|---------|
| Text2Comp直接预测物理结果 | Text2Comp生成参数→传给专家模型 |
| output_dim = 最终结果维度 | output_dim = 专家模型输入维度 |

## 四、支持任意专家模型

### 预定义专家

```python
from PierNet.training.text2comp import EXPERT_MODEL_LIBRARY

# 已支持: diff-sorp, burgers, diff-reaction, modflow, power_flow
print(EXPERT_MODEL_LIBRARY.keys())
```

### 自定义专家

```python
from PierNet.training.text2comp import register_custom_expert

# 注册自定义专家
register_custom_expert(
    name="my_custom_expert",
    output_dim=256,              # Text2Comp输出256维给专家
    domain="My Physics Domain",
    expert_type="Custom",
    description="自定义物理专家模型",
)

# 或在创建任务时直接指定
create_job({
    "simulator": "new_expert",
    "output_dim": 128,           # 自动注册临时专家
    "train_data_path": "...",
    "gpu_id": 0,
})
```

## 五、一键训练流程

```python
from PierNet.training.text2comp import run_full_pipeline

# 自动化流程：HDF5 → 训练数据 → 模型训练
result = run_full_pipeline(
    h5_path="data/my_expert/data.h5",
    simulator="my_expert",
    output_dim=128,              # 可选，自定义专家时需提供
    gpu_id=0,
)

# 输出
# {
#     "train_data_path": "...",
#     "output_dim": 128,
#     "job_id": "...",
#     "model_path": "...",
# }
```

## 六、训练数据格式

```json
{
    "prompt": "这是物理任务描述，数据如下[...]",
    "label": [128个数值],        // 专家模型输入参数
    "completion": [64个数值],    // 专家模型输出（可选，用于验证）
    "meta": {...}
}
```

**关键**：label维度 = expert_input_dim

## 七、API总结

| API | 功能 |
|-----|------|
| `register_custom_expert()` | 注册自定义专家模型 |
| `create_job()` | 创建训练任务（支持预定义和自定义） |
| `run_full_pipeline()` | 一键训练流程 |
| `list_simulators()` | 列出所有专家模型 |

---

*文档版本: v2.0*
*最后更新: 2026-05-10*
*关键修正: Text2Comp输出是专家模型输入参数，不是最终物理结果*