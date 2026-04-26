# PiERN

PiERN 是一个面向物理与工程时序数据的双平台系统：

- `/synth`：数据合成平台
- `/training`：Token Router 训练平台

当前仓库仍是**单仓库、单 FastAPI 应用、单前端包**，但产品入口和代码归属已经按 synth / training 分开。

## 文档入口

优先维护以下三份文档：

- `PROJECT_OVERVIEW.md`
  项目总览与系统边界
- `README.md`
  安装、启动、快速开始
- `CLAUDE.md`
  面向开发者和代码代理的真实实现说明

计划类文档已清理；长期事实以以上三份文档为准。

## 当前平台入口

统一入口：

- 引导页：`http://localhost:8000/`
- 数据合成平台：`http://localhost:8000/synth`
- 训练平台：`http://localhost:8000/training`

开发模式下，Vite 也会启动在：

- `http://localhost:5173/`

说明：

- `8000` 是 FastAPI 入口，也会托管构建后的前端静态资源
- `5173` 是前端开发服务，适合做样式和交互调试

## 当前能力范围

### 数据合成平台 `/synth`

覆盖 Stage 1-4：

1. 物理仿真
2. 注册与模板生成
3. 样本填充
4. Router 数据构建

主要页面：

- 数据总览
- 物理仿真
- 注册场景
- 生成模板
- 填充样本
- 构建路由
- 模板/样本/路由浏览
- 注册信息
- LLM 配置

### 训练平台 `/training`

当前只支持 Token Router 训练。

主要能力：

- 训练数据选择
- 单 GPU 训练任务创建
- GPU 状态查看
- 训练任务列表
- 训练详情、曲线、日志、checkpoint 查看

当前限制：

- 仅单 GPU
- 不支持 DDP
- 不支持通用模型训练平台

## 支持的 Stage 1 模拟器

| Simulator | Domain | Math Type | Output Shape | Scenario Count |
| --- | --- | --- | --- | --- |
| `modflow` | Groundwater | Parabolic PDE | `(5, 365)` | 7 |
| `simpeg` | Geophysics | Elliptic PDE | `(1, 100)` | 4 |
| `power_flow` | Steady-state power flow | Nonlinear algebraic system | `(43, 365)` | 5 |
| `transient` | Transient stability | DAE | `(5, 1000)` | 3 |
| `gcam` | Energy-climate planning | Dynamic algebraic / LP | `(5, 16)` | 3 |

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

前端：

```bash
cd frontend
npm install
```

MODFLOW 可执行文件：

```bash
python - <<'PY'
from pathlib import Path
from flopy.utils import get_modflow
get_modflow(str(Path.home() / '.flopy_bin'), subset='mf2005')
PY
```

然后设置：

```bash
export PIERN_MODFLOW_EXE=/path/to/mf2005
```

说明：

- 当前依赖约束已经覆盖 SimPEG / PyPSA 相关兼容版本
- `requirements.txt` 是主安装入口
- `setup.py` 仍保留，但不应单独视为唯一事实源

## 启动

### 推荐：一键启动开发环境

```bash
./start_ui.sh
./start_ui.sh --dev
```

当前 `start_ui.sh` 会：

- 尝试激活 `piern-project` conda 环境
- 启动 FastAPI：`8000`
- 启动 Vite：`5173`

### 常用访问地址

```text
Landing   http://localhost:8000/
Synth     http://localhost:8000/synth
Training  http://localhost:8000/training
Vite Dev  http://localhost:5173/
Docs      http://localhost:8000/docs
```

## 数据目录约定

### Stage 1

HDF5 文件统一命名：

```text
{simulator}_{scenario}.h5
```

例如：

```text
data/modflow/modflow_unified_aquifer.h5
data/power_flow/power_flow_ieee14_baseload.h5
data/transient/transient_ieee14_fault.h5
```

### Stage 2

```text
data/templates/{scenario}_templates.jsonl
```

### Stage 3

```text
data/text2comp/{scenario}.jsonl
data/text2comp/all_training_data.jsonl
```

### Stage 4

```text
data/router/train.jsonl
data/router/by_scenario/{scenario}.jsonl
```

### 训练产物

```text
artifacts/token_router/{simulator}/prepared/{prepared_name}/
artifacts/token_router/{simulator}/runs/{run_name}/
```

## 快速开始

### Stage 1：物理仿真

```bash
python -m piern.simulators.modflow.pipeline \
  --config configs/modflow/variants/unified_aquifer.yaml \
  --n-samples 1000

python -m piern.simulators.simpeg.pipeline \
  --config configs/simpeg/variants/dc_resistivity.yaml \
  --n-samples 1000

python -m piern.simulators.power_flow.pipeline \
  --config configs/power_flow/variants/ieee14_baseload.yaml \
  --n-samples 1000

python -m piern.simulators.transient.pipeline \
  --config configs/transient/variants/ieee14_fault.yaml \
  --n-samples 500

python -m piern.simulators.gcam.pipeline \
  --config configs/gcam/variants/energy_transition.yaml \
  --n-samples 1000
```

### Stage 2：注册与模板生成

```bash
python -m piern.synth.text2comp.auto_register \
  --config configs/text2comp/default.yaml \
  --output configs/text2comp/registry.yaml

python scripts/text2comp/generate_templates.py \
  --config configs/text2comp/default.yaml \
  --n-templates 1000
```

### Stage 3：样本填充

```bash
python scripts/text2comp/fill_samples.py \
  --config configs/text2comp/default.yaml \
  --n-samples 1000 \
  --skip-existing
```

### Stage 4：Router 数据构建

```bash
python scripts/router/build_router_data.py --seed 42
python scripts/router/build_router_data.py --seed 42 --chat-template qwen --neg-ratio 2
```

### Token Router 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/router/train_token_router.py \
  --simulator modflow \
  --device cuda:0 \
  --epochs 1
```

## 当前训练核心

当前训练平台的真实实现不是通用 Transformer 训练平台，而是：

- 模型：`FullSeqDilatedConvRouter`
- 数据切分：`train / test`
- Stage 4 约定：Qwen chat template
- Embedding backbone：默认 `/data/models/Qwen/Qwen2.5-0.5B-Instruct`
- 训练输入：训练时动态 tokenize router JSONL 的 `context`，再通过冻结的 Qwen embedding table 转成输入向量；不离线存储 embedding

核心代码位于：

- `piern/training/router/pretrained_embeddings.py`
- `piern/training/router/model.py`
- `piern/training/router/data.py`
- `piern/training/router/train.py`

## 读性能说明

Stage 2-4 的交互式读取当前采用 sidecar 读层：

- manifest：`data/.manifests/`
- index：`data/.indexes/`

约定：

- JSONL 仍然是源产物格式
- 摘要接口优先读 manifest
- 大文件分页优先读 sparse index
- 常用筛选分页优先读 filter index

手动重建：

```bash
python scripts/utils/rebuild_manifests.py
python scripts/utils/rebuild_indexes.py
python scripts/utils/rebuild_filter_indexes.py
```

统计页当前优先走：

- `/api/dashboard/summary`

## 当前项目结构

```text
piern/
  piern/
    api/                    # 统一 FastAPI app 装配
    core/                   # 通用底层
    shared/                 # 共享基础设施
    simulators/             # Stage 1
    synth/                  # synth 平台后端 + Stage 2/3 text2comp
    training/               # training 平台后端与训练核心

  frontend/src/
    platform/               # landing + 顶层平台路由
    shared/                 # 共享主题层
    lib/                    # api/types/utils/scrollAssist
    synth/                  # synth 平台前端
    training/               # training 平台前端

  scripts/
    text2comp/
    router/
    utils/

  data/
  artifacts/
```

## 当前注意点

- `piern/api/main.py` 是统一 FastAPI 装配入口；业务实现不再放在 `piern/api/`
- synth 和 training 虽然产品层分开，但仍然共享同一套部署入口
- 当前训练平台是单 GPU 平台，不要按多卡训练基础设施来理解
- 前端滚动行为依赖 `frontend/src/lib/scrollAssist.ts`，不要只改 CSS 不看滚轮链路

## 许可证

MIT
