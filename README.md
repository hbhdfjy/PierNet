# PierNet

PierNet 是一个面向物理与工程时序数据的双平台系统，当前由同一个 FastAPI + React 应用交付。

- `/synth`：Stage 1-4 数据合成工作台。
- `/training`：Token Router、Text2Comp 和模型拼装训练工作台。

两个平台在代码命名空间和产品职责上分离，但共享启动入口、静态资源托管、主题样式和少量基础设施。

## 文档地图

长期有效的项目信息集中在以下文档：

- `README.md`：安装、启动、快速命令和运行约定。
- `PROJECT_OVERVIEW.md`：系统边界、架构和数据契约。
- `CLAUDE.md`：开发者和编码 Agent 的实现注意事项。
- `docs/MIGRATION.md`：服务器迁移、便携式数据边界和恢复流程。
- `docs/patents/README.md`：专利 Markdown、附图和 Word 生成说明。

历史阶段文档不作为事实源；正式专利文本以 `docs/patents/*_PATENT_DRAFT.md` 为源文件。

## 运行入口

```text
首页       http://localhost:8000/
数据合成   http://localhost:8000/synth
训练平台   http://localhost:8000/training
文件管理   http://localhost:8000/synth/files（/files 会重定向）
API 文档   http://localhost:8000/docs
Vite 开发  http://localhost:3000/
```

`8000` 是 FastAPI 端口；当 `frontend/dist` 存在时，后端也会托管构建后的前端资源。`3000` 是 Vite 开发服务器端口。

## 安装

要求 Python 3.11 和 Node.js 20.19.0+。

```bash
pip install -r requirements.txt
pip install -e .
```

安装前端依赖：

```bash
cd frontend
npm install
```

`pyproject.toml` 保存包元数据和质量工具配置；`requirements.txt` 仍作为现有服务器的兼容安装入口。

### Conda 环境

`scripts/services/*` 和 `start_ui.sh` 会在 `.env` 存在时加载它；若仓库内存在 `.conda/env/bin/python` 会优先使用该环境，否则默认使用 `$HOME/.conda/envs/PierNet`。迁移部署时建议复制 `.env.example`：

```bash
cp .env.example .env
export PierNet_CONDA_ENV=$HOME/.conda/envs/PierNet
export PierNet_QWEN_EMBEDDING_MODEL=$HOME/Qwen/Qwen2.5-0.5B-Instruct
```

修改完变量后，在同一个 shell 中启动服务。

### MODFLOW 可执行文件

MODFLOW 场景需要 `mf2005` 可执行文件。一种安装方式：

```bash
python - <<'PY2'
from pathlib import Path
from flopy.utils import get_modflow
get_modflow(str(Path.home() / '.flopy_bin'), subset='mf2005')
PY2
```

然后设置：

```bash
export PierNet_MODFLOW_EXE=/path/to/mf2005
```

## 启动应用

共享远程服务器建议使用持久化服务脚本：

```bash
scripts/services/start.sh
scripts/services/status.sh
scripts/services/restart.sh
scripts/services/stop.sh
```

服务脚本会在 `.runlogs/services/` 下写入 PID 和日志，SSH 退出后进程仍会保留，并将配置的 conda 或 Node bin 目录放到 `PATH` 前面，确保 Vite 使用 Node.js 20.19.0+。

交互式开发启动：

```bash
./start_ui.sh
./start_ui.sh --dev
```

两种方式都会启动：

- FastAPI：`0.0.0.0:8000`
- Vite：`0.0.0.0:3000`

手动启动后端：

```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

手动启动前端：

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --strictPort
```

## 产品范围

### `/synth`

数据合成平台覆盖四个阶段：

1. Stage 1：物理仿真或 HDF5 上传。
2. Stage 2：场景注册元数据和大模型语言模板生成。
3. Stage 3：本地确定性样本填充。
4. Stage 4：Token Router 训练数据构建。

主要页面包括：数据概览、物理仿真、HDF5 上传、场景注册、模板生成、样本填充、Router 数据构建、模板/样本/Router 查看器、文件管理、注册表编辑和大模型配置。

### `/training`

训练平台覆盖三个模块：

- Token Router：消费 Stage 4 Router 数据，训练 `FullSeqDilatedConvRouter`。
- Text2Comp：消费 Text2Comp JSONL 数据，训练文本到专家输入向量的模型。
- 模型拼装：加载 LLM、Router、Text2Comp 和 FNO 专家，做端到端推理验证。

当前训练任务仍以单 GPU 为主，不支持 DDP。平台支持数据集选择、GPU 状态、任务创建、任务列表、日志、曲线、检查点、停止和删除。

## 支持的 Stage 1 仿真器

| 仿真器 | 领域 | 数学类型 | 输出形状 | 场景数 |
| --- | --- | --- | --- | --- |
| `modflow` | 地下水 | 抛物型 PDE | `(5, 365)` | 7 |
| `simpeg` | 地球物理 | 椭圆型 PDE | `(1, 100)` | 4 |
| `power_flow` | 稳态潮流 | 非线性代数系统 | `(43, 365)` | 5 |
| `transient` | 暂态稳定 | DAE | `(5, 1000)` | 3 |
| `gcam` | 能源气候规划 | 动态代数 / LP | `(5, 16)` | 3 |

场景配置位于 `configs/{simulator}/variants/`。

## 数据契约

### Stage 1 HDF5

标准生成文件：

```text
data/{simulator}/{simulator}_{scenario}.h5
```

外部仿真器或大场景上传文件：

```text
data/{big_scene}/{big_scene}_{scenario}.h5
```

注册时会校验 HDF5 契约：

- `timeseries`：数值 3D `[N, C, T]`。
- `params`：数值 2D `[N, P]`。
- `param_names`：字符串 1D `[P]`。
- 根属性 `n_samples`、`n_channels`、`n_timesteps`、`n_params` 必须与形状一致。
- `timeseries` 和 `params` 必须是有限数，不能包含 NaN 或 Inf。

### Stage 2 模板

```text
data/templates/{scenario}_templates.jsonl
```

### Stage 3 样本

默认 Parquet 分区：

```text
data/text2comp_parquet/simulator={simulator}/scenario={scenario}/part-*.parquet
```

兼容旧 JSONL：

```text
data/text2comp/{scenario}.jsonl
data/text2comp/all_training_data.jsonl
```

### Stage 4 Router 数据

默认 Parquet 分区：

```text
data/router_parquet/simulator={simulator}/scenario={scenario}/part-*.parquet
```

兼容旧 JSONL：

```text
data/router/by_scenario/{scenario}.jsonl
data/router/train.jsonl
```

### 训练与拼装产物

```text
artifacts/token_router/{simulator}/prepared/{prepared_name}/
artifacts/token_router/{simulator}/runs/{run_name}/
artifacts/text2comp_models/{simulator}/runs/{job_id}/
artifacts/fno_models/
configs/assembly/models.yaml
configs/assembly/prompt.yaml
```

## 快速命令

### Stage 1

```bash
python -m PierNet.simulators.modflow.pipeline \
  --config configs/modflow/variants/unified_aquifer.yaml \
  --n-samples 1000

python -m PierNet.simulators.simpeg.pipeline \
  --config configs/simpeg/variants/dc_resistivity.yaml \
  --n-samples 1000

python -m PierNet.simulators.power_flow.pipeline \
  --config configs/power_flow/variants/ieee14_baseload.yaml \
  --n-samples 1000

python -m PierNet.simulators.transient.pipeline \
  --config configs/transient/variants/ieee14_fault.yaml \
  --n-samples 500

python -m PierNet.simulators.gcam.pipeline \
  --config configs/gcam/variants/energy_transition.yaml \
  --n-samples 1000
```

### Stage 2

```bash
python -m PierNet.synth.text2comp.auto_register \
  --config configs/text2comp/default.yaml \
  --output configs/text2comp/registry.yaml

python scripts/text2comp/generate_templates.py \
  --config configs/text2comp/default.yaml \
  --n-templates 1000
```

模板生成会调用 `configs/text2comp/default.yaml` 或 `/synth/llm-config` 页面中配置的大模型服务。

### Stage 3

```bash
python scripts/text2comp/fill_samples.py \
  --config configs/text2comp/default.yaml \
  --n-samples 1000 \
  --skip-existing
```

Stage 3 是本地确定性填充，不调用大模型。

### Stage 4

```bash
python scripts/router/build_router_data.py --seed 42 --input-format parquet --output-format parquet
python scripts/router/build_router_data.py --seed 42 --input-format parquet --output-format parquet --chat-template qwen --neg-ratio 2
```

Stage 4 会生成 Qwen chat template 上下文，用于二分类 Token Router 训练。

### Token Router 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/router/train_token_router.py \
  --simulator modflow \
  --device cuda:0 \
  --epochs 1
```

当前训练核心：

- 模型：`FullSeqDilatedConvRouter`。
- 输入表示：Qwen tokenizer + 冻结的预训练 embedding 表。
- 默认 embedding backbone：优先使用 `PierNet_QWEN_EMBEDDING_MODEL`，否则回退到 `~/Qwen/Qwen2.5-0.5B-Instruct`。
- 数据切分：只使用 train/test。
- embedding 在训练时查表，不离线保存。

Token Router 会产生两类可再生缓存：

- `data/router/.parquet_jsonl_cache/`：Router Parquet 训练前的 JSONL 物化缓存。
- `artifacts/token_router/{simulator}/prepared/{prepared_name}/`：训练前处理好的 token cache。

这两类缓存都会写入显式 `last_used_at` / `last_built_at`。只有训练数据准备链路实际复用缓存时才刷新 `last_used_at`；普通文件系统扫描不会刷新。过期清理可用：

```bash
python scripts/cache/cleanup_training_cache.py --execute
python scripts/cache/cleanup_training_cache.py --dry-run --json
```

默认 TTL 均为 7 天，可用 `PierNet_ROUTER_JSONL_CACHE_TTL_DAYS` 和 `PierNet_TRAINING_PREPARED_CACHE_TTL_DAYS` 调整。worker 自动清理默认关闭；设置 `PierNet_CACHE_CLEANUP_ENABLED=1` 后按 `PierNet_CACHE_CLEANUP_INTERVAL_HOURS` 运行，默认真实删除，受 `PierNet_CACHE_CLEANUP_MAX_DELETE_GB` 单次上限保护。处于 `queued` / `starting` / `running` / `evaluating` / `stopping` 的训练任务引用的 prepared cache 不会被删除。

### Text2Comp 与模型拼装

Text2Comp 通过 `/training/text2comp` 或 `/api/text2comp/*` 创建训练任务，默认产物写入：

```text
artifacts/text2comp_models/{simulator}/runs/{job_id}/
.runlogs/text2comp/{job_id}.log
```

模型拼装通过 `/training/assembly` 或 `/api/assembly/*` 扫描并加载：

- LLM：`configs/assembly/models.yaml` 中的 `llm_dirs`。
- Router：默认 `artifacts/token_router/`，可通过 `PIERN_ROUTER_ARTIFACTS_DIR` 覆盖。
- Text2Comp：默认 `artifacts/text2comp_models/`，可通过 `PIERN_TEXT2COMP_MODELS_DIR` 覆盖。
- FNO：默认 `artifacts/fno_models/`，可通过 `PIERN_FNO_MODELS_DIR` 覆盖。

公共 HuggingFace 模型目录仍可指向 `/root/eb-public/...`；项目产物不要默认依赖个人开发目录。

## 读取性能层

Stage 2 模板仍是 JSONL；Stage 3/4 当前主产物是便携式 Parquet 分区，旧 JSONL 仍可读取和迁移。JSONL 读取通过旁路摘要和索引加速，Parquet 分区通过 manifest、目录数据库和文件管理器汇总：

```text
data/.manifests/
data/.indexes/
data/text2comp_parquet/
data/router_parquet/
```

手动重建：

```bash
python scripts/utils/rebuild_manifests.py
python scripts/utils/rebuild_indexes.py
```

## 仓库结构

```text
api_server.py                  # 兼容入口，导出 PierNet.api.main.app
PierNet/api/main.py              # 统一 FastAPI 应用组装
PierNet/core/                    # 共享底层存储、校验和 LLM 客户端
PierNet/shared/                  # 静态托管和运行时路径
PierNet/simulators/              # Stage 1 仿真实现
PierNet/synth/                   # 数据合成后端和 text2comp 核心
PierNet/training/                # 训练 API、管理器和 Token Router 核心
frontend/src/platform/         # 首页和顶层路由
frontend/src/synth/            # 数据合成前端
frontend/src/training/         # 自动训练前端
frontend/src/files/            # 统一文件管理界面
scripts/text2comp/             # Stage 2/3 CLI
scripts/router/                # Stage 4 和训练 CLI
scripts/utils/                 # manifest、索引、检查和工具脚本
```

## 验证

后端语法检查：

```bash
python -m compileall PierNet scripts api_server.py
```

前端构建：

```bash
cd frontend
npm run build
```

定向测试：

```bash
pytest tests/test_build_router_data_script.py \
  tests/test_hdf5_data_validation.py \
  tests/test_data_browsing_mixed_storage.py \
  tests/test_router_prepared_inputs.py \
  tests/test_storage_scripts.py \
  tests/test_training_manager_fallbacks.py
```

仓库一致性检查：

```bash
python scripts/ci/check_consistency.py
```

## 许可证

MIT，见 `LICENSE`。
