# CLAUDE.md

## 文档定位

本文件是 PiERN 仓库内面向开发者和代码代理的工作说明，强调当前真实实现、运行边界和改动约束。

文档优先级：

1. `PROJECT_OVERVIEW.md`
   项目级总览，定义系统边界、平台划分和阶段职责。
2. `README.md`
   安装、启动、运行和常用命令入口。
3. `CLAUDE.md`
   面向实现层的结构说明、工程约束、关键契约和改动提示。

历史计划文档已经清理。改代码时，事实以 `PROJECT_OVERVIEW.md`、`README.md`、本文件和当前实现为准。

---

## 一句话说明

PiERN 当前是一个**单仓库、单 FastAPI 应用、单前端包**承载的双平台系统：

- `/synth`：数据合成平台
- `/training`：Token Router 训练平台

两边在产品层已经分开，但仍共享同一套部署入口、主题系统和少量基础设施。

---

## 当前产品入口

### 前端路由

- `/`
  引导页，位于 `frontend/src/platform/LandingPage.tsx`
- `/synth`
  数据合成平台首页，当前直接展示数据总览页
- `/training`
  训练平台首页

### 旧路由兼容

`frontend/src/platform/PlatformRouter.tsx` 里保留了 synth 旧路径重定向：

- `/simulate` → `/synth/simulate`
- `/register` → `/synth/register`
- `/templates` → `/synth/templates`
- `/fill` → `/synth/fill`
- `/router` → `/synth/router`
- `/template-viewer` → `/synth/template-viewer`
- `/samples` → `/synth/samples`
- `/router-viewer` → `/synth/router-viewer`
- `/stats` → `/synth/stats`，随后又收敛到 `/synth`
- `/registry` → `/synth/registry`
- `/llm-config` → `/synth/llm-config`

如果改顶层路由，必须同时检查：

- `frontend/src/platform/PlatformRouter.tsx`
- `frontend/src/synth/SynthApp.tsx`
- `frontend/src/training/TrainingApp.tsx`
- `piern/shared/api/static.py`

---

## 当前后端装配方式

### 真实入口

- `api_server.py`
  向后兼容入口，只 re-export `piern.api.main.app`
- `piern/api/main.py`
  当前唯一真实 FastAPI 装配入口

### 路由装配

`piern/api/main.py` 当前直接挂载：

- synth 路由：来自 `piern.synth.api.routers.*`
- training 路由：来自 `piern.training.api.routers.training`
- 静态前端：`SPAStaticFiles`

### API 命名空间边界

`piern/api/` 只保留统一 app 装配入口，不再承载业务 routers、schemas 或 services。

真实实现归属：

- 数据合成：`piern/synth/api/*`、`piern/synth/services/*`、`piern/synth/text2comp/*`
- 训练平台：`piern/training/api/*`、`piern/training/services/*`、`piern/training/router/*`

---

## 当前仓库结构

### 后端

```text
piern/
  api/                     # 统一 app 装配
  core/                    # 通用底层：llm_client / storage / validation
  shared/                  # 真正共享的基础设施
    api/static.py          # SPA history fallback
    runtime/paths.py       # 全局路径常量
  simulators/              # Stage 1 五类模拟器实现
  synth/                   # 数据合成平台后端
    api/routers/           # Stage 1-4 API
    api/schemas/
    services/              # manifest / index / file / jobs
    text2comp/             # Stage 2/3 核心逻辑
  training/                # 训练平台后端和训练核心
    api/routers/
    api/schemas/
    services/
    router/                # Token Router 数据准备、模型、训练、指标
```

### 前端

```text
frontend/src/
  platform/                # 顶层平台路由与 landing
  shared/                  # 主题等共享基础设施
  lib/                     # api/types/utils/scrollAssist 等通用运行时代码
  synth/                   # 数据合成平台
    pages/
    hooks/
    components/
  training/                # 训练平台
    pages/
    shared.ts
```

### 脚本

```text
scripts/
  text2comp/               # Stage 2/3 CLI
  router/                  # Stage 4 与训练 CLI
  utils/                   # manifests / indexes / summary 等工具
```

### 数据与产物

```text
data/
  modflow/
  simpeg/
  power_flow/
  transient/
  gcam/
  templates/
  text2comp/
  router/
  .manifests/
  .indexes/

artifacts/
  token_router/
```

---

## 数据合成平台（/synth）

### 作用范围

数据合成平台覆盖 Stage 1-4：

1. 物理仿真
2. 注册与模板生成
3. 样本填充
4. Router 数据构建

### Synth 前端入口

`frontend/src/synth/SynthApp.tsx` 当前管理这些页面：

- `/synth`
  数据总览，当前使用 `DatasetStats.tsx`
- `/synth/simulate`
  `SimulationRunner.tsx`
- `/synth/upload`
  `DataUploadPage.tsx`，上传外部 Stage 1 HDF5 并校验后写入标准数据目录
- `/synth/register`
  `RegisterSimulator.tsx`
- `/synth/templates`
  `TemplateGenerator.tsx`
- `/synth/fill`
  `SampleFiller.tsx`
- `/synth/router`
  `RouterDataBuilder.tsx`
- `/synth/template-viewer`
  `TemplateViewer.tsx`
- `/synth/samples`
  `SampleViewer.tsx`
- `/synth/router-viewer`
  `RouterViewer.tsx`
- `/synth/registry`
  `RegistryPage.tsx`
- `/synth/llm-config`
  `LLMConfig.tsx`

### Synth 后端 API

真实实现位于 `piern/synth/api/routers/`：

- `config.py`
- `datasets.py`
- `files.py`
- `generation.py`
- `interview.py`
- `jobs.py`
- `registry.py`
- `router_data.py`
- `simulation.py`

### Synth 服务层

真实实现位于 `piern/synth/services/`：

- `job_manager.py`
  生成任务流、SSE、作业状态
- `file_manager.py`
  模板/样本/router 文件列表与删除
- `manifest_store.py`
  Stage 2-4 sidecar manifest
- `jsonl_index.py`
  大 JSONL 的稀疏分页索引
- `jsonl_filter_index.py`
  常用筛选索引
- `hdf5_data.py`
  Stage 1 HDF5 文件发现、上传校验、注册前校验

### Stage 1 模拟器

当前支持：

- `modflow`
- `simpeg`
- `power_flow`
- `transient`
- `gcam`

注意：`power_flow` 和 `transient` 已经拆成两个独立 simulator，结构上与 `modflow`、`simpeg`、`gcam` 一致。

### Stage 1 HDF5 上传与校验

新增仿真器/大场景数据可以从 `/synth/upload` 上传，后端接口在 `piern/synth/api/routers/simulation.py`：

- `GET /api/simulation/data-files`
- `POST /api/simulation/upload?simulator=...&scenario=...&overwrite=false`

`simulator` 在这里表示仿真器/大场景命名空间，不限制为内置 5 个 simulator，但必须是安全 slug。落盘路径固定为 `data/{big_scene}/{big_scene}_{scenario}.h5`。
上传页只返回即时预检结果；Stage 2 注册前会使用 `piern/synth/services/hdf5_data.py` 强制校验，不合规则拒绝注册：

- `timeseries`: 数值型 3 维 `[N, C, T]`
- `params`: 数值型 2 维 `[N, P]`
- `param_names`: 字符串 1 维 `[P]`
- 根属性 `n_samples`、`n_channels`、`n_timesteps`、`n_params` 必须与 shape 完全一致
- 不允许 NaN 或 Inf

### Stage 2/3 核心

主要在 `piern/synth/text2comp/`：

- `pipeline.py`
- `generator.py`
- `template_store.py`
- `auto_register.py`
- `interview_agent.py`

当前注册契约仍是：

- `configs/text2comp/registry.yaml`

当前 Stage 2/3 默认配置：

- `configs/text2comp/default.yaml`

### 数据读取层

当前页面读大数据时，不再默认全扫 JSONL。

主要约定：

- Stage 2-4 的源产物仍然是 JSONL
- 交互式摘要优先读 `data/.manifests/`
- 大文件分页优先读 `data/.indexes/`

如果改了模板、样本、router 产物格式或清理逻辑，要同时检查：

- `scripts/utils/rebuild_manifests.py`
- `scripts/utils/rebuild_indexes.py`
- `scripts/utils/rebuild_filter_indexes.py`
- `piern/synth/services/manifest_store.py`
- `piern/synth/services/jsonl_index.py`
- `piern/synth/services/jsonl_filter_index.py`

---

## 训练平台（/training）

### 当前范围

当前训练平台只支持：

- Token Router 训练
- 单 GPU 训练
- 训练任务管理
- 曲线、日志、checkpoint 查看

当前**不支持**：

- 多 GPU 分配
- DDP
- 通用模型训练平台
- 模型注册中心

### Training 前端入口

`frontend/src/training/TrainingApp.tsx` 当前管理这些页面：

- `/training`
  `TrainingOverviewPage.tsx`
- `/training/new`
  `TrainingNewJobPage.tsx`
- `/training/jobs`
  `TrainingJobsPage.tsx`
- `/training/jobs/:jobId`
  `TrainingJobDetailPage.tsx`

### Training 后端 API

真实实现位于：

- `piern/training/api/routers/training.py`
- `piern/training/api/schemas/training.py`
- `piern/training/services/training_manager.py`

API 路径前缀是：

- `/api/training/*`

当前主要接口：

- `GET /api/training/overview`
- `GET /api/training/datasets`
- `GET /api/training/gpus`
- `GET /api/training/jobs`
- `POST /api/training/jobs`
- `GET /api/training/jobs/{job_id}`
- `GET /api/training/jobs/{job_id}/curves`
- `DELETE /api/training/jobs/{job_id}`
- `GET /api/training/jobs/{job_id}/logs`
- `POST /api/training/jobs/{job_id}/stop`

### 训练任务状态

当前状态集合：

- `queued`
- `starting`
- `running`
- `evaluating`
- `done`
- `error`
- `terminated`

### 训练任务持久化

当前训练平台不是用 synth 的 `job_manager`，而是独立 file-backed registry。

关键文件：

- `artifacts/token_router/training_jobs.json`

训练日志：

- `.runlogs/`

训练输出：

- `artifacts/token_router/{simulator}/runs/{run_name}/`

### GPU 分配约束

当前 `training_manager.py` 用的是简单探测和锁定策略：

- `GPU_AVAILABLE_MEMORY_THRESHOLD_MIB = 2048`
- `GPU_AVAILABLE_UTIL_THRESHOLD = 20`

这是保守策略，不是抢占式调度。

如果要改 GPU 选择逻辑，必须同时检查：

- `piern/training/services/training_manager.py`
- `frontend/src/training/pages/TrainingNewJobPage.tsx`
- `frontend/src/training/pages/TrainingOverviewPage.tsx`

### 当前训练核心实现

不要假设当前训练平台是通用 Transformer 训练平台，也不要退回旧 tokenizer 实现。

当前真实实现是：

- embedding encoder：动态调用 Qwen tokenizer，并使用冻结的预训练 embedding table
  - 文件：`piern/training/router/pretrained_embeddings.py`
- 模型：`FullSeqDilatedConvRouter`
  - 文件：`piern/training/router/model.py`
- 数据准备：
  - 文件：`piern/training/router/data.py`
- 训练循环：
  - 文件：`piern/training/router/train.py`

### 当前训练数据约定

训练平台直接消费：

- `data/router/by_scenario/*.jsonl`
- `data/.manifests/router.json`

当前数据准备不是重新生成 Stage 4 数据，而是：

1. 读取 router JSONL
2. 按 `context + scenario` 做稳定 hash 切分
3. 生成 `train/test` 两个 split
4. 写入 source 文件、offset、长度、label、scenario id 等轻量索引
5. 训练时动态读取原始 `context`，动态 tokenize，并通过 Qwen embedding table 转成输入向量

当前没有 validation split，只有：

- train
- test

### 当前切分与模型假设

这几条很重要：

- split 由 `build_group_key(context, scenario)` 推导
- 默认 `test_ratio = 0.10`
- Stage 4 router 数据默认使用 Qwen chat template
- 默认 embedding backbone 是 `/data/fjy/Qwen2.5-0.5B-Instruct`
- 训练输入不是离线 embedding 文件，而是训练时动态 tokenize + 冻结 embedding lookup
- 模型是 dilated conv 序列分类器，不是 Transformer

如果要改训练假设，至少要同步检查：

- `piern/training/router/tokenizer.py`
- `piern/training/router/data.py`
- `piern/training/router/model.py`
- `piern/training/router/train.py`
- `scripts/router/train_token_router.py`
- `frontend/src/training/pages/TrainingNewJobPage.tsx`
- `piern/training/api/schemas/training.py`

### 训练 CLI

CLI 入口：

- `scripts/router/train_token_router.py`

当前核心参数：

- `--simulator`
- `--scenarios`
- `--router-dir`
- `--artifact-root`
- `--prepared-name`
- `--run-name`
- `--test-ratio`
- `--batch-size`
- `--test-batch-size`
- `--epochs`
- `--eval-interval`
- `--learning-rate`
- `--weight-decay`
- `--embedding-dim`
- `--model-dim`
- `--scene-dim`
- `--dropout`
- `--kernel-size`
- `--num-workers`
- `--device`
- `--seed`
- `--force-prepare`
- `--resume-from`
- `--max-train-samples`
- `--max-test-samples`
- `--input-representation embedding`
- `--stop-file`

---

## 数据目录与产物约定

### Stage 1

HDF5 命名统一为：

- `{simulator}_{scenario}.h5`

例如：

- `data/modflow/modflow_unified_aquifer.h5`
- `data/power_flow/power_flow_ieee14_baseload.h5`

### Stage 2

- `data/templates/{scenario}_templates.jsonl`

### Stage 3

- `data/text2comp/{scenario}.jsonl`
- 合并文件：`data/text2comp/all_training_data.jsonl`

### Stage 4

- `data/router/train.jsonl`
- `data/router/by_scenario/{scenario}.jsonl`

### 训练产物

- `artifacts/token_router/{simulator}/prepared/{prepared_name}/`
- `artifacts/token_router/{simulator}/runs/{run_name}/`

---

## 当前共享基础设施

### 后端 shared

`piern/shared/` 现在只放真正跨平台基础设施：

- `runtime/paths.py`
  - `PROJECT_ROOT`
  - `DATA_DIR`
  - `TEMPLATES_DIR`
  - `CONFIG_DIR`
  - `CONFIGS_ROOT`
  - `REGISTRY_PATH`
- `api/static.py`
  - `SPAStaticFiles`

### 前端 shared

当前前端共享层仍然比较薄：

- `frontend/src/shared/theme.ts`

其余公共运行时代码主要还在：

- `frontend/src/lib/`

这意味着前端还没有完全抽成一个纯粹的 `shared/ui` 组件系统，别把它误判为已经完成。

---

## 前端滚动与布局约定

当前前端存在一套全局滚轮辅助逻辑：

- `frontend/src/lib/scrollAssist.ts`
- `frontend/src/main.tsx` 中安装

用途：

- 鼠标悬停在内部滚动组件时，优先滚组件
- 组件到边界后，再把滚动量交给右侧主页面

如果修改以下内容，必须一起检查滚轮行为：

- `.page-content`
- `.workbench-main-scroll`
- `.training-page__body`
- `.training-scroll`
- `.list-scroll-*`
- `.list-table-scroll*`

对应文件：

- `frontend/src/index.css`
- `frontend/src/lib/scrollAssist.ts`

不要只改 CSS 而忽略滚轮链路。

---

## 当前测试与验证覆盖

当前自动测试覆盖很薄。

当前已有的 Python 测试文件：

- `tests/test_build_router_data_script.py`
- `tests/test_check_garbled_text.py`
- `tests/test_registry_observation_config.py`
- `tests/test_router_prepared_inputs.py`
- `tests/test_training_manager_fallbacks.py`

主要覆盖：

- Stage 4 router 数据构建与 prepared input metadata
- 注册信息中的观测维度/通道配置约束
- 乱码检查工具
- training manager 在 router manifest 缺失、`nvidia-smi` 不可用时的回退行为

这意味着：

- 训练平台 UI 行为几乎没有自动化前端测试
- synth/training 的大多数集成路径主要靠构建和手工回归

当前较可靠的回归方式仍然是：

```bash
python -m compileall piern scripts api_server.py
cd frontend && npm run build
```

如果动了训练平台或大文件读取逻辑，建议至少补一个：

- Python 层单元测试
- 或最小端到端 smoke test

---

## 常用命令

### 安装

```bash
pip install -r requirements.txt
pip install -e .
```

### 启动

```bash
./start_ui.sh
./start_ui.sh --dev
```

`start_ui.sh` 当前行为：

- 尝试激活 `piern-project` conda 环境
- 启动 FastAPI：`8000`
- 启动 Vite：`5173`

### 前端构建

```bash
cd frontend
npm run build
```

### 后端快速检查

```bash
python -m compileall piern scripts api_server.py
```

### Manifest / Index 重建

```bash
python scripts/utils/rebuild_manifests.py
python scripts/utils/rebuild_indexes.py
python scripts/utils/rebuild_filter_indexes.py
```

### Token Router 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/router/train_token_router.py \
  --simulator modflow \
  --device cuda:0 \
  --epochs 1
```

---

## 改动时必须联动检查的文档

### 改“用户怎么启动/使用”

更新：

- `README.md`

### 改“平台结构、阶段边界、目录归属”

更新：

- `PROJECT_OVERVIEW.md`
- `CLAUDE.md`

### 改“训练平台能力、训练契约、训练路线”

更新：

- `README.md` / `PROJECT_OVERVIEW.md` / `CLAUDE.md`

### 改“性能读路径、manifest/index 行为”

更新：

- `README.md` / `PROJECT_OVERVIEW.md` / `CLAUDE.md`

---

## 当前开发约束

1. synth 和 training 在产品层已经分开，但仓库层仍共享同一应用和同一前端包。
2. `piern/api/main.py` 是统一装配入口；业务代码应放入 `piern/synth/` 或 `piern/training/`。
3. training 当前是**单 GPU**系统，前端和后端都按这个边界设计。
4. 训练核心当前是基于 Qwen embedding 输入的 full-sequence conv router，不要按通用 Transformer 平台来改。
5. Stage 2-4 的在线读取优化依赖 `data/.manifests/` 与 `data/.indexes/`，不要绕开它们直接退回全量扫描。
6. 前端内部滚动和页面滚动是当前易出问题区域；只改样式经常不够，需要同时看 `scrollAssist.ts`。
7. 文档必须跟实现同步，不能继续把计划文档写成事实。

---

## 最后一句

如果开始改这个仓库，优先把它当成：

- 一个双平台产品
- 一个已完成 synth/training 命名空间拆分、仅保留 `api_server.py` 入口兼容的代码库
- 一个以 JSONL/HDF5 为源产物、以 manifest/index 为交互读取层的系统
- 一个当前只对 Token Router 单 GPU 训练负责的训练平台

不要沿用旧的“单一 Stage 1-4 工具页面”心智模型，也不要默认训练平台已经演进成通用训练基础设施。
