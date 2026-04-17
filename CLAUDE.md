# CLAUDE.md

## 文档定位

本文件是开发者与代码代理在 PiERN 仓库中工作的上下文说明，不再单独承担项目总览的全部职责。

默认文档优先级如下：

1. `PROJECT_OVERVIEW.md`：项目整体 overview，作为系统边界和阶段划分的基线。
2. `README.md`：安装、启动、运行入口。
3. `CLAUDE.md`：开发约定、代码结构、实现细节与操作提示。

`UPGRADE_PLAN.md` 只作为路线图和待办计划，不作为 overview 的事实来源。

本文件为 Claude Code 在此仓库中工作提供指导。

## 项目定位

**PiERN 多模拟器数据合成管线**。

跨物理域仿真数据合成工程，同时覆盖五种数学结构，全部使用成熟开源工具。

## 五大模拟器 · 五种数学结构

| 模拟器 | 数学类型 | 领域 | 输出形状 | 场景数 | 状态 |
|--------|---------|------|---------|--------|---------|------|
| **MODFLOW** | 抛物型PDE | 地下水 | (5, 365) | 7 | ✅ 代码完成 |
| **SimPEG** | 椭圆型PDE | 地球物理 | (1, 100) | 4 | ✅ 代码完成 |
| **pandapower** | 非线性代数方程组 | 稳态潮流 | (43, 365) | 5 | ✅ 代码完成 |
| **ANDES** | DAE系统 | 暂态稳定 | (5, 1000) | 3 | ✅ 代码完成 |
| **GCAM简化版** | 动态代数系统 | 能源-气候 | (5, 16) | 3 | ✅ 代码完成 |
| **总计** | 5种数学结构 | 3大领域 | — | **22** | — |

## 数学原理说明

```
MODFLOW:    S_s * dh/dt = ∇(K∇h) + W          → 抛物型PDE，FDM+PCG
SimPEG:     ∇·(σ∇φ) = -Iδ                     → 椭圆型PDE，FVM+直接求解
pandapower: P_i = ΣV_iV_j(G_ij cosθ_ij + ...) → 非线性代数，Newton-Raphson
ANDES:      δ̇=ω, Mω̇=Pm-Pe-Dω, 0=g(x,y)       → DAE系统，隐式梯形法
GCAM:       多期LP能源优化（PyPSA+HiGHS）        → 动态代数，LP求解
```

## 当前数据状态

```
data/modflow/     — 7场景，文件名 modflow_{scenario}.h5
data/simpeg/      — 4场景，文件名 simpeg_{scenario}.h5
data/power_flow/  — 5潮流场景，文件名 power_flow_{scenario}.h5
data/transient/   — 3暂态场景，文件名 transient_{scenario}.h5
data/gcam/        — 3场景，文件名 gcam_{scenario}.h5
data/templates/   — 多场景模板库（Stage 2 输出，{scenario}_templates.jsonl）
data/text2comp/   — 最终训练样本（Stage 3 输出，{scenario}.jsonl）
data/router/      — Token Router 训练数据（Stage 4 输出，train.jsonl + by_scenario/）
```

**HDF5 文件命名约定**：`{simulator}_{scenario}.h5`，如 `modflow_unified_aquifer.h5`。
目录名即 simulator 名，Stage 2/3 脚本自动扫描 `data/` 下各子目录，无需额外配置。

## 常用命令

```bash
# 安装
pip install -r requirements.txt
pip install -e .

# Stage 1：物理仿真（推荐通过前端管理界面启动）
./start_ui.sh   # 启动 FastAPI(8000) + Vite(5173)，访问 http://localhost:5173

# 也可 CLI 直接运行：

# MODFLOW（抛物型PDE，输出 data/modflow/modflow_{scenario}.h5）
python -m piern.simulators.modflow.pipeline \
    --config configs/modflow/variants/unified_aquifer.yaml --n-samples 1000

# SimPEG（椭圆型PDE，输出 data/simpeg/simpeg_{scenario}.h5）
python -m piern.simulators.simpeg.pipeline \
    --config configs/simpeg/variants/dc_resistivity.yaml --n-samples 1000

# pandapower 潮流（非线性代数，输出 data/power_flow/power_flow_{scenario}.h5）
python -m piern.simulators.power_flow.pipeline \
    --config configs/power_flow/variants/ieee14_baseload.yaml --n-samples 1000

# ANDES 暂态（DAE，输出 data/transient/transient_{scenario}.h5）
python -m piern.simulators.transient.pipeline \
    --config configs/transient/variants/ieee14_fault.yaml --n-samples 500

# GCAM（动态代数，输出 data/gcam/gcam_{scenario}.h5）
python -m piern.simulators.gcam.pipeline \
    --config configs/gcam/variants/energy_transition.yaml --n-samples 1000

# 汇总
python scripts/utils/summarize_all.py

# Stage 2：语言模板生成

# Step 0：自动注册元数据（新数据集必须先跑）
python -m piern.text2comp.auto_register \
    --config configs/text2comp/default.yaml \
    --output configs/text2comp/registry.yaml

# Step 1：生成语言模板（调 LLM，结果存 data/templates/）
python scripts/text2comp/generate_templates.py \
    --config configs/text2comp/default.yaml \
    --n-templates 1000
python scripts/text2comp/generate_templates.py \
    --scenarios unified_aquifer ieee14_baseload \
    --n-templates 200 --language-mix 0.6 --max-workers 8

# Stage 3：数值填充（不调 LLM，结果存 data/text2comp/）
python scripts/text2comp/fill_samples.py \
    --config configs/text2comp/default.yaml \
    --n-samples 1000 --skip-existing

# Stage 4：Token Router 数据生成（不调 LLM）
python scripts/router/build_router_data.py \
    --seed 42 --neg-ratio 1
```

## 项目结构

```
piern/
├── piern/
│   ├── core/                    # llm_client, storage, validation
│   ├── api/                     # FastAPI 后端（分层架构）
│   │   ├── main.py              #   app 入口，注册 router，CORS
│   │   ├── deps.py              #   公共常量（PROJECT_ROOT 等）
│   │   ├── routers/             #   按资源分 router
│   │   │   ├── datasets.py      #     /api/datasets, /api/samples, /api/stats
│   │   │   ├── config.py        #     /api/config, /api/llm-config, /api/config/scenarios
│   │   │   ├── registry.py      #     /api/registry CRUD, /api/register
│   │   │   ├── jobs.py          #     /api/generate/{id}/stream, status, delete（SSE）
│   │   │   ├── generation.py    #     /api/generate-templates, /api/fill-samples, /api/templates
│   │   │   ├── files.py         #     /api/files/templates, /api/files/samples
│   │   │   ├── interview.py     #     /api/interview/*
│   │   │   ├── simulation.py    #     /api/simulation/*, /api/simulate（Stage 1 仿真）
│   │   │   └── router_data.py   #     /api/router/*（Stage 4 Router 数据）
│   │   ├── schemas/             #   Pydantic 模型
│   │   └── services/            #   业务逻辑（job_manager, file_manager）
│   ├── simulators/
│   │   ├── modflow/             # ✅ 抛物型PDE，flopy/MODFLOW
│   │   ├── simpeg/              # ✅ 椭圆型PDE，SimPEG
│   │   ├── power_flow/          # ✅ 非线性代数，pandapower（稳态潮流）
│   │   ├── transient/           # ✅ DAE系统，ANDES（暂态稳定）
│   │   ├── power_system/        # 共享工具库（unified_params, generator_with_params）
│   │   └── gcam/                # ✅ 动态代数，PyPSA+HiGHS
│   └── text2comp/               # ✅ Stage 2/3 语言模板生成
│       ├── generator.py         #    LLMTextGenerator（占位符机制，并发）
│       ├── template_store.py    #    TemplateRecord，fill_sample，load_templates
│       ├── pipeline.py          #    工具函数：load_config, _scan_h5_files, _resolve_domain 等
│       ├── auto_register.py     #    LLM自动推断HDF5元数据（CLI工具）
│       └── interview_agent.py   #    多智能体交互式注册（6步状态机）
│
├── api_server.py                # 入口（单行 import，向后兼容）
│
├── configs/
│   ├── modflow/variants/        # 7个场景 YAML（output_dir: data/modflow）
│   ├── simpeg/variants/         # 4个场景 YAML（output_dir: data/simpeg）
│   ├── power_flow/variants/     # 5个潮流场景 YAML（output_dir: data/power_flow）
│   ├── transient/variants/      # 3个暂态场景 YAML（output_dir: data/transient）
│   ├── gcam/variants/           # 3个场景 YAML（output_dir: data/gcam）
│   └── text2comp/
│       ├── default.yaml         # 唯一配置文件（data_root/registry/output/llm/generation/seed）
│       └── registry.yaml        # 元数据注册表（auto_register 生成）
│
├── scripts/
│   ├── text2comp/
│   │   ├── generate_templates.py  # Stage 2：LLM生成模板（→ data/templates/）
│   │   └── fill_samples.py        # Stage 3：数值填充（→ data/text2comp/）
│   ├── router/
│   │   └── build_router_data.py   # Stage 4：Token Router 数据生成（→ data/router/）
│   ├── utils/summarize_all.py
│   └── ci/check_consistency.py
│
├── frontend/                    # React + Vite + Tailwind 管理界面
│   └── src/
│       ├── pages/
│       │   ├── SimulationRunner.tsx   # Stage 1：物理仿真运行
│       │   ├── RegisterSimulator.tsx  # Stage 2-01：注册数据集
│       │   ├── TemplateGenerator.tsx  # Stage 2-02：模板生成
│       │   ├── SampleFiller.tsx       # Stage 3：样本填充
│       │   ├── RouterDataBuilder.tsx  # Stage 4：Router 数据生成
│       │   ├── SampleViewer.tsx       # 样本浏览
│       │   ├── TemplateViewer.tsx     # 模板浏览
│       │   ├── RouterViewer.tsx       # 路由数据浏览
│       │   ├── DatasetStats.tsx       # 数据集统计
│       │   ├── RegistryPage.tsx       # 注册信息管理
│       │   └── LLMConfig.tsx          # LLM 配置
│       ├── hooks/useJobMonitor.ts     # SSE 状态监控（持久化，支持刷新/重开）
│       └── lib/                       # api.ts, types.ts, utils.ts
│
└── data/
    ├── modflow/      # modflow_{scenario}.h5，(N, 5, 365)
    ├── simpeg/       # simpeg_{scenario}.h5，(N, 1, 100)
    ├── power_flow/   # power_flow_{scenario}.h5，(N, 43, 365)
    ├── transient/    # transient_{scenario}.h5，(N, 5, 1000)
    ├── gcam/         # gcam_{scenario}.h5，(N, 5, 16)
    ├── templates/    # {scenario}_templates.jsonl（Stage 2 输出）
    ├── text2comp/    # {scenario}.jsonl（Stage 3 输出，最终训练样本）
    └── router/       # train.jsonl + by_scenario/（Stage 4 输出）
```

## 电力系统输出格式（重要）

**潮流仿真（pandapower）** 三个物理量垂直拼接：
```
V_bus(n_bus, 365) + theta_bus(n_bus, 365) + P_line(n_line, 365)
→ (n_bus×2+n_line, 365)

IEEE 14节点: (14+14+15, 365) = (43, 365)
```

**暂态仿真（ANDES）** 输出发电机转子角：
```
delta(n_gen, 1000)，10秒×100Hz
IEEE 14: (5, 1000)
```

power_flow 和 transient 是两个完全独立的 simulator：
- `piern/simulators/power_flow/` — pandapower 潮流，configs 在 `configs/power_flow/`，输出到 `data/power_flow/`
- `piern/simulators/transient/` — ANDES 暂态，configs 在 `configs/transient/`，输出到 `data/transient/`
- `piern/simulators/power_system/` — 共享工具库（18维参数转换 + 增强生成器），不含 pipeline

## 四阶段架构

```
Stage 1  物理仿真
  输入: configs/{simulator}/variants/*.yaml
  运行: python -m piern.simulators.{simulator}.pipeline
  输出: data/{simulator}/{simulator}_{scenario}.h5

Stage 2  语言模板生成（调 LLM）
  输入: HDF5（只读 param_names + timeseries_shape）+ registry.yaml
  运行: scripts/text2comp/generate_templates.py
  输出: data/templates/{scenario}_templates.jsonl

Stage 3  数值填充（不调 LLM）
  输入: data/templates/ + HDF5（读取实际数值）+ registry.yaml
  运行: scripts/text2comp/fill_samples.py
  输出: data/text2comp/{scenario}.jsonl（最终训练样本，5字段）

Stage 4  Token Router 数据生成（不调 LLM）
  输入: data/text2comp/*.jsonl（Stage 3 输出）
  运行: scripts/router/build_router_data.py
  输出: data/router/train.jsonl（全量合并）+ data/router/by_scenario/{scenario}.jsonl
  格式: {"context": str, "label": 0|1, "metadata": {...}}
  说明: label=1（chat_template(input) + 完整引导语，触发专家）
        label=0（chat_template(input) + 引导语内部随机截断，继续 LLM）
        Router 推理时永远看到完整 input，截断只发生在 assistant 侧引导语内部
        正负比例可配置（默认1:1），支持 qwen/deepseek/llama3/mistral/custom chat template
```

## 最终训练样本格式（5字段）

```json
{
  "input":              "自然语言描述（含参数数值）",
  "number":             [18维原始参数数组],
  "params_transformed": [18维变换后参数数组],
  "target":             "引导语 + 时序数值（JSON矩阵）",
  "metadata":           { "simulator", "scenario", "language", "style", ... }
}
```

## 配置文件说明

`configs/text2comp/default.yaml` 是唯一的配置文件，包含：
- `data_root`：HDF5 数据根目录（默认 `data`），自动扫描各子目录
- `registry`：元数据注册表路径
- `output_dir` / `output_file`：Stage 3 输出位置
- `llm`：LLM 提供商、模型、API key、温度等
- `generation`：并发数、语言比例、风格权重、变换概率等
- `seed`：全局随机种子

Stage 2/3 脚本通过 `piern.text2comp.pipeline.load_config()` 加载此文件。

## 前端侧边栏结构

```
── Stage 1 · 物理仿真 ──────────
  仿真运行  /simulate   (amber)

── Stage 2 · 语言模板 ──────────
  01 注册数据集  /register  (sky)
  02 模板生成    /templates (violet)

── Stage 3 · 样本填充 ──────────
  01 样本填充    /fill      (emerald)

── Stage 4 · 路由数据 ──────────
  路由数据  /router     (rose)

── 数据查看 ────────────────────
  模板浏览 /template-viewer
  样本浏览 /samples
  路由浏览 /router-viewer
  数据统计 /stats

── 设置 ────────────────────────
  注册信息 /registry   （含 LLM 配置入口）
  LLM 配置 /llm-config
```

默认路由：`/simulate`

## Key Files — Stage 1 API

- `piern/api/routers/simulation.py`
  - `SIMULATORS = ["modflow", "simpeg", "power_flow", "transient", "gcam"]`
  - `_get_run_pipeline(simulator)` 直接路由：power_flow → power_flow.pipeline，transient → transient.pipeline
  - `GET /api/simulation/scenarios` — 扫描场景，用 YAML 里的 output_dir 字段定位 HDF5
  - `POST /api/simulate` — 单场景仿真
  - `POST /api/simulate/batch` — 批量顺序仿真，每个场景完成发 scenario_done 事件
  - `GET /api/simulation/history` — 历史记录（内存 deque，重启清空）

## Key Files — Stage 2/3

- `piern/text2comp/pipeline.py` — 工具函数库
  - `load_config(cfg_path)` — 加载 default.yaml（兼容旧 generation_config 引用）
  - `_scan_h5_files(cfg, base_dir)` — 扫描 data_root 下各子目录的 HDF5 文件
  - `_scenario_name_from_path(h5_path)` — 从文件名提取场景名（去掉 `{simulator}_` 前缀）
  - `_load_registry(registry_path)` — 加载 registry.yaml
  - `_resolve_domain(simulator, scenario, registry)` — 解析 domain 元数据
- `piern/text2comp/generator.py` — LLMTextGenerator，fixed_channels 支持名字或整数索引
- `piern/text2comp/template_store.py` — TemplateRecord，fill_sample（接受 output_info 参数）
- `piern/text2comp/auto_register.py` — LLM 自动推断 HDF5 元数据
- `piern/text2comp/interview_agent.py` — 多智能体交互式注册（6步状态机）

## Important Notes — Stage 1

- `filter_sample()` 用 `.get()` 有默认值，传空 dict 不会 KeyError
- power_flow pipeline 在 `piern/simulators/power_flow/`，YAML 在 `configs/power_flow/`，输出到 `data/power_flow/`
- transient pipeline 在 `piern/simulators/transient/`，YAML 在 `configs/transient/`，输出到 `data/transient/`
- `power_system/` 目录现在只是共享工具库：`unified_params.py`（18维参数转换）和 `generator_with_params.py`（增强用）
- ANDES IEEE14 有 5 台发电机，暂态输出 (5, 1000)
- 增强循环有 max_rounds=50 上限（各 simulator 均已设置）
- macOS 上用 `ThreadPoolExecutor`（ProcessPoolExecutor+PIPE 会死锁）
- `terminate_job` 时 `os.killpg` kill 整个进程组
- 历史记录：内存 deque(maxlen=200)，重启后清空
- 批量仿真每个场景完成后发 `scenario_done` SSE 事件，前端实时刷新

## Important Notes — Stage 4

- Router 数据格式：`{"context": str, "label": 0|1, "metadata": {...}}`
- 正样本（label=1）：`user_prefix + input + user_suffix + assistant_prefix + 完整引导语`
- 负样本（label=0）：`user_prefix + input + user_suffix + assistant_prefix + 引导语[:pos]`，pos 在 `[0, len(引导语)-1]` 均匀随机
- Router 推理时永远看到完整 input，截断只发生在 assistant 侧引导语内部
- Chat template 通过 `--chat-template` CLI 参数指定，内置：`qwen`/`chatml`/`deepseek`/`llama3`/`mistral`，自定义用 `custom`
- `--neg-ratio N`：每条正样本生成 N 条负样本（默认1:1），各负样本截断位置独立随机
- 输出：`data/router/by_scenario/{scenario}.jsonl`（各场景独立）+ `data/router/train.jsonl`（全量合并打乱）
- 进度协议：`PROGRESS_INIT:scene:total` → `PROGRESS_UPDATE:scene:done:total` → `PROGRESS_DONE:scene:done:total`
- metadata 字段：`simulator`、`scenario`、`language`、`chat_template`、`trigger_prefix`（正样本）

## Important Notes — Stage 2/3

- 并发线程数：`default.yaml` 的 `generation.max_workers`，生产建议 32-100
- API key：`default.yaml` 的 `llm.api_key`，也可用 `SILICONFLOW_API_KEY` 环境变量
- `fill_sample` 接受 `output_info` 参数，从 registry 传入完整元数据（name_zh/unit）
- `generator.py` 的 `fixed_channels` 支持名字字符串（按 output_info.name 查找索引）
- `parseTimeseries.ts` 用括号深度匹配提取 target 中的 [[...]] 矩阵

## 待办

- [ ] MODFLOW：运行 Stage 1 生成所有场景数据
- [ ] SimPEG：运行 Stage 1 生成所有场景数据
- [ ] 电力潮流：运行 Stage 1 生成所有场景数据
- [ ] 电力暂态：运行 Stage 1 生成所有场景数据
- [ ] GCAM：运行 Stage 1 生成所有场景数据
- [ ] Stage 2/3：为所有场景生成语言模板和训练样本
- [ ] Stage 4：运行 scripts/router/build_router_data.py 生成 Router 训练数据
