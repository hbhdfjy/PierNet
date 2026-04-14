# CLAUDE.md

本文件为 Claude Code 在此仓库中工作提供指导。

## 项目定位

**PiERN 多模拟器数据合成管线**。

**核心卖点**：首个同时覆盖五种数学结构的跨物理域仿真数据集，全部使用权威开源工具。

## 五大模拟器 · 五种数学结构

| 模拟器 | 数学类型 | 领域 | 输出形状 | 场景数 | 目标样本 | 状态 |
|--------|---------|------|---------|--------|---------|------|
| **MODFLOW** | 抛物型PDE | 地下水 | (5, 365) | 7 | 70,000 | ✅ 代码完成 |
| **SimPEG** | 椭圆型PDE | 地球物理 | (1, 100) | 4 | 20,000 | ✅ 代码完成 |
| **pandapower** | 非线性代数方程组 | 稳态潮流 | (43, 365) | 5 | 50,000 | ✅ 代码完成 |
| **ANDES** | DAE系统 | 暂态稳定 | (5, 1000) | 3 | 15,000 | ✅ 代码完成 |
| **GCAM简化版** | 动态代数系统 | 能源-气候 | (5, 16) | 3 | 3,000 | ✅ 代码完成 |
| **总计** | 5种数学结构 | 3大领域 | — | **22** | **158,000** | — |

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
data/modflow/        — 7场景，各约1000样本（目标：各10,000）
data/simpeg/         — 4场景，各约1000样本（目标：各5,000）
data/power_system/   — 8场景（5潮流+3暂态），各约1000样本
                       目标：5潮流×10,000 + 3暂态×5,000
data/gcam/           — 3场景，各约1000样本（目标：各1,000）
data/templates/      — 多场景模板库（Stage 2 阶段一输出，*_templates.jsonl）
data/text2comp/      — 最终训练样本（Stage 3 阶段二输出，*.jsonl）
```

## 常用命令

```bash
# 安装
pip install -r requirements.txt
pip install -e .

# Stage 1：物理仿真（推荐通过前端管理界面启动）
./start_ui.sh   # 启动 FastAPI(8000) + Vite(5173)，访问 http://localhost:5173

# 也可 CLI 直接运行：

# MODFLOW（抛物型PDE）
python -m piern.simulators.modflow.pipeline \
    --config configs/modflow/variants/unified_aquifer.yaml --n-samples 1000

# SimPEG（椭圆型PDE）
python -m piern.simulators.simpeg.pipeline \
    --config configs/simpeg/variants/dc_resistivity.yaml --n-samples 1000

# pandapower 潮流（非线性代数，输出 (43, 365)）
# 注意：configs 在 power_flow/ 目录，HDF5 输出到 data/power_system/
python -m piern.simulators.power_system.pipeline \
    --config configs/power_flow/variants/ieee14_baseload.yaml --n-samples 1000

# ANDES 暂态（DAE，输出 (5, 1000)）
# 注意：configs 在 transient/ 目录，HDF5 输出到 data/power_system/
python -m piern.simulators.power_system.pipeline \
    --config configs/transient/variants/ieee14_fault.yaml --n-samples 500

# GCAM（动态代数，输出 (5, 16)）
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
│   │   │   └── simulation.py    #     /api/simulation/*, /api/simulate（Stage 1 仿真）
│   │   ├── schemas/             #   Pydantic 模型
│   │   └── services/            #   业务逻辑（job_manager, file_manager）
│   ├── simulators/
│   │   ├── modflow/             # ✅ 抛物型PDE，flopy/MODFLOW
│   │   ├── simpeg/              # ✅ 椭圆型PDE，SimPEG
│   │   ├── power_system/        # ✅ 非线性代数+DAE，pandapower+ANDES
│   │   │                        #    power_flow 和 transient 共用此 pipeline
│   │   └── gcam/                # ✅ 动态代数，PyPSA+HiGHS
│   ├── text2comp/               # ✅ Stage 2/3 语言模板生成
│   │   ├── generator.py         #    LLMTextGenerator（占位符机制，并发）
│   │   ├── template_store.py    #    TemplateRecord，fill_sample，load_templates
│   │   ├── pipeline.py          #    run_pipeline()（旧版一步流程，仍可用）
│   │   ├── auto_register.py     #    LLM自动推断HDF5元数据（CLI工具）
│   │   └── interview_agent.py   #    多智能体交互式注册（6步状态机）
│   └── router/                  # ❌ Stage 4 Token Router（待实现）
│
├── api_server.py                # 入口（单行 import，向后兼容）
│
├── configs/
│   ├── modflow/variants/        # 7个场景 YAML
│   ├── simpeg/variants/         # 4个场景 YAML
│   ├── power_flow/variants/     # 5个潮流场景 YAML（ieee14_*）
│   │                            #   输出到 data/power_system/，pipeline: power_system
│   ├── transient/variants/      # 3个暂态场景 YAML（ieee14_*）
│   │                            #   输出到 data/power_system/，pipeline: power_system
│   ├── gcam/variants/           # 3个场景 YAML
│   └── text2comp/
│       ├── generation.yaml      # 生成超参数（LLM/并发/语言/变换，任务无关）
│       ├── default.yaml         # 任务配置（data_dirs/output/registry）
│       └── registry.yaml        # 元数据注册表（auto_register生成）
│
├── scripts/
│   ├── text2comp/
│   │   ├── generate_templates.py  # Stage 2：LLM生成模板（→ data/templates/）
│   │   └── fill_samples.py        # Stage 3：数值填充（→ data/text2comp/）
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
│       │   ├── SampleViewer.tsx       # 样本浏览
│       │   ├── TemplateViewer.tsx     # 模板浏览
│       │   └── DatasetStats.tsx       # 数据集统计
│       ├── hooks/useJobMonitor.ts     # SSE 状态监控（持久化，支持刷新/重开）
│       └── lib/                       # api.ts, types.ts, utils.ts
│
└── data/
    ├── modflow/      # (N, 5, 365)
    ├── simpeg/       # (N, 1, 100)
    ├── power_system/ # (N, 43, 365) 潮流 或 (N, 5, 1000) 暂态
    │                 # power_flow 和 transient 两类场景共用此目录
    ├── gcam/         # (N, 5, 16)
    ├── templates/    # *_templates.jsonl（Stage 2 阶段一输出）
    └── text2comp/    # *.jsonl（Stage 3 阶段二输出，最终训练样本）
```

## 电力系统输出格式（重要）

潮流仿真（pandapower）三个物理量垂直拼接：
```
V_bus(n_bus, 365) + theta_bus(n_bus, 365) + P_line(n_line, 365)
→ (n_bus×2+n_line, 365)

IEEE 14节点: (14+14+15, 365) = (43, 365)
```

暂态仿真（ANDES）输出发电机转子角：
```
delta(n_gen, 1000)，10秒×100Hz
IEEE 14: (5, 1000)
```

## 三阶段架构

```
Stage 1  物理仿真
  输入: configs/{simulator}/variants/*.yaml
  运行: python -m piern.simulators.{simulator}.pipeline
  输出: data/{simulator}/*.h5

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
  输出: data/router/train.jsonl, val.jsonl, test.jsonl
  格式: {"context": str, "label": 0|1, "metadata": {...}}
  说明: label=1（input+引导语，触发专家），label=0（input 随机截断，继续 LLM）
        正负比例 1:1，按 8:1:1 划分 train/val/test
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

## 前端侧边栏结构

```
── Stage 1 · 物理仿真 ──────────
  仿真运行  /simulate

── Stage 2 · 语言模板 ──────────
  01 注册数据集  /register
  02 模板生成    /templates

── Stage 3 · 样本填充 ──────────
  01 样本填充    /fill

── 数据查看 ────────────────────
  模板浏览 / 样本浏览 / 数据统计

── 设置 ────────────────────────
  注册信息（含数据目录 Tab）/ LLM 配置
```

## 待办

- [ ] MODFLOW：7个场景各生成10,000样本
- [ ] SimPEG：4个场景各生成5,000样本
- [ ] 电力潮流：5个场景各生成10,000样本
- [ ] 电力暂态：3个场景各生成5,000样本
- [ ] GCAM：3个场景各生成1,000样本
- [ ] Stage 2/3：全量生成（每场景10,000条语言模板 + 对应训练样本）
- [ ] Stage 4：运行 scripts/router/build_router_data.py 生成 Router 训练数据
