# PiERN 多模拟器数据合成管线

**PiERN**（Physically-isolated Experts Routing Network）论文数据合成代码，投稿 ICML 2026。

> *"PiERN-Dataset is the first large-scale simulation dataset spanning three physical domains and five mathematical structures: parabolic PDEs (MODFLOW), elliptic PDEs (SimPEG), nonlinear algebraic systems (pandapower), differential-algebraic equations (ANDES), and dynamic algebraic systems (GCAM). All simulators share a unified 18-dimensional parameter representation."*

---

## 五大模拟器 · 五种数学结构

| 模拟器 | 数学类型 | 核心方程 | 领域 | 求解方法 |
|--------|---------|---------|------|---------|
| **MODFLOW** | 抛物型PDE | $S_s\frac{\partial h}{\partial t}=\nabla(K\nabla h)+W$ | 地下水流动 | FDM + PCG |
| **SimPEG** | 椭圆型PDE | $\nabla\cdot(\sigma\nabla\phi)=-I\delta$ | 地球物理勘探 | FVM + 直接求解 |
| **pandapower** | 非线性代数方程组 | $P_i=\sum V_iV_j(G_{ij}\cos\theta_{ij}+B_{ij}\sin\theta_{ij})$ | 稳态潮流 | Newton-Raphson |
| **ANDES** | DAE系统 | $\dot{\delta}=\omega,\ M\dot{\omega}=P_m-P_e-D\omega,\ 0=g(x,y)$ | 暂态稳定 | 隐式梯形法 |
| **GCAM简化版** | 动态代数系统 | 多期LP能源优化 | 能源-气候耦合 | PyPSA + HiGHS |

---

## 数据集规模

| 领域 | 模拟器 | 场景数 | 样本数目标 | 输出形状 |
|------|--------|--------|-----------|---------|
| 地质 | MODFLOW | 7 | 70,000 | (5, 365) |
| 地质 | SimPEG | 4 | 20,000 | (1, 100) |
| 电力 | pandapower（潮流） | 5 | 50,000 | (43, 365) |
| 电力 | ANDES（暂态） | 3 | 15,000 | (5, 1000) |
| 能源-气候 | GCAM | 3 | 3,000 | (5, 16) |
| **总计** | **5个模拟器** | **22个场景** | **158,000** | 统一18维参数 |

---

## 安装

```bash
# Python 依赖（含所有模拟器）
pip install -r requirements.txt
pip install -e .

# MODFLOW 可执行文件（运行地下水仿真必须）
python -c "import flopy; flopy.utils.get_modflow()"
# 或手动下载：https://water.usgs.gov/water-resources/software/MODFLOW-2005/

# 前端（Node.js 18+）
cd frontend && npm install
```

---

## 快速开始

### 推荐：前端管理界面

```bash
./start_ui.sh   # 启动 FastAPI(8000) + Vite(5173)，访问 http://localhost:5173
```

前端提供完整的三阶段工作流：Stage 1 物理仿真 → Stage 2 语言模板生成 → Stage 3 数值填充。

### CLI 方式

```bash
# Stage 1：物理仿真数据生成

# 地质（MODFLOW，抛物型PDE）
python -m piern.simulators.modflow.pipeline \
    --config configs/modflow/variants/unified_aquifer.yaml --n-samples 1000

# 地球物理（SimPEG，椭圆型PDE）
python -m piern.simulators.simpeg.pipeline \
    --config configs/simpeg/variants/dc_resistivity.yaml --n-samples 1000

# 电力潮流（pandapower，非线性代数）
# configs 在 power_flow/ 目录，HDF5 输出到 data/power_system/
python -m piern.simulators.power_system.pipeline \
    --config configs/power_flow/variants/ieee14_baseload.yaml --n-samples 1000

# 电力暂态（ANDES，DAE）
# configs 在 transient/ 目录，HDF5 输出到 data/power_system/
python -m piern.simulators.power_system.pipeline \
    --config configs/transient/variants/ieee14_fault.yaml --n-samples 500

# 能源-气候（GCAM，动态代数）
python -m piern.simulators.gcam.pipeline \
    --config configs/gcam/variants/energy_transition.yaml --n-samples 1000

# 汇总所有数据
python scripts/utils/summarize_all.py

# Stage 2：语言模板生成（调 LLM）

# Step 0：自动注册元数据（新数据集先跑此步）
python -m piern.text2comp.auto_register \
    --config configs/text2comp/default.yaml \
    --output configs/text2comp/registry.yaml

# Step 1：生成语言模板（结果存 data/templates/）
python scripts/text2comp/generate_templates.py \
    --config configs/text2comp/default.yaml \
    --n-templates 1000

# Stage 3：数值填充（不调 LLM，结果存 data/text2comp/）
python scripts/text2comp/fill_samples.py \
    --config configs/text2comp/default.yaml \
    --n-samples 1000 --skip-existing
```

---

## 项目结构

```
piern/
├── piern/
│   ├── core/                    # HDF5存储、质量过滤、LLM客户端
│   ├── api/                     # FastAPI 后端（分层架构）
│   │   ├── routers/
│   │   │   ├── simulation.py    #   /api/simulation/*, /api/simulate（Stage 1）
│   │   │   ├── generation.py    #   /api/generate-templates, /api/fill-samples
│   │   │   ├── datasets.py      #   /api/datasets, /api/samples, /api/stats
│   │   │   ├── config.py        #   /api/config, /api/llm-config
│   │   │   ├── registry.py      #   /api/registry CRUD
│   │   │   ├── jobs.py          #   /api/generate/{id}/stream（SSE）
│   │   │   ├── files.py         #   /api/files/templates, /api/files/samples
│   │   │   └── interview.py     #   /api/interview/*
│   │   └── services/            #   job_manager, file_manager
│   ├── simulators/
│   │   ├── modflow/             # ✅ 抛物型PDE，flopy/MODFLOW
│   │   ├── simpeg/              # ✅ 椭圆型PDE，SimPEG
│   │   ├── power_system/        # ✅ 非线性代数+DAE，pandapower+ANDES
│   │   │                        #    power_flow 和 transient 共用此 pipeline
│   │   └── gcam/                # ✅ 动态代数，PyPSA+HiGHS
│   └── text2comp/               # ✅ Stage 2/3 语言模板生成
│
├── configs/
│   ├── modflow/variants/        # 7个场景 YAML
│   ├── simpeg/variants/         # 4个场景 YAML
│   ├── power_flow/variants/     # 5个潮流场景（ieee14_baseload/renewable/peak/light/voltage_stress）
│   ├── transient/variants/      # 3个暂态场景（ieee14_fault/gentrip/load_step）
│   ├── gcam/variants/           # 3个场景 YAML
│   └── text2comp/               # generation.yaml, default.yaml, registry.yaml
│
├── scripts/
│   ├── text2comp/
│   │   ├── generate_templates.py  # Stage 2：LLM生成模板
│   │   └── fill_samples.py        # Stage 3：数值填充
│   └── utils/summarize_all.py
│
├── frontend/                    # React + Vite + Tailwind 管理界面
│   └── src/pages/
│       ├── SimulationRunner.tsx   # Stage 1：物理仿真
│       ├── RegisterSimulator.tsx  # Stage 2-01：注册数据集
│       ├── TemplateGenerator.tsx  # Stage 2-02：模板生成
│       ├── SampleFiller.tsx       # Stage 3：样本填充
│       ├── SampleViewer.tsx       # 样本浏览
│       └── DatasetStats.tsx       # 数据集统计
│
└── data/
    ├── modflow/      # (N, 5, 365)
    ├── simpeg/       # (N, 1, 100)
    ├── power_system/ # (N, 43, 365) 潮流 或 (N, 5, 1000) 暂态
    ├── gcam/         # (N, 5, 16)
    ├── templates/    # *_templates.jsonl（Stage 2 输出）
    └── text2comp/    # *.jsonl（Stage 3 输出，最终训练样本）
```

---

## 文档

- [`CLAUDE.md`](CLAUDE.md) — 开发指南与常用命令

---

## 许可证

MIT License
