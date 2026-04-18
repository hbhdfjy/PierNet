# PiERN 多模拟器数据合成管线

## 文档入口

后续我会优先维护以下三份文档，并把它们作为项目整体 overview 的正式入口：

- `PROJECT_OVERVIEW.md`：项目级总览，定义系统边界、阶段划分、核心组件和文档分工。
- `README.md`：面向使用者的安装、启动、快速开始和常用命令。
- `CLAUDE.md`：面向开发与代码代理的工作上下文，默认以 `PROJECT_OVERVIEW.md` 为总览基线。

其中 `UPGRADE_PLAN.md` 保留为演进路线与问题清单，不再承担项目 overview 的职责。

**PiERN**（Physically-isolated Experts Routing Network）多模拟器数据合成管线。

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

| 领域 | 模拟器 | 场景数 | 输出形状 |
|------|--------|--------|-----------|---------|
| 地质 | MODFLOW | 7 | (5, 365) |
| 地质 | SimPEG | 4 | (1, 100) |
| 电力 | pandapower（潮流） | 5 | (43, 365) |
| 电力 | ANDES（暂态） | 3 | (5, 1000) |
| 能源-气候 | GCAM | 3 | (5, 16) |
| **总计** | **5个模拟器** | **22个场景** | 统一18维参数 |

---

## 安装

```bash
# Python ??????????
pip install -r requirements.txt
pip install -e .
# ???SimPEG ????????? requirements?PyPSA ??? xarray<2026 ????

# MODFLOW ????????????????
python - <<'PY'
from pathlib import Path
from flopy.utils import get_modflow
get_modflow(str(Path.home() / '.flopy_bin'), subset='mf2005')
PY
# ??????????????
export PIERN_MODFLOW_EXE=/path/to/mf2005

# ???Node.js 18+?
cd frontend && npm install
```

---

## 快速开始

### 推荐：前端管理界面

```bash
./start_ui.sh   # 启动 FastAPI(8000) + Vite(5173)，访问 http://localhost:5173
```

前端提供完整的四阶段工作流：

```
Stage 1 物理仿真 → Stage 2 语言模板生成 → Stage 3 数值填充 → Stage 4 Router 数据
```

### CLI 方式

```bash
# ── Stage 1：物理仿真数据生成 ────────────────────────────────────

# 地质（MODFLOW，抛物型PDE）→ data/modflow/modflow_{scenario}.h5
python -m piern.simulators.modflow.pipeline \
    --config configs/modflow/variants/unified_aquifer.yaml --n-samples 1000

# 地球物理（SimPEG，椭圆型PDE）→ data/simpeg/simpeg_{scenario}.h5
python -m piern.simulators.simpeg.pipeline \
    --config configs/simpeg/variants/dc_resistivity.yaml --n-samples 1000

# 电力潮流（pandapower，非线性代数）→ data/power_flow/power_flow_{scenario}.h5
python -m piern.simulators.power_flow.pipeline \
    --config configs/power_flow/variants/ieee14_baseload.yaml --n-samples 1000

# 电力暂态（ANDES，DAE）→ data/transient/transient_{scenario}.h5
python -m piern.simulators.transient.pipeline \
    --config configs/transient/variants/ieee14_fault.yaml --n-samples 500

# 能源-气候（GCAM，动态代数）→ data/gcam/gcam_{scenario}.h5
python -m piern.simulators.gcam.pipeline \
    --config configs/gcam/variants/energy_transition.yaml --n-samples 1000

# 汇总所有数据
python scripts/utils/summarize_all.py

# ── Stage 2：语言模板生成（调 LLM）────────────────────────────────

# Step 0：自动注册元数据（新数据集先跑此步）
python -m piern.text2comp.auto_register \
    --config configs/text2comp/default.yaml \
    --output configs/text2comp/registry.yaml

# Step 1：生成语言模板 → data/templates/{scenario}_templates.jsonl
python scripts/text2comp/generate_templates.py \
    --config configs/text2comp/default.yaml \
    --n-templates 1000

# ── Stage 3：数值填充（不调 LLM）──────────────────────────────────

# → data/text2comp/{scenario}.jsonl
python scripts/text2comp/fill_samples.py \
    --config configs/text2comp/default.yaml \
    --n-samples 1000 --skip-existing

# ── Stage 4：Token Router 数据生成（不调 LLM）──────────────────────

# → data/router/train.jsonl + data/router/by_scenario/{scenario}.jsonl
# 默认不包裹 chat template，可用 --chat-template qwen/deepseek/llama3/mistral/custom
python scripts/router/build_router_data.py --seed 42
python scripts/router/build_router_data.py --seed 42 --chat-template qwen --neg-ratio 2

# 重建 Stage 2-4 摘要读取使用的 sidecar manifest
python scripts/utils/rebuild_manifests.py

# 重建 Stage 2-4 无筛选分页使用的 sparse offset indexes
python scripts/utils/rebuild_indexes.py

# 重建 Stage 2-4 常用筛选（language/style/label）索引
python scripts/utils/rebuild_filter_indexes.py
```

## Token Router ??

???????? CLI ??????????? `piern/training/router/`?????? `scripts/router/train_token_router.py`?

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/router/train_token_router.py \
    --simulator modflow \
    --device cuda:0 \
    --epochs 1
```

## 读性能说明

- Stage 2/3/4 的 `JSONL` 仍然是源产物格式。
- 前端摘要和列表接口优先读取 `data/.manifests/` 下的 sidecar manifest，避免反复全量扫描大文件。
- 无筛选分页接口优先读取 `data/.indexes/` 下的 sparse offset index，避免从文件头顺扫到目标页。
- 常用筛选分页（`language` / `style` / `label`）优先读取 `data/.indexes/` 下的 filter index。
- 批量生成或删除模板、样本、router 数据后，可手动刷新 manifest：

```bash
python scripts/utils/rebuild_manifests.py
python scripts/utils/rebuild_indexes.py
python scripts/utils/rebuild_filter_indexes.py
```

- 统计页现在优先通过单个 `/api/dashboard/summary` 聚合接口读取摘要。
- 后端摘要接口在 manifest 缺失或快照过期时会自动重建；分页索引缺失时会按需重建。

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
│   │   │   ├── router_data.py   #   /api/router/*（Stage 4）
│   │   │   ├── datasets.py      #   /api/datasets, /api/samples, /api/stats
│   │   │   ├── config.py        #   /api/config, /api/llm-config
│   │   │   ├── registry.py      #   /api/registry CRUD
│   │   │   ├── jobs.py          #   /api/generate/{id}/stream（SSE）
│   │   │   ├── files.py         #   /api/files/templates, /api/files/samples
│   │   │   └── interview.py     #   /api/interview/*
│   │   └── services/            #   job_manager, file_manager, manifest_store, jsonl_index, jsonl_filter_index
│   ├── simulators/
│   │   ├── modflow/             # ✅ 抛物型PDE，flopy/MODFLOW
│   │   ├── simpeg/              # ✅ 椭圆型PDE，SimPEG
│   │   ├── power_flow/          # ✅ 非线性代数，pandapower（稳态潮流）
│   │   ├── transient/           # ✅ DAE系统，ANDES（暂态稳定）
│   │   ├── power_system/        # 共享工具库（unified_params, generator_with_params）
│   │   └── gcam/                # ✅ 动态代数，PyPSA+HiGHS
│   └── text2comp/               # ✅ Stage 2/3 语言模板生成
│
├── configs/
│   ├── modflow/variants/        # 7个场景 YAML
│   ├── simpeg/variants/         # 4个场景 YAML
│   ├── power_flow/variants/     # 5个潮流场景（ieee14_baseload/renewable/peak/light/voltage_stress）
│   ├── transient/variants/      # 3个暂态场景（ieee14_fault/gentrip/load_step）
│   ├── gcam/variants/           # 3个场景 YAML
│   └── text2comp/
│       ├── default.yaml         # 唯一配置文件（LLM/生成超参/数据路径）
│       └── registry.yaml        # 元数据注册表（auto_register 生成）
│
├── scripts/
│   ├── text2comp/
│   │   ├── generate_templates.py  # Stage 2：LLM生成模板
│   │   └── fill_samples.py        # Stage 3：数值填充
│   ├── router/
│   │   └── build_router_data.py   # Stage 4：Router 数据生成
│   └── utils/
│       ├── summarize_all.py
│       ├── rebuild_manifests.py   # 重建 Stage 2-4 摘要 manifest
│       ├── rebuild_indexes.py     # 重建 Stage 2-4 无筛选分页索引
│       └── rebuild_filter_indexes.py # 重建 Stage 2-4 常用筛选索引
│
├── frontend/                    # React + Vite + Tailwind 管理界面
│   └── src/pages/
│       ├── SimulationRunner.tsx   # Stage 1：物理仿真
│       ├── RegisterSimulator.tsx  # Stage 2-01：注册数据集
│       ├── TemplateGenerator.tsx  # Stage 2-02：模板生成
│       ├── SampleFiller.tsx       # Stage 3：样本填充
│       ├── RouterDataBuilder.tsx  # Stage 4：Router 数据生成
│       ├── SampleViewer.tsx       # 样本浏览
│       ├── TemplateViewer.tsx     # 模板浏览
│       ├── RouterViewer.tsx       # 路由数据浏览
│       ├── DatasetStats.tsx       # 数据集统计
│       └── RegistryPage.tsx       # 注册信息管理
│
└── data/
    ├── modflow/      # modflow_{scenario}.h5，(N, 5, 365)
    ├── simpeg/       # simpeg_{scenario}.h5，(N, 1, 100)
    ├── power_flow/   # power_flow_{scenario}.h5，(N, 43, 365)
    ├── transient/    # transient_{scenario}.h5，(N, 5, 1000)
    ├── gcam/         # gcam_{scenario}.h5，(N, 5, 16)
    ├── templates/    # {scenario}_templates.jsonl（Stage 2 输出）
    ├── text2comp/    # {scenario}.jsonl（Stage 3 输出，最终训练样本）
    ├── router/       # train.jsonl + by_scenario/（Stage 4 输出）
    ├── .manifests/   # templates/samples/router 摘要 sidecar（前端摘要读取）
    └── .indexes/     # 模板/样本/router JSONL 的 sparse offset index
```

---

## HDF5 文件命名约定

所有 HDF5 文件统一命名为 `{simulator}_{scenario}.h5`：

```
data/modflow/modflow_unified_aquifer.h5
data/power_flow/power_flow_ieee14_baseload.h5
data/transient/transient_ieee14_fault.h5
```

Stage 2/3 脚本通过 `data_root` 配置自动扫描各子目录，目录名即 simulator 名，无需额外配置。

---

## 配置文件

`configs/text2comp/default.yaml` 是 Stage 2/3 的唯一配置文件（Stage 4 通过 CLI 参数配置）：

```yaml
data_root: data          # HDF5 数据根目录，自动扫描各子目录
registry: configs/text2comp/registry.yaml
output_dir: data/text2comp
output_file: all_training_data.jsonl

llm:
  provider: siliconflow
  model: deepseek-ai/DeepSeek-V3.2
  api_key: ''           # 留空，通过前端 LLM 配置页面设置，或 SILICONFLOW_API_KEY 环境变量
  temperature: 1.0
  max_tokens: 8196

generation:
  n_samples_per_scenario: 1000
  max_workers: 5
  language_mix: 0.5       # 中文比例
  transform_prob: 0.1     # 参数变换概率

seed: 42
```

---

## Stage 4 Router 数据格式

Token Router 训练数据为二分类：

```json
{"context": "<|im_start|>user\n用户输入<|im_end|>\n<|im_start|>assistant\n引导语", "label": 1, "metadata": {...}}
{"context": "<|im_start|>user\n用户输入<|im_end|>\n<|im_start|>assistant\n截断", "label": 0, "metadata": {...}}
```

- **label=1**：完整 `input + 完整引导语`，表示应触发专家模型
- **label=0**：完整 `input + 引导语内部随机截断`，表示应继续 LLM 生成
- Router 推理时永远看到完整 input，截断只发生在 assistant 侧引导语内部
- 支持 5 种内置 chat template：`qwen`、`deepseek`、`llama3`、`mistral`、`chatml`，以及 `custom`（自定义前缀/后缀）

---

## 文档

- [`CLAUDE.md`](CLAUDE.md) — 开发指南与详细架构说明

---

## 许可证

MIT License


---

## ???????????

??????????????

- ??????????? Stage 1-4 ???????
- ????????????????????????????

?????????

- ???`/training`
- ???????
- ?????????
- ?????????

????????????????????????????????

