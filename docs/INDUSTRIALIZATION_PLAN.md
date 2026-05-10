# PiERN 工业化升级计划书

版本日期：2026-05-10
适用范围：PiERN 数据合成平台、Token Router 训练平台、服务启动脚本、迁移与部署流程。

## 1. 目标

PiERN 当前已经具备完整功能闭环：原始科学数据、语言模板、样本填充、Router 数据构建、训练任务管理、文件管理和前端工作台。本计划的目标不是重写系统，而是在保持当前科研效率和迁移能力的前提下，把项目逐步提升到工业团队可维护、可测试、可部署、可迁移的标准。

最终状态应满足：

- 新服务器可以在明确步骤内恢复完整服务。
- GitHub 只保存代码、配置、文档、语言模板和最原始科学计算数据。
- Stage 3 样本、Stage 4 Router 数据、索引、manifest、训练缓存、权重默认作为派生或运行时数据处理。
- 前后端 API 契约可验证，类型尽量自动同步。
- 每次提交都可以自动跑质量门禁。
- 后台任务可以安全终止、恢复状态、避免并发冲突。
- 服务配置不绑定单台服务器路径，迁移只需改环境变量或部署配置。

## 2. 工业化原则

### 2.1 可复现优先

所有运行时派生物必须能由 Git 中保存的源数据和配置重建。不能重建的内容才允许进入备份策略，例如关键 checkpoint 或人工标注资产。

### 2.2 配置外置

服务器 IP、端口、模型路径、数据根目录、Conda 环境、CORS 来源、日志级别、最大并发、GPU 选择等不应硬编码在业务代码里。默认值可以保留，但必须可由环境变量或配置文件覆盖。

### 2.3 前后端契约优先

后端 FastAPI 是事实 API 源。前端类型应尽量由 OpenAPI schema 生成，减少手写类型漂移。

### 2.4 任务状态可恢复

长任务不能只依赖浏览器 localStorage 或后端进程内存。工业化后应将任务元数据持久化到 SQLite 或 PostgreSQL，进程重启后至少可以看到历史任务状态和日志。

### 2.5 渐进式改造

项目已经能服务当前工作流，升级必须分阶段执行。每个阶段都要有明确验收标准，避免一次性大改导致不可用。

## 3. 当前状态评估

### 3.1 已达标部分

- 前端采用 React、TypeScript strict、Vite、SWR、React Router，技术栈合理。
- 后端采用 FastAPI、Pydantic、模块化 router，主入口清晰。
- 已有 `scripts/services/start.sh`、`status.sh`、`restart.sh`、`stop.sh`，适合远程服务器持久运行。
- 已有 `scripts/ci/check_consistency.py`，可以检查路由注册、文档引用、API 前后缀一致性、配置路径等。
- Git 忽略规则已经排除大部分派生数据、Parquet、artifacts、runlogs。
- 数据迁移策略已经明确：GitHub 保存原始 HDF5 和语言模板，派生数据由平台重建。

### 3.2 主要差距

- 缺少统一 `pyproject.toml`，Python 依赖、ruff、pytest、mypy 配置分散。
- 前端缺少 ESLint、Prettier、Vitest、Playwright 质量门禁。
- 没有 GitHub Actions 或等价 CI。
- 前端 API 类型主要手写，缺少 OpenAPI 自动生成。
- 后台任务仍以进程内存为主，重启后不可完整恢复。
- 部分文件过大，长期维护成本高。
- 部署配置仍带有服务器路径和端口假设。
- 可观测性不足，缺少结构化日志、任务指标、健康检查分层和资源监控历史。

## 4. 目标架构

### 4.1 仓库结构目标

推荐最终结构：

```text
piern/
  api/                    # FastAPI app 装配
  synth/                  # 数据合成平台后端
  training/               # 训练平台后端
  shared/                 # 共享 runtime、storage、API utilities
frontend/
  src/
    lib/                  # API client、types、公共工具
    platform/             # 平台入口
    synth/                # 合成平台前端
    training/             # 训练平台前端
    shared/               # 通用组件和主题
docs/
  PORTABLE_STORAGE.md
  MIGRATION_RESTORE.md
  INDUSTRIALIZATION_PLAN.md
scripts/
  services/               # 服务生命周期
  ci/                     # 本地和 CI 一致性检查
  storage/                # 存储迁移和目录数据库
  text2comp/              # 样本生成脚本
  router/                 # Router 数据和训练脚本
tests/
  backend and script tests
```

原则：

- `piern/` 内只放可 import 的 Python 包。
- `scripts/` 内放命令行入口和批处理脚本。
- `frontend/` 内前端独立构建。
- `data/` 只允许跟踪原始 HDF5、语言模板和必要 `.gitignore`。
- `artifacts/`、`.runlogs/`、Parquet、SQLite catalog、manifest、index 默认不进 Git。

### 4.2 数据层目标

Git 保存：

- `data/{simulator}/*.h5`
- `data/templates/*.jsonl`
- 配置、registry、文档、代码

运行时重建：

- `data/text2comp_parquet/`
- `data/router_parquet/`
- `data/catalog.sqlite`
- `data/.manifests/`
- `data/.indexes/`
- `artifacts/token_router/*/prepared/`

可选备份：

- 明确需要保留的训练 checkpoint
- 训练曲线和最终模型摘要
- 重要实验配置快照

### 4.3 任务系统目标

短期：

- 继续使用当前 in-process job manager。
- 所有 progress 事件统一携带 stats。
- 同类或互斥任务启动前进行后端保护。

中期：

- 使用 SQLite 持久化任务表、任务事件表、任务日志表。
- 后端重启后可以列出历史任务。
- 对正在运行但进程消失的任务标记为 `external_terminated`。

长期：

- 将长任务执行器迁移为独立 worker。
- 可选方案：RQ/Redis、Celery/Redis、Dramatiq、Arq，或轻量本地 SQLite 队列。
- Web 进程只负责任务提交、查询和 SSE/WebSocket 推送。

### 4.4 服务部署目标

开发模式：

- FastAPI：`0.0.0.0:8000`
- Vite：`0.0.0.0:5173`

生产或长期服务模式：

- FastAPI 统一服务静态前端 `frontend/dist`。
- 可选 Nginx 反向代理。
- systemd 或 supervisor 管理进程。
- 所有路径和端口通过 `.env` 或环境变量配置。

## 5. 分阶段实施计划

## 阶段 0：基线冻结和风险清单

目标：在继续开发前固定当前基线，保证之后每次改造都能判断是否破坏功能。

任务：

1. 确认当前主分支可启动。
2. 记录服务器启动方式、端口、模型路径、数据保存策略。
3. 保留 `docs/PORTABLE_STORAGE.md` 作为迁移策略源文档。
4. 执行并记录以下检查：
   - `python scripts/ci/check_consistency.py`
   - `pytest`
   - `cd frontend && npm run build`
   - `scripts/services/status.sh`
5. 确认 `.gitignore` 不会误纳入派生数据。

验收标准：

- 上述命令通过，或失败项有明确 issue。
- Git 工作区干净。
- 当前服务可访问。

## 阶段 1：工程配置标准化

目标：让本地开发、CI、服务器部署使用同一套质量标准。

### 1.1 Python 工程配置

新增 `pyproject.toml`，统一管理：

- project metadata
- dependencies
- optional dev dependencies
- ruff 配置
- pytest 配置
- mypy 或 pyright 配置

推荐结构：

```toml
[project]
name = "piern"
requires-python = ">=3.11"

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[tool.ruff]
line-length = 120

[tool.pytest.ini_options]
testpaths = ["tests"]
```

迁移策略：

- 第一阶段保留 `requirements.txt` 和 `setup.py`，避免破坏现有服务器安装。
- `pyproject.toml` 建好后，把 `requirements.txt` 逐步变成导出文件或兼容入口。
- 最终以 `pyproject.toml` 为唯一源，`requirements.txt` 由工具生成。

验收标准：

- `pip install -e .` 可用。
- `python -m pytest` 可用。
- `ruff check .` 可用。

### 1.2 前端工程配置

新增或完善：

- ESLint
- Prettier
- Vitest
- Playwright
- `npm run lint`
- `npm run format:check`
- `npm run test`
- `npm run typecheck`

建议脚本：

```json
{
  "scripts": {
    "typecheck": "tsc --noEmit",
    "lint": "eslint src --ext .ts,.tsx",
    "format:check": "prettier --check src",
    "test": "vitest run",
    "e2e": "playwright test",
    "build": "npm run typecheck && node ../scripts/frontend/run-vite.mjs build"
  }
}
```

迁移策略：

- 先只对新增文件严格 lint。
- 再逐步修复已有文件。
- 对超大页面先不强制拆分，避免一次性重构。

验收标准：

- `npm run build` 通过。
- `npm run lint` 至少对核心目录通过。
- `npm run typecheck` 独立可运行。

## 阶段 2：CI/CD 质量门禁

目标：每次 push 都自动验证项目不会明显坏掉。

新增 GitHub Actions：

1. Backend CI
   - 安装 Python 3.11
   - 安装依赖
   - 运行 `ruff check`
   - 运行 `pytest`
   - 运行 `scripts/ci/check_consistency.py`

2. Frontend CI
   - 安装 Node 20
   - `npm ci`
   - `npm run typecheck`
   - `npm run build`

3. Optional smoke CI
   - 启动 FastAPI
   - 请求 `/api/training/gpus` 或 `/api/config`
   - 检查 OpenAPI schema 可生成

迁移策略：

- 第一版 CI 不跑大型仿真，不下载模型，不生成百万样本。
- 大型数据闭环放在手动 workflow 或服务器定期 smoke test 中。

验收标准：

- PR 或 push 后自动运行。
- 主分支不能合入明显构建失败的代码。

## 阶段 3：API 契约工业化

目标：减少前后端字段漂移。

任务：

1. 后端为所有 API response/request 建 Pydantic schema。
2. FastAPI OpenAPI schema 作为 API 契约源。
3. 使用 `openapi-typescript` 生成前端类型。
4. 前端 `api.ts` 逐步改为使用生成类型。
5. 加 API 契约检查脚本，CI 中验证 schema 能生成。

推荐流程：

```bash
python -m uvicorn api_server:app --port 8000
npx openapi-typescript http://127.0.0.1:8000/openapi.json -o frontend/src/lib/generated/openapi.d.ts
```

迁移策略：

- 先只生成类型，不立刻替换所有手写类型。
- 从训练 API 和任务 API 开始替换，因为这些最容易出现状态字段漂移。
- 最后再处理 registry 这类动态 JSON 结构。

验收标准：

- 前端核心 API response 类型来自 OpenAPI。
- 删除重复手写类型或保留为 UI view model，而非 API source of truth。

## 阶段 4：任务系统持久化

目标：解决任务状态重启丢失、浏览器恢复依赖 localStorage、长任务不可审计的问题。

### 4.1 SQLite 任务表

新增表：

```text
jobs(
  job_id text primary key,
  job_type text,
  status text,
  started_at real,
  finished_at real,
  pid integer,
  request_json text,
  scenario_totals_json text,
  progress_json text,
  stats_json text,
  error_message text
)

job_events(
  id integer primary key autoincrement,
  job_id text,
  event_type text,
  ts real,
  payload_json text
)
```

### 4.2 状态恢复

后端启动时：

- 扫描 running 状态任务。
- 如果 pid 不存在，标记 `external_terminated`。
- 如果任务有独立 worker，尝试重新订阅状态。

### 4.3 并发策略

定义互斥规则：

- `fill_samples` 运行时不允许 `router` 构建同一数据源。
- `router` 运行时可以禁止删除相关样本分区。
- 训练任务运行时禁止删除对应 router 数据和 checkpoint 目录。
- 文件管理删除操作必须检查 active job references。

验收标准：

- 重启后前端仍能看到最近任务列表。
- 任务终止、错误、完成状态可追溯。
- 互斥规则由后端强制，不依赖前端。

## 阶段 5：数据和迁移能力增强

目标：新服务器迁移时只需要 Git clone、安装环境、下载模型、平台重建派生数据。

### 5.1 数据保存边界

保持 Git 保存：

- 语言模板。
- 原始 HDF5。
- 配置和 registry。

不保存：

- text2comp parquet。
- router parquet。
- catalog sqlite。
- manifest 和 index。
- prepared training inputs。
- 默认 checkpoint。

### 5.2 迁移清单

新增 `docs/MIGRATION_CHECKLIST.md` 或扩展现有文档：

1. Clone repo。
2. 创建 conda 环境。
3. 安装 Python 和前端依赖。
4. 配置模型路径。
5. 启动服务。
6. 检查 HDF5 和模板是否完整。
7. 在平台生成 Stage 3 样本。
8. 在平台生成 Stage 4 Router 数据。
9. 开始训练。

### 5.3 环境变量标准化

建议统一环境变量：

```text
PIERN_ROOT
PIERN_DATA_ROOT
PIERN_ARTIFACT_ROOT
PIERN_MODEL_ROOT
PIERN_QWEN_EMBEDDING_MODEL
PIERN_HOST
PIERN_BACKEND_PORT
PIERN_FRONTEND_PORT
PIERN_CORS_ORIGINS
PIERN_CONDA_ENV
PIERN_NODE_BIN
PIERN_MAX_WORKERS
```

验收标准：

- 新服务器不需要改代码即可修改路径。
- `scripts/services/start.sh` 从环境变量读取配置。
- 模型路径不再散落在业务代码中。

## 阶段 6：前端结构和设计系统升级

目标：保持当前中文工作台风格，同时提高信息密度、可维护性和一致性。

任务：

1. 抽出基础组件：
   - Button
   - Select
   - Input
   - Toggle
   - Table
   - Tooltip
   - ProgressBar
   - StatusBadge
   - EmptyState
   - ConfirmDialog

2. 抽出布局组件：
   - WorkbenchShell
   - SidebarPanel
   - DetailPanel
   - Toolbar
   - SplitPane

3. 页面拆分：
   - RegistryPage 拆成 schema editor、scenario list、preview panel。
   - TrainingJobDetailPage 拆成 curves、logs、metadata、checkpoint manager。
   - JobMonitorPanel 拆成 header、progress list、logs、stats。

4. 统一文本截断策略：
   - 只对真实 overflow 的文本显示 tooltip。
   - tooltip 可选中复制。
   - 表格列设置最小宽度和最大宽度。

5. 可访问性：
   - 关键按钮有 aria-label。
   - 所有图标按钮有 tooltip。
   - 键盘可以操作主要表单。

验收标准：

- 页面在 1366x768、1920x1080、移动窄屏下无明显溢出。
- 所有关键操作有清晰状态反馈。
- 前端页面单文件逐步控制在 500 行以内，特殊复杂页需要明确拆分理由。

## 阶段 7：后端模块化和错误模型

目标：减少宽泛异常处理和超大服务文件，提高可测试性。

任务：

1. 为 API 错误建立统一模型：

```json
{
  "code": "ROUTER_BUILD_BLOCKED",
  "message": "样本填充任务仍在运行",
  "details": {}
}
```

2. 后端 router 只做：
   - 参数解析。
   - 调用 service。
   - 返回 schema。

3. service 层负责业务逻辑。
4. storage 层负责文件、Parquet、SQLite。
5. subprocess/worker 层负责长任务执行。
6. 限制裸 `except Exception`，至少记录 logger exception 和上下文。

建议拆分目标：

- `training_manager.py` 拆成 registry、process_control、checkpoint_store、prepared_cache、gpu_probe。
- `router_data.py` 拆成 status、builder、samples、deletion。
- `portable.py` 拆成 partition_writer、partition_reader、manifest_like、stats。

验收标准：

- 核心 service 有单元测试。
- API 错误前端可按 code 显示友好中文。
- 主要后端文件逐步控制在 600 行以内。

## 阶段 8：测试体系升级

目标：覆盖真实风险路径，而不是追求形式覆盖率。

### 8.1 后端测试

新增测试类型：

- API contract tests：FastAPI TestClient。
- task manager tests：终止、恢复、并发互斥、stats 更新。
- storage tests：Parquet 分区读写、manifest 重建、删除保护。
- training tests：checkpoint 保留策略、停止训练、GPU 选择逻辑。

### 8.2 前端测试

新增：

- Vitest component tests。
- React Testing Library。
- Playwright smoke tests。

关键 E2E：

1. 打开合成平台。
2. 进入样本填充页面。
3. 选择场景。
4. 启动任务。
5. 看到进度条、速度、耗时变化。
6. 终止任务。
7. 状态变为已终止。

### 8.3 迁移测试

手动或 CI 中模拟：

1. 删除派生目录。
2. 保留 HDF5 和 templates。
3. 启动平台。
4. 平台可识别源数据。
5. 可重新生成 Stage 3 和 Stage 4。

验收标准：

- 核心后端测试在 2 分钟内完成。
- 前端 smoke test 在 1 分钟内完成。
- 大型生成任务不放进默认 CI。

## 阶段 9：可观测性和运维

目标：服务器上出问题时能快速定位，不依赖猜测。

任务：

1. 结构化日志：
   - request id
   - job id
   - job type
   - scenario
   - duration
   - status

2. 健康检查分层：
   - `/api/health/live`：进程活着。
   - `/api/health/ready`：依赖可用。
   - `/api/health/storage`：数据目录可读写。
   - `/api/health/gpu`：GPU 探测。

3. 资源状态：
   - CPU。
   - 内存。
   - GPU。
   - 磁盘。
   - 当前任务。

4. 日志轮转：
   - `.runlogs/services/*.log` 需要定期截断或轮转。

5. 前端错误上报：
   - 最低限度可以在 console 记录 request id。
   - 工业部署可接 Sentry 或自建 error endpoint。

验收标准：

- 用户说“打不开”时，可以通过一个 status 命令看到端口、进程、日志尾部、磁盘、GPU。
- 长任务异常能定位到 job id 和场景。

## 阶段 10：安全和权限

目标：适合共享服务器和远程部署。

任务：

1. 禁止在仓库中保存密码、token、API key。
2. `.env` 只在服务器本地保存。
3. LLM key 后端保存时应限制文件权限。
4. 删除文件操作必须后端校验路径，不能接受任意路径。
5. 文件管理必须区分可删除和受保护资源。
6. CORS 来源通过环境变量配置。
7. 对破坏性操作加确认和后端保护。

验收标准：

- `git grep` 不出现明文密钥。
- 删除 API 无法删除项目外路径。
- 训练运行时不能删除被引用数据。

## 6. 推荐优先级

### P0：立即做

- 新增 `pyproject.toml`。
- 新增前端 lint/typecheck/test 脚本。
- 新增 GitHub Actions。
- 把服务配置改成环境变量优先。
- 完成任务互斥规则后端保护。

### P1：短期做

- OpenAPI 类型生成。
- SQLite 持久化任务状态。
- 统一错误模型。
- 日志和健康检查升级。
- 前端基础组件抽象。

### P2：中期做

- 长任务 worker 化。
- 训练管理拆模块。
- 大页面拆分。
- Playwright E2E。
- 迁移 checklist 自动化检查。

### P3：长期做

- 多服务器部署。
- 实验追踪系统。
- 模型注册和版本管理。
- 数据版本工具，例如 DVC 或对象存储。

## 7. 里程碑建议

### 里程碑 A：质量门禁可用

预计工作量：1 到 2 天。

交付：

- `pyproject.toml`
- ESLint/Prettier/Vitest 基础配置
- GitHub Actions
- README 更新

验收：

- push 后自动跑后端测试和前端构建。

### 里程碑 B：迁移配置标准化

预计工作量：1 到 2 天。

交付：

- `.env.example`
- 服务脚本读取环境变量
- 模型路径、数据路径、端口去硬编码
- 迁移 checklist

验收：

- 新服务器只改 `.env` 即可启动。

### 里程碑 C：任务系统持久化

预计工作量：3 到 5 天。

交付：

- SQLite job store
- job events store
- 后端任务列表恢复
- 互斥规则和删除保护

验收：

- 后端重启后仍可看到历史任务。
- active job 的保护规则由后端强制。

### 里程碑 D：API 契约和前端稳定性

预计工作量：3 到 5 天。

交付：

- OpenAPI 类型生成
- 前端 API 客户端逐步迁移
- Playwright smoke test
- 关键页面布局回归检查

验收：

- API 字段改动会在 CI 中暴露。
- 前端核心流程可自动冒烟。

## 8. 验收指标

工程指标：

- 默认 CI 运行时间小于 5 分钟。
- 前端 `npm run build` 必须通过。
- 后端 `pytest` 必须通过。
- `scripts/ci/check_consistency.py` 必须 0 warning。
- 核心 API response 有 schema。

迁移指标：

- 新服务器从 clone 到服务启动小于 30 分钟，不含大模型下载时间。
- 只依赖 Git 中的 HDF5 和模板即可重建 Stage 3/4。
- `.env` 是唯一需要本机修改的配置入口。

运行指标：

- 后台任务可终止。
- 任务进度、速度、耗时正常刷新。
- 服务日志可以定位 job id。
- 磁盘不足时前端或后端给出明确提示。

可维护性指标：

- 新增 API 必须同时有 schema 和前端类型。
- 超过 800 行的新文件需要拆分说明。
- 破坏性操作必须有后端保护。

## 9. 风险和应对

### 风险 1：一次性重构导致平台不可用

应对：只做渐进式改造。每个阶段都保持可启动、可回滚、可测试。

### 风险 2：CI 跑大型任务过慢

应对：默认 CI 只跑小测试和构建。百万样本、模型训练、完整数据闭环放手动 workflow 或服务器 smoke test。

### 风险 3：迁移后路径失效

应对：所有路径环境变量化，保留 `.env.example`，禁止在业务代码中写服务器私有路径。

### 风险 4：前后端类型漂移

应对：OpenAPI 类型生成和 CI schema 检查。

### 风险 5：任务状态丢失

应对：SQLite job store，进程重启后扫描并恢复终态。

### 风险 6：派生数据误提交

应对：`.gitignore`、CI 检查大文件和派生目录、文档明确 GitHub 保存范围。

## 10. 执行顺序建议

推荐按以下顺序执行：

1. 建立质量门禁。
2. 配置环境变量化。
3. 完成任务持久化。
4. 做 API 类型生成。
5. 拆分前端大页面和后端大服务。
6. 加 Playwright 冒烟测试。
7. 完善可观测性和安全保护。

这个顺序的理由是：先建立门禁，后续重构才有保护；先做配置外置，迁移风险立即降低；任务持久化和 API 契约是平台长期稳定性的核心；最后再做大规模结构优化。
