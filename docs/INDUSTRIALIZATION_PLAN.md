# PiERN 工业化升级计划书

版本日期：2026-05-16
适用范围：PiERN 数据合成平台、Token Router 自动训练平台、数据存储、任务系统、前后端工程、部署迁移和后续生产化升级。

## 1. 当前结论

PiERN 当前已经从科研脚本集合重构为一个准工业化的模块化单体平台。它已经适合单机科研生产使用，也具备较好的迁移能力和持续开发基础。当前架构的合理定位是：

- 现代化工程架构：已具备。
- 单机任务平台：已具备。
- 可迁移科研生产平台：基本具备。
- 多用户生产系统：尚未完成。
- 分布式任务平台：尚未完成。

当前工业化完成度约为 70% 到 80%。后续升级的核心不应再是零散修补，而应围绕任务执行器、部署标准化、数据库迁移、权限安全、可观测性和长期数据治理逐步推进。

## 2. 当前已完成的工业化基线

### 2.1 架构分层

已完成：

- 后端统一入口：`piern/api/main.py`。
- 合成平台后端：`piern/synth`。
- 训练平台后端：`piern/training`。
- 共享运行时、API、存储能力：`piern/shared`。
- 前端按平台划分：`frontend/src/synth`、`frontend/src/training`、`frontend/src/platform`。
- 脚本按用途划分：`scripts/services`、`scripts/ci`、`scripts/storage`、`scripts/text2comp`、`scripts/router`。

评价：

- 结构已经从脚本式项目升级为模块化单体。
- 当前拆分适合单机部署和快速迭代。
- 后续不需要推倒重来，应在现有边界上继续细化。

### 2.2 工程质量门禁

已完成：

- Python 项目配置：`pyproject.toml`。
- Pytest 配置和测试目录：`tests/`。
- Ruff 轻量规则。
- 前端 TypeScript strict。
- ESLint、Prettier、Vitest、Playwright smoke。
- GitHub Actions CI：
  - Backend Quality Gate。
  - Frontend Quality Gate。
  - OpenAPI contract check。
  - Playwright smoke。
- 本地一致性检查：`scripts/ci/check_consistency.py`。

评价：

- 当前 CI 已能防止明显构建失败、类型漂移和 API schema 漂移。
- Ruff 和 mypy 仍偏保守，属于第一阶段质量门禁，不是最终严格标准。

### 2.3 API 契约

已完成：

- FastAPI 作为 API 契约源。
- Pydantic schema 覆盖核心合成和训练接口。
- OpenAPI schema 可导出。
- 前端生成类型文件：
  - `frontend/src/lib/generated/openapi.json`
  - `frontend/src/lib/generated/openapi.d.ts`
- CI 检查 OpenAPI 生成结果是否同步。

评价：

- 前后端契约已经进入现代工程状态。
- 仍有一部分前端 UI view model 和 API 类型混合，需要继续拆分。

### 2.4 任务状态和持久化

已完成：

- 合成任务使用 SQLite `jobs` / `job_events`。
- 训练任务使用 SQLite `training_jobs` / `training_job_events`。
- 后端重启后可以看到历史任务。
- 外部消失的任务可标记为 `external_terminated`。
- 任务删除、文件删除和运行中任务引用已有保护。
- 训练停止具备平台 stop file、信号发送和强制清理逻辑。

评价：

- 已解决“任务只存在内存里”的核心风险。
- 仍然不是完整工业 worker 架构，因为 Web API 进程仍直接管理子进程。

### 2.5 数据和迁移策略

已完成：

- GitHub 只应保存：
  - 代码。
  - 配置。
  - 文档。
  - 语言模板。
  - 原始科学计算 HDF5。
- 默认不保存：
  - Stage 3 样本派生数据。
  - Stage 4 Router 数据。
  - Parquet 分区。
  - catalog SQLite。
  - manifest/index。
  - 训练 prepared cache。
  - 常规训练权重和日志。
- 已有便携式存储层：
  - Parquet 分区。
  - manifest。
  - catalog SQLite。
  - DuckDB/SQLite 辅助查询能力。
- 已有迁移文档：`docs/MIGRATION.md`。

评价：

- 数据边界已经清楚。
- 迁移风险已大幅降低。
- 后续需要补的是自动校验、备份策略和版本治理。

### 2.6 服务运行

已完成：

- 服务脚本：
  - `scripts/services/start.sh`
  - `scripts/services/stop.sh`
  - `scripts/services/restart.sh`
  - `scripts/services/status.sh`
- 支持 `.env` 和环境变量覆盖。
- 健康检查：
  - `/api/health/live`
  - `/api/health/ready`
  - `/api/health/storage`
  - `/api/health/gpu`
- 前端开发服务和后端 API 可长期运行。

评价：

- 当前适合远程服务器常驻开发和单机使用。
- 还不是生产部署标准，缺少 systemd/Docker/Nginx 等正式部署层。

### 2.7 前端工作台

已完成：

- 数据合成平台和训练平台统一入口。
- 平台切换组件统一。
- 文件管理系统接入训练平台。
- 训练保存后 N 个 epoch 的策略已接入。
- 训练随机种子已接入前端、API 和后端训练命令。
- 进度条、悬浮窗、文本截断和布局一致性经过多轮修正。

评价：

- 已经从原型界面升级为可用工作台。
- 后续重点是组件化、可访问性、布局回归和性能。

## 3. 最终目标架构

### 3.1 短期目标形态

短期目标是强化当前单机平台：

- 一个 FastAPI Web 进程。
- 一个前端工作台。
- SQLite 作为本地任务和事件存储。
- 文件系统保存大数据、日志、权重和派生物。
- 服务脚本或 systemd 管理前后端进程。
- GitHub Actions 负责轻量 CI。

该阶段适合个人或小团队科研开发。

### 3.2 中期目标形态

中期目标是形成单机工业任务平台：

- FastAPI 只负责 API、鉴权、任务提交和查询。
- 独立 worker 负责数据合成、Router 构建和训练进程。
- SQLite 或 PostgreSQL 保存任务、事件、锁、运行元数据。
- 前端通过轮询或 SSE/WebSocket 订阅任务状态。
- systemd 或 Docker Compose 管理 Web、worker、前端和反向代理。

该阶段适合稳定长期运行和多人协作。

### 3.3 长期目标形态

长期目标是可横向扩展的生产系统：

- PostgreSQL 作为主元数据数据库。
- Redis/RabbitMQ/NATS 作为任务队列或消息层。
- 多 worker、多 GPU 节点。
- 对象存储保存大文件、权重和数据快照。
- 完整认证、权限、审计、告警和监控。
- 数据集、模型、实验和专利文档都有版本治理。

该阶段适合团队化生产部署。

## 4. 剩余工业化事项总清单

本节列出后续还需要做的所有重要事项。优先级含义：

- P0：影响稳定运行或迁移安全，应该优先做。
- P1：影响长期维护和多人协作，建议尽快做。
- P2：提升工程成熟度、效率或体验，可排期做。
- P3：面向更大规模生产化，当前不是刚需。

## 5. P0：运行稳定性和迁移安全

### 5.1 建立正式服务守护方案

现状：

- 已有 `scripts/services/*.sh`。
- 服务可启动、停止、重启和查看状态。
- 进程守护依赖脚本和人工调用。

需要做：

1. 增加 `deploy/systemd/piern-backend.service`。
2. 增加 `deploy/systemd/piern-frontend.service` 或生产静态前端服务方案。
3. 增加 `deploy/systemd/piern-worker.service`，为未来 worker 预留。
4. 增加 `scripts/services/install-systemd.sh`，生成用户级 systemd 配置。
5. 明确日志位置、重启策略、环境变量加载方式。
6. 在 `docs/MIGRATION.md` 中加入 systemd 部署步骤。

验收标准：

- 服务器重启后服务可自动恢复。
- `systemctl --user status piern-backend` 可查看状态。
- `.env` 修改后可重启服务生效。

### 5.2 完成 `.env` 配置全覆盖审计

现状：

- `PIERN_ROOT`、`PIERN_DATA_ROOT`、`PIERN_ARTIFACT_ROOT`、`PIERN_RUNLOG_ROOT` 已支持。
- 服务端口、Conda、Node 路径支持环境变量。
- 部分模型路径、任务参数、并发参数仍可能散落在代码默认值里。

需要做：

1. 审计所有 `os.getenv`、硬编码路径和默认端口。
2. 将模型路径、tokenizer 路径、最大并发、默认 batch size、GPU 阈值整理到 `.env.example`。
3. 给每个变量加注释和默认值说明。
4. 启动时输出有效配置摘要，但不能打印敏感 token。
5. 增加配置校验函数，缺失关键路径时给出明确错误。

验收标准：

- 新服务器只需复制 `.env.example` 为 `.env` 并修改路径即可启动。
- 代码中不出现服务器私有路径。

### 5.3 建立迁移验收脚本

现状：

- 有迁移文档和健康检查。
- 缺少一键迁移验收。

需要做：

1. 新增 `scripts/ci/check_migration_ready.py`。
2. 检查 Git 跟踪数据是否只包含允许的数据类型。
3. 检查 `.env.example` 是否覆盖必要变量。
4. 检查原始 HDF5 和模板是否齐全。
5. 检查派生目录是否可重建但不应进入 Git。
6. 检查模型路径是否存在或给出下载提示。

验收标准：

- `python scripts/ci/check_migration_ready.py` 在 CI 和服务器上可运行。
- 迁移风险能在启动训练前暴露。

### 5.4 清理运行时缓存进入仓库的风险

现状：

- `.gitignore` 已排除多数运行时文件。
- 工作区中仍会出现 `.pytest_cache`、`.ruff_cache`、`__pycache__`、`frontend/test-results` 等本地缓存。

需要做：

1. 审查 `.gitignore` 对 Python、Node、Playwright、Vite、训练产物的覆盖。
2. 增加 CI 检查，禁止提交缓存目录、SQLite runtime、训练日志、权重和 Parquet 派生物。
3. 增加 `scripts/ci/check_repo_hygiene.py`。

验收标准：

- CI 能阻止误提交运行时文件。
- `git status --ignored` 中的缓存目录符合预期。

## 6. P0：任务执行器工业化

### 6.1 抽离独立 worker

现状：

- Web API 进程直接启动合成、Router 构建和训练子进程。
- SQLite 已记录任务状态，但任务执行仍和 API 进程耦合。

需要做：

1. 新增 `piern/worker` 包。
2. 定义统一任务表：
   - job id。
   - job type。
   - status。
   - priority。
   - payload。
   - lock key。
   - created/started/finished timestamps。
3. API 创建任务只写入队列表。
4. worker 循环领取任务、加锁、执行、写事件。
5. 支持 worker 心跳。
6. API 根据心跳判断任务是否失联。
7. 训练任务仍可以启动子进程，但由 worker 管理。

验收标准：

- 重启 API 不影响正在运行的 worker。
- 重启前端不影响任务。
- worker 消失时任务可标记为 `external_terminated` 或 `worker_lost`。

### 6.2 统一任务状态机

现状：

- 合成任务和训练任务状态基本相似，但实现分散。
- 状态转换规则没有统一定义。

需要做：

1. 定义统一状态：
   - `queued`
   - `starting`
   - `running`
   - `stopping`
   - `succeeded`
   - `failed`
   - `cancelled`
   - `external_terminated`
2. 保留前端兼容映射，如 `done -> succeeded`。
3. 定义合法转换表。
4. 非法转换写入错误事件。
5. 将合成和训练任务都接入同一状态机。

验收标准：

- 所有任务状态转换可审计。
- 前端不需要为不同任务系统维护多套状态逻辑。

### 6.3 统一任务锁和资源引用

现状：

- 已有部分 active job protection。
- 锁逻辑分散在不同 service 中。

需要做：

1. 新增 `job_locks` 表。
2. 锁 key 设计：
   - `dataset:{simulator}:{scenario}`
   - `router:{simulator}:{scenario}`
   - `gpu:{index}`
   - `checkpoint:{job_id}`
3. 文件删除、Router 构建、样本填充、训练启动都必须申请锁。
4. worker 崩溃后根据心跳清理过期锁。

验收标准：

- Router 构建不会误影响正在填充的同一数据。
- 删除操作不会删除运行任务依赖的数据。
- GPU 不会被两个训练任务同时占用。

### 6.4 增加任务事件和日志统一接口

现状：

- 前端能看日志和进度。
- 合成和训练的事件结构不完全统一。

需要做：

1. 统一 `/api/jobs/{job_id}`。
2. 统一 `/api/jobs/{job_id}/events`。
3. 统一 `/api/jobs/{job_id}/logs`。
4. 训练和合成保留各自详情接口，但底层事件模型一致。
5. 支持分页查询事件，避免长任务事件过多。

验收标准：

- 前端任务列表可以跨平台展示任务。
- 调试任务不需要手动翻 `.runlogs`。

## 7. P1：数据库和存储治理

### 7.1 引入数据库迁移机制

现状：

- SQLite 表由代码 `CREATE TABLE IF NOT EXISTS` 创建。
- schema 演进缺少版本记录。

需要做：

1. 建立 `piern/shared/db`。
2. 选择轻量迁移方案：
   - 短期：自研 `schema_migrations` 表。
   - 中期：Alembic。
3. 为合成任务、训练任务、文件 catalog 建 migration。
4. 服务启动时检查 schema version。
5. 迁移失败时拒绝启动并给出修复方式。

验收标准：

- 数据库结构变化可追踪。
- 老服务器升级不会静默写坏旧数据。

### 7.2 统一 catalog 和 manifest 版本

现状：

- Parquet manifest、catalog SQLite 和 `.manifests` 已存在。
- 版本字段仍不够严格。

需要做：

1. 为每类 manifest 定义 schema version。
2. 为 text2comp、router、template、hdf5 原始数据建立统一 catalog 条目。
3. catalog 中记录：
   - 数据来源。
   - 生成配置 hash。
   - 输入模板版本。
   - 原始 HDF5 mtime/hash。
   - 记录数。
   - 分区路径。
4. 前端展示 catalog schema version 和可重建状态。

验收标准：

- 任意派生数据都能追踪到原始 HDF5、模板和生成参数。
- 迁移后能判断哪些数据需要重建。

### 7.3 数据完整性校验

现状：

- 已有部分 HDF5 校验和行数校验。
- 大规模数据缺少统一 checksum。

需要做：

1. 为原始 HDF5 生成 checksum manifest。
2. 为模板 JSONL 生成 checksum manifest。
3. 为 Parquet 分区记录 row count、file count、schema hash。
4. 增加 `scripts/storage/verify_data_integrity.py`。
5. 前端文件管理显示数据是否通过校验。

验收标准：

- 迁移后能快速判断数据是否损坏或缺失。
- 训练前能拒绝使用损坏的 Router 数据。

### 7.4 权重和实验产物保留策略

现状：

- 训练可保留后 N 个 epoch。
- 文件管理可查看和删除权重。

需要做：

1. 定义权重保留策略：
   - latest。
   - best。
   - last N。
   - final。
2. 训练配置记录保留策略。
3. 文件管理区分：
   - 受保护权重。
   - 可删除权重。
   - 正在运行任务依赖的权重。
4. 增加训练产物 manifest。
5. 支持导出单次实验 bundle。

验收标准：

- 不会误删可复现实验所需的最小文件。
- 大量 checkpoint 不会无限占用磁盘。

## 8. P1：部署和生产化

### 8.1 生产静态前端方案

现状：

- 当前主要用 Vite dev server。
- 后端可服务 `frontend/dist`，但生产部署流程未正式化。

需要做：

1. 明确开发模式和生产模式。
2. 新增 `scripts/services/start-prod.sh`。
3. 生产模式：
   - `npm run build`。
   - FastAPI mount `frontend/dist`。
   - 只开放一个端口或通过 Nginx 反代。
4. 文档说明 5173 只用于开发。

验收标准：

- 生产模式不依赖 Vite dev server。
- 用户访问一个稳定入口即可使用完整平台。

### 8.2 Docker Compose 可选部署

现状：

- 依赖 conda 和服务器脚本。
- 迁移可行，但环境复刻仍需人工。

需要做：

1. 新增 `Dockerfile.backend`。
2. 新增 `Dockerfile.frontend` 或前端 build stage。
3. 新增 `compose.yaml`。
4. 支持挂载：
   - data。
   - artifacts。
   - runlogs。
   - model cache。
5. GPU 运行说明使用 NVIDIA Container Toolkit。

验收标准：

- 新机器可以用 Docker Compose 启动非训练功能。
- GPU 训练路径有明确说明。

### 8.3 Nginx 反向代理配置

现状：

- 前端直接暴露 5173。
- 后端直接暴露 8000。

需要做：

1. 增加 `deploy/nginx/piern.conf`。
2. 支持：
   - `/` 前端。
   - `/api` 后端。
   - 大请求体上传。
   - 长轮询或 SSE/WebSocket。
3. 文档说明防火墙端口。

验收标准：

- 外部只需访问 80/443 或指定统一端口。
- 5173/8000 可只在本机监听。

## 9. P1：安全、权限和审计

### 9.1 基础认证

现状：

- 平台假设只在可信服务器或 SSH 隧道访问。
- 没有登录、权限和 CSRF 策略。

需要做：

1. 增加最小认证层：
   - 单用户 token。
   - 或用户名密码登录。
2. API 支持 Bearer token。
3. 前端登录态保存和过期处理。
4. 服务端敏感 API 保护：
   - 删除文件。
   - 启动训练。
   - 停止任务。
   - 修改 LLM 配置。
5. `.env.example` 中配置 secret。

验收标准：

- 未认证访问不能启动任务或删除文件。
- 内网部署也有基本保护。

### 9.2 敏感信息治理

现状：

- GitHub token、API key、模型 token 等依赖本机文件或环境。
- 文档中不能暴露真实密钥。

需要做：

1. 增加 secret 扫描：
   - gitleaks。
   - 或自定义轻量检查。
2. CI 阻止提交 token。
3. 日志脱敏：
   - Authorization。
   - API keys。
   - GitHub token。
   - 模型下载 token。
4. 文档统一使用占位符。

验收标准：

- 误提交 token 时 CI 失败。
- 日志不会打印敏感配置。

### 9.3 操作审计

现状：

- 有 request id 和任务事件。
- 删除、停止、修改配置等操作缺少统一审计模型。

需要做：

1. 新增 `audit_events` 表。
2. 记录：
   - 操作人。
   - 操作类型。
   - 目标资源。
   - request id。
   - 时间。
   - 结果。
3. 前端文件管理和任务管理的危险操作写审计。

验收标准：

- 能追踪谁删除了哪个文件或停止了哪个任务。

## 10. P1：后端模块化和类型质量

### 10.1 拆分 `training_manager.py`

现状：

- `training_manager.py` 承担 GPU 查询、任务创建、状态刷新、checkpoint 管理、停止逻辑和日志读取。

需要做：

1. 拆出 `gpu_inventory.py`。
2. 拆出 `training_launcher.py`。
3. 拆出 `training_status.py`。
4. 拆出 `training_stop.py`。
5. 拆出 `training_artifacts.py`。
6. 保留 `training_manager.py` 作为 facade。

验收标准：

- 单文件职责清晰。
- 训练启动、停止、状态刷新可单独测试。

### 10.2 拆分合成 job manager

现状：

- 合成任务流程包含样本填充、Router 构建、进度统计和互斥保护。

需要做：

1. 拆出任务定义。
2. 拆出执行器。
3. 拆出进度聚合。
4. 拆出文件引用检查。
5. 与统一 worker 状态机对接。

验收标准：

- 合成任务和训练任务共享任务基础设施。

### 10.3 扩大 Ruff 和 mypy 覆盖

现状：

- Ruff 当前偏 correctness-only。
- mypy 配置较宽松。

需要做：

1. 分阶段扩大 Ruff 规则：
   - import 排序。
   - 未使用变量。
   - 复杂度告警。
   - 常见 bugbear 规则。
2. 为核心 shared/training/synth service 增加类型标注。
3. CI 中先以 warning 方式运行 mypy，再逐步强制。

验收标准：

- 新增核心代码必须有类型。
- 关键 service 能通过 mypy。

### 10.4 统一错误模型

现状：

- 已有 `ApiError` 和 request id。
- 部分接口仍可能返回普通字符串或默认异常。

需要做：

1. 所有业务错误返回：
   - `code`
   - `message`
   - `details`
   - `request_id`
2. 前端 API client 统一解析错误。
3. 页面错误提示显示可读中文。
4. 日志中保留 request id 便于定位。

验收标准：

- 前端不再显示生硬 traceback 或英文内部错误。
- API 错误格式稳定。

## 11. P1：前端结构和体验治理

### 11.1 建立前端设计系统基础层

现状：

- 已有若干 shared UI。
- 页面中仍存在重复 card、toolbar、badge、table、tooltip 逻辑。

需要做：

1. 抽象基础组件：
   - `PageShell`
   - `Panel`
   - `MetricCard`
   - `DataTable`
   - `StatusBadge`
   - `ConfirmDialog`
   - `TruncatedText`
   - `ProgressRow`
2. 统一 spacing、字号、圆角、颜色 token。
3. 禁止页面内临时写复杂样式。
4. 写组件使用规范。

验收标准：

- 新页面优先使用 shared component。
- 同类组件高度、交互和视觉一致。

### 11.2 拆分大型页面

现状：

- `RegistryPage.tsx`、`TrainingJobDetailPage.tsx`、部分 synth 页面仍偏大。

需要做：

1. `TrainingJobDetailPage` 拆为：
   - summary。
   - curves。
   - logs。
   - checkpoints。
   - config。
2. `RegistryPage` 拆为：
   - simulator list。
   - dataset table。
   - metadata drawer。
   - upload/register dialog。
3. 合成页面按任务区块拆 hooks 和 components。

验收标准：

- 单个页面文件控制在可维护范围。
- 页面逻辑和展示逻辑分离。

### 11.3 布局回归测试

现状：

- 有 Playwright smoke。
- 还没有系统性的截图回归。

需要做：

1. 增加关键页面截图：
   - 数据总览。
   - 样本填充。
   - Router 构建。
   - 新建训练。
   - 训练详情。
   - 文件管理。
2. 覆盖桌面和较窄屏幕。
3. 检查：
   - 文字溢出。
   - 高度不统一。
   - 弹窗被遮挡。
   - tooltip 可选择文本。
4. CI 中先保存 artifacts，不强制 pixel diff。

验收标准：

- 前端布局问题能在合入前发现。

### 11.4 前端性能优化

现状：

- 大列表和大日志已有部分优化。
- 多页面仍依赖轮询。

需要做：

1. 大列表统一虚拟滚动。
2. 日志按增量读取。
3. 任务进度可选 SSE/WebSocket。
4. API 请求去重和缓存策略统一。
5. 对大型图表做降采样。

验收标准：

- 百万样本 metadata 场景下前端不卡死。
- 训练日志增长时页面仍可操作。

## 12. P1：可观测性

### 12.1 结构化日志完善

现状：

- API request 已有结构化日志。
- 任务日志仍主要是文本行。

需要做：

1. 任务事件写 JSONL。
2. 训练、合成、Router 构建统一 event schema。
3. 日志按 job id、request id、worker id 关联。
4. 增加日志级别配置。

验收标准：

- 单个任务可完整还原运行过程。

### 12.2 指标和监控

现状：

- 健康检查有实时状态。
- 没有历史指标。

需要做：

1. 增加 `/api/metrics/summary`。
2. 记录：
   - 任务耗时。
   - 成功/失败次数。
   - 样本生成速率。
   - Router 构建速率。
   - GPU 利用率。
   - 磁盘剩余。
3. 可选 Prometheus endpoint。
4. 前端增加运维页。

验收标准：

- 能判断瓶颈在 CPU、GPU、磁盘还是数据读取。

### 12.3 告警策略

需要做：

1. 磁盘低于阈值告警。
2. GPU 被锁但进程消失告警。
3. 任务长时间 0 throughput 告警。
4. 训练 loss NaN 告警。
5. 后端健康检查失败告警。

验收标准：

- 长任务异常不会静默挂住。

## 13. P2：测试体系升级

### 13.1 后端集成测试

需要做：

1. 使用 FastAPI TestClient 覆盖主要 API。
2. 使用临时目录模拟 data/artifacts/runlogs。
3. 覆盖任务创建、取消、删除保护、状态恢复。
4. 覆盖 OpenAPI schema 生成。

验收标准：

- 后端核心流程无需真实 GPU 即可测试。

### 13.2 训练最小闭环测试

需要做：

1. 构造极小 router 数据。
2. CPU 或 mock 模式跑 1 epoch。
3. 检查：
   - checkpoint。
   - metrics。
   - logs。
   - config seed。
   - keep_last_epochs。
4. CI 默认可跳过 GPU 训练，服务器可跑 nightly smoke。

验收标准：

- 训练入口改动不会破坏最小闭环。

### 13.3 数据合成最小闭环测试

需要做：

1. 使用小模板和小 HDF5 fixture。
2. 跑样本填充。
3. 跑 Router 构建。
4. 检查 manifest、catalog、Parquet、前端 API 数据。

验收标准：

- 数据链路改动可快速验证。

### 13.4 前端组件测试

需要做：

1. 测试 truncation tooltip。
2. 测试任务进度组件。
3. 测试训练参数表单。
4. 测试文件管理删除保护提示。

验收标准：

- 曾经出现过的 UI bug 有回归测试。

## 14. P2：性能优化

### 14.1 样本填充性能

需要做：

1. 分析瓶颈：
   - JSONL 写入。
   - 模板采样。
   - HDF5 读取。
   - Python 序列化。
2. 支持按场景并行。
3. 支持每个场景内部 chunk 并行。
4. 控制写入合并，避免小文件过多。
5. 进度统计从频繁写 DB 改为节流写入。

验收标准：

- 大规模填充时 CPU 利用率合理。
- 前端进度实时但不拖慢任务。

### 14.2 Router 构建性能

需要做：

1. Parquet 优先读取。
2. chunk size 自动调优。
3. embedding cache 可复用。
4. 多进程准备数据。
5. 进度中显示每个场景、chunk、records、throughput。

验收标准：

- 构建进度可读。
- 大场景处理速度稳定。

### 14.3 训练性能

需要做：

1. DataLoader 参数自动建议。
2. pinned memory、prefetch、persistent workers 可配置。
3. embedding-only 训练路径减少重复读取。
4. test interval 和 test batch size 给出推荐。
5. 训练曲线和评估指标写入节流。

验收标准：

- GPU 利用率和数据加载瓶颈可解释。
- 默认参数对 TITAN V 这类 GPU 合理。

## 15. P2：实验和模型治理

### 15.1 实验记录

需要做：

1. 每次训练保存完整 config snapshot。
2. 记录 git commit。
3. 记录数据 manifest hash。
4. 记录模型路径和 tokenizer 路径。
5. 记录 seed。
6. 记录硬件信息。

验收标准：

- 任意训练结果都能解释来源。

### 15.2 模型注册表

需要做：

1. 新增 model registry。
2. 注册：
   - run id。
   - checkpoint path。
   - metrics。
   - data version。
   - created_at。
   - tags。
3. 文件管理读取 registry 而不是只扫目录。

验收标准：

- 最优模型、最新模型、可部署模型有明确状态。

### 15.3 推理服务预留

需要做：

1. 设计 Token Router 推理 API。
2. 支持加载指定 checkpoint。
3. 支持批量输入。
4. 记录推理耗时和命中场景。

验收标准：

- 训练平台产物能进入推理验证流程。

## 16. P2：文档和知识资产

### 16.1 README 和快速开始

需要做：

1. README 保持中文主文档。
2. 加入：
   - 环境安装。
   - 服务启动。
   - 数据生成。
   - 训练。
   - 常见故障。
3. 明确哪些文件进入 Git，哪些不进入 Git。

验收标准：

- 新用户能按 README 跑起最小功能。

### 16.2 运维手册

需要做：

1. 服务无法访问。
2. 端口占用。
3. GPU 不可用。
4. 磁盘不足。
5. 任务卡住。
6. GitHub CI 失败。
7. 模型路径错误。

验收标准：

- 常见问题不需要重新排查。

### 16.3 专利文档流水线

现状：

- 已有专利 md 和 docx 生成脚本。
- 已有专利化附图版本。

需要做：

1. 将专利文档生成纳入 Make/script 命令。
2. 检查 md 到 docx 的模板一致性。
3. 图片资源加 manifest。
4. 文档生成不应依赖不可追踪的 Word 手工修改。

验收标准：

- 专利 md 是唯一源，docx 可重复生成。

## 17. P3：多用户和分布式能力

### 17.1 多用户权限

需要做：

1. 用户表。
2. 角色：
   - admin。
   - operator。
   - viewer。
3. 任务和文件操作权限。
4. 审计操作人。

验收标准：

- 多人同时使用时有基本隔离。

### 17.2 分布式 worker

需要做：

1. 队列服务：
   - Redis。
   - RabbitMQ。
   - NATS。
2. worker 注册和心跳。
3. GPU 资源调度。
4. 数据路径共享或对象存储。

验收标准：

- 多台 GPU 机器可以同时承担训练任务。

### 17.3 对象存储

需要做：

1. 支持 S3/MinIO。
2. 大文件上传下载走对象存储。
3. 权重和数据集快照可归档。
4. 本地文件系统作为开发模式。

验收标准：

- 大规模训练产物不依赖单机磁盘。

## 18. 推荐执行顺序

### 第一批：稳定运行

1. 完成 systemd 用户级服务。
2. 完成 `.env` 配置审计。
3. 完成迁移验收脚本。
4. 完成 repo hygiene 检查。

理由：

- 这些事项直接降低服务器重启、迁移和误提交风险。

### 第二批：任务 worker 化

1. 新增统一任务状态机。
2. 新增统一任务锁。
3. 抽离 worker 进程。
4. API 和 worker 解耦。
5. 合成和训练任务逐步接入。

理由：

- 这是从单机脚本平台升级为工业任务平台的关键。

### 第三批：数据库和数据治理

1. 引入 schema migration。
2. 统一 catalog/manifest version。
3. 增加数据完整性校验。
4. 增加训练产物 manifest。

理由：

- 这些事项决定项目迁移和长期维护的可靠性。

### 第四批：前端和后端结构深化

1. 拆分 `training_manager.py`。
2. 拆分大型前端页面。
3. 抽象设计系统组件。
4. 增加布局回归测试。

理由：

- 当前功能已经可用，下一步要降低维护成本。

### 第五批：安全和观测

1. 基础认证。
2. secret 扫描。
3. audit events。
4. metrics。
5. 告警。

理由：

- 面向多人和长期运行必须补齐。

### 第六批：生产部署和扩展

1. 生产静态前端。
2. Nginx。
3. Docker Compose。
4. PostgreSQL 方案。
5. 对象存储。
6. 分布式 worker。

理由：

- 这些是规模化部署需要的能力，当前可作为中长期目标。

## 19. 验收命令

每次完成一批工业化事项后，至少运行：

```bash
python -m ruff check .
python scripts/ci/check_consistency.py
python -m pytest
cd frontend
npm run typecheck
npm run lint
npm run format:check
npm run test
npm run build
npm run e2e:smoke
```

如涉及服务启动，还需运行：

```bash
scripts/services/restart.sh
scripts/services/status.sh
```

如涉及迁移和数据，还需运行：

```bash
python scripts/ci/check_migration_ready.py
python scripts/storage/verify_data_integrity.py
```

上述两个命令是计划新增命令，完成对应阶段前可暂时不存在。

## 20. 当前最重要的下一步

如果只选一个方向，优先做任务 worker 化。原因是当前系统最大的架构风险已经不是前端显示，也不是 API 类型，而是长任务执行仍与 Web API 进程耦合。

建议下一步执行：

1. 建立统一任务状态机。
2. 建立统一任务锁表。
3. 建立 worker 进程骨架。
4. 先接入样本填充或 Router 构建。
5. 再接入训练任务。

完成后，PiERN 将从“准工业化单机平台”升级为“工业化单机任务平台”。
