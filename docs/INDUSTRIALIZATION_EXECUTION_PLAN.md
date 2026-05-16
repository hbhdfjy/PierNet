# PiERN 工业化执行计划

本文件是 `INDUSTRIALIZATION_PLAN.md` 的执行版，用于后续按批次推进升级。主计划负责说明全量目标和剩余事项，本文件负责记录当前状态、下一批任务和验收命令。

## 1. 当前工业化状态

当前项目已经完成准工业化基线：

- 后端已经按 `api`、`synth`、`training`、`shared` 分层。
- 前端已经按平台拆分为 `platform`、`synth`、`training`、`shared`。
- 已有 Python/前端工程配置。
- 已有 GitHub Actions 后端和前端质量门禁。
- 已有 OpenAPI 生成类型和契约检查。
- 合成任务和训练任务都已使用 SQLite 持久化任务快照和事件。
- 已有服务启动、停止、重启、状态检查脚本。
- 已有健康检查接口。
- 已有迁移文档和便携式数据存储策略。
- GitHub 数据边界已明确：保留代码、配置、文档、语言模板和原始 HDF5，派生数据默认不入库。

当前定位：

- 适合单机科研生产。
- 适合持续开发。
- 适合迁移到新服务器。
- 尚未达到多用户生产系统或分布式任务系统标准。

## 2. 执行原则

- 不一次性重写系统。
- 每批升级都必须保持服务可启动、可测试、可回滚。
- 默认 CI 不跑百万样本生成、不下载模型、不跑真实 GPU 训练。
- GitHub 只保存代码、配置、文档、语言模板和原始 HDF5。
- 服务器私有路径只允许出现在 `.env` 或部署配置中。
- 所有长任务必须能被查询、停止、审计。
- 前后端 API 字段必须以 OpenAPI 为契约源。
- 每次大改前先确认服务状态，每次大改后跑验收命令。

## 3. 已交付阶段

| 阶段 | 目标 | 当前状态 |
| --- | --- | --- |
| 0 | 基线冻结 | 已完成 |
| 1 | 工程配置标准化 | 已完成 |
| 2 | CI 质量门禁 | 已完成 |
| 3 | 迁移配置标准化 | 已完成基础版 |
| 4 | 后端删除保护和任务互斥 | 已完成基础版 |
| 5 | API 契约 | 已完成基础版，仍需扩大覆盖和清理手写类型 |
| 6 | 任务持久化 | 合成和训练均已接入 SQLite |
| 7 | 前端结构升级 | 已完成多轮 UI 修复，仍需系统性组件化 |
| 8 | 后端模块化与观测 | 已完成错误模型、request id、健康检查基础版 |

## 4. 下一批 P0 任务

### 4.1 正式服务守护

任务：

1. 新增 `deploy/systemd/piern-backend.service`。
2. 新增 `deploy/systemd/piern-frontend.service` 或生产前端服务方案。
3. 新增 `deploy/systemd/piern-worker.service` 占位。
4. 新增安装或生成 systemd 用户服务的脚本。
5. 更新 `docs/MIGRATION.md`。

验收：

```bash
scripts/services/restart.sh
scripts/services/status.sh
```

后续 systemd 完成后增加：

```bash
systemctl --user status piern-backend
systemctl --user status piern-frontend
```

### 4.2 `.env` 全覆盖审计

任务：

1. 审计所有服务器路径和端口。
2. 将模型路径、tokenizer 路径、并发参数、GPU 阈值加入 `.env.example`。
3. 启动时校验关键配置。
4. 文档解释每个变量。

验收：

```bash
python scripts/ci/check_consistency.py
```

### 4.3 迁移验收脚本

任务：

1. 新增 `scripts/ci/check_migration_ready.py`。
2. 检查 Git 跟踪数据边界。
3. 检查原始 HDF5、模板、配置、模型路径。
4. 检查派生数据没有误入 Git。

验收：

```bash
python scripts/ci/check_migration_ready.py
```

### 4.4 仓库卫生检查

任务：

1. 新增 `scripts/ci/check_repo_hygiene.py`。
2. 阻止提交缓存、SQLite runtime、训练日志、权重、Parquet 派生物。
3. 将检查接入 CI。

验收：

```bash
python scripts/ci/check_repo_hygiene.py
```

## 5. 下一批 P1 任务

### 5.1 统一任务状态机

任务：

1. 定义统一状态枚举。
2. 定义合法状态转换表。
3. 合成任务和训练任务共享状态转换逻辑。
4. 前端状态展示使用统一映射。

验收：

```bash
python -m pytest tests/test_synth_job_store.py tests/test_training_job_store.py
```

### 5.2 统一任务锁

任务：

1. 新增任务锁表。
2. 定义 dataset、router、gpu、checkpoint 锁。
3. 文件删除、样本填充、Router 构建、训练启动统一申请锁。
4. 处理 worker 或进程消失后的锁回收。

验收：

```bash
python -m pytest tests/test_training_manager_fallbacks.py
```

### 5.3 独立 worker 骨架

任务：

1. 新增 `piern/worker`。
2. API 只写任务队列。
3. Worker 领取任务并写事件。
4. 先接入样本填充或 Router 构建。
5. 再接入训练任务。

验收：

```bash
python -m pytest
scripts/services/status.sh
```

### 5.4 数据库迁移机制

任务：

1. 新增 `schema_migrations`。
2. 给合成任务、训练任务、catalog 建立 schema version。
3. 服务启动时检查并执行迁移。
4. 迁移失败时给出明确错误。

验收：

```bash
python -m pytest tests/test_synth_job_store.py tests/test_training_job_store.py
```

## 6. 后续批次

### 6.1 前端组件化和布局回归

任务：

1. 抽象 `PageShell`、`Panel`、`DataTable`、`StatusBadge`、`TruncatedText`、`ConfirmDialog`。
2. 拆分 `TrainingJobDetailPage.tsx`。
3. 拆分 `RegistryPage.tsx`。
4. 增加 Playwright 截图检查。

验收：

```bash
cd frontend
npm run typecheck
npm run lint
npm run format:check
npm run test
npm run build
npm run e2e:smoke
```

### 6.2 可观测性

任务：

1. 统一任务 JSONL event schema。
2. 增加任务耗时、吞吐、GPU、磁盘指标。
3. 增加 `/api/metrics/summary`。
4. 前端增加运维视图。
5. 增加磁盘不足、任务卡住、GPU 锁异常告警。

验收：

```bash
curl -fsS http://127.0.0.1:8000/api/health/live
curl -fsS http://127.0.0.1:8000/api/health/ready
```

### 6.3 安全和审计

任务：

1. 增加基础认证。
2. 保护删除、训练启动、任务停止、配置修改接口。
3. 增加 secret 扫描。
4. 增加 `audit_events`。
5. 日志脱敏。

验收：

```bash
python scripts/ci/check_consistency.py
python -m pytest
```

### 6.4 生产部署

任务：

1. 增加生产静态前端启动方式。
2. 增加 Nginx 配置。
3. 增加 Docker Compose 可选部署。
4. 文档说明开发模式和生产模式区别。

验收：

```bash
npm --prefix frontend run build
scripts/services/restart.sh
scripts/services/status.sh
```

## 7. 默认验收命令

每次完成工业化改造后运行：

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

服务相关改造额外运行：

```bash
scripts/services/restart.sh
scripts/services/status.sh
```

OpenAPI 相关改造额外运行：

```bash
cd frontend
npm run openapi:check
```

## 8. 当前最推荐的下一步

优先做 P0 第一批：

1. systemd 服务守护。
2. `.env` 全覆盖审计。
3. 迁移验收脚本。
4. 仓库卫生检查。

完成后再做 P1 的 worker 化。这样顺序最稳，因为先把服务器运行和迁移风险压低，再重构任务执行层。
