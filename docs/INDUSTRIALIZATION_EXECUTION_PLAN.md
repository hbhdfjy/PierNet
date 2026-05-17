# PiERN 工业化执行计划

本文件是 `INDUSTRIALIZATION_PLAN.md` 的执行版，用于记录已完成批次、剩余事项和验收命令。主计划说明长期目标，本文件约束近期执行。

## 1. 当前状态

截至 2026-05-17，PiERN 已完成准工业化单机平台基线：

- 后端按 `api`、`synth`、`training`、`shared` 分层。
- 前端按 `platform`、`synth`、`training`、`shared` 分层。
- 合成任务和训练任务均使用 SQLite 持久化快照和事件。
- 服务已支持 `.env`、健康检查、用户级 systemd 和 user linger。
- CI 已覆盖后端、前端、OpenAPI、迁移检查和仓库卫生检查。
- GitHub 数据边界已明确：只保留代码、配置、文档、语言模板和原始 HDF5；派生数据、训练权重、日志、缓存默认不入库。

当前定位：

- 适合单机科研生产和远程服务器长期运行。
- 适合迁移到新服务器。
- 尚不是多用户生产系统，也不是分布式任务系统。

## 2. 已完成批次

### 2.1 P0 运行和迁移基线

已完成：

1. 用户级 systemd 模板：backend、frontend、maintenance worker。
2. 服务脚本：启动、停止、重启、状态、前台 runner、生产静态启动。
3. `.env.example` 覆盖路径、端口、模型、GPU、并发、安全和日志变量。
4. `/api/health/ready` 返回配置校验摘要。
5. `scripts/ci/check_migration_ready.py`。
6. `scripts/ci/check_repo_hygiene.py`。
7. `docs/MIGRATION.md` 迁移说明。

### 2.2 P1 任务基础设施

已完成基础版：

1. `piern/shared/tasks/state.py`：统一任务状态、别名归一化和合法转换校验。
2. `piern/shared/tasks/locks.py`：SQLite 协作锁、TTL、释放和过期清理。
3. 合成任务接入资源锁：模板、样本数据、Router 构建。
4. 训练任务接入 GPU 锁，并在启动失败、任务结束、删除时释放。
5. `piern/worker`：维护型 worker，负责过期锁清理。

未完成：

1. API 创建任务只写队列。
2. Worker 领取并执行全部合成、Router 和训练任务。
3. Worker 心跳、租约续期、任务重投递。

### 2.3 P1 数据库和完整性

已完成基础版：

1. `piern/shared/db/migrations.py`：轻量 SQLite schema migration。
2. 合成任务 store 接入 `schema_migrations`。
3. 训练任务 store 接入 `schema_migrations`。
4. `scripts/storage/verify_data_integrity.py`：源模板和原始 HDF5 checksum 扫描/校验。

未完成：

1. catalog SQLite 和各类 manifest 的统一 schema migration。
2. Parquet 分区 schema hash、row count、file count 的统一校验。
3. 前端文件管理展示完整性状态。

### 2.4 P1 安全和生产部署

已完成基础版：

1. 可选 `PIERN_AUTH_TOKEN`。
2. 写操作支持 Bearer token 或 `X-PIERN-Token`。
3. `Dockerfile` 多阶段构建，包含前端静态产物。
4. `compose.yaml` 可选部署示例。
5. `deploy/nginx/piern.conf` 生产反代示例。

未完成：

1. 登录页和前端 token 管理。
2. 角色权限。
3. 审计事件表。
4. secret 扫描。
5. Docker GPU 训练镜像的完整验证。

## 3. 下一批必须做的事情

优先级按实际收益排序：

1. 统一跨平台任务 API：`/api/jobs`、`/api/jobs/{id}`、`/api/jobs/{id}/events`、`/api/jobs/{id}/logs`。
2. 真正 worker 队列化：先接 Router 构建，再接样本填充，最后接训练。
3. 审计事件：记录删除、停止、启动训练、修改配置等危险操作。
4. 前端截图回归：把关键页面截图审查加入 Playwright，避免 UI 回退。
5. 拆分 `training_manager.py`：GPU、launcher、status、stop、artifacts 分模块。
6. 数据治理增强：catalog/manifest 版本化、派生数据可追溯、完整性状态上屏。
7. 严格类型质量：扩大 Ruff 规则，逐步引入 mypy 强制区。
8. 安全增强：前端登录态、secret 扫描、日志脱敏。

## 4. 默认验收命令

后端和仓库：

```bash
python -m ruff check .
python scripts/ci/check_repo_hygiene.py
python scripts/ci/check_migration_ready.py
python scripts/ci/check_consistency.py
python -m pytest
python scripts/storage/verify_data_integrity.py
```

前端：

```bash
cd frontend
npm run typecheck
npm run lint
npm run format:check
npm run test
npm run build
npm run e2e:smoke
```

服务：

```bash
python -m piern.shared.runtime.config
curl -fsS http://127.0.0.1:8000/api/health/ready
systemctl --user status piern-backend piern-frontend
```

生产部署：

```bash
scripts/services/start-prod.sh
docker compose config
```
