# PiERN 工业化执行计划

本文件是 `INDUSTRIALIZATION_PLAN.md` 的执行版。升级必须按阶段推进，每个阶段都需要通过验收命令后再进入下一阶段。

## 执行原则

- 不一次性重写系统。
- 每个阶段保持服务可启动、可测试、可回滚。
- 默认 CI 不跑百万样本生成、不下载模型、不训练。
- GitHub 只保存代码、配置、文档、语言模板和原始 HDF5。
- 服务器私有路径只允许出现在 `.env` 中。

## 阶段和验收

| 阶段 | 目标 | 验收 |
| --- | --- | --- |
| 0 | 基线冻结 | `check_consistency.py`、`pytest`、前端 build、服务 status 通过 |
| 1 | 工程配置标准化 | `pyproject.toml`、前端 lint/format/test/typecheck/build 可执行 |
| 2 | CI 质量门禁 | GitHub Actions 覆盖后端和前端轻量检查 |
| 3 | 迁移配置标准化 | `.env.example`、服务脚本环境变量化、迁移清单 |
| 4 | 后端保护 | 任务互斥、删除保护、active job reference 检查 |
| 5 | API 契约 | OpenAPI 类型生成并先覆盖训练 API |
| 6 | 任务持久化 | SQLite jobs/events，重启可见历史任务 |
| 7 | 前端结构升级 | 基础组件抽象、关键页面拆分、布局回归 |
| 8 | 后端模块化与观测 | 统一错误模型、健康检查分层、结构化日志 |

## 已交付范围

1. 阶段 0：记录并验证当前基线。
2. 阶段 1：新增 Python/前端工程配置和最小前端测试。
3. 阶段 2：新增轻量 CI。
4. 阶段 3：新增 `.env.example`、迁移清单和服务脚本环境变量入口。
5. 阶段 4：后端强制样本填充、Router 构建、文件删除的 active job 互斥和删除保护。
6. 阶段 5：新增 OpenAPI schema 导出、前端类型生成脚本，并将合成任务状态类型接入生成 schema。
7. 阶段 6：新增合成任务 SQLite `jobs` / `job_events` 存储，后端重启后保留历史任务并标记未完成任务为 `external_terminated`。
8. 阶段 8 部分：新增 `/api/health/live`、`/api/health/ready`、`/api/health/storage`、`/api/health/gpu`。

## 后续升级顺序

1. 扩大前端 lint/format 覆盖范围，逐页修复历史 warning。
2. 继续拆分 `training_manager.py`、`RegistryPage.tsx`、`TrainingJobDetailPage.tsx`。
3. 将训练任务 registry 从 JSON 进一步迁移到 SQLite，或在 JSON registry 外加 SQLite 事件审计。
4. 为主要 API 引入统一错误模型 `{ code, message, details }`。
5. 加 Playwright 冒烟测试和布局回归检查。

## 默认命令

```bash
python -m ruff check .
python scripts/ci/check_consistency.py
python -m pytest
cd frontend
npm run lint
npm run format:check
npm run test
npm run build
```
