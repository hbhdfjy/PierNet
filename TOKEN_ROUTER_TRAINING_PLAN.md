# Token Router 训练平台改造计划

## 1. 文档定位

本文件描述 Token Router 训练平台的当前状态、明确边界和后续演进方向。

它不是事实源。

当前事实以这些文件为准：

- `README.md`
- `PROJECT_OVERVIEW.md`
- `CLAUDE.md`
- 实际代码实现

## 2. 当前状态

训练平台已经落地，入口为：

- 前端：`/training`
- 后端：`/api/training/*`

当前已实现：

- 训练总览页
- 新建训练页
- 训练任务列表页
- 训练任务详情页
- GPU 状态查看
- 单 GPU 分配
- 训练任务创建、停止、详情读取
- 训练曲线、测试曲线、日志、checkpoint 查看
- file-backed training job registry

当前未实现：

- 多 GPU 分配
- DDP
- 独立 run 管理页
- 通用模型训练平台
- 模型注册中心

## 3. 产品边界

### 平台边界

训练平台和数据合成平台继续保持产品层分离：

- 数据合成平台：`/synth/*`
- 训练平台：`/training/*`

要求：

- 数据合成平台不混入训练任务管理 UI
- 训练平台不承载 Stage 1-4 操作逻辑
- 两边只通过链接跳转

### 数据边界

训练平台当前只依赖：

- `data/router/by_scenario/*.jsonl`
- `data/router/train.jsonl`
- `data/.manifests/router.json`
- `artifacts/token_router/`

训练平台不应直接依赖：

- Stage 1 HDF5 内部结构
- Stage 2 模板实现细节
- Stage 3 填充实现细节

## 4. 当前实现结构

### 前端

当前页面：

- `/training`
- `/training/new`
- `/training/jobs`
- `/training/jobs/:jobId`

主要文件：

- `frontend/src/training/TrainingApp.tsx`
- `frontend/src/training/pages/TrainingOverviewPage.tsx`
- `frontend/src/training/pages/TrainingNewJobPage.tsx`
- `frontend/src/training/pages/TrainingJobsPage.tsx`
- `frontend/src/training/pages/TrainingJobDetailPage.tsx`

### 后端

主要文件：

- `piern/training/api/routers/training.py`
- `piern/training/api/schemas/training.py`
- `piern/training/services/training_manager.py`

当前 API：

- `GET /api/training/overview`
- `GET /api/training/datasets`
- `GET /api/training/gpus`
- `GET /api/training/jobs`
- `POST /api/training/jobs`
- `GET /api/training/jobs/{job_id}`
- `GET /api/training/jobs/{job_id}/curves`
- `GET /api/training/jobs/{job_id}/logs`
- `POST /api/training/jobs/{job_id}/stop`

## 5. 当前训练核心

当前训练核心不是 Transformer 平台，而是轻量字符级序列分类器。

组成：

- tokenizer：`CharTokenizer`
- model：`FullSeqDilatedConvRouter`
- data prepare：router JSONL → prepared token cache
- split：`train / test`
- device：单 GPU

实现文件：

- `piern/training/router/tokenizer.py`
- `piern/training/router/model.py`
- `piern/training/router/data.py`
- `piern/training/router/train.py`
- `scripts/router/train_token_router.py`

## 6. 当前任务生命周期

训练平台使用独立的 file-backed registry 管理任务。

状态集合：

- `queued`
- `starting`
- `running`
- `evaluating`
- `done`
- `error`
- `terminated`

当前持久化文件：

- `artifacts/token_router/training_jobs.json`

训练输出目录：

- `artifacts/token_router/{simulator}/runs/{run_name}/`

训练日志目录：

- `.runlogs/`

## 7. 当前已知边界

- 当前 GPU 分配是保守阈值策略，不是抢占式调度
- 当前没有 validation split
- 当前没有多卡训练
- 当前没有独立 runs 页面
- 当前没有模型注册能力

## 8. 后续演进方向

### 优先级 1

- 增加 run 清理与任务记录清理能力
- 增加训练任务恢复与导入能力
- 增加更多训练产物摘要展示

### 优先级 2

- 将 GPU inventory / registry / curves 进一步从 `training_manager.py` 拆分
- 增加更稳定的任务持久化与恢复机制
- 增加更多训练 smoke tests

### 优先级 3

- 多 GPU / DDP
- 独立 run 页面
- 更通用的训练平台能力

## 9. 修改训练平台时必须同步检查

- `README.md`
- `PROJECT_OVERVIEW.md`
- `CLAUDE.md`
- `frontend/src/training/*`
- `piern/training/*`
- `scripts/router/train_token_router.py`