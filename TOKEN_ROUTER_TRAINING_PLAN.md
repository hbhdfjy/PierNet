# Token Router 训练平台改造计划

## 1. 目标

在不影响现有数据合成平台的前提下，建设一个独立的 Token Router 训练平台，满足以下能力：

- 可以自由选择训练数据
- 可以创建多个并行训练任务
- 可以查看训练进程与资源占用
- 可以分配单张空闲 GPU
- 可以停止和查看训练任务
- 点击单个训练任务后，可以查看完整训练曲线与测试曲线
- 数据平台与训练平台继续保持产品层隔离，只通过超链接互相跳转

## 2. 当前现状

### 已有能力

- 训练平台前端入口已存在：`/training`
- 现有训练后端是 CLI-first：
  - `piern/training/router/`
  - `scripts/router/train_token_router.py`
- 当前模型已经可以在服务器上训练，并输出：
  - `train_log.jsonl`
  - `router_latest.pt`
  - `router_epoch_xxxx.pt`
  - `test_metrics_epoch_xxxx.json`
  - `test_metrics_latest.json`
- 当前已有可复用任务基础设施：
  - `piern/synth/services/job_manager.py`
  - `/api/generate/{id}/stream` 这一套 SSE 机制
- 当前已有可复用前端作业监控模式：
  - `frontend/src/synth/hooks/useJobMonitor.ts`
  - `frontend/src/synth/components/generation/JobMonitorPanel.tsx`

### 当前实现状态

- 训练后端 Web API 已启用：
  - `/api/training/overview`
  - `/api/training/datasets`
  - `/api/training/gpus`
  - `/api/training/jobs`
  - `/api/training/jobs/{job_id}`
  - `/api/training/jobs/{job_id}/curves`
  - `/api/training/jobs/{job_id}/logs`
  - `/api/training/jobs/{job_id}/stop`
- 训练任务已落地 file-backed job registry
- 前端训练任务管理页和曲线展示页已可用
- 后端 GPU 资源探测与分配接口已可用
- 当前训练代码仍是单进程单卡逻辑，多 GPU 训练还没实现

## 3. 产品边界

### 平台边界

继续保持双平台结构：

- 数据合成平台：`/simulate` 等现有页面
- 训练平台：`/training/*`

要求：

- 数据合成平台不混入训练任务的复杂 UI
- 训练平台不直接承载 Stage 1-4 的操作逻辑
- 两边只通过链接跳转
- 训练平台只消费 `data/router/` 及其后续衍生产物

### 数据边界

训练平台只依赖以下输入：

- `data/router/train.jsonl`
- `data/router/by_scenario/*.jsonl`
- `data/.manifests/router.json`
- 后续新增的训练 prepared cache / run artifact

不让训练平台直接依赖：

- Stage 1 HDF5 细节
- Stage 2 模板细节
- Stage 3 样本填充细节

## 4. 前端目标界面

训练平台建议拆为 4 个页面。

### 4.1 训练首页 `/training`

作用：训练控制台首页与概览。

展示内容：

- 当前可训练数据集摘要
- 当前运行中的训练任务数
- 当前可用 GPU / 已占用 GPU
- 最近完成的训练 run
- 快捷入口：新建训练 / 查看任务 / 查看 runs

### 4.2 新建训练页 `/training/new`

作用：提交新的训练任务。

当前需要的配置表单：

- 训练数据选择
  - 大场景选择，例如 `modflow`
  - 子场景多选，例如 4 个 MODFLOW 子场景
  - 是否用全部现有 Router 数据
- 训练参数
  - batch size
  - test ratio
  - learning rate
  - eval interval
  - checkpoint 策略
  - 是否 resume
  - resume checkpoint 路径
- 资源配置
  - 空闲 GPU 列表
  - 选定单张 GPU
- 启动动作
  - 提交任务
  - 提交并跳转任务详情

### 4.3 训练任务列表页 `/training/jobs`

作用：管理当前和历史训练任务。

列表字段：

- `job_id`
- 任务名称
- 数据集范围
- 状态：`queued / starting / running / evaluating / done / error / terminated`
- 分配 GPU
- 当前 epoch / step
- 最新 loss
- 最近测试 epoch
- 启动时间 / 运行时长
- 停止按钮 / 查看详情按钮

支持操作：

- 查看任务详情
- 停止任务
- 删除任务记录
- 只看运行中 / 只看失败 / 只看完成

### 4.4 训练任务详情页 `/training/jobs/:jobId`

作用：查看单个训练任务的完整过程。

展示内容：

- 任务基础信息
  - 训练数据
  - 模型配置
  - 分配 GPU
  - checkpoint 策略
- 实时状态卡片
  - 当前 epoch
  - 当前 step
  - step/s
  - loss
  - ETA
- 训练曲线
  - `train loss vs global_step`
  - `train loss vs epoch`
  - `throughput vs step`
- 测试曲线
  - `accuracy vs epoch`
  - `precision vs epoch`
  - `recall vs epoch`
  - `f1 vs epoch`
  - `pr_auc vs epoch`
- 子场景测试曲线
  - 每个子场景一组指标折线
- 日志面板
  - 原始 stdout / stderr
- 操作区
  - 停止
  - 复制命令
  - 打开产物目录

### 4.5 训练 run 列表页 `/training/runs`

当前未实现，先不开放。

已完成 run 的信息暂时通过任务列表页和任务详情页查看：

- job id
- checkpoint 列表
- 最新测试指标
- 完成时间

## 5. 后端改造方案

后端建议新增独立训练路由，不继续把逻辑堆进现有 Stage 1-4 路由中。

### 5.1 新增模块

当前已新增：

- `piern/training/api/routers/training.py`
- `piern/training/api/schemas/training.py`
- `piern/training/services/training_manager.py`

当前没有继续拆成 `gpu_inventory / training_registry / training_curves` 三个服务文件，而是先集中落在 `training_manager.py`，避免在第一版过早拆分。

### 5.2 训练任务生命周期

当前训练平台不复用 Stage 1-4 的 SSE job manager，而是使用独立的 file-backed registry 管理训练进程。

状态建议统一为：

- `queued`
- `starting`
- `running`
- `evaluating`
- `done`
- `error`
- `terminated`

训练 manager 负责：

- 校验训练请求
- 锁定 GPU
- 生成 job record
- 启动训练子进程
- 持久化 job metadata
- 读取日志并解析训练进度
- 释放 GPU 锁
- 维护 run 记录

### 5.3 建议 API

#### 训练数据与资源

- `GET /api/training/overview`
  - 首页摘要
- `GET /api/training/datasets`
  - 可训练数据集列表
- `GET /api/training/gpus`
  - 当前 GPU 状态
- `POST /api/training/gpus/refresh`
  - 手动刷新 GPU 探测

#### 训练任务

- `POST /api/training/jobs`
  - 创建训练任务
- `GET /api/training/jobs`
  - 获取任务列表
- `GET /api/training/jobs/{job_id}`
  - 获取任务详情
- `GET /api/training/jobs/{job_id}/stream`
  - SSE 日志与进度流
- `POST /api/training/jobs/{job_id}/stop`
  - 停止任务
- `DELETE /api/training/jobs/{job_id}`
  - 删除任务记录

#### 曲线与指标

- `GET /api/training/jobs/{job_id}/curves`
  - 返回当前 job 的训练曲线 + 测试曲线
- `GET /api/training/jobs/{job_id}/logs`
  - 分页读取原始日志
- `GET /api/training/runs`
  - run 列表
- `GET /api/training/runs/{run_id}`
  - run 详情
- `GET /api/training/runs/{run_id}/curves`
  - 已完成 run 的全部曲线

## 6. GPU 资源管理设计

这是这次改造里最需要单独设计的一块。

### 6.1 GPU 探测

后端通过 `nvidia-smi` 定期采样，输出每张卡：

- `index`
- `name`
- `memory_used`
- `memory_total`
- `utilization`
- `running_processes`
- `available` 标记

### 6.2 空闲判定

建议使用保守策略：

- 显存占用低于阈值
- 利用率低于阈值
- 未被平台内其它训练任务锁定

平台内再维护一层 GPU 锁文件或 registry，避免两个训练任务同时抢到同一张表面上“空闲”的卡。

### 6.3 单卡与多卡

这里要明确分两阶段。

#### Phase 1

- 前后端都支持“选择 GPU 集合”
- 但训练执行先只支持：
  - 单卡训练
  - 或“多卡占位但只使用第一张卡”这种伪多卡模式不开放给用户
- 也就是说，Phase 1 的 UI 可以只允许选 1 张卡，先把训练平台跑稳

#### Phase 2

- 后端训练脚本升级到真正的多 GPU 分布式训练
- 采用 `torchrun` / DDP
- 前端正式开放“任意数量 GPU”训练

这个顺序必须明确，因为“任意数量 GPU”不是调度问题，而是训练代码本身要支持 DDP。

## 7. 曲线数据设计

当前训练后端已经有两类曲线源。

### 7.1 训练曲线来源

文件：`train_log.jsonl`

字段：

- `epoch`
- `step`
- `global_step`
- `avg_loss`
- `steps_per_sec`
- `eta_seconds`

前端可展示：

- `train loss vs global_step`
- `train loss vs epoch`
- `throughput vs global_step`

### 7.2 测试曲线来源

文件：

- `test_metrics_epoch_xxxx.json`
- `test_metrics_latest.json`

字段：

- `overall.accuracy`
- `overall.precision`
- `overall.recall`
- `overall.f1`
- `overall.pr_auc`
- 各子场景对应指标

前端可展示：

- 总体指标曲线
- 子场景分指标曲线

### 7.3 曲线服务层

新增 `training_curves.py`，负责：

- 读取 `train_log.jsonl`
- 读取所有 `test_metrics_epoch_*.json`
- 标准化返回前端曲线结构
- 支持 downsample，避免大日志一次性把前端打爆

建议统一接口结构：

- `series_name`
- `x`
- `y`
- `kind`：`train` / `test`
- `scope`：`overall` / `scenario`

## 8. 多任务管理设计

### 8.1 任务 registry

新增训练任务 registry，建议落盘到：

- `artifacts/token_router/job_registry.json`
- 或拆成 `artifacts/token_router/jobs/{job_id}.json`

每个任务至少记录：

- `job_id`
- `run_id`
- `status`
- `created_at`
- `started_at`
- `ended_at`
- `requested_gpu_ids`
- `allocated_gpu_ids`
- `command`
- `pid`
- `dataset_spec`
- `training_config`
- `artifact_dir`
- `log_path`

### 8.2 停止任务

停止时需要做三件事：

- 杀训练进程/进程组
- 刷新任务状态为 `terminated`
- 释放 GPU 锁

### 8.3 多任务并发

平台应允许多个训练任务并发，只要：

- GPU 不冲突
- 平台内锁允许
- 系统资源校验通过

## 9. 前端实现结构建议

建议把训练平台从单个占位页拆成以下结构：

```text
frontend/src/training/
├── TrainingApp.tsx
├── pages/
│   ├── TrainingOverviewPage.tsx
│   ├── TrainingNewJobPage.tsx
│   ├── TrainingJobsPage.tsx
│   ├── TrainingJobDetailPage.tsx
│   └── TrainingRunsPage.tsx
├── components/
│   ├── TrainingJobTable.tsx
│   ├── TrainingLaunchForm.tsx
│   ├── GPUInventoryPanel.tsx
│   ├── CurvePanel.tsx
│   ├── MetricCards.tsx
│   ├── LogViewer.tsx
│   └── CheckpointList.tsx
└── hooks/
    ├── useTrainingJobs.ts
    ├── useTrainingJobDetail.ts
    ├── useTrainingCurves.ts
    └── useGPUInventory.ts
```

曲线组件建议优先选轻量可控方案：

- `recharts`
- 或 `visx`

不建议自己手动画复杂坐标轴。

## 10. 实施顺序

### Phase 1：后端基础

先做：

- 训练 job registry
- 训练任务 API
- GPU 探测 API
- 训练曲线 API
- SSE 训练日志流

目标：后端先能稳定创建/管理训练任务。

### Phase 2：前端训练平台骨架

先做：

- `/training`
- `/training/new`
- `/training/jobs`
- `/training/jobs/:jobId`

目标：可以创建任务、看到任务、能点进去看详情。

### Phase 3：曲线可视化

接入：

- 训练曲线
- 测试曲线
- 子场景曲线
- 原始日志折叠面板

目标：任务详情页完整可用。

### Phase 4：运行管理

补齐：

- 停止任务
- 删除任务
- 查看 checkpoint
- run 列表页
- run 详情页

### Phase 5：多 GPU 训练

这是单独阶段。

需要：

- 训练脚本升级 DDP
- 后端正确分配多 GPU
- 前端开放多 GPU 选择
- 任务详情显示 world size / rank 信息

在 DDP 完成之前，前端不要承诺“任意数量 GPU 都可立即训练”。

## 11. 当前建议

如果按最稳路线推进，下一步应先实现：

- 后端训练任务 API
- GPU inventory API
- 训练任务详情 API
- 训练曲线 API
- 前端训练任务列表页
- 前端训练任务详情页

也就是先把：

- “能发任务”
- “能看任务”
- “能看曲线”

这三件事做实，再扩到真正的多 GPU 分布式训练。
