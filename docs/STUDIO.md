# PiERN Studio

PiERN Studio 是与旧前端完全隔离的用户工作台。用户建立项目后，上传自己的成对科学计算数据和 Python 专家模型；Studio 自动完成数据解析、真实专家前向验证、项目专属训练内容生成、任务选择模型与参数生成模型训练、拼装、自动验证和对话推理。

## 服务与边界

| 服务 | 端口 | 入口 |
| --- | ---: | --- |
| 旧前端 | 3000 | `/` |
| Studio 开发服务 | 3001 | `/studio/` |
| FastAPI 后端 | 8000 | `/api/studio/*` |
| Studio 生产静态页面 | 8000 | `/studio/` |

`frontend-studio` 拥有独立的依赖、路由、组件、样式、状态、测试与构建产物，不导入 `frontend/src`。浏览器只访问 `/api/studio/*`；`PierNet/studio` 负责聚合底层能力。项目由 30 天 HttpOnly 会话 Cookie 隔离，上传、训练产物、日志和对话记录均按项目保存。

## 启动

开发服务一次启动：

```bash
scripts/services/start.sh
scripts/services/status.sh
```

只运行 Studio 前端：

```bash
scripts/services/run-studio.sh
```

安装并立即启用后端、旧前端、Studio 和 worker 的用户级持久化服务：

```bash
scripts/services/install-systemd.sh --now
```

当前开发容器未提供 `systemctl`，因此本机由 `start.sh` 使用 detached 进程托管，退出 SSH 后仍保持运行；`deploy/systemd/PierNet-studio.service` 用于具备用户级 systemd 的正式部署环境。

生产构建会同时生成 `frontend/dist` 和 `frontend-studio/dist`：

```bash
scripts/services/start-prod.sh
```

可用 `PierNet_STUDIO_PORT` 覆盖 Studio 端口，默认值为 `3001`。

## 数据格式

单文件和仅包含一个可识别数据文件的 `.zip`、`.tar.gz`、`.tgz` 均可上传，单次上传上限为 1 GB。

- NPZ：优先读取 `inputs`/`outputs`，也兼容 `params`/`timeseries`；首维必须是样本数。
- HDF5：需要 `inputs`/`outputs` 或 `params`/`timeseries` 数据集。
- CSV、Parquet：选择数值输入列和输出列；以 `input`、`output` 开头的列会自动建议。
- 输入与输出样本数必须一致，至少包含 4 条样本，且所有值必须为有限数。

上传后的数据会规范化为：

```text
inputs:  [samples, ...input_shape]
outputs: [samples, ...output_shape]
```

## 专家模型接口

支持 `.py`、`.zip`、`.tar.gz`、`.tgz`。单个 `.py` 文件可直接上传。多文件压缩包中，如果只有一个 Python 文件在顶层定义了 `predict(inputs)`，Studio 会自动识别入口并生成清单。

存在多个候选入口或需要自定义调用名时，模型包应包含
`piernet_expert_model.json`：

```json
{
  "runtime": "python",
  "entrypoint": "model.py",
  "callable": "predict",
  "batch_mode": "auto"
}
```

模型接收 `float32` NumPy 数组。默认 `batch_mode=auto`，Studio 会优先尝试
批量调用，并在接口只支持单个样本时自动逐样本执行。模型应返回数值数组；
也可返回 `{"outputs": value}`、`{"output": value}` 或单元素 tuple/list。

Studio 会在独立子进程中用项目真实样本调用专家模型，并校验 batch、输出形状、有限值和参考输出误差。第一版不会自动安装模型包的额外依赖，运行环境需已具备所需依赖。

## 项目流程

1. 新建项目并说明计算目标。
2. 上传数据和专家模型；表格数据按需确认字段。
3. 执行兼容性检查，真实调用专家模型。
4. 启动训练。服务器本地 Qwen 作为冻结语言基座，项目数据训练独立的任务选择头和参数生成头。
5. 保存模型、训练指标和拼装清单，并自动执行验证。
6. 使用自动填入的推荐问题对话；结果包含文字回答、真实输入、完整数值输出及折线图或热力图。

重新上传数据或专家模型会使下游训练、拼装和对话结果失效，必须重新检查并训练。运行期间禁止替换项目资源；失败任务可重试，运行任务可取消。

## API

主要接口如下，均位于 `/api/studio`：

- `POST /session`
- `GET /presets`
- `GET|POST /projects`
- `GET /projects/{project_id}`
- `DELETE /projects/{project_id}`
- `POST /projects/{project_id}/data`
- `POST /projects/{project_id}/expert`
- `POST /projects/{project_id}/mapping`
- `POST /projects/{project_id}/inspect`
- `POST /projects/{project_id}/compatibility-check`
- `POST /projects/{project_id}/run`
- `POST /projects/{project_id}/retry`
- `POST /projects/{project_id}/cancel`
- `GET /projects/{project_id}/events`
- `POST /projects/{project_id}/chat`
- `GET /health`

文件上传使用原始二进制请求体，文件名通过 URL 编码后的 `X-File-Name` 请求头传递。项目事件使用 SSE 推送。

## 存储与复现

```text
data/studio/{project_id}/       用户资源、规范化数据和训练语句
artifacts/studio/{project_id}/  训练数据、模型、指标、拼装清单和验证结果
.runlogs/studio/{project_id}/   任务与错误日志
.runlogs/studio/projects.sqlite 项目状态、事件、对话和审计记录
```

删除项目需要当前会话拥有该项目，运行中的项目必须先停止。删除操作只清理该
项目的上述三棵目录和关联数据库记录，不扫描或删除其他项目。创建、上传、字段
确认、兼容性检查、训练启动与结果、取消、对话和删除等关键操作写入独立
`audit_events` 审计表；审计记录不保存原始科学数据或完整对话内容。

两组不同形状的端到端验证可复现为：

```bash
.conda/env/bin/python scripts/studio/e2e_demo.py
```

验证摘要写入 `.runlogs/studio/e2e-full.json`。当前已验证 `3 -> 8` 和 `5 -> 4x6` 两条真实链路，均完成上传专家前向、Qwen 特征提取、训练、持久化、拼装、自动验证及对话推理。

质量检查：

```bash
.conda/env/bin/python -m pytest -q tests/test_studio_flow.py tests/test_studio_service_scripts.py
npm --prefix frontend-studio run check
HOME=/root/data TMPDIR=/root/data npm --prefix frontend-studio run e2e
```

Playwright 覆盖 `390x844`、`768x1024`、`1280x720`、`1440x900` 四档视口，并检查关键页面不存在横向溢出。

构建默认最多并行运行 1 个 Studio 项目，单次构建上限 30 分钟，启动前至少保留 2 GB 可用空间。可通过 `PIERN_STUDIO_MAX_CONCURRENT_RUNS`、`PIERN_STUDIO_MAX_RUN_SECONDS` 和 `PIERN_STUDIO_MIN_FREE_DISK_BYTES` 调整。项目事件用于实时进度，独立审计表保留关键操作轨迹。

## 当前限制

第一版采用匿名会话隔离，不等同于正式账号与权限系统。上传的 Python 模型运行在受控子进程中，但尚未使用容器级沙箱，因此只应运行可信模型；面向不可信多租户开放前，应增加身份认证、对象存储、任务队列、容器资源限额、依赖镜像，以及集中审计导出与保留策略。当前 Qwen 基座冻结，训练项目专属轻量头，适合建立可复现 Demo；大规模联合微调需要单独的分布式训练与模型版本治理。
