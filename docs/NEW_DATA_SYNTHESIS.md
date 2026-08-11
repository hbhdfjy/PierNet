# 新数据合成

“新数据合成”是 PierNet 的独立顶层工作区，生产路径为 `/new-synth/`。它与旧数据合成、简洁训练和复杂训练并列，不复用 `frontend/src/synth` 的页面、状态或样式；前端源码、依赖清单、测试和构建产物位于 `frontend-new-synth/`。

## 用户流程

用户只需完成三个步骤：

1. **接入数据**：上传 HDF5、复用或运行内置仿真，或者调用已注册且支持数据生成的专家模型。三种方式互斥，选择一种即可。
2. **确认定义**：核对领域、场景、专家输入参数、物理输出通道和时间采样范围。HDF5 中可推导的样本数、维度和形状由服务器识别。
3. **生成数据**：一次操作完成模板、Text2Comp 数据、当前二分类 Router 数据和专家评测数据的生成、校验与登记。

生成完成后，“进入简洁训练”和“复杂训练”会携带稳定的 Router 数据集 ID。两个训练页面按该 ID 自动选择同一份数据，不通过目录名称猜测。

## 数据契约

PierNet 的调用顺序是：

```text
用户问题 -> Router -> Text2Comp -> Expert -> 物理结果
```

因此三个产物分别承担不同职责：

| schema | 用途 | 关键字段 |
| --- | --- | --- |
| `piernet.text2comp` | 训练自然语言到专家输入参数 | `prompt`, `label`, `expert_input`, `metadata.parameter_names` |
| `piernet.router.binary` | 训练当前二分类路由 | `context`, `label`, `metadata.route_label`, `metadata.class_names` |
| `piernet.expert-evaluation` | 专家验证和端到端评测 | `expert_input`, `expected_expert_output` |

`piernet.text2comp.label` 与 `expert_input` 相同，都是按定义顺序排列的专家模型输入参数。物理 `timeseries` 或旧数据中的 `target` 只进入 `piernet.expert-evaluation.expected_expert_output`，不得作为 Text2Comp 标签。Router 记录只保存路由训练所需的文本、标签和元数据，不复制完整物理时序。

所有正式数据集都包含稳定 `dataset_id`、`schema_name`、`schema_version`、`artifact_type`、来源哈希、定义版本和生成配置。路径和元数据同时使用 `simulator/scenario`，避免不同模拟器的同名场景互相覆盖。

## API 与状态

聚合接口位于 `/api/new-synth/*`，主要包括：

- `POST /session`：创建匿名工作会话。
- `GET|POST /workflows`：列出或创建工作流。
- `GET /workflows/{workflow_id}`：恢复持久化状态。
- `POST /workflows/{workflow_id}/source/upload`：上传 HDF5。
- `POST /workflows/{workflow_id}/source/simulation`：接入内置仿真。
- `POST /workflows/{workflow_id}/source/expert`：通过专家模型生成来源数据。
- `PUT /workflows/{workflow_id}/definition`：保存确认后的数据定义。
- `POST /workflows/{workflow_id}/definition/suggest`：用已配置的 LLM 补全人类可读说明，不修改数据契约。
- `POST /workflows/{workflow_id}/generate`：幂等启动生成。
- `POST /workflows/{workflow_id}/retry`、`cancel`：失败重试或取消。
- `GET /workflows/{workflow_id}/events`、`datasets`：查询真实进度和登记结果。

工作流状态保存在 `data/new_synth/<workflow_id>` 和 `.runlogs/new_synth/state.sqlite`。状态包含当前阶段、进度、错误代码、来源版本、定义版本和产物 ID，页面刷新或后端重启后可以恢复。

## 存储与缓存

原始来源、规范化数据、已确认定义和正式训练数据位于 `data/new_synth/<workflow_id>`，不会被缓存清理删除。仅可重建内容进入 `data/.cache/new_synth`。

派生缓存 TTL 由 `PIERN_NEW_SYNTH_CACHE_TTL_SECONDS` 配置，默认 14 天；读取会更新最近访问时间。清理逻辑只遍历新数据合成缓存根目录，并校验真实路径仍在该目录内。训练数据列表对旧 Parquet 清单使用 15 秒内存缓存，并跳过页面不消费的全量 Router 标签统计；新数据合成登记结果始终实时合并。

## 开发与验证

```bash
scripts/services/start.sh
npm --prefix frontend-new-synth run check
NEW_SYNTH_E2E_HDF5=/path/to/sample.h5 npm --prefix frontend-new-synth run e2e
```

开发端口为 `3002`，主前端仍使用 `3000`，后端使用 `8000`。生产构建由 `scripts/services/start-prod.sh` 生成，FastAPI 在 `/new-synth/` 托管独立构建产物。

浏览器 E2E 使用真实小型 HDF5，并验证上传、定义、生成、登记及简洁/复杂训练自动选中；同时通过界面走通内置仿真来源。视觉回归覆盖桌面与 `390x844` 移动视口，并为长页面保存分段截图。
