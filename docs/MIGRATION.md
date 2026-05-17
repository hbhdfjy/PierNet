# PiERN 迁移与便携式存储手册

本项目需要频繁迁移服务器。迁移原则是：GitHub 只保存可复现源数据和工程源文件，派生数据在新服务器上通过平台重新生成。

## 保存边界

GitHub 主分支保存：

- 代码、配置和文档。
- 语言模板：`data/templates/*.jsonl`。
- 原始科学计算数据：`data/*/*.h5`。

GitHub 不保存：

- Stage 3 样本数据：`data/text2comp/`、`data/text2comp_parquet/`。
- Stage 4 Router 数据：`data/router/`、`data/router_parquet/`。
- 目录数据库：`data/catalog.sqlite*`。
- 索引和 manifest：`data/.manifests/`、`data/.indexes/`。
- 训练缓存和权重：`artifacts/`。
- 运行日志：`.runlogs/`。

这些目录都是可重建产物，默认由 `.gitignore` 排除。

## 新服务器恢复

```bash
git clone git@github-hbhdfjy:hbhdfjy/piern.git piern
cd piern
git checkout main
```

创建环境：

```bash
conda create -n piern python=3.11 -y
conda activate piern
pip install -r requirements.txt
pip install -e .
cd frontend
npm ci
cd ..
```

复制并修改本机配置：

```bash
cp .env.example .env
```

必须按服务器实际情况确认：

- `PIERN_ROOT`
- `PIERN_DATA_ROOT`
- `PIERN_ARTIFACT_ROOT`
- `PIERN_CONDA_ENV`
- `PIERN_PYTHON`
- `PIERN_NODE_BIN`
- `PIERN_QWEN_EMBEDDING_MODEL`
- `PIERN_QWEN_EMBEDDING_TOKENIZER`
- `PIERN_MODFLOW_EXE`

## 迁移前验收

在旧服务器提交或迁移前运行：

```bash
python scripts/ci/check_repo_hygiene.py
python scripts/ci/check_migration_ready.py
python -m piern.shared.runtime.config
```

这些检查只做静态和轻量文件检查，不生成样本、不下载模型、不启动 GPU 训练。模型目录缺失会给出警告和下载提示，不阻止 CI。

## 启动服务

开发模式继续使用：

```bash
scripts/services/start.sh
scripts/services/status.sh
```

默认端口：

- FastAPI：`0.0.0.0:8000`
- Vite：`0.0.0.0:5173`

端口、CORS 来源、数据目录和模型路径都可以通过 `.env` 修改。

## 用户级 systemd 守护

无需管理员权限时，可以安装用户级 systemd 单元：

```bash
scripts/services/install-systemd.sh --dry-run
scripts/services/install-systemd.sh --enable --now
systemctl --user status piern-backend
systemctl --user status piern-frontend
curl -fsS http://127.0.0.1:8000/api/health/ready
```

如果服务器未启用用户 linger，重启后用户级服务可能不会自动恢复，需要管理员执行：

```bash
loginctl enable-linger $USER
```

当前 worker 单元负责共享维护任务，例如过期任务锁清理；默认不启用，可使用 `scripts/services/install-systemd.sh --enable --now --worker` 安装。

## 重建派生数据

启动平台后先检查：

- `data/templates/` 中的模板是否可见。
- `data/` 下各仿真器的 HDF5 是否可见。
- 文件管理页面是否能看到源数据。

然后在平台中按顺序执行：

1. 样本填充：由模板和 HDF5 生成 Stage 3 样本。
2. Router 数据构建：由 Stage 3 样本生成 Stage 4 Router 数据。
3. 训练平台：选择 Router 数据并启动训练。

命令行批处理入口仍保留：

```bash
python scripts/text2comp/fill_samples.py --config configs/text2comp/default.yaml
python scripts/router/build_router_data.py --input-format parquet --output-format parquet
python scripts/storage/build_catalog_db.py
```

## 目录数据库

`data/catalog.sqlite` 是派生目录数据库，可以在新服务器上重建：

```bash
python scripts/storage/build_catalog_db.py
```

它不需要提交到 Git。

## 旧 JSONL 迁移

如果旧服务器只剩 JSONL，可以转换为 Parquet：

```bash
python scripts/storage/migrate_jsonl_to_parquet.py --kind all --validate
```

当前主流程优先使用平台重新生成派生数据，不依赖迁移 JSONL。

## 验收

```bash
python scripts/storage/build_catalog_db.py
python scripts/ci/check_consistency.py
python -m pytest
cd frontend && npm run build
scripts/services/status.sh
```

以上命令通过，且平台能看到源数据并能重新合成派生数据，即迁移完成。
