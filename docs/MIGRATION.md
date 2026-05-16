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

## 启动服务

```bash
scripts/services/start.sh
scripts/services/status.sh
```

默认端口：

- FastAPI：`0.0.0.0:8000`
- Vite：`0.0.0.0:5173`

端口、CORS 来源、数据目录和模型路径都可以通过 `.env` 修改。

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
