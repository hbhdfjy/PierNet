# 便携式数据存储方案

这个项目需要频繁迁移服务器，因此存储目标不是引入长期运行的数据库服务，而是使用容易拷贝、校验和重建的文件式数据层。

## 分层

- Git：代码、配置、文档、小型基础 HDF5、模板文件。
- Parquet：`data/text2comp_parquet/` 与 `data/router_parquet/`，作为阶段 3/4 的主数据。
- SQLite：`data/catalog.sqlite`，派生目录数据库，可随时重建。
- 可重建缓存：`data/.manifests/`、`data/.indexes/`、`artifacts/token_router/*/prepared/`。
- 训练权重：只迁移明确需要保留的 checkpoint。


## 新数据生成默认路径

阶段 3 样本填充默认直接写 Parquet：

```bash
python scripts/text2comp/fill_samples.py --config configs/text2comp/default.yaml
```

如需旧 JSONL 兼容输出，显式指定：

```bash
python scripts/text2comp/fill_samples.py --config configs/text2comp/default.yaml --output-format jsonl
```

阶段 4 Router 数据构建默认读取阶段 3 Parquet，并直接写 `data/router_parquet/`：

```bash
python scripts/router/build_router_data.py --input-format parquet --output-format parquet
```

训练平台和数据合成平台会优先读取 Parquet；旧 JSONL 目录可以在完成备份后删除。

## 迁移 JSONL 到 Parquet

先确保依赖已安装：

```bash
pip install pyarrow duckdb
```

按场景逐步迁移，避免磁盘短时间双份占用过高：

```bash
python scripts/storage/migrate_jsonl_to_parquet.py --kind text2comp --scenarios coastal_seawater --validate
python scripts/storage/migrate_jsonl_to_parquet.py --kind router --scenarios coastal_seawater --validate
```

全部迁移：

```bash
python scripts/storage/migrate_jsonl_to_parquet.py --kind all --validate
```

如果使用 DVC 管理大数据，Git 中只提交 `.dvc` 指针文件，不直接提交 Parquet 内容：

```bash
pip install 'dvc[ssh]'
dvc init
dvc add data/text2comp_parquet data/router_parquet
git add .dvc .dvcignore data/.gitignore data/text2comp_parquet.dvc data/router_parquet.dvc
```

配置远端对象存储或另一台 SSH 服务器后再推送数据对象：

```bash
dvc remote add -d storage ssh://USER@HOST/path/to/dvc-store
dvc push
```

如果暂时没有 DVC 远端，可以先生成迁移包和校验文件：

```bash
tar -czf migration_exports/piern-parquet-YYYYMMDD.tar.gz data/text2comp_parquet data/router_parquet data/text2comp_parquet.dvc data/router_parquet.dvc data/.gitignore .dvc/config
sha256sum migration_exports/piern-parquet-YYYYMMDD.tar.gz > migration_exports/piern-parquet-YYYYMMDD.tar.gz.sha256
```

## 重建目录数据库

```bash
python scripts/storage/build_catalog_db.py
```

`data/catalog.sqlite` 是派生文件，新服务器上可以重建，不需要提交到 Git。

## 新服务器恢复建议

```bash
git clone https://github.com/hbhdfjy/piern
cd piern
pip install -r requirements.txt
# 如果使用 DVC 且已配置远端：dvc pull
# 如果使用迁移包：tar -xzf migration_exports/piern-parquet-YYYYMMDD.tar.gz
python scripts/storage/build_catalog_db.py
python scripts/utils/rebuild_manifests.py
```

训练平台会优先识别 Parquet 路由数据；如果旧训练路径需要 JSONL，会在 `data/router/.parquet_jsonl_cache/` 下生成兼容缓存。这个目录属于可重建缓存，不需要迁移。


## 当前离机备份

当前已将 `migration_exports/piern-parquet-20260508.tar.gz` 拆分并推送到 GitHub 分支 `data-backup-20260508`。恢复步骤见 `docs/MIGRATION_RESTORE.md`。
