# 便携式数据存储方案

这个项目需要频繁迁移服务器，因此存储目标不是引入长期运行的数据库服务，而是使用容易拷贝、校验和重建的文件式数据层。

## 分层

- Git：代码、配置、文档、小型基础 HDF5、模板文件。
- Parquet：`data/text2comp_parquet/` 与 `data/router_parquet/`，用于替代大体量 JSONL。
- SQLite：`data/catalog.sqlite`，派生目录数据库，可随时重建。
- 可重建缓存：`data/.manifests/`、`data/.indexes/`、`artifacts/token_router/*/prepared/`。
- 训练权重：只迁移明确需要保留的 checkpoint。

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

如果使用 DVC 管理大数据：

```bash
pip install 'dvc[ssh]'
dvc init
python scripts/storage/migrate_jsonl_to_parquet.py --kind all --validate --dvc-add
dvc remote add -d storage ssh://USER@HOST/path/to/dvc-store
dvc push
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
# 如果使用 DVC：dvc pull
python scripts/storage/build_catalog_db.py
python scripts/utils/rebuild_manifests.py
```

训练平台会优先识别 Parquet 路由数据；如果旧训练路径需要 JSONL，会在 `data/router/.parquet_jsonl_cache/` 下生成兼容缓存。这个目录属于可重建缓存，不需要迁移。
