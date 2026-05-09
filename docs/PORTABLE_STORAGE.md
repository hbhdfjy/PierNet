# 便携式数据存储方案

这个项目需要频繁迁移服务器。迁移策略是只保存可复现源数据，派生数据在新服务器上通过平台重新合成。

## GitHub 保存范围

GitHub 主分支只保存：

- 代码、配置、文档。
- 语言模板：data/templates/*.jsonl。
- 原始科学计算数据：data/*/*.h5。

GitHub 不保存：

- Stage 3 样本数据：data/text2comp_parquet/。
- Stage 4 Router 数据：data/router_parquet/。
- 目录数据库：data/catalog.sqlite。
- 索引、manifest、训练缓存、训练权重。

这些都是派生物，可以由模板和 HDF5 重新生成，保持 Git 忽略。

## 本机运行时数据层

- Parquet：data/text2comp_parquet/ 与 data/router_parquet/，作为平台运行时主数据。
- SQLite：data/catalog.sqlite，派生目录数据库，可随时重建。
- 可重建缓存：data/.manifests/、data/.indexes/、artifacts/token_router/*/prepared/。
- 训练权重：只迁移明确需要保留的 checkpoint。

## 平台生成流程

迁移到新服务器后，先打开数据合成平台：

1. 确认模板库和 HDF5 原始数据已经显示。
2. 在样本填充页面生成 Stage 3 样本，默认会写入 data/text2comp_parquet/。
3. 在 Router 数据构建页面生成 Stage 4 Router 数据，默认会写入 data/router_parquet/。
4. 平台会优先读取 Parquet 数据。

命令行仍然保留给需要批处理时使用：

    python scripts/text2comp/fill_samples.py --config configs/text2comp/default.yaml
    python scripts/router/build_router_data.py --input-format parquet --output-format parquet
    python scripts/storage/build_catalog_db.py

## 目录数据库

    python scripts/storage/build_catalog_db.py

data/catalog.sqlite 是派生文件，新服务器上可以重建，不需要提交到 Git。

## 旧 JSONL 迁移

如果以后遇到旧服务器只剩 JSONL，可以用迁移脚本转换为 Parquet：

    python scripts/storage/migrate_jsonl_to_parquet.py --kind all --validate

当前主流程不再依赖 JSONL。
