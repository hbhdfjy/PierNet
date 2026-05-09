# 迁移恢复手册

当前项目的代码在 `main` 分支，Parquet 主数据在 GitHub 独立备份分支 `data-backup-20260508`。

## 新服务器恢复

```bash
git clone git@github-hbhdfjy:hbhdfjy/piern.git piern
cd piern
git checkout main

# 拉取数据备份分支到相邻目录
git clone --branch data-backup-20260508 --single-branch git@github-hbhdfjy:hbhdfjy/piern.git ../piern-data-backup-20260508

# 拼接分片、校验 sha256、解压 Parquet 数据
bash scripts/storage/restore_data_backup.sh ../piern-data-backup-20260508

# 安装依赖并重建派生目录库
pip install -r requirements.txt
python scripts/storage/build_catalog_db.py
```

恢复完成后应存在：

- `data/text2comp_parquet/`
- `data/router_parquet/`
- `data/text2comp_parquet.dvc`
- `data/router_parquet.dvc`

## 校验

```bash
sha256sum -c migration_exports/piern-parquet-20260508.tar.gz.sha256
python scripts/storage/build_catalog_db.py
python scripts/ci/check_consistency.py
```

当前数据备份包：

- archive: `piern-parquet-20260508.tar.gz`
- sha256: `0b4a29257954a4c0bfa5830a28d6a9fa7ab1d2403c9504d8c53c17892a027ff7`
- backup branch: `data-backup-20260508`

## 说明

GitHub 备份分支存放的是拆分后的迁移包，不参与主分支开发。后续如果配置了真正的 DVC 远端，优先使用 `dvc push/pull`；该备份分支作为无外部对象存储时的兜底恢复路径。
