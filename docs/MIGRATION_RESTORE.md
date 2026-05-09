# 迁移恢复手册

迁移时只依赖 GitHub 主分支中的源数据：语言模板和原始 HDF5。Stage 3 样本、Router 数据、目录库和训练缓存都不迁移，在新服务器上通过平台重新合成。

## 新服务器恢复

    git clone git@github-hbhdfjy:hbhdfjy/piern.git piern
    cd piern
    git checkout main
    pip install -r requirements.txt
    python scripts/storage/build_catalog_db.py

启动平台后检查：

- data/templates 里的模板是否可见。
- data 下各模拟器的 HDF5 是否可见。
- 文件管理页面是否能看到源数据。

## 重新合成派生数据

在平台里按顺序执行：

1. 样本填充：由模板和 HDF5 生成 Stage 3 Parquet。
2. Router 数据构建：由 Stage 3 Parquet 生成 Router Parquet。
3. 训练平台：使用 Router Parquet 启动训练。

对应输出是：

- data/text2comp_parquet/
- data/router_parquet/
- data/catalog.sqlite

这些输出不提交到 GitHub。

## 校验

    python scripts/storage/build_catalog_db.py
    python scripts/ci/check_consistency.py

如果平台显示源数据正常，就可以直接重新合成派生数据。
