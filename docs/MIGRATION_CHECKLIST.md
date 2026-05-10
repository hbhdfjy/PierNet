# PiERN 迁移清单

本清单落实 `INDUSTRIALIZATION_PLAN.md` 的迁移边界：GitHub 只保存代码、配置、文档、语言模板和最原始科学计算数据；Stage 3 样本、Stage 4 Router 数据、索引、manifest、训练缓存和权重默认由平台重建。

## 1. 获取代码

```bash
git clone git@github.com:hbhdfjy/piern.git
cd piern
```

## 2. 创建环境

```bash
conda create -n piern python=3.11 -y
conda activate piern
pip install -r requirements.txt
pip install -e .
cd frontend
npm ci
cd ..
```

## 3. 配置本机变量

```bash
cp .env.example .env
```

必须按服务器实际情况修改：

- `PIERN_ROOT`
- `PIERN_DATA_ROOT`
- `PIERN_ARTIFACT_ROOT`
- `PIERN_CONDA_ENV`
- `PIERN_PYTHON`
- `PIERN_NODE_BIN`
- `PIERN_QWEN_EMBEDDING_MODEL`
- `PIERN_QWEN_EMBEDDING_TOKENIZER`
- `PIERN_MODFLOW_EXE`

## 4. 检查 Git 数据边界

Git 应只跟踪：

- `data/{simulator}/*.h5`
- `data/templates/*.jsonl`
- 代码、配置、文档

不应跟踪：

- `data/text2comp/`
- `data/text2comp_parquet/`
- `data/router/`
- `data/router_parquet/`
- `data/catalog.sqlite*`
- `data/.manifests/`
- `data/.indexes/`
- `artifacts/`
- `.runlogs/`

## 5. 启动服务

```bash
scripts/services/start.sh
scripts/services/status.sh
```

默认端口：

- FastAPI: `0.0.0.0:8000`
- Vite: `0.0.0.0:5173`

端口和 CORS 来源均可通过 `.env` 修改。

## 6. 重建派生数据

在平台中按顺序执行：

1. 检查 Stage 1 HDF5 和 Stage 2 模板是否完整。
2. 进入样本填充页面，生成 Stage 3 样本。
3. 进入 Router 数据构建页面，生成 Stage 4 Router 数据。
4. 进入训练平台，选择 Router 数据并启动训练。

## 7. 验收

```bash
python scripts/ci/check_consistency.py
python -m pytest
cd frontend && npm run build
scripts/services/status.sh
```

以上命令通过后，迁移完成。
