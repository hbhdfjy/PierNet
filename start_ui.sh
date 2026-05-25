#!/bin/bash
# PierNet 前后端一键启动脚本
# 用法：
#   ./start_ui.sh          # 启动后端和前端开发服务
#   ./start_ui.sh --dev    # 后端启用 --reload，便于修改 Python 代码后自动重载

set -e
cd "$(dirname "$0")"

ENV_FILE=${PierNet_ENV_FILE:-"$PWD/.env"}
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

DEV_MODE=false
if [[ "${1:-}" == "--dev" ]]; then
    DEV_MODE=true
fi

CONDA_BASE="${PierNet_CONDA_BASE:-/usr/local/miniconda3}"
DEFAULT_CONDA_ENV="$HOME/.conda/envs/PierNet"
if [[ -x "$PWD/.conda/env/bin/python" ]]; then
    DEFAULT_CONDA_ENV="$PWD/.conda/env"
fi
CONDA_ENV_PATH="${PierNet_CONDA_ENV:-$DEFAULT_CONDA_ENV}"
if [[ -f "$CONDA_BASE/bin/activate" && -d "$CONDA_ENV_PATH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/bin/activate" "$CONDA_ENV_PATH"
elif [[ -x "$CONDA_ENV_PATH/bin/python" ]]; then
    export PATH="$CONDA_ENV_PATH/bin:$PATH"
fi

if ! command -v python >/dev/null 2>&1; then
    echo "未找到 python，请先激活正确环境"
    exit 1
fi

DEFAULT_NODE_BIN=""
if [[ -x "$PWD/.node/current/bin/node" ]]; then
    DEFAULT_NODE_BIN="$PWD/.node/current/bin/node"
elif compgen -G "$PWD/.node/node-v*/bin/node" >/dev/null; then
    for candidate in "$PWD"/.node/node-v*/bin/node; do
        [[ -x "$candidate" ]] && DEFAULT_NODE_BIN="$candidate" && break
    done
fi
NODE_BIN_CANDIDATE="${PierNet_NODE_BIN:-${PierNet_NODE:-$DEFAULT_NODE_BIN}}"
NODE_BIN_DIR="${PierNet_NODE_BIN_DIR:-}"
if [[ -n "$NODE_BIN_CANDIDATE" ]]; then
    if [[ ! -x "$NODE_BIN_CANDIDATE" ]]; then
        echo "配置的 Node 不存在或不可执行：$NODE_BIN_CANDIDATE"
        exit 1
    fi
    NODE_BIN_DIR="$(dirname "$NODE_BIN_CANDIDATE")"
fi
if [[ -n "$NODE_BIN_DIR" ]]; then
    export PATH="$NODE_BIN_DIR:$PATH"
fi

NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ -z "$NODE_BIN_CANDIDATE" && -z "$NODE_BIN_DIR" && -s "$NVM_DIR/nvm.sh" ]]; then
    # shellcheck disable=SC1090
    source "$NVM_DIR/nvm.sh"
    nvm use "${PierNet_NODE_VERSION:-default}" >/dev/null 2>&1 || \
        nvm use --lts >/dev/null 2>&1 || \
        nvm use 20 >/dev/null 2>&1 || true
fi
if ! command -v node >/dev/null 2>&1; then
    echo "未找到 node，请先安装 Node.js 20.19.0+"
    exit 1
fi
NPM_BIN="${PierNet_NPM:-npm}"
if ! command -v "$NPM_BIN" >/dev/null 2>&1; then
    echo "未找到 npm：$NPM_BIN"
    exit 1
fi

MIN_NODE_VERSION="20.19.0"
BACKEND_PORT="${PierNet_BACKEND_PORT:-8000}"
FRONTEND_PORT="${PierNet_FRONTEND_PORT:-3000}"
HOST="${PierNet_HOST:-0.0.0.0}"
if ! node -e "const min='$MIN_NODE_VERSION'.split('.').map(Number); const cur=process.versions.node.split('.').map(Number); process.exit(cur[0] > min[0] || (cur[0] === min[0] && (cur[1] > min[1] || (cur[1] === min[1] && cur[2] >= min[2]))) ? 0 : 1)" >/dev/null 2>&1; then
    echo "当前 Node.js 版本过低：$(node --version 2>&1)，请安装或切换到 Node.js ${MIN_NODE_VERSION}+"
    exit 1
fi

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo "启动 PierNet 前后端开发环境..."
echo ""

echo "Python: $(python --version 2>&1)"
echo "Node:   $(node --version 2>&1)"
echo ""

if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "检测到缺少 FastAPI 依赖，正在补装..."
    python -m pip install fastapi "uvicorn[standard]" -q
fi

if [ "$DEV_MODE" = true ]; then
    echo "启动 FastAPI 后端 (port $BACKEND_PORT, --reload 已开启)..."
    echo "后端代码改动后会自动重载，前端仍保持 Vite 热更新"
    python -m uvicorn api_server:app --host "$HOST" --port "$BACKEND_PORT" --reload &
else
    echo "启动 FastAPI 后端 (port $BACKEND_PORT)..."
    python -m uvicorn api_server:app --host "$HOST" --port "$BACKEND_PORT" &
fi
BACKEND_PID=$!
echo "   PID: $BACKEND_PID"

sleep 1.5

echo ""
echo "启动 Vite 前端 (port $FRONTEND_PORT)..."
cd frontend
"$NPM_BIN" run dev -- --host "$HOST" --port "$FRONTEND_PORT" --strictPort &
FRONTEND_PID=$!

echo ""
echo "访问地址："
echo "   前端：http://localhost:$FRONTEND_PORT"
echo "   后端 API：http://localhost:$BACKEND_PORT"
echo "   API 文档：http://localhost:$BACKEND_PORT/docs"
if [[ -n "$HOST_IP" ]]; then
    echo "   局域网前端：http://$HOST_IP:$FRONTEND_PORT"
    echo "   局域网后端：http://$HOST_IP:$BACKEND_PORT"
fi
echo ""
echo "按 Ctrl+C 可同时停止前后端"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '已停止'" INT TERM
wait
