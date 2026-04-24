#!/bin/bash
# PiERN 前后端一键启动脚本
# 用法：
#   ./start_ui.sh          # 启动后端和前端开发服务
#   ./start_ui.sh --dev    # 后端启用 --reload，便于修改 Python 代码后自动重载

set -e
cd "$(dirname "$0")"

DEV_MODE=false
if [[ "$1" == "--dev" ]]; then
    DEV_MODE=true
fi

CONDA_BASE="${PIERN_CONDA_BASE:-/home/fjy/miniconda3}"
CONDA_ENV_PATH="${PIERN_CONDA_ENV:-/home/fjy/miniconda3/envs/piern-project}"
if [[ -f "$CONDA_BASE/bin/activate" && -d "$CONDA_ENV_PATH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/bin/activate" "$CONDA_ENV_PATH"
fi

if ! command -v python >/dev/null 2>&1; then
    echo "未找到 python，请先激活正确环境"
    exit 1
fi

if ! command -v node >/dev/null 2>&1; then
    NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    if [[ -s "$NVM_DIR/nvm.sh" ]]; then
        # shellcheck disable=SC1090
        source "$NVM_DIR/nvm.sh"
        nvm use 20 >/dev/null 2>&1 || nvm use --lts >/dev/null 2>&1 || true
    fi
fi
if ! command -v node >/dev/null 2>&1; then
    echo "未找到 node，请先安装 Node.js 18+"
    exit 1
fi

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo "启动 PiERN 前后端开发环境..."
echo ""

echo "Python: $(python --version 2>&1)"
echo "Node:   $(node --version 2>&1)"
echo ""

if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "检测到缺少 FastAPI 依赖，正在补装..."
    python -m pip install fastapi "uvicorn[standard]" -q
fi

if [ "$DEV_MODE" = true ]; then
    echo "启动 FastAPI 后端 (port 8000, --reload 已开启)..."
    echo "后端代码改动后会自动重载，前端仍保持 Vite 热更新"
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload &
else
    echo "启动 FastAPI 后端 (port 8000)..."
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 &
fi
BACKEND_PID=$!
echo "   PID: $BACKEND_PID"

sleep 1.5

echo ""
echo "启动 Vite 前端 (port 5173)..."
cd frontend
npm run dev -- --host 0.0.0.0 --strictPort &
FRONTEND_PID=$!

echo ""
echo "访问地址："
echo "   前端：http://localhost:5173"
echo "   后端 API：http://localhost:8000"
echo "   API 文档：http://localhost:8000/docs"
if [[ -n "$HOST_IP" ]]; then
    echo "   局域网前端：http://$HOST_IP:5173"
    echo "   局域网后端：http://$HOST_IP:8000"
fi
echo ""
echo "按 Ctrl+C 可同时停止前后端"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '已停止'" INT TERM
wait
