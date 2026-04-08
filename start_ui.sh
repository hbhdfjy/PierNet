#!/bin/bash
# PiERN Stage 2 UI 一键启动脚本
# 用法：
#   ./start_ui.sh          # 生产模式（生成任务不会因代码改动中断）
#   ./start_ui.sh --dev    # 开发模式（--reload，代码改动自动重载，但会中断正在运行的任务）

set -e
cd "$(dirname "$0")"

DEV_MODE=false
if [[ "$1" == "--dev" ]]; then
    DEV_MODE=true
fi

echo "启动 PiERN Stage 2 UI..."
echo ""

# 检查依赖
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "安装后端依赖..."
    pip install fastapi "uvicorn[standard]" -q
fi

# 启动后端（后台）
if [ "$DEV_MODE" = true ]; then
    echo "启动 FastAPI 后端 (port 8000, --reload 开发模式)..."
    echo "注意：开发模式下修改代码会重载后端，正在运行的生成任务将被中断"
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload &
else
    echo "启动 FastAPI 后端 (port 8000)..."
    uvicorn api_server:app --host 0.0.0.0 --port 8000 &
fi
BACKEND_PID=$!
echo "   PID: $BACKEND_PID"

# 等待后端就绪
sleep 1.5

# 启动前端
echo ""
echo "启动前端开发服务器 (port 5173)..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "服务已启动："
echo "   前端：http://localhost:5173"
echo "   后端 API：http://localhost:8000"
echo "   API 文档：http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待 Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '已停止'" INT TERM
wait
