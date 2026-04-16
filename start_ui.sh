#!/bin/bash
# PiERN Stage 2 UI ??????
# ???
#   ./start_ui.sh          # ???????????????????
#   ./start_ui.sh --dev    # ?????--reload??????????????????????

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
    echo "??? python?????????"
    exit 1
fi
if ! command -v node >/dev/null 2>&1; then
    echo "??? node?????????"
    exit 1
fi

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo "?? PiERN Stage 2 UI..."
echo ""

echo "Python: $(python --version 2>&1)"
echo "Node:   $(node --version 2>&1)"
echo ""

if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "??????..."
    python -m pip install fastapi "uvicorn[standard]" -q
fi

if [ "$DEV_MODE" = true ]; then
    echo "?? FastAPI ?? (port 8000, --reload ????)..."
    echo "???????????????????????????????"
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload &
else
    echo "?? FastAPI ?? (port 8000)..."
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 &
fi
BACKEND_PID=$!
echo "   PID: $BACKEND_PID"

sleep 1.5

echo ""
echo "????????? (port 5173)..."
cd frontend
npm run dev -- --host 0.0.0.0 --strictPort &
FRONTEND_PID=$!

echo ""
echo "??????"
echo "   ???http://localhost:5173"
echo "   ?? API?http://localhost:8000"
echo "   API ???http://localhost:8000/docs"
if [[ -n "$HOST_IP" ]]; then
    echo "   ????????http://$HOST_IP:5173"
    echo "   ????????http://$HOST_IP:8000"
fi
echo ""
echo "? Ctrl+C ??????"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '???'" INT TERM
wait
