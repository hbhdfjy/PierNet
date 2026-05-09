#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$ROOT"

if backend_pid=$(service_pid "$BACKEND_PID_FILE" find_backend_pid); then
  echo "backend: already running pid=$backend_pid"
else
  rm -f "$BACKEND_PID_FILE"
  : > "$BACKEND_LOG"
  echo "backend: starting on $HOST:$BACKEND_PORT"
  nohup env PATH="$SERVICE_PATH" "$PYTHON" -m uvicorn api_server:app --host "$HOST" --port "$BACKEND_PORT" \
    > "$BACKEND_LOG" 2>&1 < /dev/null &
  write_pid "$BACKEND_PID_FILE" "$!"
fi

if frontend_pid=$(service_pid "$FRONTEND_PID_FILE" find_frontend_pid); then
  echo "frontend: already running pid=$frontend_pid"
else
  rm -f "$FRONTEND_PID_FILE"
  : > "$FRONTEND_LOG"
  echo "frontend: starting on $HOST:$FRONTEND_PORT"
  nohup env PATH="$SERVICE_PATH" npm --prefix frontend run dev -- --host "$HOST" --port "$FRONTEND_PORT" --strictPort \
    > "$FRONTEND_LOG" 2>&1 < /dev/null &
  write_pid "$FRONTEND_PID_FILE" "$!"
fi

sleep 1
if ! service_alive "$BACKEND_PID_FILE" find_backend_pid; then
  echo "backend process exited before health check"
  tail -n 80 "$BACKEND_LOG" || true
  exit 1
fi
if ! service_alive "$FRONTEND_PID_FILE" find_frontend_pid; then
  echo "frontend process exited before health check"
  tail -n 80 "$FRONTEND_LOG" || true
  exit 1
fi

backend_url="http://127.0.0.1:$BACKEND_PORT/api/training/gpus"
frontend_url="http://127.0.0.1:$FRONTEND_PORT/"
wait_http "$backend_url" 30 || { echo "backend health check failed: $backend_url"; tail -n 80 "$BACKEND_LOG" || true; exit 1; }
wait_http "$frontend_url" 30 || { echo "frontend health check failed: $frontend_url"; tail -n 80 "$FRONTEND_LOG" || true; exit 1; }

scripts/services/status.sh
