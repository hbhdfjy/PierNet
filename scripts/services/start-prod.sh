#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$ROOT"

echo "prod: validating runtime configuration"
env PATH="$SERVICE_PATH" "$PYTHON" -m piern.shared.runtime.config

echo "prod: building frontend static bundle"
env PATH="$SERVICE_PATH" "$NPM" --prefix frontend run build

if backend_pid=$(service_pid "$BACKEND_PID_FILE" find_backend_pid); then
  echo "backend: already running pid=$backend_pid"
else
  rm -f "$BACKEND_PID_FILE"
  : > "$BACKEND_LOG"
  echo "backend: starting production API/static server on $HOST:$BACKEND_PORT"
  nohup env PATH="$SERVICE_PATH" "$PYTHON" -m uvicorn api_server:app --host "$HOST" --port "$BACKEND_PORT" \
    > "$BACKEND_LOG" 2>&1 < /dev/null &
  write_pid "$BACKEND_PID_FILE" "$!"
fi

backend_url="http://127.0.0.1:$BACKEND_PORT/api/health/ready"
wait_http "$backend_url" 30 || { echo "backend health check failed: $backend_url"; tail -n 80 "$BACKEND_LOG" || true; exit 1; }
echo "prod: serving app through backend on http://127.0.0.1:$BACKEND_PORT/"
