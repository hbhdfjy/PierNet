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
  start_detached backend_pid "$BACKEND_LOG" \
    env PATH="$SERVICE_PATH" "$PYTHON" -m uvicorn api_server:app --host "$HOST" --port "$BACKEND_PORT"
  write_pid "$BACKEND_PID_FILE" "$backend_pid"
fi

if frontend_pid=$(service_pid "$FRONTEND_PID_FILE" find_frontend_pid); then
  echo "frontend: already running pid=$frontend_pid"
else
  rm -f "$FRONTEND_PID_FILE"
  : > "$FRONTEND_LOG"
  echo "frontend: starting on $HOST:$FRONTEND_PORT"
  start_detached frontend_pid "$FRONTEND_LOG" \
    env PATH="$SERVICE_PATH" "$NPM" --prefix frontend run dev -- --host "$HOST" --port "$FRONTEND_PORT" --strictPort
  write_pid "$FRONTEND_PID_FILE" "$frontend_pid"
fi

if studio_pid=$(service_pid "$STUDIO_PID_FILE" find_studio_pid); then
  echo "studio: already running pid=$studio_pid"
else
  rm -f "$STUDIO_PID_FILE"
  : > "$STUDIO_LOG"
  echo "studio: starting on $HOST:$STUDIO_PORT"
  start_detached studio_pid "$STUDIO_LOG" \
    env PATH="$SERVICE_PATH" "$NPM" --prefix frontend-studio run dev
  write_pid "$STUDIO_PID_FILE" "$studio_pid"
fi

if new_synth_pid=$(service_pid "$NEW_SYNTH_PID_FILE" find_new_synth_pid); then
  echo "new-synth: already running pid=$new_synth_pid"
else
  rm -f "$NEW_SYNTH_PID_FILE"
  : > "$NEW_SYNTH_LOG"
  echo "new-synth: starting on $HOST:$NEW_SYNTH_PORT"
  start_detached new_synth_pid "$NEW_SYNTH_LOG" \
    env PATH="$SERVICE_PATH" "$NPM" --prefix frontend-new-synth run dev -- --host "$HOST" --port "$NEW_SYNTH_PORT" --strictPort
  write_pid "$NEW_SYNTH_PID_FILE" "$new_synth_pid"
fi

if worker_should_start; then
  if worker_pid=$(service_pid "$WORKER_PID_FILE" find_worker_pid); then
    echo "worker: already running pid=$worker_pid"
  else
    rm -f "$WORKER_PID_FILE"
    : > "$WORKER_LOG"
    echo "worker: starting"
    start_detached worker_pid "$WORKER_LOG" \
      env PATH="$SERVICE_PATH" "$PYTHON" -m PierNet.worker --interval 2
    write_pid "$WORKER_PID_FILE" "$worker_pid"
  fi
else
  echo "worker: disabled"
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
if ! service_alive "$STUDIO_PID_FILE" find_studio_pid; then
  echo "studio process exited before health check"
  tail -n 80 "$STUDIO_LOG" || true
  exit 1
fi
if ! service_alive "$NEW_SYNTH_PID_FILE" find_new_synth_pid; then
  echo "new-synth process exited before health check"
  tail -n 80 "$NEW_SYNTH_LOG" || true
  exit 1
fi
if worker_should_start && ! service_alive "$WORKER_PID_FILE" find_worker_pid; then
  echo "worker process exited before health check"
  tail -n 80 "$WORKER_LOG" || true
  exit 1
fi

backend_url="http://127.0.0.1:$BACKEND_PORT/api/health/ready"
frontend_url="http://127.0.0.1:$FRONTEND_PORT/"
studio_url="http://127.0.0.1:$STUDIO_PORT/studio/"
new_synth_url="http://127.0.0.1:$NEW_SYNTH_PORT/new-synth/"
wait_http "$backend_url" 30 || { echo "backend health check failed: $backend_url"; tail -n 80 "$BACKEND_LOG" || true; exit 1; }
wait_http "$frontend_url" 30 || { echo "frontend health check failed: $frontend_url"; tail -n 80 "$FRONTEND_LOG" || true; exit 1; }
wait_http "$studio_url" 30 || { echo "studio health check failed: $studio_url"; tail -n 80 "$STUDIO_LOG" || true; exit 1; }
wait_http "$new_synth_url" 30 || { echo "new-synth health check failed: $new_synth_url"; tail -n 80 "$NEW_SYNTH_LOG" || true; exit 1; }

scripts/services/status.sh
