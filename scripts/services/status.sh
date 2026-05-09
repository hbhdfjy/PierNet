#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

print_status() {
  local name=$1
  local file=$2
  local fallback_fn=$3
  local pid
  if pid=$(service_pid "$file" "$fallback_fn"); then
    echo "$name: running pid=$pid"
  else
    echo "$name: stopped"
  fi
}

print_status backend "$BACKEND_PID_FILE" find_backend_pid
print_status frontend "$FRONTEND_PID_FILE" find_frontend_pid

backend_url="http://127.0.0.1:$BACKEND_PORT/api/training/gpus"
frontend_url="http://127.0.0.1:$FRONTEND_PORT/"
if curl -fsS --max-time 2 "$backend_url" >/dev/null 2>&1; then
  echo "backend health: ok $backend_url"
else
  echo "backend health: failed $backend_url"
fi
if curl -fsS --max-time 2 -I "$frontend_url" >/dev/null 2>&1; then
  echo "frontend health: ok $frontend_url"
else
  echo "frontend health: failed $frontend_url"
fi
