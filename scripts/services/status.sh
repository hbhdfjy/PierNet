#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATA_ROOT=${PierNet_DATA_ROOT:-"$ROOT/data"}
STATUS_TAIL_LINES=${PierNet_STATUS_TAIL_LINES:-20}

section() {
  printf '\n== %s ==\n' "$1"
}

indent() {
  sed "s/^/  /"
}

print_status() {
  local name=$1
  local file=$2
  local fallback_fn=$3
  local pid
  if pid=$(service_pid "$file" "$fallback_fn"); then
    echo "$name: running pid=$pid"
    ps -p "$pid" -o pid=,ppid=,etime=,%cpu=,%mem=,rss=,cmd= 2>/dev/null | indent || true
  else
    echo "$name: stopped"
  fi
}

pretty_json() {
  local body=$1
  if [[ -x "$PYTHON" ]]; then
    printf '%s\n' "$body" | "$PYTHON" -m json.tool 2>/dev/null | indent || printf '%s\n' "$body" | indent
  else
    printf '%s\n' "$body" | indent
  fi
}

probe_api() {
  local label=$1
  local url=$2
  local body
  if body=$(curl -fsS --max-time 3 "$url" 2>&1); then
    echo "$label: ok $url"
    pretty_json "$body"
  else
    echo "$label: failed $url"
    printf '%s\n' "$body" | indent
  fi
}

probe_head() {
  local label=$1
  local url=$2
  local head
  if head=$(curl -fsS --max-time 3 -I "$url" 2>&1); then
    echo "$label: ok $url"
    printf '%s\n' "$head" | sed -n "1,6p" | indent
  else
    echo "$label: failed $url"
    printf '%s\n' "$head" | indent
  fi
}

print_disk() {
  df -h "$ROOT" "$DATA_ROOT" 2>/dev/null | awk "NR == 1 || !seen[\$1 \" \" \$6]++" | indent || true
}

print_gpu_cli() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi: unavailable"
    return 0
  fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=, read -r idx name used total util; do
        printf 'GPU %s: %s memory=%s/%s MiB util=%s%%\n' \
          "${idx// /}" "${name# }" "${used// /}" "${total// /}" "${util// /}"
      done
}

tail_log() {
  local name=$1
  local file=$2
  echo "$name: $file"
  if [[ -f "$file" ]]; then
    tail -n "$STATUS_TAIL_LINES" "$file" | indent
  else
    echo "  missing"
  fi
}

section "processes"
print_status backend "$BACKEND_PID_FILE" find_backend_pid
print_status "frontend dev" "$FRONTEND_PID_FILE" find_frontend_pid
print_status "studio dev" "$STUDIO_PID_FILE" find_studio_pid
print_status "new-synth dev" "$NEW_SYNTH_PID_FILE" find_new_synth_pid
if worker_should_start; then
  print_status worker "$WORKER_PID_FILE" find_worker_pid
else
  echo "worker: disabled"
fi

backend_base="http://127.0.0.1:$BACKEND_PORT/api"
backend_app_url="http://127.0.0.1:$BACKEND_PORT/"
frontend_url="http://127.0.0.1:$FRONTEND_PORT/"
studio_url="http://127.0.0.1:$STUDIO_PORT/studio/"
new_synth_url="http://127.0.0.1:$NEW_SYNTH_PORT/new-synth/"

section "health"
probe_api "backend live" "$backend_base/health/live"
probe_api "backend ready" "$backend_base/health/ready"
probe_api "backend storage" "$backend_base/health/storage"
probe_api "backend gpu" "$backend_base/health/gpu"
if service_alive "$FRONTEND_PID_FILE" find_frontend_pid; then
  probe_head "frontend dev" "$frontend_url"
elif [[ -f "$ROOT/frontend/dist/index.html" ]]; then
  probe_head "frontend static" "$backend_app_url"
else
  probe_head "frontend" "$frontend_url"
fi
probe_head "studio dev" "$studio_url"
if service_alive "$NEW_SYNTH_PID_FILE" find_new_synth_pid; then
  probe_head "new-synth dev" "$new_synth_url"
elif [[ -f "$ROOT/frontend-new-synth/dist/index.html" ]]; then
  probe_head "new-synth static" "${backend_app_url}new-synth/"
else
  probe_head "new-synth" "$new_synth_url"
fi

section "disk"
print_disk

section "gpu cli"
print_gpu_cli | indent

section "recent logs"
tail_log backend "$BACKEND_LOG"
tail_log "frontend dev" "$FRONTEND_LOG"
tail_log "studio dev" "$STUDIO_LOG"
tail_log "new-synth dev" "$NEW_SYNTH_LOG"
tail_log worker "$WORKER_LOG"
