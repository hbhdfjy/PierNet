#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

ENV_FILE=${PIERN_ENV_FILE:-"$ROOT/.env"}
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

RUN_DIR=${PIERN_SERVICE_RUN_DIR:-"$ROOT/.runlogs/services"}
BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
WORKER_PID_FILE="$RUN_DIR/worker.pid"
BACKEND_LOG="$RUN_DIR/backend.log"
FRONTEND_LOG="$RUN_DIR/frontend.log"
WORKER_LOG="$RUN_DIR/worker.log"
HOST=${PIERN_HOST:-${PIERN_SERVICE_HOST:-0.0.0.0}}
BACKEND_PORT=${PIERN_BACKEND_PORT:-8000}
FRONTEND_PORT=${PIERN_FRONTEND_PORT:-5173}
CONDA_ENV=${PIERN_CONDA_ENV:-"$HOME/.conda/envs/piern"}
PYTHON=${PIERN_PYTHON:-"$CONDA_ENV/bin/python"}
NPM=${PIERN_NPM:-npm}
NODE_BIN=${PIERN_NODE_BIN:-}
NODE_BIN_DIR=${PIERN_NODE_BIN_DIR:-}
[[ -n "$NODE_BIN" && -z "$NODE_BIN_DIR" ]] && NODE_BIN_DIR=$(dirname "$NODE_BIN")
SERVICE_PATH="${NODE_BIN_DIR:+$NODE_BIN_DIR:}$CONDA_ENV/bin:$PATH"
SERVICE_USER=${PIERN_SERVICE_USER:-$(id -un)}

mkdir -p "$RUN_DIR"

pid_alive() {
  local pid=${1:-}
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" >/dev/null 2>&1
}

read_pid() {
  local file=$1
  [[ -f "$file" ]] && tr -d "[:space:]" < "$file" || true
}

write_pid() {
  local file=$1
  local pid=$2
  mkdir -p "$(dirname "$file")"
  printf '%s\n' "$pid" > "$file"
}

first_matching_pid() {
  local pattern=$1
  local pid
  while IFS= read -r pid; do
    if pid_alive "$pid"; then
      printf '%s\n' "$pid"
      return 0
    fi
  done < <(pgrep -u "$SERVICE_USER" -f "$pattern" 2>/dev/null || true)
  return 1
}

find_backend_pid() {
  first_matching_pid "uvicorn api_server:app.*--port[ =]*$BACKEND_PORT" || true
}

find_frontend_pid() {
  first_matching_pid "vite.*--port[ =]*$FRONTEND_PORT" || true
}

find_worker_pid() {
  first_matching_pid "python.*-m piern.worker" || true
}

worker_should_start() {
  local setting=${PIERN_SERVICE_WORKER:-auto}
  case "${setting,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
  esac
  local synth=${PIERN_WORKER_QUEUE_SYNTH:-0}
  local training=${PIERN_WORKER_QUEUE_TRAINING:-0}
  case "${synth,,}" in
    1|true|yes|on) return 0 ;;
  esac
  case "${training,,}" in
    1|true|yes|on) return 0 ;;
  esac
  return 1
}

service_pid() {
  local file=$1
  local fallback_fn=${2:-}
  local pid
  pid=$(read_pid "$file")
  if pid_alive "$pid"; then
    printf '%s\n' "$pid"
    return 0
  fi
  if [[ -n "$fallback_fn" ]]; then
    pid=$($fallback_fn || true)
    if pid_alive "$pid"; then
      write_pid "$file" "$pid"
      printf '%s\n' "$pid"
      return 0
    fi
  fi
  rm -f "$file"
  return 1
}

service_alive() {
  service_pid "$@" >/dev/null 2>&1
}

wait_http() {
  local url=$1
  local timeout=${2:-20}
  local start
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) - start >= timeout )); then
      return 1
    fi
    sleep 1
  done
}

collect_matching_pids() {
  local pid
  declare -A seen=()
  for pattern in "$@"; do
    while IFS= read -r pid; do
      if [[ -n "$pid" && -z "${seen[$pid]:-}" ]]; then
        seen[$pid]=1
        printf '%s\n' "$pid"
      fi
    done < <(pgrep -u "$SERVICE_USER" -f "$pattern" 2>/dev/null || true)
  done
}

stop_pid() {
  local pid=$1
  pid_alive "$pid" || return 0
  kill "$pid" >/dev/null 2>&1 || true
}

force_stop_pid() {
  local pid=$1
  pid_alive "$pid" || return 0
  kill -9 "$pid" >/dev/null 2>&1 || true
}

stop_service() {
  local name=$1
  local file=$2
  shift 2
  local pid
  local pids=()
  declare -A seen=()
  pid=$(read_pid "$file")
  if [[ -n "$pid" && -z "${seen[$pid]:-}" ]]; then
    seen[$pid]=1
    pids+=("$pid")
  fi
  while IFS= read -r pid; do
    if [[ -n "$pid" && -z "${seen[$pid]:-}" ]]; then
      seen[$pid]=1
      pids+=("$pid")
    fi
  done < <(collect_matching_pids "$@")

  if [[ ${#pids[@]} -eq 0 ]]; then
    rm -f "$file"
    echo "$name: not running"
    return 0
  fi

  echo "$name: stopping ${pids[*]}"
  for pid in "${pids[@]}"; do
    stop_pid "$pid"
  done
  for _ in $(seq 1 15); do
    local any_alive=0
    for pid in "${pids[@]}"; do
      if pid_alive "$pid"; then
        any_alive=1
      fi
    done
    if [[ "$any_alive" == "0" ]]; then
      rm -f "$file"
      echo "$name: stopped"
      return 0
    fi
    sleep 1
  done

  echo "$name: force killing remaining processes"
  for pid in "${pids[@]}"; do
    force_stop_pid "$pid"
  done
  rm -f "$file"
}
