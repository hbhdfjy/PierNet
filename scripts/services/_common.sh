#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

ENV_FILE=${PierNet_ENV_FILE:-"$ROOT/.env"}
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

RUN_DIR=${PierNet_SERVICE_RUN_DIR:-"$ROOT/.runlogs/services"}
BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
WORKER_PID_FILE="$RUN_DIR/worker.pid"
BACKEND_LOG="$RUN_DIR/backend.log"
FRONTEND_LOG="$RUN_DIR/frontend.log"
WORKER_LOG="$RUN_DIR/worker.log"
HOST=${PierNet_HOST:-${PierNet_SERVICE_HOST:-0.0.0.0}}
BACKEND_PORT=${PierNet_BACKEND_PORT:-8000}
FRONTEND_PORT=${PierNet_FRONTEND_PORT:-3000}
DEFAULT_CONDA_ENV="$HOME/.conda/envs/PierNet"
if [[ -x "$ROOT/.conda/env/bin/python" ]]; then
  DEFAULT_CONDA_ENV="$ROOT/.conda/env"
fi
CONDA_ENV=${PierNet_CONDA_ENV:-"$DEFAULT_CONDA_ENV"}
PYTHON=${PierNet_PYTHON:-"$CONDA_ENV/bin/python"}
NPM=${PierNet_NPM:-npm}
DEFAULT_NODE_BIN=""
if [[ -x "$ROOT/.node/current/bin/node" ]]; then
  DEFAULT_NODE_BIN="$ROOT/.node/current/bin/node"
elif compgen -G "$ROOT/.node/node-v*/bin/node" >/dev/null; then
  for candidate in "$ROOT"/.node/node-v*/bin/node; do
    [[ -x "$candidate" ]] && DEFAULT_NODE_BIN="$candidate" && break
  done
fi
NODE_BIN=${PierNet_NODE_BIN:-${PierNet_NODE:-$DEFAULT_NODE_BIN}}
NODE_BIN_DIR=${PierNet_NODE_BIN_DIR:-}
[[ -n "$NODE_BIN" && -z "$NODE_BIN_DIR" ]] && NODE_BIN_DIR=$(dirname "$NODE_BIN")
SERVICE_PATH="${NODE_BIN_DIR:+$NODE_BIN_DIR:}$CONDA_ENV/bin:$PATH"
SERVICE_USER=${PierNet_SERVICE_USER:-$(id -un)}

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

start_detached() {
  local __pid_var=$1
  local log_file=$2
  shift 2
  (
    # When start.sh is called by the watchdog, fd 9 holds watchdog.lock.
    # Detached services must not inherit it, otherwise the watchdog exits while
    # backend/frontend/worker keep the lock forever and future watchdog starts fail.
    exec 9>&- 2>/dev/null || true
    if command -v setsid >/dev/null 2>&1; then
      exec nohup setsid "$@"
    fi
    exec nohup "$@"
  ) > "$log_file" 2>&1 < /dev/null &
  printf -v "$__pid_var" "%s" "$!"
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
  first_matching_pid "python.*-m PierNet.worker" || true
}

worker_should_start() {
  local setting=${PierNet_SERVICE_WORKER:-auto}
  case "${setting,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
  esac
  local synth=${PierNet_WORKER_QUEUE_SYNTH:-1}
  local training=${PierNet_WORKER_QUEUE_TRAINING:-1}
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
