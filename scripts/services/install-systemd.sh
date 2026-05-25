#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
UNIT_DIR=${PIERN_SYSTEMD_USER_DIR:-"$HOME/.config/systemd/user"}
DRY_RUN=0
ENABLE=0
START_NOW=0
INSTALL_WORKER=${PIERN_INSTALL_WORKER:-1}

usage() {
  cat <<USAGE
Usage: scripts/services/install-systemd.sh [--dry-run] [--enable] [--now] [--worker] [--no-worker]

Installs user-level systemd units for PiERN backend, frontend, and worker.
The worker consumes queued synthesis/training jobs and performs shared housekeeping; use --no-worker only for API/UI-only deployments.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --enable) ENABLE=1 ;;
    --now) ENABLE=1; START_NOW=1 ;;
    --worker) INSTALL_WORKER=1 ;;
    --no-worker) INSTALL_WORKER=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

units=(piern-backend.service piern-frontend.service)
if [[ "$INSTALL_WORKER" == "1" ]]; then
  units+=(piern-worker.service)
fi

render_unit() {
  local unit=$1
  sed "s#__ROOT__#$ROOT#g" "$ROOT/deploy/systemd/$unit"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "systemd user unit dir: $UNIT_DIR"
  for unit in "${units[@]}"; do
    echo "--- $unit ---"
    render_unit "$unit"
  done
  exit 0
fi

mkdir -p "$UNIT_DIR"
for unit in "${units[@]}"; do
  render_unit "$unit" > "$UNIT_DIR/$unit"
  echo "installed: $UNIT_DIR/$unit"
done

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found; units were installed but not loaded."
  exit 0
fi

systemctl --user daemon-reload
if [[ "$ENABLE" == "1" ]]; then
  systemctl --user enable "${units[@]}"
fi
if [[ "$START_NOW" == "1" ]]; then
  systemctl --user restart "${units[@]}"
fi
systemctl --user status "${units[@]}" --no-pager || true
