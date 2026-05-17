#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$ROOT"
echo "frontend: foreground on $HOST:$FRONTEND_PORT"
exec env PATH="$SERVICE_PATH" "$NPM" --prefix frontend run dev -- --host "$HOST" --port "$FRONTEND_PORT" --strictPort
