#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$ROOT"
echo "studio: foreground on $HOST:$STUDIO_PORT"
exec env PATH="$SERVICE_PATH" "$NPM" --prefix frontend-studio run dev -- --host "$HOST" --port "$STUDIO_PORT" --strictPort
