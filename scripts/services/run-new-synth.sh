#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$ROOT"
echo "new-synth: foreground on $HOST:$NEW_SYNTH_PORT"
exec env PATH="$SERVICE_PATH" "$NPM" --prefix frontend-new-synth run dev -- --host "$HOST" --port "$NEW_SYNTH_PORT" --strictPort
