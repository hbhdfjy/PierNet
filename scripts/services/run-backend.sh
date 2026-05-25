#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$ROOT"
echo "backend: validating runtime configuration"
env PATH="$SERVICE_PATH" "$PYTHON" -m PierNet.shared.runtime.config
echo "backend: foreground on $HOST:$BACKEND_PORT"
exec env PATH="$SERVICE_PATH" "$PYTHON" -m uvicorn api_server:app --host "$HOST" --port "$BACKEND_PORT"
