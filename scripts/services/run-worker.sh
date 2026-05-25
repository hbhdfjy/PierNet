#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$ROOT"
exec env PATH="$SERVICE_PATH" "$PYTHON" -m PierNet.worker
