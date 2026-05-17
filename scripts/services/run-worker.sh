#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$ROOT"
if env PATH="$SERVICE_PATH" "$PYTHON" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('piern.worker') else 1)"; then
  exec env PATH="$SERVICE_PATH" "$PYTHON" -m piern.worker
fi
echo "worker: piern.worker is not implemented yet; unit is a reserved placeholder."
exit 0
