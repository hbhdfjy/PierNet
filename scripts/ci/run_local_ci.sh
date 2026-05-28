#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_BACKEND=1
RUN_FRONTEND=1
RUN_E2E=1

usage() {
  cat <<'EOF'
Usage: scripts/ci/run_local_ci.sh [--backend] [--frontend] [--no-e2e]

Runs the local equivalent of the GitHub CI quality gates. By default it runs
both backend and frontend gates, including Playwright smoke and visual checks.

Options:
  --backend   Run only backend checks.
  --frontend  Run only frontend checks.
  --no-e2e    Skip Playwright checks.
  -h, --help  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      RUN_BACKEND=1
      RUN_FRONTEND=0
      ;;
    --frontend)
      RUN_BACKEND=0
      RUN_FRONTEND=1
      ;;
    --no-e2e)
      RUN_E2E=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

resolve_python() {
  if [[ -n "${PIERN_PYTHON:-}" ]]; then
    if [[ -x "$PIERN_PYTHON" ]]; then
      printf '%s\n' "$PIERN_PYTHON"
      return
    fi
    echo "PIERN_PYTHON is set but not executable: $PIERN_PYTHON" >&2
    exit 2
  fi
  if [[ -x "$ROOT/.conda/env/bin/python" ]]; then
    printf '%s\n' "$ROOT/.conda/env/bin/python"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "python not found" >&2
  exit 2
}

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run_env() {
  printf '\n==> %s\n' "$*"
  env "$@"
}

PYTHON_BIN="$(resolve_python)"
CI_TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$CI_TMPDIR"

if [[ "$RUN_BACKEND" -eq 1 ]]; then
  run "$PYTHON_BIN" -m ruff check .
  run "$PYTHON_BIN" scripts/ci/check_consistency.py
  run "$PYTHON_BIN" scripts/ci/check_repo_hygiene.py
  run "$PYTHON_BIN" scripts/ci/check_migration_ready.py
  run "$PYTHON_BIN" scripts/ci/export_openapi.py "$CI_TMPDIR/PierNet-openapi-local.json"
  run npm --prefix frontend run openapi:check
  run "$PYTHON_BIN" -m pytest
fi

if [[ "$RUN_FRONTEND" -eq 1 ]]; then
  run npm --prefix frontend run typecheck
  run npm --prefix frontend run lint
  run npm --prefix frontend run format:check
  run npm --prefix frontend run test
  run npm --prefix frontend run build

  if [[ "$RUN_E2E" -eq 1 ]]; then
    (
      cd frontend
      if node -e 'const fs = require("fs"); const { chromium } = require("playwright"); process.exit(fs.existsSync(chromium.executablePath()) ? 0 : 1)' >/dev/null 2>&1; then
        echo
        echo "==> playwright chromium already installed"
      else
        run npx playwright install chromium
      fi
    )
    run_env PierNet_E2E_START_SERVER=1 npm --prefix frontend run e2e:smoke
    run_env PierNet_E2E_START_SERVER=1 npm --prefix frontend run e2e:visual
  fi
fi

printf '\nLocal CI passed.\n'
