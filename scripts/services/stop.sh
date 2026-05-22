#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

stop_service worker "$WORKER_PID_FILE" \
  "python.*-m piern.worker"

stop_service frontend "$FRONTEND_PID_FILE" \
  "npm.*--prefix frontend run dev" \
  "sh -c vite.*--port[ =]*$FRONTEND_PORT" \
  "vite.*--port[ =]*$FRONTEND_PORT"

stop_service backend "$BACKEND_PID_FILE" \
  "uvicorn api_server:app.*--port[ =]*$BACKEND_PORT"
