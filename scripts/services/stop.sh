#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

stop_service worker "$WORKER_PID_FILE" \
  "python.*-m PierNet.worker"

stop_service frontend "$FRONTEND_PID_FILE" \
  "npm.*--prefix frontend run dev" \
  "sh -c vite.*--port[ =]*$FRONTEND_PORT" \
  "vite.*--port[ =]*$FRONTEND_PORT"

stop_service studio "$STUDIO_PID_FILE" \
  "npm.*--prefix frontend-studio run dev" \
  "vite.*--port[ =]*$STUDIO_PORT"

stop_service new-synth "$NEW_SYNTH_PID_FILE" \
  "npm.*--prefix frontend-new-synth run dev" \
  "vite.*--port[ =]*$NEW_SYNTH_PORT"

stop_service backend "$BACKEND_PID_FILE" \
  "uvicorn api_server:app.*--port[ =]*$BACKEND_PORT"
