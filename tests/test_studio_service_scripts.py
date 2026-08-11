from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_studio_has_independent_service_files_and_port() -> None:
    common = (ROOT / "scripts/services/_common.sh").read_text(encoding="utf-8")
    start = (ROOT / "scripts/services/start.sh").read_text(encoding="utf-8")
    stop = (ROOT / "scripts/services/stop.sh").read_text(encoding="utf-8")
    status = (ROOT / "scripts/services/status.sh").read_text(encoding="utf-8")

    assert 'STUDIO_PID_FILE="$RUN_DIR/studio.pid"' in common
    assert "STUDIO_PORT=${PierNet_STUDIO_PORT:-3001}" in common
    assert 'first_matching_pid "vite.*--port[ =]*$STUDIO_PORT"' in common
    assert '"$NPM" --prefix frontend-studio run dev' in start
    assert 'studio_url="http://127.0.0.1:$STUDIO_PORT/studio/"' in start
    assert 'stop_service studio "$STUDIO_PID_FILE"' in stop
    assert 'print_status "studio dev" "$STUDIO_PID_FILE" find_studio_pid' in status
    assert 'probe_head "studio dev" "$studio_url"' in status


def test_studio_has_independent_systemd_unit() -> None:
    runner = (ROOT / "scripts/services/run-studio.sh").read_text(encoding="utf-8")
    unit = (ROOT / "deploy/systemd/PierNet-studio.service").read_text(
        encoding="utf-8"
    )
    installer = (ROOT / "scripts/services/install-systemd.sh").read_text(
        encoding="utf-8"
    )
    production = (ROOT / "scripts/services/start-prod.sh").read_text(encoding="utf-8")

    assert '"$NPM" --prefix frontend-studio run dev' in runner
    assert "ExecStart=__ROOT__/scripts/services/run-studio.sh" in unit
    assert "Restart=on-failure" in unit
    assert "INSTALL_STUDIO=${PierNet_INSTALL_STUDIO:-1}" in installer
    assert "PierNet-studio.service" in installer
    assert '"$NPM" --prefix frontend-studio run build' in production
