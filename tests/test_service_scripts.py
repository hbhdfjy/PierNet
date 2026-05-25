from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _worker_should_start(tmp_path: Path, **overrides: str) -> bool:
    env = os.environ.copy()
    for key in (
        "PierNet_ENV_FILE",
        "PierNet_SERVICE_WORKER",
        "PierNet_WORKER_QUEUE_SYNTH",
        "PierNet_WORKER_QUEUE_TRAINING",
    ):
        env.pop(key, None)
    env.update(
        {
            "PierNet_ENV_FILE": str(tmp_path / "missing.env"),
            "PierNet_SERVICE_RUN_DIR": str(tmp_path / "services"),
        }
    )
    env.update(overrides)
    result = subprocess.run(
        ["bash", "-c", "source scripts/services/_common.sh; worker_should_start"],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return result.returncode == 0


def test_worker_auto_starts_when_queue_env_is_unset(tmp_path):
    assert _worker_should_start(tmp_path)


def test_worker_auto_starts_when_synth_queue_enabled(tmp_path):
    assert _worker_should_start(
        tmp_path,
        PierNet_WORKER_QUEUE_SYNTH="1",
        PierNet_WORKER_QUEUE_TRAINING="0",
    )


def test_worker_auto_starts_when_training_queue_enabled(tmp_path):
    assert _worker_should_start(
        tmp_path,
        PierNet_WORKER_QUEUE_SYNTH="0",
        PierNet_WORKER_QUEUE_TRAINING="true",
    )


def test_worker_auto_stays_disabled_when_all_queues_disabled(tmp_path):
    assert not _worker_should_start(
        tmp_path,
        PierNet_WORKER_QUEUE_SYNTH="0",
        PierNet_WORKER_QUEUE_TRAINING="false",
    )


def test_worker_override_disables_auto_queue_start(tmp_path):
    assert not _worker_should_start(tmp_path, PierNet_SERVICE_WORKER="off")


def test_worker_override_enables_worker_when_queues_disabled(tmp_path):
    assert _worker_should_start(tmp_path, PierNet_SERVICE_WORKER="on")


def test_service_common_accepts_legacy_PierNet_node_env(tmp_path):
    node = tmp_path / "bin" / "node"
    node.parent.mkdir(parents=True)
    node.write_text("#!/bin/sh\n", encoding="utf-8")
    node.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PierNet_ENV_FILE": str(tmp_path / "missing.env"),
            "PierNet_CONDA_ENV": str(tmp_path / "conda"),
            "PierNet_NODE": str(node),
        }
    )
    env.pop("PierNet_NODE_BIN", None)
    env.pop("PierNet_NODE_BIN_DIR", None)

    result = subprocess.run(
        ["bash", "-c", "source scripts/services/_common.sh; printf '%s\n' \"$NODE_BIN_DIR\""],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(node.parent)


def test_service_common_prefers_repo_local_conda_env(tmp_path):
    fake_root = tmp_path / "repo"
    script_dir = fake_root / "scripts" / "services"
    script_dir.mkdir(parents=True)
    (fake_root / ".conda" / "env" / "bin").mkdir(parents=True)
    python_bin = fake_root / ".conda" / "env" / "bin" / "python"
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    python_bin.chmod(0o755)
    (script_dir / "_common.sh").write_text((ROOT / "scripts/services/_common.sh").read_text(encoding="utf-8"), encoding="utf-8")
    env = os.environ.copy()
    env.update({"PierNet_ENV_FILE": str(fake_root / "missing.env")})
    env.pop("PierNet_CONDA_ENV", None)
    env.pop("PierNet_PYTHON", None)

    result = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/services/_common.sh; printf '%s\\n%s\\n' \"$CONDA_ENV\" \"$PYTHON\"",
        ],
        cwd=fake_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        str(fake_root / ".conda" / "env"),
        str(python_bin),
    ]


def test_service_common_prefers_repo_local_node_when_unset(tmp_path):
    fake_root = tmp_path / "repo"
    script_dir = fake_root / "scripts" / "services"
    script_dir.mkdir(parents=True)
    node_bin_dir = fake_root / ".node" / "current" / "bin"
    node_bin_dir.mkdir(parents=True)
    node = node_bin_dir / "node"
    node.write_text("#!/bin/sh\n", encoding="utf-8")
    node.chmod(0o755)
    (script_dir / "_common.sh").write_text(
        (ROOT / "scripts/services/_common.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({"PierNet_ENV_FILE": str(fake_root / "missing.env")})
    for key in ("PierNet_NODE_BIN", "PierNet_NODE", "PierNet_NODE_BIN_DIR"):
        env.pop(key, None)

    result = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/services/_common.sh; printf '%s\\n%s\\n' \"$NODE_BIN\" \"$NODE_BIN_DIR\"",
        ],
        cwd=fake_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [str(node), str(node_bin_dir)]


def test_install_systemd_includes_worker_by_default_and_reports_all_units():
    install_script = (ROOT / "scripts/services/install-systemd.sh").read_text(encoding="utf-8")

    assert "INSTALL_WORKER=${PierNet_INSTALL_WORKER:-1}" in install_script
    assert "--no-worker) INSTALL_WORKER=0 ;;" in install_script
    assert 'systemctl --user enable "${units[@]}"' in install_script
    assert 'systemctl --user restart "${units[@]}"' in install_script
    assert 'systemctl --user status "${units[@]}" --no-pager' in install_script


def test_status_script_checks_backend_static_frontend_in_prod_mode():
    status_script = (ROOT / "scripts/services/status.sh").read_text(encoding="utf-8")

    assert 'backend_app_url="http://127.0.0.1:$BACKEND_PORT/"' in status_script
    assert 'probe_head "frontend static" "$backend_app_url"' in status_script
    assert '[[ -f "$ROOT/frontend/dist/index.html" ]]' in status_script


def test_compose_worker_uses_package_entrypoint():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert '["python", "-m", "PierNet.worker", "--interval", "5"]' in compose
    assert "PierNet.worker.runner" not in compose


def test_start_script_uses_backend_readiness_endpoint():
    start_script = (ROOT / "scripts/services/start.sh").read_text(encoding="utf-8")

    assert 'backend_url="http://127.0.0.1:$BACKEND_PORT/api/health/ready"' in start_script
    assert "/api/training/gpus" not in start_script


def test_start_ui_prefers_repo_local_conda_env():
    start_ui = (ROOT / "start_ui.sh").read_text(encoding="utf-8")

    assert 'DEFAULT_CONDA_ENV="$HOME/.conda/envs/PierNet"' in start_ui
    assert 'if [[ -x "$PWD/.conda/env/bin/python" ]]; then' in start_ui
    assert 'DEFAULT_CONDA_ENV="$PWD/.conda/env"' in start_ui
    assert 'CONDA_ENV_PATH="${PierNet_CONDA_ENV:-$DEFAULT_CONDA_ENV}"' in start_ui
    assert 'export PATH="$CONDA_ENV_PATH/bin:$PATH"' in start_ui
    assert 'DEFAULT_NODE_BIN="$PWD/.node/current/bin/node"' in start_ui
    assert 'NODE_BIN_CANDIDATE="${PierNet_NODE_BIN:-${PierNet_NODE:-$DEFAULT_NODE_BIN}}"' in start_ui
