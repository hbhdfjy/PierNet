from __future__ import annotations

import os
import signal
from pathlib import Path


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        fields = stat_path.read_text(encoding="utf-8", errors="replace").split()
    except OSError:
        return True
    return len(fields) < 3 or fields[2] != "Z"


def safe_kill_process_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
