"""Small .env loader used before runtime paths are resolved."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set

_LOADED_ENV_FILES: Set[Path] = set()


def _project_root() -> Path:
    configured = os.getenv("PierNet_ROOT", "").strip()
    if configured:
        return Path(os.path.expandvars(configured)).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _clean_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Optional[Path] = None) -> Optional[Path]:
    """Load KEY=VALUE pairs from .env without overriding existing environment."""

    if path is None:
        configured = os.getenv("PierNet_ENV_FILE", "").strip()
        path = Path(os.path.expandvars(configured)).expanduser() if configured else _project_root() / ".env"
    resolved = path.expanduser().resolve()
    if resolved in _LOADED_ENV_FILES:
        return resolved if resolved.exists() else None
    _LOADED_ENV_FILES.add(resolved)
    if not resolved.exists():
        return None

    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _clean_value(value)
    return resolved
