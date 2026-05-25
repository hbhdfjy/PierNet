"""Stable path identifiers for data artifacts stored inside or outside the project tree."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def source_relative_path(source_path: Path, *, roots: Iterable[Path]) -> Path:
    """Return a stable path relative to a known root, with a hashed fallback for external files."""
    resolved = source_path.resolve(strict=False)
    for root in roots:
        try:
            return resolved.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    return Path("__external__") / digest / resolved.name
