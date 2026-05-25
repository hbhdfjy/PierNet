from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

HDF5_SUFFIXES = (".h5", ".hdf5")


def is_hdf5_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in HDF5_SUFFIXES


def hdf5_scenario_from_path(path: str | Path, simulator: str | None = None) -> str:
    scenario = Path(path).stem
    if simulator:
        prefix = f"{simulator}_"
        if scenario.startswith(prefix):
            return scenario[len(prefix):]
    return scenario


def iter_hdf5_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and is_hdf5_path(path))


def iter_hdf5_files_in_child_dirs(directory: Path, *, skip_dirs: Iterable[str] = ()) -> list[Path]:
    if not directory.exists():
        return []
    skipped = set(skip_dirs)
    files: list[Path] = []
    for child in sorted(directory.iterdir()):
        if child.is_dir() and child.name not in skipped:
            files.extend(iter_hdf5_files(child))
    return sorted(files)
