from __future__ import annotations

from pathlib import Path


def scenario_name_from_output(output_dir: str | None, output_file: str | None) -> str:
    output_stem = Path(output_file or "unknown").stem
    simulator_dir = Path(output_dir or "").name
    prefix = f"{simulator_dir}_" if simulator_dir else ""
    return output_stem[len(prefix):] if prefix and output_stem.startswith(prefix) else output_stem
