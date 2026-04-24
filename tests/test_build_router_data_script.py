from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from piern.training.router.data import DEFAULT_QWEN_EMBEDDING_MODEL


def test_build_router_data_script_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "text2comp"
    output_dir = tmp_path / "router"
    source_path = data_dir / "coastal_seawater.jsonl"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "input": "describe the aquifer",
                "metadata": {
                    "simulator": "modflow",
                    "scenario": "coastal_seawater",
                    "language": "en",
                    "target_template": "groundwater response {output_0}",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "router" / "build_router_data.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--scenarios",
            "coastal_seawater",
            "--seed",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    scenario_path = output_dir / "by_scenario" / "coastal_seawater.jsonl"
    meta_path = output_dir / "by_scenario" / "coastal_seawater.meta.json"
    train_path = output_dir / "train.jsonl"

    assert scenario_path.exists()
    assert train_path.exists()

    lines = [line for line in scenario_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["chat_template"] == "qwen"
    assert payload["embedding_model"] == DEFAULT_QWEN_EMBEDDING_MODEL
    assert payload["embedding_tokenizer"] == DEFAULT_QWEN_EMBEDDING_MODEL

