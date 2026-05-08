from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

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

    scenario_dir = output_dir / "simulator=modflow" / "scenario=coastal_seawater"
    parquet_path = scenario_dir / "part-00000.parquet"
    meta_path = scenario_dir / "_manifest.json"

    assert parquet_path.exists()
    assert pq.ParquetFile(parquet_path).metadata.num_rows == 2

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["chat_template"] == "qwen"
    assert payload["embedding_model"] == DEFAULT_QWEN_EMBEDDING_MODEL
    assert payload["embedding_tokenizer"] == DEFAULT_QWEN_EMBEDDING_MODEL
    assert payload["row_count"] == 2
    assert payload["by_label"] == {"0": 1, "1": 1}

