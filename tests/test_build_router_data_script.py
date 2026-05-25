from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from piern.shared.storage import portable
from piern.training.router.data import DEFAULT_QWEN_EMBEDDING_MODEL


def test_build_router_data_script_auto_reads_mixed_parquet_and_jsonl_sources(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "data"
    jsonl_dir = data_root / "text2comp"
    parquet_dir = data_root / "text2comp_parquet"
    output_dir = tmp_path / "router"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    parquet_record = {
        "input": "describe modflow",
        "metadata": {
            "simulator": "modflow",
            "scenario": "parquet_case",
            "language": "en",
            "target_template": "groundwater response {output_0}",
        },
    }
    portable.write_records_partition(
        "text2comp",
        [parquet_record],
        simulator="modflow",
        scenario="parquet_case",
        output_root=parquet_dir,
        compression="none",
    )

    jsonl_record = {
        "input": "describe simpeg",
        "metadata": {
            "simulator": "simpeg",
            "scenario": "jsonl_case",
            "language": "en",
            "target_template": "resistivity response {output_0}",
        },
    }
    (jsonl_dir / "simpeg_jsonl_case.jsonl").write_text(
        json.dumps(jsonl_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "router" / "build_router_data.py"),
            "--data-dir",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--input-format",
            "auto",
            "--output-format",
            "parquet",
            "--scenarios",
            "modflow/parquet_case",
            "simpeg/jsonl_case",
            "--seed",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "[router-build] input_storage=mixed" in result.stdout
    assert "PROGRESS_INIT:modflow/parquet_case:2" in result.stdout
    assert "PROGRESS_INIT:simpeg/jsonl_case:2" in result.stdout
    assert (output_dir / "simulator=modflow" / "scenario=parquet_case" / "part-00000.parquet").exists()
    assert (output_dir / "simulator=simpeg" / "scenario=jsonl_case" / "part-00000.parquet").exists()


def test_build_router_data_script_filters_composite_scenario_selector(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "text2comp"
    output_dir = tmp_path / "router"
    data_dir.mkdir(parents=True, exist_ok=True)

    for simulator in ["gcam", "simpeg"]:
        (data_dir / f"{simulator}_shared.jsonl").write_text(
            json.dumps(
                {
                    "input": f"describe {simulator}",
                    "metadata": {
                        "simulator": simulator,
                        "scenario": "shared",
                        "language": "en",
                        "target_template": "response {output_0}",
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
            "gcam/shared",
            "--seed",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "PROGRESS_INIT:gcam/shared:2" in result.stdout
    assert "PROGRESS_INIT:simpeg/shared" not in result.stdout
    assert (output_dir / "simulator=gcam" / "scenario=shared" / "part-00000.parquet").exists()
    assert not (output_dir / "simulator=simpeg" / "scenario=shared").exists()


def test_build_router_data_script_rejects_duplicate_scenarios_for_jsonl_output(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "text2comp"
    output_dir = tmp_path / "router"
    data_dir.mkdir(parents=True, exist_ok=True)

    for simulator in ["gcam", "simpeg"]:
        (data_dir / f"{simulator}_shared.jsonl").write_text(
            json.dumps(
                {
                    "input": f"describe {simulator}",
                    "metadata": {
                        "simulator": simulator,
                        "scenario": "shared",
                        "language": "en",
                        "target_template": "response {output_0}",
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
            "--output-format",
            "jsonl",
            "--seed",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "cannot represent duplicate scenario names" in result.stdout
    assert not (output_dir / "by_scenario" / "shared.jsonl").exists()


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
