"""Worker-safe executor for Token Router dataset builds."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from piern.shared.runtime.paths import DATA_ROOT, PROJECT_ROOT
from piern.shared.storage import portable
from piern.synth.services import job_manager, manifest_store
from piern.synth.services.job_manager import JobRecord, publish

DEFAULT_LOCAL_QWEN_DIR = str(Path.home() / "Qwen" / "Qwen2.5-0.5B-Instruct")
DEFAULT_QWEN_EMBEDDING_MODEL = os.getenv("PIERN_QWEN_EMBEDDING_MODEL", DEFAULT_LOCAL_QWEN_DIR)
DEFAULT_QWEN_EMBEDDING_TOKENIZER = os.getenv("PIERN_QWEN_EMBEDDING_TOKENIZER", DEFAULT_QWEN_EMBEDDING_MODEL)


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _has_jsonl_stage3_sources() -> bool:
    jsonl_root = DATA_ROOT / "text2comp"
    return jsonl_root.exists() and any(
        path.name != "all_training_data.jsonl" for path in jsonl_root.glob("*.jsonl")
    )


def _router_input_source() -> tuple[Path, str]:
    has_parquet = portable.has_partitions("text2comp")
    has_jsonl = _has_jsonl_stage3_sources()
    if has_parquet and has_jsonl:
        return DATA_ROOT, "auto"
    if has_parquet:
        return portable.TEXT2COMP_PARQUET_DIR, "parquet"
    return DATA_ROOT / "text2comp", "jsonl"


def _router_build_command(seed: int, neg_ratio: int, max_workers: int, scenario_list: list[str]) -> list[str]:
    script = PROJECT_ROOT / "scripts" / "router" / "build_router_data.py"
    input_dir, input_format = _router_input_source()
    cmd = [
        sys.executable,
        str(script),
        "--data-dir",
        str(input_dir),
        "--output-dir",
        str(portable.ROUTER_PARQUET_DIR),
        "--input-format",
        input_format,
        "--output-format",
        "parquet",
        "--seed",
        str(seed),
        "--neg-ratio",
        str(neg_ratio),
        "--chat-template",
        "qwen",
        "--batch-size",
        "32768",
        "--max-workers",
        str(max_workers),
    ]
    if scenario_list:
        cmd += ["--scenarios", *scenario_list]
    return cmd


def run_router_build_job(record: JobRecord, payload: dict[str, Any]) -> None:
    seed = int(payload.get("seed", 42))
    neg_ratio = int(payload.get("neg_ratio", 1))
    max_workers = max(1, int(payload.get("max_workers") or os.getenv("PIERN_ROUTER_BUILD_WORKERS", "8")))
    scenario_list = [str(item) for item in payload.get("scenarios", []) if str(item)]
    proc: subprocess.Popen[str] | None = None
    try:
        sc_desc = f"scenarios: {', '.join(scenario_list)}" if scenario_list else "all scenarios"
        publish(
            record,
            {
                "type": "log",
                "line": f"[Stage 4] start building Token Router data: {sc_desc}, chat_template=qwen",
                "ts": time.time(),
            },
        )
        publish(
            record,
            {
                "type": "log",
                "line": (
                    "[Stage 4] embedding backbone: "
                    f"model={DEFAULT_QWEN_EMBEDDING_MODEL} "
                    f"tokenizer={DEFAULT_QWEN_EMBEDDING_TOKENIZER}"
                ),
                "ts": time.time(),
            },
        )

        cmd = _router_build_command(seed, neg_ratio, max_workers, scenario_list)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
        record.proc = proc
        record.proc_uses_process_group = True

        if proc.stdout is None:
            raise RuntimeError("router build subprocess did not provide stdout")

        scenario_totals: dict[str, int] = {}

        def _handle_line(line: str) -> None:
            line = line.rstrip()
            if not line:
                return
            if line.startswith("PROGRESS_INIT:"):
                parts = line.split(":", 2)
                if len(parts) == 3:
                    sc_name, total_str = parts[1], parts[2]
                    try:
                        total = int(total_str)
                    except ValueError:
                        total = 0
                    scenario_totals[sc_name] = total
                    record.scenario_totals[sc_name] = total
                    publish(record, {"type": "init", "scenario_totals": dict(record.scenario_totals), "ts": time.time()})
                    publish(record, {"type": "log", "line": f"[init] {sc_name} expected {total} rows", "ts": time.time()})
                return
            if line.startswith("PROGRESS_UPDATE:"):
                parts = line.split(":", 3)
                if len(parts) == 4:
                    sc_name = parts[1]
                    try:
                        done = int(parts[2])
                        total = int(parts[3])
                    except ValueError:
                        done, total = 0, 0
                    publish(
                        record,
                        {
                            "type": "log",
                            "line": f"  {sc_name}: {done}/{total}",
                            "ts": time.time(),
                            "progress": {"scenario": sc_name, "done": done, "total": total},
                        },
                    )
                return
            if line.startswith("PROGRESS_DONE:"):
                parts = line.split(":", 3)
                if len(parts) >= 3:
                    sc_name = parts[1]
                    try:
                        done = int(parts[2])
                        total = int(parts[3]) if len(parts) == 4 else scenario_totals.get(sc_name, done)
                    except ValueError:
                        done, total = 0, 0
                    publish(
                        record,
                        {
                            "type": "log",
                            "line": f"  {sc_name}: {done}/{total}",
                            "ts": time.time(),
                            "progress": {"scenario": sc_name, "done": done, "total": total},
                        },
                    )
                return
            publish(record, {"type": "log", "line": line, "ts": time.time()})

        while True:
            if job_manager.should_stop(record):
                _kill_process_group(proc)
                record.status = "terminated"
                publish(record, {"type": "terminated", "ts": time.time(), "message": "任务已由平台终止。"})
                return
            if proc.poll() is not None:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    _handle_line(line)
                break
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if ready:
                line = proc.stdout.readline()
                if line:
                    _handle_line(line)

        proc.wait()
        if job_manager.should_stop(record):
            record.status = "terminated"
            publish(record, {"type": "terminated", "ts": time.time(), "message": "任务已由平台终止。"})
            return
        if proc.returncode == 0:
            try:
                manifest_store.rebuild_router_manifest()
            except Exception as exc:
                publish(record, {"type": "log", "line": f"[warn] Router manifest rebuild failed: {exc}", "ts": time.time()})
            record.status = "done"
            publish(record, {"type": "done", "ts": time.time(), "message": "Router build completed"})
        else:
            record.status = "error"
            publish(record, {"type": "error", "ts": time.time(), "message": f"Router build failed with exit code {proc.returncode}"})
    except Exception as exc:
        if not job_manager.should_stop(record):
            record.status = "error"
            publish(record, {"type": "error", "ts": time.time(), "message": str(exc)})
    finally:
        record.proc = None
        record.proc_uses_process_group = False
