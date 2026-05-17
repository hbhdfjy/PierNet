"""Worker-safe executor for Token Router dataset builds."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from piern.shared.runtime.paths import PROJECT_ROOT
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


def run_router_build_job(record: JobRecord, payload: dict[str, Any]) -> None:
    seed = int(payload.get("seed", 42))
    neg_ratio = int(payload.get("neg_ratio", 1))
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

        script = PROJECT_ROOT / "scripts" / "router" / "build_router_data.py"
        cmd = [
            sys.executable,
            str(script),
            "--data-dir",
            "data/text2comp_parquet",
            "--output-dir",
            "data/router_parquet",
            "--input-format",
            "parquet",
            "--output-format",
            "parquet",
            "--seed",
            str(seed),
            "--neg-ratio",
            str(neg_ratio),
            "--chat-template",
            "qwen",
        ]
        if scenario_list:
            cmd += ["--scenarios", *scenario_list]

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
        for line in proc.stdout:
            if job_manager.should_stop(record):
                _kill_process_group(proc)
                record.status = "terminated"
                publish(record, {"type": "terminated", "ts": time.time(), "message": "任务已由平台终止。"})
                return
            line = line.rstrip()
            if not line:
                continue
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
                continue
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
                continue
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
                continue
            publish(record, {"type": "log", "line": line, "ts": time.time()})

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
