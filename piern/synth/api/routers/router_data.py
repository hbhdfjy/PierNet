"""Stage 4 Token Router routes under /api/router/*."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Query

from piern.shared.runtime.paths import PROJECT_ROOT
from piern.shared.storage import portable
from piern.synth.services import job_manager, jsonl_filter_index, jsonl_index, manifest_store
from piern.synth.services.job_manager import publish

router = APIRouter()

ROUTER_DIR = PROJECT_ROOT / "data" / "router"
SCENARIO_DIR = ROUTER_DIR / "by_scenario"
TEXT2COMP_DIR = PROJECT_ROOT / "data" / "text2comp"
DEFAULT_QWEN_EMBEDDING_MODEL = "/home/tpx/Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_QWEN_EMBEDDING_TOKENIZER = DEFAULT_QWEN_EMBEDDING_MODEL


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return 0


def _load_jsonl_samples(path: Path) -> list[dict]:
    samples: list[dict] = []
    if not path.exists():
        return samples
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _rewrite_train_from_scenarios(seed: int = 0) -> int:
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for path in sorted(SCENARIO_DIR.glob("*.jsonl")):
        try:
            samples.extend(_load_jsonl_samples(path))
        except Exception:
            continue

    rng = random.Random(seed)
    rng.shuffle(samples)

    out_path = ROUTER_DIR / "train.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples)


def _build_router_status_from_manifests(router_manifest: dict, sample_manifest: dict) -> dict:
    sample_items = {item["scenario"]: item for item in sample_manifest.get("items", [])}
    source_by_scenario = {
        scenario: item.get("sample_count", 0)
        for scenario, item in sample_items.items()
    }

    scenario_map: dict[str, dict] = {
        scenario: {
            "scenario": scenario,
            "simulator": item.get("simulator", "unknown"),
            "source_count": item.get("sample_count", 0),
        }
        for scenario, item in sample_items.items()
    }

    for item in router_manifest.get("scenarios", []):
        entry = scenario_map.setdefault(
            item["scenario"],
            {
                "scenario": item["scenario"],
                "simulator": item.get("simulator", "unknown"),
                "source_count": 0,
            },
        )
        if entry.get("simulator", "unknown") == "unknown":
            entry["simulator"] = item.get("simulator", "unknown")
        entry["router_count"] = item.get("router_count", 0)
        entry["file_size_bytes"] = item.get("file_size_bytes", 0)
        entry["mtime"] = item.get("mtime", 0)
        entry["storage"] = item.get("storage", "jsonl")

    return {
        "splits": router_manifest.get(
            "splits",
            {
                "train": {
                    "exists": False,
                    "count": 0,
                    "file_size_bytes": 0,
                    "mtime": 0,
                }
            },
        ),
        "total": router_manifest.get("total", 0),
        "label_counts": router_manifest.get("label_counts", {"0": 0, "1": 0}),
        "scenarios": sorted(scenario_map.values(), key=lambda item: item["scenario"]),
        "source_count": sample_manifest.get("summary", {}).get("total_samples", 0),
        "source_by_scenario": source_by_scenario,
        "router_dir": str(portable.ROUTER_PARQUET_DIR if router_manifest.get("storage") == "parquet" else ROUTER_DIR),
        "storage": router_manifest.get("storage", "jsonl"),
    }


def _legacy_get_router_status() -> dict:
    splits: dict[str, dict] = {}
    total = 0
    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        stat = train_path.stat()
        count = _count_lines(train_path)
        splits["train"] = {
            "exists": True,
            "count": count,
            "file_size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
        }
        total = count
    else:
        splits["train"] = {"exists": False, "count": 0, "file_size_bytes": 0, "mtime": 0}

    label_counts = {"0": 0, "1": 0}
    if train_path.exists():
        try:
            with train_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    label = json.loads(line).get("label", -1)
                    key = str(label)
                    if key in label_counts:
                        label_counts[key] += 1
        except Exception:
            pass

    scenarios: list[dict] = []
    if SCENARIO_DIR.exists():
        for path in sorted(SCENARIO_DIR.glob("*.jsonl")):
            stat = path.stat()
            simulator = "unknown"
            count = 0
            try:
                with path.open("r", encoding="utf-8") as handle:
                    first = None
                    for raw in handle:
                        line = raw.strip()
                        if not line:
                            continue
                        count += 1
                        if first is None:
                            first = line
                    if first:
                        simulator = json.loads(first).get("metadata", {}).get("simulator", "unknown")
            except Exception:
                pass
            scenarios.append(
                {
                    "scenario": path.stem,
                    "simulator": simulator,
                    "count": count,
                    "file_size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )

    source_by_scenario: dict[str, int] = {}
    source_scenarios: list[dict] = []
    if TEXT2COMP_DIR.exists():
        for path in sorted(TEXT2COMP_DIR.glob("*.jsonl")):
            if path.name == "all_training_data.jsonl":
                continue
            simulator = "unknown"
            count = 0
            try:
                with path.open("r", encoding="utf-8") as handle:
                    first = None
                    for raw in handle:
                        line = raw.strip()
                        if not line:
                            continue
                        count += 1
                        if first is None:
                            first = line
                    if first:
                        simulator = json.loads(first).get("metadata", {}).get("simulator", "unknown")
            except Exception:
                pass
            source_by_scenario[path.stem] = count
            source_scenarios.append(
                {
                    "scenario": path.stem,
                    "simulator": simulator,
                    "source_count": count,
                }
            )

    scenario_map = {item["scenario"]: item for item in source_scenarios}
    for item in scenarios:
        entry = scenario_map.setdefault(
            item["scenario"],
            {
                "scenario": item["scenario"],
                "simulator": item.get("simulator", "unknown"),
                "source_count": 0,
            },
        )
        if entry.get("simulator", "unknown") == "unknown":
            entry["simulator"] = item.get("simulator", "unknown")
        entry["router_count"] = item["count"]
        entry["file_size_bytes"] = item["file_size_bytes"]
        entry["mtime"] = item["mtime"]

    return {
        "splits": splits,
        "total": total,
        "label_counts": label_counts,
        "scenarios": sorted(scenario_map.values(), key=lambda item: item["scenario"]),
        "source_count": sum(source_by_scenario.values()),
        "source_by_scenario": source_by_scenario,
        "router_dir": str(ROUTER_DIR),
    }


def _combined_sample_manifest() -> dict:
    try:
        jsonl_manifest = manifest_store.ensure_sample_manifest()
    except Exception:
        jsonl_manifest = {}
    parquet_manifest = portable.text2comp_manifest_like()
    if not parquet_manifest:
        return jsonl_manifest
    if not jsonl_manifest:
        return parquet_manifest

    parquet_items = list(parquet_manifest.get("items", []))
    parquet_scenarios = {item.get("scenario") for item in parquet_items}
    items = [*parquet_items]
    items.extend(
        {**item, "storage": "jsonl"}
        for item in jsonl_manifest.get("items", [])
        if item.get("scenario") not in parquet_scenarios
    )
    total = sum(int(item.get("sample_count") or 0) for item in items)
    return {
        "version": 2,
        "kind": "sample_manifest",
        "storage": "mixed" if len(items) != len(parquet_items) else "parquet",
        "items": sorted(items, key=lambda item: item.get("scenario", "")),
        "summary": {"total_samples": total},
    }


def _combined_router_manifest() -> dict:
    try:
        jsonl_manifest = manifest_store.ensure_router_manifest()
    except Exception:
        jsonl_manifest = {}
    parquet_manifest = portable.router_manifest_like()
    if not parquet_manifest:
        return jsonl_manifest
    if not jsonl_manifest:
        return parquet_manifest

    parquet_scenarios = {item.get("scenario") for item in parquet_manifest.get("scenarios", [])}
    scenarios = [*parquet_manifest.get("scenarios", [])]
    scenarios.extend(
        {**item, "storage": "jsonl"}
        for item in jsonl_manifest.get("scenarios", [])
        if item.get("scenario") not in parquet_scenarios
    )
    total = sum(int(item.get("router_count") or 0) for item in scenarios)
    size = sum(int(item.get("file_size_bytes") or 0) for item in scenarios)
    mtime = max((float(item.get("mtime") or 0) for item in scenarios), default=0.0)
    label_counts = parquet_manifest.get("label_counts", {"0": 0, "1": 0})
    if len(scenarios) != len(parquet_manifest.get("scenarios", [])):
        label_counts = jsonl_manifest.get("label_counts", label_counts)
    return {
        "version": 2,
        "kind": "router_manifest",
        "storage": "mixed" if len(scenarios) != len(parquet_manifest.get("scenarios", [])) else "parquet",
        "generated_at": max(float(jsonl_manifest.get("generated_at") or 0), float(parquet_manifest.get("generated_at") or 0)),
        "splits": {"train": {"exists": bool(scenarios), "count": total, "file_size_bytes": size, "mtime": mtime}},
        "total": total,
        "label_counts": label_counts,
        "scenarios": sorted(scenarios, key=lambda item: item.get("scenario", "")),
    }


def _router_total_from_manifest(split: str, scenario: str) -> int | None:
    try:
        manifest = _combined_router_manifest()
    except Exception:
        return None

    if scenario:
        for item in manifest.get("scenarios", []):
            if item.get("scenario") == scenario:
                return int(item.get("router_count", 0))
        return 0

    splits = manifest.get("splits", {})
    split_info = splits.get(split)
    if not split_info:
        return 0
    return int(split_info.get("count", 0))


@router.get("/router/status")
def get_router_status():
    """Return router dataset status and per-scenario stats."""
    try:
        router_manifest = _combined_router_manifest()
        sample_manifest = _combined_sample_manifest()
        return _build_router_status_from_manifests(router_manifest, sample_manifest)
    except Exception:
        return _legacy_get_router_status()


@router.post("/router/build")
async def build_router_data(
    seed: int = Query(42),
    neg_ratio: int = Query(1, ge=1, le=10),
    scenarios: str = Query(""),
):
    """Start Stage 4 router build and return a job id for SSE."""
    record = job_manager.create_job("router")
    scenario_list = [s.strip() for s in scenarios.split(",") if s.strip()] if scenarios else []

    def _run():
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

            scenario_totals: dict[str, int] = {}

            if proc.stdout is None:
                raise RuntimeError("router build subprocess did not provide stdout")

            for line in proc.stdout:
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
                        publish(
                            record,
                            {
                                "type": "init",
                                "scenario_totals": dict(record.scenario_totals),
                                "ts": time.time(),
                            },
                        )
                        publish(
                            record,
                            {
                                "type": "log",
                                "line": f"[init] {sc_name} expected {total} rows",
                                "ts": time.time(),
                            },
                        )
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
        except Exception as exc:
            if not record.stop_event.is_set():
                record.status = "error"
                publish(record, {"type": "error", "ts": time.time(), "message": str(exc)})
            return
        finally:
            record.proc = None
            record.proc_uses_process_group = False

        if record.stop_event.is_set():
            return

        if proc.returncode == 0:
            try:
                manifest_store.rebuild_router_manifest()
            except Exception as exc:
                publish(
                    record,
                    {
                        "type": "log",
                        "line": f"[warn] Router manifest rebuild failed: {exc}",
                        "ts": time.time(),
                    },
                )
            record.status = "done"
            publish(record, {"type": "done", "ts": time.time(), "message": "Router build completed"})
        else:
            record.status = "error"
            publish(
                record,
                {
                    "type": "error",
                    "ts": time.time(),
                    "message": f"Router build failed with exit code {proc.returncode}",
                },
            )

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": record.job_id, "status": "running"}


@router.delete("/router/scenario/{scenario}")
def delete_router_scenario(scenario: str):
    """Delete one scenario file and rewrite train.jsonl."""
    path = SCENARIO_DIR / f"{scenario}.jsonl"
    meta_path = path.with_suffix(".meta.json")
    if portable.delete_partition("router", scenario):
        return {"ok": True, "train_count": 0, "storage": "parquet"}
    if not path.exists():
        return {"ok": False, "message": "scenario file does not exist"}
    path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    total = _rewrite_train_from_scenarios(seed=0)
    try:
        manifest_store.rebuild_router_manifest()
    except Exception:
        pass
    return {"ok": True, "train_count": total}


@router.delete("/router/all")
def delete_all_router_data():
    """Delete all per-scenario router files and train.jsonl."""
    deleted = 0
    if SCENARIO_DIR.exists():
        for path in SCENARIO_DIR.glob("*.jsonl"):
            path.unlink()
            deleted += 1
        for meta_path in SCENARIO_DIR.glob("*.meta.json"):
            meta_path.unlink()
            deleted += 1
    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        train_path.unlink()
        deleted += 1
    try:
        manifest_store.rebuild_router_manifest()
    except Exception:
        pass
    return {"ok": True, "deleted": deleted}


@router.get("/router/samples")
def get_router_samples(
    split: str = Query("train"),
    scenario: str = Query(""),
    page: int = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
    label: int = Query(-1, ge=-1, le=1),
):
    """Read router samples page-wise, optionally filtered by split, scenario and label."""
    if portable.has_partitions("router"):
        try:
            parquet_page = portable.read_router_page(
                scenario=scenario,
                page=page,
                page_size=page_size,
                label=label,
            )
            if parquet_page is not None:
                total, items = parquet_page
                return {"total": total, "page": page, "page_size": page_size, "items": items, "storage": "parquet"}
        except Exception as exc:
            return {"total": 0, "page": page, "page_size": page_size, "items": [], "error": f"读取 Parquet 失败: {exc}"}

    path = SCENARIO_DIR / f"{scenario}.jsonl" if scenario else ROUTER_DIR / f"{split}.jsonl"
    if not path.exists():
        return {"total": 0, "page": page, "page_size": page_size, "items": []}

    has_label_filter = label in (0, 1)
    start = page * page_size
    end = start + page_size

    try:
        if not has_label_filter:
            total_hint = _router_total_from_manifest(split=split, scenario=scenario)
            try:
                total, items = jsonl_index.read_page(
                    path,
                    page=page,
                    page_size=page_size,
                    total_rows=total_hint,
                )
                return {"total": total, "page": page, "page_size": page_size, "items": items}
            except Exception:
                pass

        if has_label_filter:
            try:
                total, items = jsonl_filter_index.read_filtered_page(
                    path,
                    profile="router_label",
                    key=f"label={label}",
                    page=page,
                    page_size=page_size,
                )
                return {"total": total, "page": page, "page_size": page_size, "items": items}
            except Exception:
                pass

        items: list[dict] = []
        total = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if has_label_filter and obj.get("label") != label:
                    continue
                if start <= total < end:
                    items.append(obj)
                total += 1
    except Exception as exc:
        return {"total": 0, "page": page, "page_size": page_size, "items": [], "error": str(exc)}

    return {"total": total, "page": page, "page_size": page_size, "items": items}
