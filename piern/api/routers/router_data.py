"""Stage 4 Token Router 数据路由：/api/router/*"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Query

from piern.api.deps import PROJECT_ROOT
from piern.api.services import job_manager
from piern.api.services.job_manager import publish

router = APIRouter()

ROUTER_DIR    = PROJECT_ROOT / "data" / "router"
SCENARIO_DIR  = ROUTER_DIR / "by_scenario"
TEXT2COMP_DIR = PROJECT_ROOT / "data" / "text2comp"


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in open(path, "rb") if line.strip())
    except Exception:
        return 0


# ── 状态查询 ──────────────────────────────────────────────────────

@router.get("/router/status")
def get_router_status():
    """返回 Router 数据目录的整体状态 + 按场景统计。"""

    # train 汇总（只有 train，不再划分 val/test）
    splits = {}
    total = 0
    path = ROUTER_DIR / "train.jsonl"
    if path.exists():
        stat = path.stat()
        count = _count_lines(path)
        splits["train"] = {
            "exists": True,
            "count": count,
            "file_size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
        }
        total = count
    else:
        splits["train"] = {"exists": False, "count": 0, "file_size_bytes": 0, "mtime": 0}

    # 正负样本分布（从 train 统计）
    # 注意：JSON 序列化会将 int key 转为 string，前端用字符串 key 访问
    label_counts = {"0": 0, "1": 0}
    train_path = ROUTER_DIR / "train.jsonl"
    if train_path.exists():
        try:
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        label = json.loads(line).get("label", -1)
                        key = str(label)
                        if key in label_counts:
                            label_counts[key] += 1
        except Exception:
            pass

    # 按场景统计（from by_scenario/）
    scenarios: list[dict] = []
    if SCENARIO_DIR.exists():
        for f in sorted(SCENARIO_DIR.glob("*.jsonl")):
            count = _count_lines(f)
            stat  = f.stat()
            # 读第一行取 simulator
            simulator = "unknown"
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    first = fh.readline().strip()
                    if first:
                        simulator = json.loads(first).get("metadata", {}).get("simulator", "unknown")
            except Exception:
                pass
            scenarios.append({
                "scenario": f.stem,
                "simulator": simulator,
                "count": count,
                "file_size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            })

    # Stage 3 源数据（按场景），同时读取 simulator 信息
    source_by_scenario: dict[str, int] = {}
    source_scenarios: list[dict] = []
    if TEXT2COMP_DIR.exists():
        for f in sorted(TEXT2COMP_DIR.glob("*.jsonl")):
            if f.name == "all_training_data.jsonl":
                continue
            count = _count_lines(f)
            source_by_scenario[f.stem] = count
            # 读第一行取 simulator
            simulator = "unknown"
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    first = fh.readline().strip()
                    if first:
                        simulator = json.loads(first).get("metadata", {}).get("simulator", "unknown")
            except Exception:
                pass
            source_scenarios.append({
                "scenario": f.stem,
                "simulator": simulator,
                "source_count": count,
            })

    # 合并：以 Stage 3 场景为基础，附加 Router 已生成的条数
    scenario_map = {s["scenario"]: s for s in source_scenarios}
    for sc in scenarios:
        if sc["scenario"] in scenario_map:
            scenario_map[sc["scenario"]]["router_count"] = sc["count"]
            scenario_map[sc["scenario"]]["file_size_bytes"] = sc["file_size_bytes"]
            scenario_map[sc["scenario"]]["mtime"] = sc["mtime"]
    merged_scenarios = list(scenario_map.values())

    return {
        "splits": splits,
        "total": total,
        "label_counts": label_counts,
        "scenarios": merged_scenarios,
        "source_count": sum(source_by_scenario.values()),
        "source_by_scenario": source_by_scenario,
        "router_dir": str(ROUTER_DIR),
    }


# ── 触发生成 ──────────────────────────────────────────────────────

@router.post("/router/build")
async def build_router_data(
    seed: int = Query(42),
    neg_ratio: int = Query(1, ge=1, le=10),
    scenarios: str = Query(""),        # 逗号分隔的场景名，空=全部
    chat_template: str = Query("plain"),  # chat template 类型
    user_prefix: str = Query(""),      # 自定义 template 前缀
    user_suffix: str = Query(""),      # 自定义 template 后缀
    assistant_prefix: str = Query(""), # 自定义 assistant 前缀
):
    """触发 Stage 4 Router 数据生成，返回 job_id 供 SSE 监听。"""
    record = job_manager.create_job("router")
    scenario_list = [s.strip() for s in scenarios.split(",") if s.strip()] if scenarios else []

    def _run():
        try:
            sc_desc = f"场景：{', '.join(scenario_list)}" if scenario_list else "全部场景"
            publish(record, {"type": "log", "line": f"[Stage 4] 开始生成 Token Router 训练数据（{sc_desc}，template={chat_template}）…", "ts": time.time()})
            script = PROJECT_ROOT / "scripts" / "router" / "build_router_data.py"
            cmd = [
                sys.executable, str(script),
                "--data-dir",      "data/text2comp",
                "--output-dir",    "data/router",
                "--seed",          str(seed),
                "--neg-ratio",     str(neg_ratio),
                "--chat-template", chat_template,
            ]
            if chat_template == "custom":
                cmd += [
                    "--user-prefix",      user_prefix,
                    "--user-suffix",      user_suffix,
                    "--assistant-prefix", assistant_prefix,
                ]
            if scenario_list:
                cmd += ["--scenarios"] + scenario_list
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(PROJECT_ROOT),
            )
            record.proc = proc

            scenario_totals: dict[str, int] = {}

            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue

                # PROGRESS_INIT:场景名:预期总条数 — 初始化进度
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
                        publish(record, {
                            "type": "init",
                            "scenario_totals": dict(record.scenario_totals),
                            "ts": time.time(),
                        })
                        publish(record, {
                            "type": "log",
                            "line": f"[场景] {sc_name}（预计 {total} 条）",
                            "ts": time.time(),
                        })
                    continue

                # PROGRESS_UPDATE:场景名:当前条数:预期总条数 — 中间进度
                if line.startswith("PROGRESS_UPDATE:"):
                    parts = line.split(":", 3)
                    if len(parts) == 4:
                        sc_name = parts[1]
                        try:
                            done  = int(parts[2])
                            total = int(parts[3])
                        except ValueError:
                            done, total = 0, 0
                        publish(record, {
                            "type": "log",
                            "line": f"  {sc_name}: {done}/{total}",
                            "ts": time.time(),
                            "progress": {"scenario": sc_name, "done": done, "total": total},
                        })
                    continue

                # PROGRESS_DONE:场景名:实际条数:预期条数 — 场景完成
                if line.startswith("PROGRESS_DONE:"):
                    parts = line.split(":", 3)
                    if len(parts) >= 3:
                        sc_name = parts[1]
                        try:
                            done  = int(parts[2])
                            total = int(parts[3]) if len(parts) == 4 else scenario_totals.get(sc_name, done)
                        except ValueError:
                            done, total = 0, 0
                        publish(record, {
                            "type": "log",
                            "line": f"  {sc_name}: {done}/{total}",
                            "ts": time.time(),
                            "progress": {"scenario": sc_name, "done": done, "total": total},
                        })
                    continue

                # 普通日志行
                publish(record, {"type": "log", "line": line, "ts": time.time()})

            proc.wait()
            if record.stop_event.is_set():
                pass
            elif proc.returncode == 0:
                record.status = "done"
                publish(record, {"type": "done", "ts": time.time(), "message": "Router 数据生成完成"})
            else:
                record.status = "error"
                publish(record, {"type": "error", "ts": time.time(), "message": f"脚本退出码 {proc.returncode}"})
        except Exception as e:
            if not record.stop_event.is_set():
                record.status = "error"
                publish(record, {"type": "error", "ts": time.time(), "message": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": record.job_id, "status": "running"}


# ── 文件管理 ──────────────────────────────────────────────────────

@router.delete("/router/scenario/{scenario}")
def delete_router_scenario(scenario: str):
    """删除指定场景的路由数据文件（by_scenario/{scenario}.jsonl）。"""
    path = SCENARIO_DIR / f"{scenario}.jsonl"
    if not path.exists():
        return {"ok": False, "message": "文件不存在"}
    path.unlink()
    return {"ok": True}


@router.delete("/router/all")
def delete_all_router_data():
    """清空所有路由数据（by_scenario/ 目录及 train/val/test.jsonl）。"""
    deleted = 0
    if SCENARIO_DIR.exists():
        for f in SCENARIO_DIR.glob("*.jsonl"):
            f.unlink()
            deleted += 1
    p = ROUTER_DIR / "train.jsonl"
    if p.exists():
        p.unlink()
        deleted += 1
    return {"ok": True, "deleted": deleted}


# ── 样本浏览 ──────────────────────────────────────────────────────

@router.get("/router/samples")
def get_router_samples(
    split: str    = Query("train"),
    scenario: str = Query(""),        # 空=全部场景；非空=从 by_scenario/ 读取
    page: int     = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
    label: int    = Query(-1, ge=-1, le=1),
):
    """分页读取 Router 样本，支持按场景或按 split 筛选。"""

    # 按场景读取（from by_scenario/）
    if scenario:
        path = SCENARIO_DIR / f"{scenario}.jsonl"
    else:
        path = ROUTER_DIR / f"{split}.jsonl"

    if not path.exists():
        return {"total": 0, "page": page, "page_size": page_size, "items": []}

    has_label_filter = label in (0, 1)
    start = page * page_size
    end   = start + page_size

    items: list[dict] = []
    total = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if has_label_filter and obj.get("label") != label:
                        continue
                    if start <= total < end:
                        items.append(obj)
                    total += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"total": 0, "page": page, "page_size": page_size, "items": [], "error": str(e)}

    return {"total": total, "page": page, "page_size": page_size, "items": items}
