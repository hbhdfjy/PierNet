"""
Text2Comp 训练管理器

架构说明：
- Text2Comp：文本 → 数值预测（生成专家模型输入参数）
- 专家模型：接收Text2Comp输出 → 计算最终物理结果

关键概念：
- output_dim = expert_input_dim（Text2Comp输出维度等于专家模型输入维度）

提供功能：
1. 任务创建和管理
2. GPU资源调度
3. 训练进度监控
4. 支持任意专家模型类型
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

from pathlib import Path
from threading import RLock
from typing import Any

# 路径配置 - 使用项目内固定路径，避免数据丢失
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Checkpoint和模型存储在大容量存储（/root/data有1.2TB可用）
ARTIFACTS_ROOT = Path("/root/data/zyx/piern_artifacts/text2comp_models")
RUNLOGS_ROOT = PROJECT_ROOT / ".runlogs" / "text2comp"
REGISTRY_PATH = ARTIFACTS_ROOT / "training_jobs.json"

# 数据路径 - 统一使用项目内data/text2comp目录
TEXT2COMP_DATA_DIR = PROJECT_ROOT / "data" / "text2comp"
BASE_MODEL_DIR = Path("/root/eb-public/huggingface-models")

# GPU配置
GPU_FREE_MEMORY_THRESHOLD_MIB = 8192  # Text2Comp需要更多显存
GPU_AVAILABLE_UTIL_THRESHOLD = 20

# 默认训练参数
DEFAULT_TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "loss_fn": "mse",
    "max_length": 2048,
    "eval_interval": 10,
    "log_interval": 5,  # 降低日志间隔，确保小数据集也能记录进度
}

# 专家模型配置库
# 注意：output_dim = expert_input_dim（Text2Comp输出维度 = 专家模型输入维度）
EXPERT_MODEL_LIBRARY = {
    "diff-sorp": {
        "domain": "1D diffusion-sorption",
        "expert_type": "FNO",
        "output_dim": 128,          # Text2Comp输出128维 → FNO输入
        "expert_output_dim": 64,    # FNO最终输出64维
        "spatial_points": 64,
        "time_steps": 2,
        "channels": 1,
        "description": "扩散-吸附PDE预测，Text2Comp输出128维(64点×2帧)给FNO",
    },
    "diff-reaction": {
        "domain": "1D diffusion-reaction",
        "expert_type": "FNO",
        "output_dim": 32,           # Text2Comp输出32维 → FNO输入
        "expert_output_dim": 32,
        "spatial_points": 32,
        "time_steps": 1,
        "channels": 1,
        "description": "扩散-反应PDE预测",
    },
    "burgers": {
        "domain": "1D Burgers equation",
        "expert_type": "FNO",
        "output_dim": 32,
        "expert_output_dim": 32,
        "spatial_points": 32,
        "time_steps": 1,
        "channels": 1,
        "description": "Burgers方程预测",
    },
    "modflow": {
        "domain": "groundwater flow",
        "expert_type": "MODFLOW",
        "output_dim": 60,           # 5观测井 × 12时间点
        "expert_output_dim": 60,
        "spatial_points": 5,
        "time_steps": 12,
        "channels": 1,
        "description": "地下水流预测",
    },
    "power_flow": {
        "domain": "power system flow",
        "expert_type": "PowerFlow",
        "output_dim": 43,           # IEEE 14节点参数
        "expert_output_dim": 43,
        "spatial_points": 43,
        "time_steps": 1,
        "channels": 1,
        "description": "电力系统潮流计算",
    },
}

# 支持自定义专家模型的注册
CUSTOM_EXPERT_REGISTRY: dict[str, dict[str, Any]] = {}


def register_custom_expert(
    name: str,
    output_dim: int,
    domain: str = "",
    expert_type: str = "Custom",
    description: str = "",
    **kwargs
) -> None:
    """
    注册自定义专家模型

    Args:
        name: 专家模型名称
        output_dim: Text2Comp输出维度（专家模型输入维度）
        domain: 物理领域描述
        expert_type: 专家模型类型
        description: 描述信息
        **kwargs: 其他配置参数
    """
    CUSTOM_EXPERT_REGISTRY[name] = {
        "domain": domain,
        "expert_type": expert_type,
        "output_dim": output_dim,
        "expert_output_dim": kwargs.get("expert_output_dim", output_dim),
        "spatial_points": kwargs.get("spatial_points", 0),
        "time_steps": kwargs.get("time_steps", 1),
        "channels": kwargs.get("channels", 1),
        "description": description or f"自定义专家模型: {name}",
        **kwargs
    }


def get_all_experts() -> dict[str, dict[str, Any]]:
    """获取所有专家模型配置（包括自定义）"""
    return {**EXPERT_MODEL_LIBRARY, **CUSTOM_EXPERT_REGISTRY}


_REGISTRY_LOCK = RLock()
LOGGER = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    """确保必要的目录存在"""
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNLOGS_ROOT.mkdir(parents=True, exist_ok=True)
    TEXT2COMP_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_registry() -> list[dict[str, Any]]:
    """加载训练任务注册表"""
    _ensure_dirs()
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_registry(entries: list[dict[str, Any]]) -> None:
    """保存训练任务注册表"""
    _ensure_dirs()
    tmp_path = REGISTRY_PATH.with_suffix(REGISTRY_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(REGISTRY_PATH)


def _append_launch_log(path: Path, *lines: str) -> None:
    """追加启动日志"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _with_registry_lock(func):
    """注册表锁装饰器"""
    def wrapper(*args, **kwargs):
        with _REGISTRY_LOCK:
            return func(*args, **kwargs)
    return wrapper


def _find_job(entries: list[dict[str, Any]], job_id: str) -> dict[str, Any]:
    """查找任务"""
    for entry in entries:
        if entry["job_id"] == job_id:
            return entry
    raise KeyError(job_id)


def _make_job_id(name_base: str = "") -> str:
    """生成任务ID: 使用数据集名称+时间戳格式，方便模型拼装"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name_base:
        # 清理名称中的特殊字符
        safe_name = name_base.replace(" ", "_").replace("-", "_")[:40]
        return f"text2comp-{safe_name}_{timestamp}"
    return f"text2comp-{timestamp}"


def _normalize_job_name(value: Any, *, fallback: str) -> str:
    """规范化任务名称"""
    if value is None:
        return fallback
    name = str(value).strip()
    return name[:80] if name else fallback


def _pid_alive(pid: int | None) -> bool:
    """检查进程是否存活"""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _tail_lines(path: Path, limit: int = 200) -> list[str]:
    """读取文件末尾若干行"""
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            file_size = handle.tell()
            if file_size == 0:
                return []
            chunk_size = 8192
            buffers: list[bytes] = []
            remaining = limit + 1
            pos = file_size
            while pos > 0 and len(buffers) < remaining:
                read_size = min(chunk_size, pos)
                pos -= read_size
                handle.seek(pos)
                chunk = handle.read(read_size)
                buffers.append(chunk)
            tail_bytes = b"".join(reversed(buffers))
            lines = tail_bytes.decode("utf-8", errors="replace").splitlines()
            return lines[-limit:]
    except OSError:
        return []


def _latest_training_point(run_dir: Path) -> dict[str, Any] | None:
    """获取最新训练进度"""
    log_path = run_dir / "train_log.jsonl"
    lines = _tail_lines(log_path, 1)
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _latest_test_metrics(run_dir: Path) -> dict[str, Any] | None:
    """获取最新测试指标"""
    latest = run_dir / "test_metrics_latest.json"
    if not latest.exists():
        # 查找最近的评估文件
        eval_files = sorted(run_dir.glob("eval_epoch_*.json"))
        if eval_files:
            latest = eval_files[-1]
        else:
            return None
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None


def _checkpoint_entries(run_dir: Path) -> list[dict[str, Any]]:
    """获取checkpoint列表"""
    entries: list[dict[str, Any]] = []
    if not run_dir.exists():
        return entries
    for path in sorted(run_dir.glob("*model*.pt")):
        stat = path.stat()
        epoch = None
        stem = path.stem
        if "epoch" in stem:
            try:
                epoch = int(stem.split("_")[-1].replace("epoch", ""))
            except ValueError:
                epoch = None
        entries.append({
            "name": path.name,
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "epoch": epoch,
        })
    return sorted(entries, key=lambda item: item["mtime"], reverse=True)


def _refresh_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """刷新任务状态"""
    entry["name"] = _normalize_job_name(entry.get("name"), fallback=entry["job_id"])

    # 确保simulator和scenario字段存在（前端期望格式）
    entry["simulator"] = entry.get("simulator") or entry.get("expert_model", "unknown")
    entry["scenario"] = entry.get("scenario") or entry.get("simulator", "unknown")

    run_dir_str = entry.get("run_dir")
    log_path_str = entry.get("log_path")
    if not run_dir_str or not log_path_str:
        return entry

    run_dir = Path(run_dir_str)
    log_path = Path(log_path_str)
    pid = entry.get("pid")
    alive = _pid_alive(pid)

    # 更新训练进度
    latest_point = _latest_training_point(run_dir)
    if latest_point:
        entry["latest_epoch"] = int(latest_point.get("epoch", 0)) or None
        entry["latest_step"] = int(latest_point.get("step", 0)) or None
        entry["global_step"] = int(latest_point.get("global_step", 0)) or None
        entry["avg_loss"] = float(latest_point.get("avg_loss", latest_point.get("loss", 0.0)))

    # 更新测试指标
    latest_metrics = _latest_test_metrics(run_dir)
    if latest_metrics:
        entry["latest_test_epoch"] = int(latest_metrics.get("epoch", 0)) or None
        entry["latest_metrics"] = {
            "loss": latest_metrics.get("loss"),
            "mse": latest_metrics.get("mse"),
            "mae": latest_metrics.get("mae"),
            "mean_relative_error": latest_metrics.get("mean_relative_error"),
        }

    # 更新状态
    if alive:
        entry["status"] = "running"
    else:
        if entry.get("status") not in {"done", "error", "terminated"}:
            if entry.get("terminated"):
                entry["status"] = "terminated"
            elif (run_dir / "final_model.pt").exists() or (run_dir / "best_model.pt").exists():
                entry["status"] = "done"
            else:
                last_lines = _tail_lines(log_path, 20)
                if any("complete" in line.lower() or "done" in line.lower() for line in last_lines):
                    entry["status"] = "done"
                else:
                    entry["status"] = "error"
                    if last_lines:
                        entry["error_message"] = last_lines[-1]
            entry["ended_at"] = entry.get("ended_at") or time.time()

    entry["checkpoints"] = _checkpoint_entries(run_dir)
    return entry


# ==================== 公开API ====================

@_with_registry_lock
def list_jobs(refresh: bool = True) -> list[dict[str, Any]]:
    """列出所有训练任务"""
    entries = _load_registry()
    if refresh:
        entries = [_refresh_entry(entry) for entry in entries]
        _save_registry(entries)
    return sorted(entries, key=lambda item: item["created_at"], reverse=True)


@_with_registry_lock
def get_job(job_id: str, refresh: bool = True) -> dict[str, Any]:
    """获取单个任务详情"""
    entries = _load_registry()
    entry = _find_job(entries, job_id)
    if refresh:
        entry = _refresh_entry(entry)
        _save_registry(entries)
    return entry


def get_gpu_inventory() -> list[dict[str, Any]]:
    """获取GPU资源状态"""
    try:
        raw_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        LOGGER.info("nvidia-smi unavailable: %s", exc)
        return []

    rows = [line.strip() for line in raw_output.splitlines() if line.strip()]
    jobs = list_jobs(refresh=True)
    locked = {
        int(job["gpu_id"]): job["job_id"]
        for job in jobs
        if job["status"] in {"starting", "running"}
    }

    gpus: list[dict[str, Any]] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",", maxsplit=4)]
        if len(parts) != 5:
            continue
        idx_s, name, mem_used_s, mem_total_s, util_s = parts
        index = int(idx_s)
        memory_used = int(mem_used_s)
        memory_total = int(mem_total_s)
        utilization = int(util_s)

        available = True
        reason = None
        locked_by_job_id = locked.get(index)

        if locked_by_job_id:
            available = False
            reason = f"locked by {locked_by_job_id}"
        elif (memory_total - memory_used) < GPU_FREE_MEMORY_THRESHOLD_MIB:
            available = False
            reason = "memory busy"
        elif utilization >= GPU_AVAILABLE_UTIL_THRESHOLD:
            available = False
            reason = "utilization busy"

        gpus.append({
            "index": index,
            "name": name,
            "memory_used_mib": memory_used,
            "memory_total_mib": memory_total,
            "utilization_gpu": utilization,
            "available": available,
            "locked_by_job_id": locked_by_job_id,
            "reason": reason,
        })
    return gpus


def list_simulators() -> list[dict[str, Any]]:
    """列出所有支持的专家模型类型（包括自定义）"""
    all_experts = get_all_experts()
    return [
        {
            "name": name,
            "domain": info["domain"],
            "expert_type": info.get("expert_type", "generic"),
            "output_dim": info["output_dim"],       # Text2Comp输出维度
            "expert_output_dim": info.get("expert_output_dim", info["output_dim"]),
            "spatial_points": info.get("spatial_points", 0),
            "time_steps": info.get("time_steps", 1),
            "description": info["description"],
        }
        for name, info in all_experts.items()
    ]


def list_datasets() -> list[dict[str, Any]]:
    """列出可用的训练数据集"""
    datasets: list[dict[str, Any]] = []
    if not TEXT2COMP_DATA_DIR.exists():
        return datasets

    for path in TEXT2COMP_DATA_DIR.glob("*.jsonl"):
        # 解析数据集名称（如 diff-sorp_train.jsonl）
        stem = path.stem
        parts = stem.split("_")
        simulator = parts[0] if parts else stem
        scenario = stem  # 使用整个文件名作为scenario

        # 统计样本数
        try:
            with path.open("r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
        except Exception:
            count = 0

        datasets.append({
            "path": str(path),
            "name": stem,
            "simulator": simulator,
            "scenario": scenario,  # 添加scenario字段
            "n_samples": count,
            "sample_count": count,  # 兼容字段
            "size_bytes": path.stat().st_size,
            "file_size_bytes": path.stat().st_size,  # 兼容字段
            "mtime": path.stat().st_mtime,
        })

    return sorted(datasets, key=lambda item: item["simulator"])


def get_overview() -> dict[str, Any]:
    """获取训练概览"""
    jobs = list_jobs(refresh=True)
    running_count = sum(1 for job in jobs if job["status"] == "running")
    completed_count = sum(1 for job in jobs if job["status"] == "done")

    # 兼容前端schema：expert_models = simulators
    return {
        "expert_models": list_simulators(),
        "datasets": list_datasets(),
        "gpus": get_gpu_inventory(),
        "jobs": jobs[:12],
        "running_job_count": running_count,
        "completed_job_count": completed_count,
    }


def validate_training_data(data_path: str, expected_dim: int) -> dict[str, Any]:
    """验证训练数据"""
    path = Path(data_path)
    if not path.exists():
        raise ValueError(f"Training data not found: {data_path}")

    # 检查格式和维度
    valid_count = 0
    invalid_count = 0
    label_dims = set()

    try:
        with path.open("r", encoding="utf-8") as reader:
            for idx, line in enumerate(reader):
                if idx >= 100:  # 只检查前100条
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    label = item.get("label", []) if isinstance(item, dict) else []
                    if isinstance(label, list) and len(label) == expected_dim:
                        valid_count += 1
                        label_dims.add(len(label))
                    else:
                        invalid_count += 1
                except json.JSONDecodeError:
                    invalid_count += 1
    except Exception as e:
        raise ValueError(f"Failed to parse training data: {e}")

    return {
        "valid_samples": valid_count,
        "invalid_samples": invalid_count,
        "expected_dim": expected_dim,
        "actual_dims": sorted(label_dims),
        "is_valid": invalid_count == 0 and valid_count > 0,
    }


def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    """
    创建训练任务

    支持预定义和自定义专家模型：
    - 预定义：直接使用simulator名称（如 "diff-sorp"）
    - 自定义：提供output_dim参数或先通过register_custom_expert注册

    支持多种字段名（兼容前端和后端）：
    - train_path / train_data_path
    - simulator / expert_model
    """
    # 字段名兼容处理
    simulator = payload.get("simulator") or payload.get("expert_model", "unknown")
    train_data_path = payload.get("train_path") or payload.get("train_data_path")
    gpu_id = int(payload.get("gpu_id", 0))

    # 验证GPU
    gpu_map = {gpu["index"]: gpu for gpu in get_gpu_inventory()}
    gpu = gpu_map.get(gpu_id)
    if gpu is None:
        raise ValueError(f"GPU {gpu_id} not found")
    if not gpu["available"]:
        raise ValueError(f"GPU {gpu_id} is not available: {gpu['reason']}")

    # 获取专家模型配置（支持预定义和自定义）
    all_experts = get_all_experts()
    expert_info = all_experts.get(simulator)

    # 如果是自定义专家且未注册，尝试从payload创建
    if expert_info is None:
        custom_output_dim = payload.get("output_dim", 0)
        if custom_output_dim > 0:
            # 自动注册临时专家配置
            expert_info = {
                "domain": payload.get("domain", "Custom"),
                "expert_type": payload.get("expert_type", "Custom"),
                "output_dim": custom_output_dim,
                "expert_output_dim": payload.get("expert_output_dim", custom_output_dim),
                "spatial_points": payload.get("spatial_points", 0),
                "time_steps": payload.get("time_steps", 1),
                "channels": payload.get("channels", 1),
                "description": payload.get("description", f"自定义专家模型: {simulator}"),
            }
            # 临时注册
            register_custom_expert(
                name=simulator,
                output_dim=custom_output_dim,
                domain=expert_info["domain"],
                expert_type=expert_info["expert_type"],
                description=expert_info["description"],
                **{k: v for k, v in payload.items() if k in ["spatial_points", "time_steps", "channels", "expert_output_dim"]}
            )
            expert_info = CUSTOM_EXPERT_REGISTRY.get(simulator)
        else:
            raise ValueError(
                f"Unknown simulator: {simulator}. "
                f"Available predefined: {list(EXPERT_MODEL_LIBRARY.keys())}. "
                f"For custom experts, provide 'output_dim' parameter."
            )

    # 验证训练数据
    if train_data_path:
        validation = validate_training_data(train_data_path, expert_info["output_dim"])
        if not validation["is_valid"]:
            raise ValueError(f"Invalid training data: {validation}")

    # 构建任务 - 使用simulator+scenario作为任务名称基础
    scenario = payload.get("scenario") or simulator
    name_base = f"{simulator}_{scenario}"
    job_id = _make_job_id(name_base)
    job_name = _normalize_job_name(payload.get("name"), fallback=job_id)
    run_dir = ARTIFACTS_ROOT / simulator / "runs" / job_id
    log_path = RUNLOGS_ROOT / f"{job_id}.log"

    # 合并配置
    config = {**DEFAULT_TRAINING_CONFIG}
    for key in ["epochs", "batch_size", "learning_rate", "weight_decay", "loss_fn",
                "max_length", "eval_interval", "log_interval", "output_dim"]:
        if key in payload:
            config[key] = payload[key]

    # 输出维度自动推断
    if config.get("output_dim", 0) == 0:
        config["output_dim"] = expert_info["output_dim"]

    # 构建训练命令
    # 使用模块方式运行
    train_script = "-m"
    train_module = "PierNet.training.text2comp.train"
    base_model_path = payload.get("base_model") or payload.get("base_model_path") or "/root/eb-public/huggingface-models/Qwen/Qwen3-0.6B"

    # 使用当前Python解释器路径（解决conda环境问题）
    python_executable = sys.executable

    command = [
        python_executable,
        "-u",
        train_script,
        train_module,
        "--base-model", base_model_path,
        "--simulator", simulator,
        "--train-data", str(train_data_path) if train_data_path else "",
        "--output-dim", str(config["output_dim"]),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--loss-fn", config["loss_fn"],
        "--max-length", str(config["max_length"]),
        "--eval-interval", str(config["eval_interval"]),
        "--log-interval", str(config["log_interval"]),
        "--device", f"cuda:0",
        "--output-dir", str(ARTIFACTS_ROOT),
        "--run-name", job_id,
    ]

    # 创建任务记录并启动
    with _REGISTRY_LOCK:
        entries = _load_registry()

        # 再次检查GPU可用性
        current_gpu_map = {item["index"]: item for item in get_gpu_inventory()}
        current_gpu = current_gpu_map.get(gpu_id)
        if current_gpu is None or not current_gpu["available"]:
            raise ValueError(f"GPU {gpu_id} is not available")

        # 启动日志
        launch_started_at = time.time()
        _append_launch_log(
            log_path,
            f"[launch] job_id={job_id} name={job_name} created_at={launch_started_at:.3f}",
            f"[launch] status=starting simulator={simulator} gpu={gpu_id}",
            f"[launch] run_dir={run_dir}",
            f"[launch] log_path={log_path}",
            "[launch] spawning training subprocess...",
        )

        # 启动进程
        log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
                text=True,
            )
        except Exception as exc:
            log_handle.write(f"[error] failed to spawn subprocess: {exc}\n")
            log_handle.flush()
            log_handle.close()
            raise

        log_handle.write(f"[launch] subprocess pid={process.pid}\n")
        log_handle.flush()
        log_handle.close()

        # 保存任务记录
        scenario = payload.get("scenario") or simulator
        entry = {
            "job_id": job_id,
            "name": job_name,
            "status": "starting",
            "simulator": simulator,
            "scenario": scenario,
            "gpu_id": gpu_id,
            "created_at": time.time(),
            "started_at": time.time(),
            "ended_at": None,
            "pid": process.pid,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "config": config,
            "command": command,
            "terminated": False,
            "latest_epoch": None,
            "latest_step": None,
            "global_step": None,
            "avg_loss": None,
            "latest_test_epoch": None,
            "latest_metrics": None,
            "error_message": None,
            "checkpoints": [],
            "train_data_path": str(train_data_path) if train_data_path else None,
            "base_model_path": base_model_path,
        }
        entries.append(entry)
        _save_registry(entries)

        return _refresh_entry(entry)


@_with_registry_lock
def stop_job(job_id: str) -> dict[str, Any]:
    """停止训练任务"""
    entries = _load_registry()
    entry = _find_job(entries, job_id)
    pid = entry.get("pid")

    if pid and _pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    entry["terminated"] = True
    entry["status"] = "terminated"
    entry["ended_at"] = time.time()
    _save_registry(entries)
    return entry


@_with_registry_lock
def delete_job(job_id: str) -> dict[str, Any]:
    """删除训练任务"""
    entries = _load_registry()
    entry = _find_job(entries, job_id)
    entry = _refresh_entry(entry)

    if entry.get("status") in {"starting", "running"} or _pid_alive(entry.get("pid")):
        raise ValueError(f"Training job is still active: {job_id}")

    remaining = [item for item in entries if item["job_id"] != job_id]
    _save_registry(remaining)

    # 删除运行目录
    run_dir = Path(entry.get("run_dir", ""))
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except OSError:
            LOGGER.exception("Failed to delete run_dir %s", run_dir)

    # 删除日志文件
    log_path = Path(entry.get("log_path", ""))
    if log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            LOGGER.exception("Failed to delete log_path %s", log_path)

    return entry


def get_job_logs(job_id: str, limit: int = 300) -> list[str]:
    """获取任务日志"""
    entry = get_job(job_id, refresh=True)
    return _tail_lines(Path(entry["log_path"]), limit=limit)


def _downsample(points: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    """降采样训练曲线"""
    if len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    sampled = [points[idx] for idx in range(0, len(points), stride)]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def get_curves(job_id: str, max_points: int = 2000) -> dict[str, Any]:
    """获取训练曲线"""
    entry = get_job(job_id, refresh=True)
    run_dir = Path(entry["run_dir"])

    # 读取训练日志
    training_points: list[dict[str, Any]] = []
    train_log_path = run_dir / "train_log.jsonl"

    if train_log_path.exists():
        with train_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                training_points.append({
                    "epoch": int(payload.get("epoch", 0)),
                    "step": int(payload.get("step", 0)),
                    "global_step": int(payload.get("global_step", 0)),
                    "loss": float(payload.get("loss", payload.get("avg_loss", 0.0))),
                })

    # 读取评估指标
    eval_points: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("eval_epoch_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            eval_points.append({
                "epoch": int(payload.get("epoch", 0)),
                "loss": float(payload.get("loss", 0.0)),
                "mse": float(payload.get("mse", 0.0)),
                "mae": float(payload.get("mae", 0.0)),
            })
        except Exception:
            continue

    return {
        "job_id": job_id,
        "training_points": _downsample(training_points, max_points=max_points),
        "eval_points": _downsample(eval_points, max_points=max_points),
        "checkpoints": _checkpoint_entries(run_dir),
    }