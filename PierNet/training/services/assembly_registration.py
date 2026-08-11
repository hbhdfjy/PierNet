from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from PierNet.shared.runtime.paths import ARTIFACT_ROOT, PROJECT_ROOT, RUNLOG_ROOT
from PierNet.synth.services import expert_models as uploaded_expert_models


PROFILE_STORE_PATH = Path(
    os.getenv(
        "PIERN_CONVERSATION_ASSEMBLY_PROFILES",
        str(RUNLOG_ROOT / "conversation_assembly_profiles.yaml"),
    )
).expanduser()
DEFAULT_LLM_PATH = Path(
    os.getenv(
        "PIERN_CONVERSATION_LLM_PATH",
        str(PROJECT_ROOT / "models" / "Qwen" / "Qwen2.5-0.5B-Instruct"),
    )
).expanduser()
BUILTIN_EXPERT_ROOT = Path(
    os.getenv(
        "PIERN_CONVERSATION_EXPERT_ROOT",
        str(ARTIFACT_ROOT / "conversation_experts"),
    )
).expanduser()

_STORE_LOCK = RLock()

_TASK_LABELS = {
    "modflow": "地下水流动与水头预测",
    "diff_sorp": "扩散吸附过程预测",
    "diff-sorp": "扩散吸附过程预测",
    "diff_reaction": "扩散反应过程预测",
    "diff-reaction": "扩散反应过程预测",
    "burgers": "Burgers 方程状态预测",
    "gcam": "能源与气候系统预测",
    "power_flow": "电力潮流计算",
    "transient": "电力暂态过程预测",
}

_TASK_KEYWORDS = {
    "modflow": ["modflow", "地下水", "水头", "含水层", "渗透", "抽水", "补给", "groundwater", "aquifer", "hydraulic"],
    "diff_sorp": ["diff_sorp", "diff-sorp", "扩散吸附", "吸附", "sorption"],
    "diff-sorp": ["diff_sorp", "diff-sorp", "扩散吸附", "吸附", "sorption"],
    "diff_reaction": ["diff_reaction", "diff-reaction", "扩散反应", "reaction"],
    "diff-reaction": ["diff_reaction", "diff-reaction", "扩散反应", "reaction"],
    "burgers": ["burgers", "伯格斯"],
    "gcam": ["gcam", "能源", "气候", "碳排放"],
    "power_flow": ["power_flow", "潮流", "电网", "power flow"],
    "transient": ["transient", "暂态", "故障", "transient stability"],
}

_PREDICTION_KEYWORDS = ["预测", "计算", "求解", "模拟", "下一时刻", "下一步", "predict", "forecast", "simulate", "solve"]


def _load_profiles() -> list[dict[str, Any]]:
    if not PROFILE_STORE_PATH.exists():
        return []
    data = yaml.safe_load(PROFILE_STORE_PATH.read_text(encoding="utf-8")) or {}
    items = data.get("profiles", data if isinstance(data, list) else [])
    return [dict(item) for item in items if isinstance(item, dict)]


def _write_profiles(profiles: list[dict[str, Any]]) -> None:
    PROFILE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROFILE_STORE_PATH.with_name(f".{PROFILE_STORE_PATH.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        yaml.safe_dump({"profiles": profiles}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, PROFILE_STORE_PATH)


def _existing_path(value: Any, label: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _resolve_expert(job: dict[str, Any], expected_input_dim: int) -> dict[str, Any]:
    uploaded_expert_id = str(job.get("uploaded_expert_id") or "").strip()
    if uploaded_expert_id:
        model = uploaded_expert_models.get_model(uploaded_expert_id)
        if model.get("status") != "active" or not model.get("assembly_enabled"):
            raise ValueError(f"Uploaded Expert is not available for assembly: {uploaded_expert_id}")
        input_dim = int(model.get("input_dim") or 0)
        if input_dim != expected_input_dim:
            raise ValueError(
                f"Text2Comp output dimension does not match Uploaded Expert input: "
                f"{expected_input_dim} != {input_dim}"
            )
        return {
            "expert_kind": "uploaded",
            "uploaded_expert_id": uploaded_expert_id,
            "expert_path": str(_existing_path(model.get("path"), "Uploaded Expert")),
            "expert_input_dim": input_dim,
            "expert_output_dim": int(model.get("output_dim") or 0),
            "output_shape": [int(model.get("output_dim") or 0)],
            "parameter_names": list(model.get("input_names") or model.get("parameter_names") or []),
        }

    simulator = str(job.get("simulator") or "").strip().lower()
    expert_root = BUILTIN_EXPERT_ROOT / simulator
    manifest_path = expert_root / "manifest.json"
    expert_path = expert_root / "expert_model.pt"
    _existing_path(manifest_path, f"{simulator} expert manifest")
    _existing_path(expert_path, f"{simulator} expert checkpoint")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    input_dim = len(manifest.get("param_mean") or [])
    if input_dim != expected_input_dim:
        raise ValueError(
            f"Text2Comp output dimension does not match built-in Expert input: "
            f"{expected_input_dim} != {input_dim}"
        )
    output_shape = [int(item) for item in manifest.get("target_shape") or []]
    output_dim = 1
    for size in output_shape:
        output_dim *= size
    return {
        "expert_kind": "modflow_dnn",
        "uploaded_expert_id": None,
        "expert_path": str(expert_path),
        "expert_input_dim": input_dim,
        "expert_output_dim": output_dim,
        "output_shape": output_shape,
        "parameter_names": [str(item) for item in manifest.get("param_names") or []],
    }


def _build_profile_prompt(job: dict[str, Any], expert: dict[str, Any]) -> dict[str, Any]:
    simulator = str(job.get("simulator") or "unknown").strip().lower()
    task_label = _TASK_LABELS.get(simulator, f"{simulator} 科学计算")
    scenarios = [str(item).strip() for item in job.get("scenarios") or [] if str(item).strip()]
    parameter_names = [str(item) for item in expert.get("parameter_names") or []]
    expected_input_dim = int(expert.get("expert_input_dim") or 0)
    keywords = list(dict.fromkeys([simulator, *_TASK_KEYWORDS.get(simulator, []), *scenarios]))
    parameter_text = "、".join(parameter_names) if parameter_names else f"按训练数据约定顺序排列的 {expected_input_dim} 个数值"
    scenario_text = "、".join(scenarios) if scenarios else "训练任务登记的场景"
    system_prompt = f"""你是“{task_label}”模型的对话助手。本模型来源于简易训练任务，适用场景为：{scenario_text}。

你需要同时处理普通对话、知识问答和科学计算预测请求，并严格遵守以下规则：
1. 普通问候、概念解释和知识问答应直接自然回答，不要声称已经进行预测。
   如果问题与{task_label}无关，应使用通用知识回答，不要主动提及本任务、地下水或预测流程。
2. 只有用户明确要求预测、计算、求解或模拟，并提供完整数值输入时，才将请求视为{task_label}计算任务。
3. 本模型的专家链路需要至少 {expected_input_dim} 个数值参数。参数顺序为：{parameter_text}。
4. 如果用户表达了预测意图但参数不足，应明确提示需要补充哪些参数，不得编造结果。
5. 禁止输出任何占位内容，禁止在专家模型未返回结果时声称预测已经完成。
6. 对完整有效的计算请求，可以用“好的，”作为简短开头；平台随后会通过 Router 和专家模型生成真实数值结果。
7. 回答应简洁、准确，不要把普通领域介绍误当成预测任务。
"""
    return {
        "task_label": task_label,
        "task_keywords": keywords,
        "prediction_keywords": list(_PREDICTION_KEYWORDS),
        "parameter_names": parameter_names,
        "min_user_numeric_values": expected_input_dim,
        "system_prompt": system_prompt,
    }


def register_training_job(job: dict[str, Any]) -> dict[str, Any]:
    if str(job.get("status")) != "done" or str(job.get("pipeline_stage")) != "done":
        raise ValueError("Only a completed simple training pipeline can be registered")
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    if not config.get("simple_pipeline_enabled"):
        raise ValueError("The training job is not a complete simple-training pipeline")

    run_dir = _existing_path(job.get("run_dir"), "Router run directory")
    router_path = _existing_path(run_dir / "router_final.pt", "Router checkpoint")
    text2comp_path = _existing_path(job.get("text2comp_model_path"), "Text2Comp checkpoint")
    llm_path = _existing_path(DEFAULT_LLM_PATH, "Conversation LLM")
    expected_input_dim = int(job.get("text2comp_output_dim") or job.get("uploaded_expert_input_dim") or 0)
    if expected_input_dim <= 0:
        raise ValueError("Training job does not declare a Text2Comp output dimension")
    expert = _resolve_expert(job, expected_input_dim)
    prompt_profile = _build_profile_prompt(job, expert)
    completed_at = float(job.get("ended_at") or job.get("created_at") or time.time())
    completed_label = time.strftime("%m-%d %H:%M:%S", time.localtime(completed_at))

    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("Training job has no job_id")
    profile = {
        "model_id": f"conversation_{job_id}",
        "name": f"{prompt_profile['task_label']} · {completed_label}",
        "description": "由对话训练工作流自动注册的 LLM、Router、Text2Comp 与 Expert 完整链路。",
        "executor": "training_job_profile",
        "simulator": str(job.get("simulator") or "unknown"),
        "root": str(run_dir),
        "llm_path": str(llm_path),
        "router_path": str(router_path),
        "text2comp_path": str(text2comp_path),
        "expert_path": expert["expert_path"],
        "expert_kind": expert["expert_kind"],
        "uploaded_expert_id": expert["uploaded_expert_id"],
        "expert_input_dim": expert["expert_input_dim"],
        "expert_output_dim": expert["expert_output_dim"],
        "param_count": expected_input_dim,
        "output_shape": expert["output_shape"],
        **prompt_profile,
        "chat_enabled": True,
        "force_split": False,
        "max_new_tokens": 256,
        "source": "simple_training",
        "source_job_id": job_id,
        "metrics": {
            "router_f1": (job.get("router_metrics") or job.get("latest_metrics") or {}).get("f1"),
            "text2comp_normalized_rmse": (job.get("text2comp_metrics") or {}).get("normalized_rmse"),
        },
    }

    with _STORE_LOCK:
        profiles = [item for item in _load_profiles() if item.get("model_id") != profile["model_id"]]
        profiles.append(profile)
        _write_profiles(profiles)
    return profile
