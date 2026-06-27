"""
PiERN Assembly API - 完整正确流程版本

关键设计（遵循 single_eval.py）：
1. LLM和Router必须在同一GPU上 - 因为generated_ids在LLM推理过程中产生，
   Router需要直接判断这些token_ids，跨GPU传输会增加延迟
2. 支持LLM单独GPU切分（使用device_map="auto"）
3. 支持Router/Text2Comp/FNO指定与LLM不同的GPU（但会自动处理设备同步）
4. 加载完成后立即更新GPU状态（使用pynvml实时获取）

架构流程：
- 用户输入 → LLM逐步生成token
- 每生成一个token，Router判断当前generated_ids
- Router触发PDE类别 → Text2Comp编码 → FNO预测
"""

from __future__ import annotations

import os
import re
import time
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from PierNet.synth.services import expert_models as uploaded_expert_models
from PierNet.training.api.modflow_assembly import ModflowAssemblyProfilePipeline


# ===== 配置文件动态扫描 =====
import yaml as _yaml
import glob as _glob

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _REPO_ROOT / "configs" / "assembly" / "models.yaml"
_ASSEMBLED_PROFILES_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "assembly" / "assembled_profiles.yaml"
_ARTIFACTS_ROOT = Path(os.getenv("PIERN_ARTIFACTS_ROOT", str(_REPO_ROOT / "artifacts"))).expanduser()
_TEXT2COMP_MODELS_ROOT = Path(
    os.getenv("PIERN_TEXT2COMP_MODELS_DIR", str(_ARTIFACTS_ROOT / "text2comp_models"))
).expanduser()
_ROUTER_ARTIFACTS_ROOT = Path(
    os.getenv("PIERN_ROUTER_ARTIFACTS_DIR", str(_ARTIFACTS_ROOT / "token_router"))
).expanduser()
_FNO_MODELS_ROOT = Path(os.getenv("PIERN_FNO_MODELS_DIR", str(_ARTIFACTS_ROOT / "fno_models"))).expanduser()
_DEFAULT_TEXT2COMP_BASE_MODEL = "/root/data/PierNet/models/Qwen/Qwen2.5-0.5B-Instruct"

def _load_config():
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return _yaml.safe_load(f) or {}
    return {}


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _assembly_profiles_config_path() -> Path:
    config = _load_config()
    raw = config.get("assembled_profiles") or config.get("assembly_profiles") or str(_ASSEMBLED_PROFILES_DEFAULT_CONFIG)
    return _resolve_repo_path(raw)


def _profile_artifact_path(root: Path, manifest: dict[str, Any], key: str, default_name: str) -> str:
    rel = (manifest.get("artifacts") or {}).get(key) or f"artifacts/{default_name}"
    path = Path(str(rel)).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path)


def _scan_assembly_profiles() -> list[dict[str, Any]]:
    config_path = _assembly_profiles_config_path()
    if not config_path.exists():
        return []
    data = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    items = data.get("profiles", data if isinstance(data, list) else [])
    profiles: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        root_value = item.get("root") or item.get("path")
        if not root_value:
            continue
        root = _resolve_repo_path(str(root_value))
        manifest_rel = item.get("manifest", "artifacts/manifest.json")
        manifest_path = Path(str(manifest_rel)).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        llm_path = str(item.get("llm_path") or manifest.get("llm_path") or "")
        chat_llm_path = str(item.get("chat_llm_path") or "")
        router_path = _profile_artifact_path(root, manifest, "router", "router_modflow.pt")
        text2comp_path = _profile_artifact_path(root, manifest, "text2comp", "text2comp_modflow.pt")
        expert_path = _profile_artifact_path(root, manifest, "expert", "expert_modflow_dnn.pt")
        required_paths = [str(root), str(manifest_path), llm_path, router_path, text2comp_path, expert_path]
        if chat_llm_path:
            required_paths.append(chat_llm_path)
        trained = all(bool(p) and os.path.exists(p) for p in required_paths)
        model_id = str(item.get("model_id") or item.get("id") or root.name)
        target_shape = manifest.get("target_shape") or []
        profiles.append({
            "model_id": model_id,
            "name": str(item.get("name") or manifest.get("name") or model_id),
            "description": str(item.get("description") or manifest.get("description") or "PierNet assembled model"),
            "executor": str(item.get("executor") or "modflow_profile"),
            "simulator": str(item.get("simulator") or "modflow"),
            "root": str(root),
            "manifest_path": str(manifest_path),
            "llm_path": llm_path,
            "chat_llm_path": chat_llm_path or None,
            "router_path": router_path,
            "text2comp_path": text2comp_path,
            "expert_path": expert_path,
            "feature_dim": int(manifest.get("feature_dim") or 0),
            "param_count": len(manifest.get("param_names") or []),
            "output_shape": target_shape,
            "sample_count": int(manifest.get("sample_count") or 0),
            "metrics": manifest.get("metrics") or {},
            "trained": trained,
            "chat_enabled": bool(item.get("chat_enabled", True)) and trained,
            "source_thread_id": item.get("source_thread_id"),
            "source": item.get("source") or "registered_profile",
            "missing_paths": [p for p in required_paths if not p or not os.path.exists(p)],
        })
    return profiles


def _get_assembly_profile(model_id: str) -> dict[str, Any]:
    for profile in _scan_assembly_profiles():
        if profile["model_id"] == model_id:
            return profile
    raise HTTPException(status_code=404, detail=f"Assembly profile not found: {model_id}")

def _infer_simulator_from_path(path: str) -> str:
    lowered = path.lower()
    if "diff-sorp" in lowered or "diff_sorp" in lowered:
        return "diff_sorp"
    if "diff-reaction" in lowered or "diff_reaction" in lowered:
        return "diff_reaction"
    if "burgers" in lowered:
        return "burgers"
    return "unknown"


def _text2comp_model_name(path: str) -> str:
    stem = os.path.basename(path).replace(".pt", "")
    if stem == "final_model":
        parent = os.path.basename(os.path.dirname(path))
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if parent and parent != "text2comp_models":
            return f"{parent}_model"
        if grandparent:
            return f"{grandparent}_model"
    return stem


def _text2comp_metadata(path: str) -> tuple[str, int, str]:
    model_path = Path(path)
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
        if isinstance(config, dict):
            simulator = str(config.get("simulator") or _infer_simulator_from_path(path))
            output_dim = int(config.get("output_dim") or (128 if simulator == "diff_sorp" else 32))
            name = str(config.get("task_name") or _text2comp_model_name(path))
            return simulator, output_dim, name
    simulator = _infer_simulator_from_path(path)
    return simulator, 128 if simulator == "diff_sorp" else 32, _text2comp_model_name(path)


def _scan_text2comp():
    config = _load_config()
    models = []
    seen = set()
    for d in config.get("text2comp_dirs", []):
        p = d.get("path", "")
        pat = d.get("pattern", "*.pt")
        if p and os.path.exists(p):
            for f in sorted(_glob.glob(os.path.join(p, pat), recursive="**" in pat)):
                if f in seen:
                    continue
                seen.add(f)
                sim, od, name = _text2comp_metadata(f)
                models.append({
                    "name": name,
                    "simulator": sim,
                    "output_dim": od,
                    "path": f,
                })
    if not models:
        art_dir = str(_TEXT2COMP_MODELS_ROOT)
        if os.path.exists(art_dir):
            for f in sorted(_glob.glob(os.path.join(art_dir, "**/final_model.pt"), recursive=True)):
                if f in seen:
                    continue
                seen.add(f)
                sim, od, name = _text2comp_metadata(f)
                models.append({
                    "name": name,
                    "simulator": sim,
                    "output_dim": od,
                    "path": f,
                })
    return models

def _scan_router():
    config = _load_config()
    rdir = config.get("router_dir") or str(_ROUTER_ARTIFACTS_ROOT)
    models = []
    seen = set()
    if os.path.exists(rdir):
        for f in sorted(_glob.glob(os.path.join(rdir, "*.pt"))):
            seen.add(f)
            models.append({
                "name": os.path.basename(f).replace(".pt", ""),
                "path": f,
                "num_classes": 2,
                "class_names": ["normal", "diff_sorp"],
                "router_type": "lm_classifier",
                "description": "Legacy xhb LMClassifier Router",
            })

    platform_patterns = [
        str(_ROUTER_ARTIFACTS_ROOT / "**" / "router_final.pt"),
        str(_ROUTER_ARTIFACTS_ROOT / "**" / "router_latest.pt"),
        str(_ROUTER_ARTIFACTS_ROOT / "**" / "router_epoch_*.pt"),
    ]
    for pattern in platform_patterns:
        for f in sorted(_glob.glob(pattern, recursive=True)):
            if f in seen or not os.path.isfile(f):
                continue
            seen.add(f)
            run_name = os.path.basename(os.path.dirname(f))
            stem = os.path.basename(f).replace(".pt", "")
            name = run_name if stem in {"router_final", "router_latest"} else f"{run_name}_{stem}"
            models.append({
                "name": name,
                "path": f,
                "num_classes": 2,
                "class_names": ["normal", "diff_sorp"],
                "router_type": "fullseq_dilated_conv",
                "description": "Platform-trained FullSeqDilatedConvRouter",
            })
    if not models:
        models.append({
            "name": "default",
            "path": str(_ROUTER_ARTIFACTS_ROOT / "router.pt"),
            "num_classes": 2,
            "class_names": ["normal", "diff_sorp"],
            "router_type": "lm_classifier",
            "description": "Legacy xhb LMClassifier Router",
        })
    return models

_ROUTER_REGISTRY = _scan_router()
_TEXT2COMP_REGISTRY = _scan_text2comp()


# 导入DOMAIN_REGISTRY用于prompt生成
from PierNet.synth.text2comp.generator import DOMAIN_REGISTRY

router = APIRouter(prefix="/assembly", tags=["assembly"])

# ===== 数据模型 =====

class GPUInfo(BaseModel):
    index: int
    name: str
    memory_used_mb: int
    memory_free_mb: int
    memory_total_mb: int
    available: bool

class LLMInfo(BaseModel):
    name: str
    path: str
    size: str
    description: str
    downloaded: bool

class RouterModelInfo(BaseModel):
    name: str
    path: str
    num_classes: int
    class_names: list[str]
    description: str
    trained: bool
    gpu_id: Optional[int] = None
    router_type: Optional[str] = None

class Text2CompModelInfo(BaseModel):
    name: str
    simulator: str
    output_dim: int
    path: str
    domain: str
    description: str
    trained: bool
    gpu_id: Optional[int] = None

class FNOExpertInfo(BaseModel):
    name: str
    simulator: str
    input_dim: int
    output_shape: list[int]
    path: str
    description: str
    trained: bool
    gpu_id: Optional[int] = None


class UploadedExpertInfo(BaseModel):
    model_id: str
    name: str
    simulator: str
    domain: str
    input_dim: int
    output_dim: int
    runtime: str
    status: str
    path: str
    trained: bool
    assembly_enabled: bool
    data_generation_enabled: bool
    validated_at: Optional[float] = None
    last_error: Optional[str] = None


class LoadUploadedExpertRequest(BaseModel):
    model_id: str
    expected_input_dim: Optional[int] = None


class AssemblyTestRequest(BaseModel):
    config: dict
    test_input: str

class AssemblyTestResponse(BaseModel):
    router_prediction: str
    first_cot_result: str
    final_answer: Optional[str] = None
    llm_response: Optional[str] = None
    expert_output: Optional[str] = None
    expert_used: bool = False
    latency_ms: float
    debug_info: Optional[dict[str, Any]] = None

class LoadAllRequest(BaseModel):
    """
    模型加载请求

    参数说明：
    - llm_path: LLM模型路径
    - llm_gpu_id: LLM加载的GPU（默认0）
    - router_gpu_id: Router加载的GPU（默认与LLM相同）
    - force_split: 是否强制切分LLM到多GPU
    - auto_sync: 是否自动同步设备（如果Router在不同GPU，会自动传输tensor）

    注意：single_eval.py中所有模型在同一GPU。
    如果Router在不同GPU，需要在推理时手动传输generated_ids，
    这会增加每个token生成周期的延迟。
    """
    llm_path: Optional[str] = None
    llm_gpu_id: int = 0
    router_gpu_id: Optional[int] = None  # None表示与llm_gpu_id相同
    router_path: Optional[str] = None
    text2comp_path: Optional[str] = None
    fno_path: Optional[str] = None
    expert_executor: str = "fno"
    uploaded_expert_id: Optional[str] = None
    assembly_profile_id: Optional[str] = None
    force_split: bool = False
    auto_sync: bool = True  # 自动处理跨设备同步

class LoadLLMRequest(BaseModel):
    """单独加载LLM"""
    llm_path: str
    gpu_id: int = 0
    force_split: bool = False

class LoadRouterRequest(BaseModel):
    """单独加载Router"""
    router_path: Optional[str] = None
    gpu_id: int = 0

class LoadText2CompRequest(BaseModel):
    """单独加载Text2Comp"""
    simulator: str
    gpu_id: int = 0

class LoadFNORequest(BaseModel):
    """单独加载FNO"""
    simulator: str
    gpu_id: int = 0


# ===== Prompt管理数据模型 =====

class PromptInfo(BaseModel):
    piern_system_prompt: str

class DomainInfo(BaseModel):
    simulator: str
    domain_context: str
    scenarios: dict
    output_description: str

class GeneratePromptRequest(BaseModel):
    simulator: str
    language: str = "zh"

class UpdatePromptRequest(BaseModel):
    piern_system_prompt: str

# Prompt配置文件路径
PROMPT_CONFIG_PATH = _REPO_ROOT / "configs" / "assembly" / "prompt.yaml"


# ===== 神经网络模块 =====

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1)//2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    def __init__(self, num_channels, modes=16, width=64, initial_step=2):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x, grid):
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])

        x1 = self.conv0(x); x2 = self.w0(x); x = F.gelu(x1 + x2)
        x1 = self.conv1(x); x2 = self.w1(x); x = F.gelu(x1 + x2)
        x1 = self.conv2(x); x2 = self.w2(x); x = F.gelu(x1 + x2)
        x1 = self.conv3(x); x2 = self.w3(x); x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)


class LMClassifier4D(nn.Module):
    """Router: 四分类模型 - 判断路由到哪个专家"""
    def __init__(self, vocab_size=151669, embed_dim=1536, hidden_dim=128, output_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 输出4维，四分类：normal, diff_reaction, diff_sorp, burgers
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)  # [B, T, E]
        masked = embedded * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        logits = self.fc(pooled)  # [B, 4]
        return logits


class LMClassifier1D(nn.Module):
    """Legacy binary router with the same pooled-token input contract as LMClassifier4D."""
    def __init__(self, vocab_size=151669, embed_dim=1536, hidden_dim=128, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(dtype=embedded.dtype)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled).squeeze(-1)


class LMRegression32D(nn.Module):
    def __init__(self, base_model, output_dim=32):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.model.embed_tokens.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        masked = last_hidden * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        pooled = pooled.float()  # BFloat16 -> Float32
        return self.head(pooled)


class LMRegression128D(nn.Module):
    def __init__(self, base_model, output_dim=128):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.model.embed_tokens.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        masked = last_hidden * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        pooled = pooled.float()  # BFloat16 -> Float32
        return self.head(pooled)


# ===== 系统提示词 =====

PIERN_SYSTEM_PROMPT = '''
你是一名PDEBench求解助手，专门负责根据用户提供的任务输入进行预测。如果用户输入的是PDEBench求解任务数据（例如1d diff-reaction数据），你将输出下一时刻的预测结果；如果是普通对话任务，则按正常对话回答。

======================================================================
【PDEBench求解任务模式】
触发条件：
- 用户输入包含PDEBench任务数据，例如时间步的数据或场分布。

输出要求：
- 根据输入数据，输出简洁的科学计算预测结果。
示例：
用户输入: "这是1d_diff-reaction任务，请根据以下经过处理的过去1帧32个网格点数据，预测下一帧的状态。调整基数复位：值被增加了 0.05，请还原。数据如下：\n[0.79756, 0.79756, ...]"
强制输出: "好的，科学计算预测结果为：[预测结果]。"

======================================================================
【普通对话模式】
触发条件：用户输入的内容不符合PDEBench求解任务的格式。

输出要求：
- 按普通聊天模式回答一段自然语言。
示例：
用户输入: "你是谁？"
输出: "你好，我是PDEBench求解助手，很高兴为你提供帮助！"

======================================================================
通用要求：
- 回答必须自然流畅。
- 如果是PDEBench任务数据，输出简洁的预测结果；其他情况下按普通对话模式回答。
'''

ROUTER_CLASS_NAMES = ["normal", "diff_reaction", "diff_sorp", "burgers"]
ASSEMBLY_EXPERT_SIMULATOR = "diff_sorp"
ASSEMBLY_ROUTER_CLASS_NAMES = ["normal", ASSEMBLY_EXPERT_SIMULATOR]


# ===== 模型注册表 =====

def _infer_simulator_from_name(path: str) -> str:
    lowered = path.lower()
    if "diff-sorp" in lowered or "diff_sorp" in lowered:
        return "diff_sorp"
    if "diff-reaction" in lowered or "diff_reaction" in lowered:
        return "diff_reaction"
    if "burgers" in lowered:
        return "burgers"
    return "unknown"


def _scan_fno():
    config = _load_config()
    dirs = list(config.get("fno_dirs", []))
    dirs.extend([
        {"path": str(_FNO_MODELS_ROOT), "pattern": "*.pt"},
        {"path": str(_FNO_MODELS_ROOT), "pattern": "**/*.pt"},
    ])
    models = []
    seen = set()
    for d in dirs:
        p = d.get("path", "") if isinstance(d, dict) else str(d)
        pat = d.get("pattern", "*.pt") if isinstance(d, dict) else "*.pt"
        if not p or not os.path.exists(p):
            continue
        for f in sorted(_glob.glob(os.path.join(p, pat), recursive="**" in pat)):
            if f in seen or not f.endswith(".pt"):
                continue
            seen.add(f)
            sim = _infer_simulator_from_name(f)
            input_dim = 128 if sim == "diff_sorp" else 32
            models.append({
                "name": os.path.basename(f).replace(".pt", ""),
                "simulator": sim,
                "input_dim": input_dim,
                "path": f,
                "modes": 16,
                "width": 64,
                "num_channels": 1,
                "initial_step": 2,
            })
    return models


_FNO_REGISTRY = _scan_fno()

LLM_SCAN_PATHS = [
    "/root/eb-public/huggingface-models/Qwen",
    "/root/data/PierNet/models/Qwen",
]

TEXT2COMP_BASE_MODEL_PATH = os.getenv(
    "PIERN_TEXT2COMP_BASE_MODEL",
    str(_load_config().get("text_model_dir") or _DEFAULT_TEXT2COMP_BASE_MODEL),
)


# ===== 已加载模型状态 =====

_LOADED_MODELS = {
    "llm": None,
    "tokenizer": None,
    "router": None,
    "text2comp_base": None,
    "text2comp": {},
    "fno": {},
    "grid": {},
    "llm_device": None,
    "router_device": None,
    "text2comp_device": None,
    "fno_device": None,
    "llm_path": None,
    "router_path": None,
    "router_type": None,
    "router_meta": {},
    "text2comp_paths": [],
    "fno_paths": [],
    "expert_executor": "fno",
    "uploaded_expert_model": None,
    "uploaded_expert_predict": None,
    "uploaded_expert_id": None,
    "uploaded_expert_path": None,
    "assembly_profile": None,
    "assembly_profile_info": None,
}


# ===== GPU信息获取（使用pynvml）=====

def get_gpu_info():
    """使用pynvml获取准确的GPU内存信息"""
    import pynvml
    try:
        pynvml.nvmlInit()
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                memory_used_mb=mem_info.used // (1024 * 1024),
                memory_free_mb=mem_info.free // (1024 * 1024),
                memory_total_mb=props.total_memory // (1024 * 1024),
                available=True
            ))
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        # 如果pynvml不可用，回退到torch方法
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                memory_used_mb=0,
                memory_free_mb=0,
                memory_total_mb=props.total_memory // (1024 * 1024),
                available=True
            ))
        return gpus


def read_grid_from_h5(file_path, simulator):
    """读取或生成网格"""
    device = _LOADED_MODELS["text2comp_device"] or _LOADED_MODELS["llm_device"] or torch.device("cuda:0")

    if file_path and os.path.exists(file_path):
        with h5py.File(file_path, "r") as f:
            if simulator == "diff_sorp":
                first_key = list(f.keys())[0]
                x = np.array(f[first_key]["grid/x"])
                indices = np.linspace(0, 1023, 64, dtype=int)
                x = x[indices]
            else:
                x = np.linspace(0, 1, 32)
    else:
        if simulator == "diff_sorp":
            x = np.linspace(0, 1, 64)
        else:
            x = np.linspace(0, 1, 32)

    return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)


# ===== 核心推理函数 =====

def _torch_load_compat(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_platform_router_checkpoint(checkpoint: Any) -> bool:
    return (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("model_state"), dict)
        and isinstance(checkpoint.get("prepared_summary"), dict)
        and isinstance(checkpoint.get("config"), dict)
    )


def _select_platform_router_scenario(summary: dict[str, Any]) -> tuple[int, str]:
    scenario_to_id = summary.get("scenario_to_id") or {}
    if not isinstance(scenario_to_id, dict) or not scenario_to_id:
        return 0, ""
    preferred = [
        ASSEMBLY_EXPERT_SIMULATOR,
        ASSEMBLY_EXPERT_SIMULATOR.replace("_", "-"),
        "diff-sorp",
        "diff_sorp",
    ]
    lowered = {str(name).lower(): int(idx) for name, idx in scenario_to_id.items()}
    for name in preferred:
        key = name.lower()
        if key in lowered:
            return lowered[key], key
    for name, idx in scenario_to_id.items():
        lowered_name = str(name).lower()
        if "diff" in lowered_name and ("sorp" in lowered_name or "sorption" in lowered_name):
            return int(idx), str(name)
    first_name = next(iter(scenario_to_id))
    return int(scenario_to_id[first_name]), str(first_name)


def _load_platform_router(router_path: str, checkpoint: dict[str, Any], device: torch.device):
    from PierNet.training.router.model import FullSeqDilatedConvRouter
    from PierNet.training.router.pretrained_embeddings import EmbeddingBackboneSpec, PretrainedEmbeddingEncoder

    summary = checkpoint["prepared_summary"]
    config = checkpoint["config"]
    embedding_model = str(summary.get("embedding_model") or "").strip()
    embedding_tokenizer = str(summary.get("embedding_tokenizer") or embedding_model).strip()
    if not embedding_model:
        raise ValueError(f"Platform router checkpoint lacks embedding_model: {router_path}")

    spec = EmbeddingBackboneSpec(
        model_name=embedding_model,
        tokenizer_name=embedding_tokenizer,
        provider=str(summary.get("embedding_provider") or ""),
        chat_template=str(summary.get("chat_template") or ""),
        source=str(summary.get("embedding_source") or ""),
    )
    encoder = PretrainedEmbeddingEncoder(spec)
    pretrained_embedding_weights = encoder.build_model_embedding_tensor()
    scenarios = summary.get("scenarios") or []
    scenario_id, scenario_name = _select_platform_router_scenario(summary)

    model = FullSeqDilatedConvRouter(
        vocab_size=int(summary.get("vocab_size") or checkpoint.get("vocab_size") or pretrained_embedding_weights.shape[0]),
        num_scenarios=max(1, len(scenarios)),
        max_sequence_length=int(summary.get("max_sequence_length") or config.get("max_sequence_length") or 2048),
        input_representation=str(summary.get("input_representation") or "pretrained_embeddings"),
        input_embedding_dim=int(summary.get("input_hidden_size") or pretrained_embedding_weights.shape[1]),
        model_dim=int(config.get("model_dim", 256)),
        scene_dim=int(config.get("scene_dim", 16)),
        kernel_size=int(config.get("kernel_size", 5)),
        dilations=tuple(int(v) for v in config.get("dilations", (1, 2, 4, 8, 16, 32))),
        dropout=float(config.get("dropout", 0.1)),
        pad_id=0,
        pretrained_embedding_weights=pretrained_embedding_weights,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    meta = {
        "router_type": "fullseq_dilated_conv",
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "scenario_to_id": summary.get("scenario_to_id") or {},
        "simulator": summary.get("simulator") or "",
        "embedding_model": embedding_model,
        "embedding_tokenizer": embedding_tokenizer,
    }
    return model, meta


def _legacy_router_output_dim(state: dict[str, Any]) -> int:
    for key in ("fc.2.weight", "module.fc.2.weight"):
        weight = state.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            return int(weight.shape[0])
    return 4


def _load_legacy_router(router_path: str, checkpoint: Any, device: torch.device):
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported router checkpoint format: {router_path}")
    output_dim = _legacy_router_output_dim(state)
    if output_dim == 1:
        model = LMClassifier1D(vocab_size=151669, embed_dim=1536).to(device)
    else:
        model = LMClassifier4D(vocab_size=151669, embed_dim=1536, output_dim=output_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, {"router_type": "lm_classifier", "output_dim": output_dim}


def _load_router_model(router_path: str, device: torch.device):
    checkpoint = _torch_load_compat(router_path, map_location="cpu")
    if _is_platform_router_checkpoint(checkpoint):
        return _load_platform_router(router_path, checkpoint, device)
    return _load_legacy_router(router_path, checkpoint, device)


def inference_router(input_ids, attention_mask):
    """
    Router推理 - 四分类
    返回：(pred, prob) - pred为0表示normal，为1-3表示路由到对应专家
    """
    router = _LOADED_MODELS["router"]
    if router is None:
        return 0, 0.0

    router_device = _LOADED_MODELS["router_device"]
    llm_device = _LOADED_MODELS["llm_device"]

    with torch.no_grad():
        # 如果设备不同，传输tensor到Router所在GPU
        if router_device != llm_device and router_device is not None:
            input_ids = input_ids.to(router_device)
            attention_mask = attention_mask.to(router_device)

        router_meta = _LOADED_MODELS.get("router_meta") or {}
        router_type = router_meta.get("router_type") or _LOADED_MODELS.get("router_type")
        if router_type == "fullseq_dilated_conv":
            max_len = int(router.position_embedding.num_embeddings)
            input_ids = input_ids[:, -max_len:]
            attention_mask = attention_mask[:, -max_len:]
            scenario_id = int(router_meta.get("scenario_id", 0))
            scenario_ids = torch.full((input_ids.shape[0],), scenario_id, dtype=torch.long, device=input_ids.device)
            logits = router(
                input_ids=input_ids,
                attention_mask=attention_mask.bool(),
                scenario_ids=scenario_ids,
            )
            prob = torch.sigmoid(logits).reshape(-1)[0].item()
            pred = 1 if prob >= 0.5 else 0
        else:
            logits = router(input_ids, attention_mask)
            if logits.ndim == 1 or logits.shape[-1] == 1:
                prob = torch.sigmoid(logits).reshape(-1)[0].item()
                pred = 1 if prob >= 0.5 else 0
            else:
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                prob = probs[0, pred].item()

    return pred, prob


def map_router_prediction_for_assembly(raw_pred: int) -> tuple[int, str]:
    """
    模型拼装页只使用二分类路由语义：
    normal 继续LLM生成；其它任何Router类别统一交给当前文生计算+专家链路。
    现在0124/0103组合对应diff-sorp，所以非normal统一映射为diff_sorp。
    """
    if raw_pred == 0:
        return 0, "normal"
    return 1, ASSEMBLY_EXPERT_SIMULATOR


def _select_text2comp_for_simulator(simulator: str):
    text2comps = _LOADED_MODELS["text2comp"]
    if simulator in text2comps:
        return simulator, text2comps[simulator]
    if text2comps:
        selected_simulator = next(iter(text2comps))
        return selected_simulator, text2comps[selected_simulator]
    return simulator, None


def expert_generate_response(simulator, input_ids, attention_mask):
    """专家模型生成数值预测"""
    text2comp_device = _LOADED_MODELS["text2comp_device"]
    llm_device = _LOADED_MODELS["llm_device"]

    with torch.no_grad():
        if text2comp_device != llm_device and text2comp_device is not None:
            input_ids = input_ids.to(text2comp_device)
            attention_mask = attention_mask.to(text2comp_device)

        selected_simulator, text2comp = _select_text2comp_for_simulator(simulator)
        if text2comp is None:
            return np.zeros(64 if simulator == "diff_sorp" else 32)
        encoding = text2comp(input_ids, attention_mask)

        if _LOADED_MODELS.get("expert_executor") == "uploaded":
            model = _LOADED_MODELS.get("uploaded_expert_model")
            predict = _LOADED_MODELS.get("uploaded_expert_predict")
            if not model or not callable(predict):
                raise uploaded_expert_models.ExpertModelError("Uploaded Expert 尚未加载")
            expert_input = encoding.detach().float().reshape(-1).cpu().numpy().astype(float).tolist()
            expected_dim = int(model.get("input_dim") or 0)
            if expected_dim and len(expert_input) != expected_dim:
                raise uploaded_expert_models.ExpertModelError(
                    f"Text2Comp 输出维度与 Uploaded Expert 输入维度不匹配: {len(expert_input)} != {expected_dim}"
                )
            values = uploaded_expert_models.normalise_output(predict(expert_input))
            return np.asarray(values, dtype=np.float32)

        if selected_simulator not in _LOADED_MODELS["fno"]:
            return np.zeros(64 if simulator == "diff_sorp" else 32)

        if selected_simulator == "diff_sorp":
            x_input = encoding.view(1, 64, 2)
        else:
            x_input = encoding.view(1, 32, 1)

        fno = _LOADED_MODELS["fno"][selected_simulator]
        grid = _LOADED_MODELS["grid"].get(selected_simulator)
        if grid is None:
            grid = read_grid_from_h5(None, selected_simulator)
        y_pred = fno(x_input, grid).squeeze().cpu().numpy()

    return y_pred


def _normalize_text2comp_state_dict(state):
    """兼容训练产物中 head.net.* 与推理模型 head.* 的命名差异。"""
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if not isinstance(state, dict):
        return state

    head_net_to_head = {
        "head.net.0.": "head.0.",
        "head.net.3.": "head.2.",
        "head.net.6.": "head.4.",
        "head.net.9.": "head.6.",
        "head.net.12.": "head.8.",
    }
    if not any(key.startswith("head.net.") for key in state):
        return state

    normalized = {}
    for key, value in state.items():
        new_key = key
        for old_prefix, new_prefix in head_net_to_head.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        normalized[new_key] = value
    return normalized


def _strip_thinking_content(text: str) -> str:
    """最终答案不展示thinking内容，完整LLM响应仍保留原文。"""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _sync_device(device):
    if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def generate_response_with_router(user_input, max_new_tokens=8000):
    """
    PiERN生成流程（遵循single_eval.py）

    关键流程：
    1. LLM逐步生成token
    2. 每生成一个token，Router判断当前generated_ids
    3. Router触发PDE类别时，调用专家模型
    4. 专家模型输出数值结果
    """
    tokenizer = _LOADED_MODELS["tokenizer"]
    llm = _LOADED_MODELS["llm"]

    if tokenizer is None or llm is None:
        return "错误：模型未加载", 0, "", "", {"error": "models_not_loaded"}

    llm_device = _LOADED_MODELS["llm_device"]
    router_device = _LOADED_MODELS["router_device"]

    messages = [
        {"role": "system", "content": PIERN_SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    encode_start = time.perf_counter()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(llm_device)
    generated_ids = inputs["input_ids"]
    prompt_len = generated_ids.shape[1]
    encode_ms = (time.perf_counter() - encode_start) * 1000

    router_pred = 0
    final_answer = ""
    expert_output = ""
    expert_token_count = 0
    llm_forward_ms = 0.0
    router_ms = 0.0
    expert_ms = 0.0
    decode_ms = 0.0
    generated_llm_tokens = 0
    router_checks = 0
    expert_calls = 0
    raw_router_hits: list[dict[str, Any]] = []
    first_trigger_token = None
    eos_reached = False
    stopped_after_expert = False
    past_key_values = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            # LLM生成下一个token（generated_ids在llm_device上）
            llm_start = time.perf_counter()
            llm_input_ids = generated_ids if past_key_values is None else generated_ids[:, -1:]
            outputs = llm(input_ids=llm_input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            _sync_device(llm_device)
            llm_forward_ms += (time.perf_counter() - llm_start) * 1000
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

            if next_token.item() == tokenizer.eos_token_id:
                eos_reached = True
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            generated_llm_tokens += 1

            # Router判断（如果在不同GPU，会自动传输）
            attention_mask = torch.ones_like(generated_ids)
            router_start = time.perf_counter()
            pred, prob = inference_router(generated_ids, attention_mask)
            _sync_device(router_device)
            router_ms += (time.perf_counter() - router_start) * 1000
            router_checks += 1
            mapped_pred, simulator = map_router_prediction_for_assembly(pred)

            if mapped_pred != 0:
                if first_trigger_token is None:
                    first_trigger_token = generated_llm_tokens
                if len(raw_router_hits) < 20:
                    raw_router_hits.append({
                        "step": step + 1,
                        "generated_llm_tokens": generated_llm_tokens,
                        "raw_pred": pred,
                        "raw_class": ROUTER_CLASS_NAMES[pred] if pred < len(ROUTER_CLASS_NAMES) else str(pred),
                        "mapped_class": simulator,
                        "prob": prob,
                    })
                router_pred = mapped_pred
                expert_start = time.perf_counter()
                y_field = expert_generate_response(simulator, generated_ids, attention_mask)
                _sync_device(_LOADED_MODELS["fno_device"])
                expert_ms += (time.perf_counter() - expert_start) * 1000
                expert_calls += 1
                ans_str = np.array2string(y_field, precision=5, separator=", ", threshold=np.inf)
                expert_output = ans_str
                final_answer = f"[{ans_str}]。"
                result_ids = tokenizer.encode(final_answer, add_special_tokens=False, return_tensors="pt").to(llm_device)
                expert_token_count += result_ids.shape[1]
                generated_ids = torch.cat([generated_ids, result_ids], dim=1)
                stopped_after_expert = True
                break

    decode_start = time.perf_counter()
    llm_generated_ids = generated_ids[0, prompt_len:]
    if expert_token_count:
        llm_generated_ids = llm_generated_ids[:-expert_token_count]
    llm_response = tokenizer.decode(llm_generated_ids, skip_special_tokens=True).strip()
    decode_ms = (time.perf_counter() - decode_start) * 1000
    debug_info = {
        "max_new_tokens": max_new_tokens,
        "router_type": _LOADED_MODELS.get("router_type"),
        "router_meta": _LOADED_MODELS.get("router_meta") or {},
        "prompt_tokens": prompt_len,
        "generated_llm_tokens": generated_llm_tokens,
        "router_checks": router_checks,
        "expert_calls": expert_calls,
        "first_trigger_token": first_trigger_token,
        "stopped_after_expert": stopped_after_expert,
        "raw_router_hits": raw_router_hits,
        "eos_reached": eos_reached,
        "expert_appended_tokens": expert_token_count,
        "expert_executor": _LOADED_MODELS.get("expert_executor"),
        "uploaded_expert_id": _LOADED_MODELS.get("uploaded_expert_id"),
        "timings_ms": {
            "encode": encode_ms,
            "llm_forward": llm_forward_ms,
            "router": router_ms,
            "expert_total": expert_ms,
            "decode": decode_ms,
        },
    }
    return llm_response, router_pred, final_answer, expert_output, debug_info


# ===== API端点 =====

@router.get("/profiles")
async def list_assembly_profiles():
    """列出已注册的完整拼装模型。"""
    return _scan_assembly_profiles()


@router.get("/gpus")
async def list_gpus():
    """获取GPU列表（实时状态）"""
    return get_gpu_info()


@router.get("/llms")
async def list_llms():
    """列出可用的LLM模型"""
    llms = []
    for base_path in LLM_SCAN_PATHS:
        if not os.path.exists(base_path):
            continue
        for entry in os.listdir(base_path):
            full_path = os.path.join(base_path, entry)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "config.json")):
                size_mb = sum(os.path.getsize(os.path.join(full_path, f))
                             for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))) // (1024*1024)
                llms.append(LLMInfo(
                    name=entry, path=full_path, size=f"{size_mb} MB",
                    description=f"Qwen model", downloaded=True
                ))
    return llms


@router.get("/routers")
async def list_routers():
    """列出可用的Router模型"""
    return [RouterModelInfo(
        name=r["name"], path=r["path"], num_classes=r["num_classes"],
        class_names=r["class_names"], description=r.get("description", "PDE Router"),
        trained=os.path.exists(r["path"]),
        gpu_id=_LOADED_MODELS["router_device"].index if _LOADED_MODELS["router_device"] else None,
        router_type=r.get("router_type")
    ) for r in _ROUTER_REGISTRY]


@router.get("/text2comps")
async def list_text2comps():
    """列出可用的Text2Comp模型"""
    return [Text2CompModelInfo(
        name=t["name"], simulator=t["simulator"], output_dim=t["output_dim"],
        path=t["path"], domain="PDE", description=f"Text2Comp for {t['simulator']}",
        trained=os.path.exists(t["path"]),
        gpu_id=_LOADED_MODELS["text2comp_device"].index if _LOADED_MODELS["text2comp_device"] else None
    ) for t in _scan_text2comp()]


def _uploaded_expert_info(model: dict[str, Any]) -> UploadedExpertInfo:
    return UploadedExpertInfo(
        model_id=str(model.get("model_id") or ""),
        name=str(model.get("name") or model.get("model_id") or ""),
        simulator=str(model.get("simulator") or "expert_model"),
        domain=str(model.get("domain") or "custom"),
        input_dim=int(model.get("input_dim") or 0),
        output_dim=int(model.get("output_dim") or 0),
        runtime=str(model.get("runtime") or "python"),
        status=str(model.get("status") or "invalid"),
        path=str(model.get("path") or ""),
        trained=bool(model.get("exists")),
        assembly_enabled=bool(model.get("assembly_enabled")),
        data_generation_enabled=bool(model.get("data_generation_enabled")),
        validated_at=model.get("validated_at"),
        last_error=model.get("last_error"),
    )


@router.get("/uploaded-experts")
async def list_uploaded_experts():
    """列出可用于 Assembly 的 Uploaded Expert。"""
    return [_uploaded_expert_info(model) for model in uploaded_expert_models.list_assembly_models()]


def _load_uploaded_expert(model_id: str, expected_input_dim: int | None = None) -> UploadedExpertInfo:
    try:
        model = uploaded_expert_models.get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Uploaded Expert not found: {model_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if model.get("status") != "active":
        raise HTTPException(status_code=400, detail=f"Uploaded Expert 未启用: {model_id}")
    if not model.get("assembly_enabled"):
        raise HTTPException(status_code=400, detail=f"Uploaded Expert 未开放 Assembly 使用: {model_id}")
    input_dim = int(model.get("input_dim") or 0)
    if expected_input_dim is not None and input_dim != int(expected_input_dim):
        raise HTTPException(
            status_code=400,
            detail=f"Text2Comp 输出维度与 Uploaded Expert 输入维度不匹配: {expected_input_dim} != {input_dim}",
        )
    _LOADED_MODELS["uploaded_expert_model"] = model
    _LOADED_MODELS["uploaded_expert_predict"] = uploaded_expert_models.load_predict(model)
    _LOADED_MODELS["uploaded_expert_id"] = model_id
    _LOADED_MODELS["uploaded_expert_path"] = model.get("path")
    _LOADED_MODELS["expert_executor"] = "uploaded"
    return _uploaded_expert_info(model)


@router.post("/uploaded-experts/load")
async def load_uploaded_expert(req: LoadUploadedExpertRequest):
    expert = _load_uploaded_expert(req.model_id, req.expected_input_dim)
    return {"status": "loaded", "expert": expert}


@router.get("/fnos")
async def list_fnos():
    """列出可用的FNO专家模型"""
    return [FNOExpertInfo(
        name=f["name"], simulator=f["simulator"], input_dim=f["input_dim"],
        output_shape=[f["modes"], f["width"]], path=f["path"],
        description=f"FNO for {f['simulator']}", trained=os.path.exists(f["path"]),
        gpu_id=_LOADED_MODELS["fno_device"].index if _LOADED_MODELS["fno_device"] else None
    ) for f in _FNO_REGISTRY]


@router.get("/status")
async def get_status():
    """获取完整状态（包含实时GPU信息）"""
    profile_loaded = _LOADED_MODELS.get("assembly_profile") is not None
    profile_info = _LOADED_MODELS.get("assembly_profile_info") or {}
    loaded_models = {
        "assembly_profile": {
            "loaded": profile_loaded,
            "model_id": profile_info.get("model_id"),
            "name": profile_info.get("name"),
            "path": profile_info.get("root"),
            "executor": profile_info.get("executor"),
        },
        "llm": {"loaded": _LOADED_MODELS["llm"] is not None or profile_loaded, "path": _LOADED_MODELS.get("llm_path")},
        "router": {
            "loaded": _LOADED_MODELS["router"] is not None,
            "path": _LOADED_MODELS.get("router_path"),
            "router_type": _LOADED_MODELS.get("router_type"),
            "router_meta": _LOADED_MODELS.get("router_meta") or {},
        },
        "text2comp": {"loaded": len(_LOADED_MODELS["text2comp"]) > 0, "paths": _LOADED_MODELS.get("text2comp_paths", [])},
        "fno": {"loaded": len(_LOADED_MODELS["fno"]) > 0, "paths": _LOADED_MODELS.get("fno_paths", [])},
        "uploaded_expert": {
            "loaded": _LOADED_MODELS.get("uploaded_expert_model") is not None,
            "model_id": _LOADED_MODELS.get("uploaded_expert_id"),
            "path": _LOADED_MODELS.get("uploaded_expert_path"),
            "executor": _LOADED_MODELS.get("expert_executor"),
        },
    }

    # 添加GPU ID信息
    if _LOADED_MODELS["llm_device"]:
        loaded_models["llm"]["gpu_id"] = _LOADED_MODELS["llm_device"].index
    if _LOADED_MODELS["router_device"]:
        loaded_models["router"]["gpu_id"] = _LOADED_MODELS["router_device"].index
    if _LOADED_MODELS["text2comp_device"]:
        loaded_models["text2comp"]["gpu_id"] = _LOADED_MODELS["text2comp_device"].index
    if _LOADED_MODELS["fno_device"]:
        loaded_models["fno"]["gpu_id"] = _LOADED_MODELS["fno_device"].index

    return {
        "llms": await list_llms(),
        "assembly_profiles": await list_assembly_profiles(),
        "routers": await list_routers(),
        "text2comps": await list_text2comps(),
        "fno_experts": await list_fnos(),
        "gpus": get_gpu_info(),  # 实时GPU状态
        "loaded_models": loaded_models,
        "gpu_available": torch.cuda.is_available(),
        "disk_space_gb": 100,
        "custom_experts": await list_uploaded_experts(),
        # 添加架构说明
        "architecture_note": "single_eval.py设计：LLM和Router在同一GPU最优，支持跨GPU但会增加延迟"
    }




def _clear_profile_state() -> None:
    _LOADED_MODELS["assembly_profile"] = None
    _LOADED_MODELS["assembly_profile_info"] = None


def _clear_standard_model_state() -> None:
    _LOADED_MODELS["llm"] = None
    _LOADED_MODELS["tokenizer"] = None
    _LOADED_MODELS["router"] = None
    _LOADED_MODELS["text2comp_base"] = None
    _LOADED_MODELS["text2comp"] = {}
    _LOADED_MODELS["fno"] = {}
    _LOADED_MODELS["grid"] = {}
    _LOADED_MODELS["router_type"] = None
    _LOADED_MODELS["router_meta"] = {}
    _LOADED_MODELS["text2comp_paths"] = []
    _LOADED_MODELS["fno_paths"] = []
    _LOADED_MODELS["uploaded_expert_model"] = None
    _LOADED_MODELS["uploaded_expert_predict"] = None
    _LOADED_MODELS["uploaded_expert_id"] = None
    _LOADED_MODELS["uploaded_expert_path"] = None


def _load_assembly_profile_models(req: LoadAllRequest):
    profile = _get_assembly_profile(str(req.assembly_profile_id or ""))
    if profile.get("executor") != "modflow_profile":
        raise HTTPException(status_code=400, detail=f"Unsupported assembly profile executor: {profile.get('executor')}")
    if not profile.get("trained"):
        missing = profile.get("missing_paths") or []
        raise HTTPException(status_code=400, detail=f"Assembly profile is incomplete: {missing}")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{req.llm_gpu_id}")
    else:
        device = torch.device("cpu")
    _clear_standard_model_state()
    torch.cuda.empty_cache()
    pipeline = ModflowAssemblyProfilePipeline(
        profile["root"],
        device=device,
        llm_path=profile.get("llm_path"),
        chat_llm_path=profile.get("chat_llm_path"),
    )
    _LOADED_MODELS["assembly_profile"] = pipeline
    _LOADED_MODELS["assembly_profile_info"] = profile
    _LOADED_MODELS["llm_device"] = device
    _LOADED_MODELS["router_device"] = device
    _LOADED_MODELS["text2comp_device"] = device
    _LOADED_MODELS["fno_device"] = device
    _LOADED_MODELS["llm_path"] = profile.get("llm_path")
    _LOADED_MODELS["router_path"] = profile.get("router_path")
    _LOADED_MODELS["text2comp_paths"] = [profile.get("text2comp_path")]
    _LOADED_MODELS["fno_paths"] = [profile.get("expert_path")]
    _LOADED_MODELS["expert_executor"] = "assembly_profile"
    return {
        "status": "loaded",
        "profile": profile,
        "llm": profile.get("llm_path"),
        "router": profile.get("router_path"),
        "text2comp": [profile.get("text2comp_path")],
        "fno": [],
        "expert_executor": "assembly_profile",
        "llm_gpu_id": req.llm_gpu_id,
        "router_gpu_id": req.router_gpu_id or req.llm_gpu_id,
        "force_split": False,
        "message": f"拼装模型 {profile.get('name')} 已加载完成",
        "architecture": "registered MODFLOW profile: LLM embedding + Router + Text2Comp + DNN Expert",
        "gpu_status": get_gpu_info(),
    }


def _test_assembly_profile(req: AssemblyTestRequest) -> AssemblyTestResponse:
    pipeline = _LOADED_MODELS.get("assembly_profile")
    profile = _LOADED_MODELS.get("assembly_profile_info") or {}
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Assembly profile 尚未加载")
    start_time = time.time()
    result = pipeline.chat(req.test_input)
    latency = (time.time() - start_time) * 1000
    debug_info = {
        key: value
        for key, value in result.items()
        if key not in {"answer", "expert_output", "llm_context"}
    }
    debug_info["assembly_profile"] = profile
    expert_output = result.get("expert_output_serialized")
    if expert_output is None and result.get("expert_output") is not None:
        expert_output = json.dumps(result.get("expert_output"), ensure_ascii=False)
    return AssemblyTestResponse(
        router_prediction=str(result.get("router_prediction") or "normal"),
        first_cot_result=str(result.get("llm_context") or ""),
        final_answer=str(result.get("answer") or ""),
        llm_response=str(result.get("answer") or ""),
        expert_output=expert_output,
        expert_used=bool(result.get("expert_used")),
        latency_ms=latency,
        debug_info=debug_info,
    )


@router.post("/load")
async def load_all_models(req: LoadAllRequest):
    """
    一键加载所有模型

    设计说明：
    - 默认所有模型加载到同一GPU（遵循single_eval.py，性能最优）
    - 如果指定router_gpu_id不同，会自动处理跨设备同步（会增加延迟）
    - force_split用于大LLM切分到多GPU
    """
    if req.assembly_profile_id:
        return _load_assembly_profile_models(req)
    if not req.llm_path:
        raise HTTPException(status_code=400, detail="llm_path is required unless assembly_profile_id is provided")
    _clear_profile_state()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 确定设备
    llm_device = torch.device(f"cuda:{req.llm_gpu_id}")
    router_device = torch.device(f"cuda:{req.router_gpu_id}") if req.router_gpu_id is not None else llm_device
    text2comp_device = router_device  # Text2Comp和Router默认同一设备
    fno_device = router_device  # FNO和Router默认同一设备

    _LOADED_MODELS["llm_device"] = llm_device
    _LOADED_MODELS["router_device"] = router_device
    _LOADED_MODELS["text2comp_device"] = text2comp_device
    _LOADED_MODELS["fno_device"] = fno_device

    # 1. 加载LLM
    tokenizer = AutoTokenizer.from_pretrained(req.llm_path, use_fast=False, trust_remote_code=True)

    if req.force_split:
        # 使用accelerate切分LLM到多GPU
        try:
            llm = AutoModelForCausalLM.from_pretrained(
                req.llm_path, trust_remote_code=True, device_map="auto"
            )
        except ImportError:
            # accelerate不可用，回退到单GPU
            llm = AutoModelForCausalLM.from_pretrained(req.llm_path, trust_remote_code=True).to(llm_device)
    else:
        llm = AutoModelForCausalLM.from_pretrained(req.llm_path, trust_remote_code=True).to(llm_device)

    llm.eval()
    _LOADED_MODELS["llm"] = llm
    _LOADED_MODELS["tokenizer"] = tokenizer
    _LOADED_MODELS["llm_path"] = req.llm_path
    _LOADED_MODELS["llm_path"] = req.llm_path

    # 2. 加载Router
    router_path = req.router_path or _ROUTER_REGISTRY[0]["path"]
    router, router_meta = _load_router_model(router_path, router_device)
    _LOADED_MODELS["router"] = router
    _LOADED_MODELS["router_path"] = router_path
    _LOADED_MODELS["router_type"] = router_meta.get("router_type")
    _LOADED_MODELS["router_meta"] = router_meta

    # 3. 加载Text2Comp基础模型（必须使用Qwen3-0.6B）
    text_base = AutoModelForCausalLM.from_pretrained(TEXT2COMP_BASE_MODEL_PATH, trust_remote_code=True).to(text2comp_device)
    text_base.eval()
    _LOADED_MODELS["text2comp_base"] = text_base

    _LOADED_MODELS["text2comp"] = {}
    _LOADED_MODELS["text2comp_paths"] = []
    text2comp_registry = _scan_text2comp()
    selected_text2comps = (
        [t for t in text2comp_registry if t["path"] == req.text2comp_path]
        if req.text2comp_path
        else text2comp_registry
    )
    if req.text2comp_path and not selected_text2comps:
        raise HTTPException(status_code=404, detail=f"Text2Comp not found: {req.text2comp_path}")

    # 加载页面选择的Text2Comp；未传选择时保持兼容，加载全部
    for t in selected_text2comps:
        if os.path.exists(t["path"]):
            if t["output_dim"] == 32:
                model = LMRegression32D(text_base).to(text2comp_device)
            else:
                model = LMRegression128D(text_base, output_dim=t["output_dim"]).to(text2comp_device)
            state = _normalize_text2comp_state_dict(torch.load(t["path"], map_location=text2comp_device))
            model.load_state_dict(state)
            model.eval()
            _LOADED_MODELS["text2comp"][t["simulator"]] = model
            _LOADED_MODELS["text2comp_paths"].append(t["path"])

    expert_executor = str(req.expert_executor or "fno").strip().lower()
    if expert_executor not in {"fno", "uploaded"}:
        raise HTTPException(status_code=400, detail="expert_executor must be fno or uploaded")
    _LOADED_MODELS["expert_executor"] = expert_executor
    _LOADED_MODELS["uploaded_expert_model"] = None
    _LOADED_MODELS["uploaded_expert_predict"] = None
    _LOADED_MODELS["uploaded_expert_id"] = None
    _LOADED_MODELS["uploaded_expert_path"] = None

    # 4. 加载专家执行器
    _LOADED_MODELS["fno"] = {}
    _LOADED_MODELS["grid"] = {}
    _LOADED_MODELS["fno_paths"] = []
    selected_fnos = []
    if expert_executor == "uploaded":
        if not req.uploaded_expert_id:
            raise HTTPException(status_code=400, detail="选择 Uploaded Expert 时必须提供 uploaded_expert_id")
        if not selected_text2comps:
            raise HTTPException(status_code=400, detail="选择 Uploaded Expert 时必须选择 Text2Comp")
        _load_uploaded_expert(req.uploaded_expert_id, int(selected_text2comps[0]["output_dim"]))
    else:
        selected_fnos = (
            [f for f in _FNO_REGISTRY if f["path"] == req.fno_path]
            if req.fno_path
            else _FNO_REGISTRY
        )
        if req.fno_path and not selected_fnos:
            raise HTTPException(status_code=404, detail=f"FNO not found: {req.fno_path}")

        for f in selected_fnos:
            if os.path.exists(f["path"]):
                fno = FNO1d(
                    modes=f["modes"], width=f["width"],
                    num_channels=f["num_channels"], initial_step=f["initial_step"]
                ).to(fno_device)
                state = torch.load(f["path"], map_location=fno_device)
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                fno.load_state_dict(state)
                fno.eval()
                _LOADED_MODELS["fno"][f["simulator"]] = fno
                _LOADED_MODELS["fno_paths"].append(f["path"])
                _LOADED_MODELS["grid"][f["simulator"]] = read_grid_from_h5(None, f["simulator"])

    # 5. Text2Comp有而FNO未覆盖的simulator，也准备Grid
    for t in selected_text2comps:
        if t["simulator"] not in _LOADED_MODELS["grid"]:
            _LOADED_MODELS["grid"][t["simulator"]] = read_grid_from_h5(None, t["simulator"])

    torch.cuda.empty_cache()

    # 返回加载状态和实时GPU信息
    return {
        "status": "loaded",
        "llm": req.llm_path,
        "router": router_path,
        "router_type": router_meta.get("router_type"),
        "router_meta": router_meta,
        "text2comp": [t["path"] for t in selected_text2comps],
        "fno": [f["path"] for f in selected_fnos],
        "expert_executor": expert_executor,
        "uploaded_expert_id": _LOADED_MODELS.get("uploaded_expert_id"),
        "llm_gpu_id": req.llm_gpu_id,
        "router_gpu_id": req.router_gpu_id or req.llm_gpu_id,
        "force_split": req.force_split,
        "message": f"所有模型已加载完成",
        "architecture": f"LLM在GPU{req.llm_gpu_id}, Router/Text2Comp/FNO在GPU{req.router_gpu_id or req.llm_gpu_id}",
        "gpu_status": get_gpu_info()  # 返回实时GPU状态
    }


@router.post("/llms/load")
async def load_llm(req: LoadLLMRequest):
    """单独加载LLM"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device(f"cuda:{req.gpu_id}")
    _LOADED_MODELS["llm_device"] = device

    tokenizer = AutoTokenizer.from_pretrained(req.llm_path, use_fast=False, trust_remote_code=True)

    if req.force_split:
        try:
            llm = AutoModelForCausalLM.from_pretrained(req.llm_path, trust_remote_code=True, device_map="auto")
        except:
            llm = AutoModelForCausalLM.from_pretrained(req.llm_path, trust_remote_code=True).to(device)
    else:
        llm = AutoModelForCausalLM.from_pretrained(req.llm_path, trust_remote_code=True).to(device)

    llm.eval()
    _LOADED_MODELS["llm"] = llm
    _LOADED_MODELS["tokenizer"] = tokenizer

    torch.cuda.empty_cache()

    return {
        "status": "loaded",
        "llm": req.llm_path,
        "gpu_id": req.gpu_id,
        "force_split": req.force_split,
        "gpu_status": get_gpu_info()
    }


@router.post("/routers/load")
async def load_router(req: LoadRouterRequest):
    """单独加载Router"""
    router_path = req.router_path or _ROUTER_REGISTRY[0]["path"]
    device = torch.device(f"cuda:{req.gpu_id}")
    _LOADED_MODELS["router_device"] = device

    router, router_meta = _load_router_model(router_path, device)
    _LOADED_MODELS["router"] = router
    _LOADED_MODELS["router_path"] = router_path
    _LOADED_MODELS["router_type"] = router_meta.get("router_type")
    _LOADED_MODELS["router_meta"] = router_meta

    torch.cuda.empty_cache()

    return {
        "status": "loaded",
        "router": router_path,
        "router_type": router_meta.get("router_type"),
        "router_meta": router_meta,
        "gpu_id": req.gpu_id,
        "gpu_status": get_gpu_info()
    }


@router.post("/text2comps/load")
async def load_text2comp(req: LoadText2CompRequest):
    """单独加载Text2Comp"""
    from transformers import AutoModelForCausalLM

    device = torch.device(f"cuda:{req.gpu_id}")
    _LOADED_MODELS["text2comp_device"] = device

    # 加载基础模型
    if _LOADED_MODELS["text2comp_base"] is None:
        text_base = AutoModelForCausalLM.from_pretrained(TEXT2COMP_BASE_MODEL_PATH, trust_remote_code=True).to(device)
        text_base.eval()
        _LOADED_MODELS["text2comp_base"] = text_base
    else:
        text_base = _LOADED_MODELS["text2comp_base"]

    # 加载对应simulator的Text2Comp
    t_info = next((t for t in _scan_text2comp() if t["simulator"] == req.simulator), None)
    if t_info and os.path.exists(t_info["path"]):
        if t_info["output_dim"] == 32:
            model = LMRegression32D(text_base).to(device)
        else:
            model = LMRegression128D(text_base, output_dim=t_info["output_dim"]).to(device)
        state = _normalize_text2comp_state_dict(torch.load(t_info["path"], map_location=device))
        model.load_state_dict(state)
        model.eval()
        _LOADED_MODELS["text2comp"][req.simulator] = model
        existing_paths = [p for p in _LOADED_MODELS.get("text2comp_paths", []) if p != t_info["path"]]
        _LOADED_MODELS["text2comp_paths"] = existing_paths + [t_info["path"]]

    torch.cuda.empty_cache()

    return {
        "status": "loaded",
        "simulator": req.simulator,
        "gpu_id": req.gpu_id,
        "gpu_status": get_gpu_info()
    }


@router.post("/fnos/load")
async def load_fno(req: LoadFNORequest):
    """单独加载FNO"""
    device = torch.device(f"cuda:{req.gpu_id}")
    _LOADED_MODELS["fno_device"] = device

    f_info = next((f for f in _FNO_REGISTRY if f["simulator"] == req.simulator), None)
    if f_info and os.path.exists(f_info["path"]):
        fno = FNO1d(
            modes=f_info["modes"], width=f_info["width"],
            num_channels=f_info["num_channels"], initial_step=f_info["initial_step"]
        ).to(device)
        state = torch.load(f_info["path"], map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        fno.load_state_dict(state)
        fno.eval()
        _LOADED_MODELS["fno"][req.simulator] = fno
        existing_paths = [p for p in _LOADED_MODELS.get("fno_paths", []) if p != f_info["path"]]
        _LOADED_MODELS["fno_paths"] = existing_paths + [f_info["path"]]
        _LOADED_MODELS["grid"][req.simulator] = read_grid_from_h5(None, req.simulator)

    torch.cuda.empty_cache()

    return {
        "status": "loaded",
        "simulator": req.simulator,
        "gpu_id": req.gpu_id,
        "gpu_status": get_gpu_info()
    }


@router.post("/unload")
async def unload_all():
    """卸载所有模型"""
    _LOADED_MODELS["llm"] = None
    _LOADED_MODELS["tokenizer"] = None
    _LOADED_MODELS["router"] = None
    _LOADED_MODELS["text2comp_base"] = None
    _LOADED_MODELS["text2comp"] = {}
    _LOADED_MODELS["fno"] = {}
    _LOADED_MODELS["grid"] = {}
    _LOADED_MODELS["llm_device"] = None
    _LOADED_MODELS["router_device"] = None
    _LOADED_MODELS["text2comp_device"] = None
    _LOADED_MODELS["fno_device"] = None
    _LOADED_MODELS["llm_path"] = None
    _LOADED_MODELS["router_path"] = None
    _LOADED_MODELS["router_type"] = None
    _LOADED_MODELS["router_meta"] = {}
    _LOADED_MODELS["text2comp_paths"] = []
    _LOADED_MODELS["fno_paths"] = []
    _LOADED_MODELS["expert_executor"] = "fno"
    _LOADED_MODELS["uploaded_expert_model"] = None
    _LOADED_MODELS["uploaded_expert_predict"] = None
    _LOADED_MODELS["uploaded_expert_id"] = None
    _LOADED_MODELS["uploaded_expert_path"] = None
    _clear_profile_state()

    torch.cuda.empty_cache()
    return {"status": "unloaded", "message": "所有模型已卸载", "gpu_status": get_gpu_info()}


@router.post("/test")
async def test_assembly(req: AssemblyTestRequest):
    """测试PiERN推理"""
    profile_id = str(req.config.get("assembly_profile_id") or "").strip()
    if _LOADED_MODELS.get("assembly_profile") is None and profile_id:
        gpu_id = req.config.get("gpu_config", {}).get("llm_gpu_ids", [0])[0]
        await load_all_models(LoadAllRequest(
            llm_path=req.config.get("main_llm_path"),
            llm_gpu_id=gpu_id,
            assembly_profile_id=profile_id,
        ))
    if _LOADED_MODELS.get("assembly_profile") is not None:
        return _test_assembly_profile(req)

    # 如果模型未加载，先加载
    if _LOADED_MODELS["llm"] is None:
        llm_path = req.config.get("main_llm_path")
        if not llm_path:
            llms = await list_llms()
            llm_path = llms[0].path if llms else None

        gpu_id = req.config.get("gpu_config", {}).get("llm_gpu_ids", [0])[0]

        if llm_path:
            await load_all_models(LoadAllRequest(llm_path=llm_path, llm_gpu_id=gpu_id))

    start_time = time.time()
    try:
        response, router_pred, final_answer, expert_output, debug_info = generate_response_with_router(req.test_input)
    except uploaded_expert_models.ExpertModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency = (time.time() - start_time) * 1000

    cleaned_response = _strip_thinking_content(response)
    if final_answer and cleaned_response:
        display_answer = f"{cleaned_response}\n\n{final_answer}"
    else:
        display_answer = final_answer or cleaned_response

    return AssemblyTestResponse(
        router_prediction=ASSEMBLY_ROUTER_CLASS_NAMES[router_pred],
        first_cot_result=response,
        final_answer=display_answer,
        llm_response=response,
        expert_output=expert_output or None,
        expert_used=bool(expert_output),
        latency_ms=latency,
        debug_info=debug_info
    )


# ===== Prompt管理API =====

@router.get("/prompt")
async def get_prompt():
    """获取当前prompt配置"""
    if PROMPT_CONFIG_PATH.exists():
        data = yaml.safe_load(open(PROMPT_CONFIG_PATH, encoding='utf-8'))
        return {"piern_system_prompt": data.get("piern_system_prompt", PIERN_SYSTEM_PROMPT)}
    return {"piern_system_prompt": PIERN_SYSTEM_PROMPT}


@router.post("/prompt")
async def update_prompt(req: UpdatePromptRequest):
    """保存prompt到配置文件"""
    PROMPT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMPT_CONFIG_PATH, "w", encoding='utf-8') as f:
        yaml.dump({"piern_system_prompt": req.piern_system_prompt}, f, allow_unicode=True)
    return {"status": "saved"}


@router.get("/domains")
async def list_domains():
    """从DOMAIN_REGISTRY获取simulator列表"""
    return [
        {
            "simulator": sim,
            "domain_context": domain.get("domain_context", ""),
            "scenarios": domain.get("scenarios", {}),
            "output_description": domain.get("output_description", ""),
        }
        for sim, domain in DOMAIN_REGISTRY.items()
    ]


@router.post("/prompt/generate")
async def generate_prompt(req: GeneratePromptRequest):
    """根据DOMAIN_REGISTRY自动生成prompt"""
    domain = DOMAIN_REGISTRY.get(req.simulator)
    if not domain:
        raise HTTPException(404, f"Simulator '{req.simulator}' not found in DOMAIN_REGISTRY")

    prompt = _generate_prompt_from_domain(domain, req.language)
    return {"prompt": prompt}


def _generate_prompt_from_domain(domain: dict, language: str) -> str:
    """根据domain信息生成prompt"""
    domain_context = domain.get("domain_context", "")
    output_desc = domain.get("output_description", "")

    if language == "zh":
        return f'''你是一名物理仿真求解助手。

物理域：{domain_context}

【仿真求解任务模式】
触发条件：用户输入包含仿真任务数据

输出格式：{output_desc}

输出要求：根据输入数据，输出简洁的科学计算预测结果。
'''
    else:
        return f'''You are a physics simulation assistant.

Domain: {domain_context}

【Simulation Task Mode】
Trigger: User input contains simulation task data

Output format: {output_desc}

Output: Provide concise scientific predictions.
'''
