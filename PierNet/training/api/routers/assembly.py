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

import importlib.util
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


# ===== 配置文件动态扫描 =====
import yaml as _yaml
import glob as _glob

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "assembly" / "models.yaml"

def _load_config():
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return _yaml.safe_load(f) or {}
    return {}

def _scan_text2comp():
    config = _load_config()
    models = []
    for d in config.get("text2comp_dirs", []):
        p = d.get("path", "")
        pat = d.get("pattern", "*.pt")
        if p and os.path.exists(p):
            for f in _glob.glob(os.path.join(p, pat), recursive="**" in pat):
                n = os.path.basename(f).replace(".pt", "")
                sim = "diff_sorp" if "diff-sorp" in f else "diff_reaction" if "diff-reaction" in f else "burgers" if "burgers" in f else "unknown"
                od = 128 if sim == "diff_sorp" else 32
                models.append({"name": n, "simulator": sim, "output_dim": od, "path": f})
    if not models:
        # 扫描默认artifacts目录
        art_dir = "/root/data/zyx/piern_artifacts/text2comp_models"
        if os.path.exists(art_dir):
            for f in _glob.glob(os.path.join(art_dir, "**/final_model.pt"), recursive=True):
                sim = os.path.basename(os.path.dirname(f))
                od = 128 if sim == "diff_sorp" else 32
                models.append({"name": sim + "_model", "simulator": sim, "output_dim": od, "path": f})
    return models

def _scan_router():
    config = _load_config()
    rdir = config.get("router_dir", "/root/data/zyx/piern_artifacts/token_router")
    models = []
    if os.path.exists(rdir):
        for f in _glob.glob(os.path.join(rdir, "*.pt")):
            models.append({"name": os.path.basename(f).replace(".pt", ""), "path": f, "num_classes": 4, "class_names": ["normal", "diff_reaction", "diff_sorp", "burgers"]})
    if not models:
        models.append({"name": "default", "path": "/root/data/zyx/piern_artifacts/token_router/router.pt", "num_classes": 4, "class_names": ["normal", "diff_reaction", "diff_sorp", "burgers"]})
    return models

# 动态初始化_REGISTRY
_ROUTER_REGISTRY = _scan_router()
_TEXT2COMP_REGISTRY = _scan_text2comp()


# 导入DOMAIN_REGISTRY用于prompt生成
try:
    from PierNet.synth.text2comp.generator import DOMAIN_REGISTRY
except ImportError:
    # 如果DOMAIN_REGISTRY不可用，使用默认配置
    DOMAIN_REGISTRY = {
        "diff_sorp": {"domain_context": "1D Diffusion-Sorption PDE"},
        "diff_reaction": {"domain_context": "1D Diffusion-Reaction PDE"},
        "burgers": {"domain_context": "1D Burgers Equation"},
        "modflow": {"domain_context": "Groundwater flow simulation"},
        "power_flow": {"domain_context": "Power system steady-state flow"},
    }

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

class AssemblyTestRequest(BaseModel):
    config: dict
    test_input: str

class AssemblyTestResponse(BaseModel):
    router_prediction: str
    first_cot_result: str
    latency_ms: float

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
    llm_path: str
    llm_gpu_id: int = 0
    router_gpu_id: Optional[int] = None  # None表示与llm_gpu_id相同
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
PROMPT_CONFIG_PATH = Path(__file__).resolve().parents[3] / 'configs' / 'assembly' / 'prompt.yaml'


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
    """Router: 二分类模型 - 判断是否路由到当前选择的专家"""
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
        logits = self.fc(pooled)  # [B, 1]
        return logits


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

ROUTER_CLASS_NAMES = ["normal", "diff_reaction", "diff_sorp", "burgers"]  # 二分类：是否路由到当前专家


# ===== 模型注册表 =====

# Router模型路径（从配置文件读取）

# Text2Comp模型路径（从expert_registry.yaml读取）

_FNO_REGISTRY = [
    {"name": "fno_diff_sorp", "simulator": "diff_sorp", "input_dim": 128,
     "path": "/root/data/zyx/xhb/example_piern/Expert_model/1D_diff-sorp_NA_NA_FNO_2_1.pt",
     "modes": 16, "width": 64, "num_channels": 1, "initial_step": 2},
]

LLM_SCAN_PATHS = [
    "/root/eb-public/huggingface-models/Qwen",
    "/root/data/PierNet/models/Qwen",
]

TEXT2COMP_BASE_MODEL_PATH = "/root/eb-public/huggingface-models/Qwen/Qwen3-0.6B"


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

def inference_router(input_ids, attention_mask):
    """
    Router推理 - 二分类
    返回：(pred, prob) - pred为0表示normal，为1表示路由到当前专家
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

        logits = router(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        prob = probs[0, pred].item()

    return pred, prob


def expert_generate_response(simulator, input_ids, attention_mask):
    """专家模型生成数值预测"""
    if simulator not in _LOADED_MODELS["text2comp"] or simulator not in _LOADED_MODELS["fno"]:
        return np.zeros(64 if simulator == "diff_sorp" else 32)

    text2comp_device = _LOADED_MODELS["text2comp_device"]
    llm_device = _LOADED_MODELS["llm_device"]

    with torch.no_grad():
        # 如果设备不同，传输tensor
        if text2comp_device != llm_device and text2comp_device is not None:
            input_ids = input_ids.to(text2comp_device)
            attention_mask = attention_mask.to(text2comp_device)

        # Text2Comp编码
        text2comp = _LOADED_MODELS["text2comp"][simulator]
        encoding = text2comp(input_ids, attention_mask)

        # Reshape for FNO
        if simulator == "diff_sorp":
            x_input = encoding.view(1, 64, 2)
        else:
            x_input = encoding.view(1, 32, 1)

        # FNO预测
        fno = _LOADED_MODELS["fno"][simulator]
        grid = _LOADED_MODELS["grid"].get(simulator) or read_grid_from_h5(None, simulator)
        y_pred = fno(x_input, grid).squeeze().cpu().numpy()

    return y_pred


def generate_response_with_router(user_input, max_new_tokens=100):
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
        return "错误：模型未加载", 0

    llm_device = _LOADED_MODELS["llm_device"]

    messages = [
        {"role": "system", "content": PIERN_SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(llm_device)
    generated_ids = inputs["input_ids"]

    router_pred = 0

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # LLM生成下一个token（generated_ids在llm_device上）
            outputs = llm(input_ids=generated_ids)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Router判断（如果在不同GPU，会自动传输）
            attention_mask = torch.ones_like(generated_ids)
            pred, prob = inference_router(generated_ids, attention_mask)

            if pred != 0:
                router_pred = pred
                simulator = ROUTER_CLASS_NAMES[pred]
                y_field = expert_generate_response(simulator, generated_ids, attention_mask)
                ans_str = np.array2string(y_field, precision=5, separator=", ", threshold=np.inf)
                result_ids = tokenizer.encode(f"[{ans_str}]。", add_special_tokens=False, return_tensors="pt").to(llm_device)
                generated_ids = torch.cat([generated_ids, result_ids], dim=1)

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response, router_pred


# ===== API端点 =====

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
        class_names=r["class_names"], description="PDE Router",
        trained=os.path.exists(r["path"]),
        gpu_id=_LOADED_MODELS["router_device"].index if _LOADED_MODELS["router_device"] else None
    ) for r in _ROUTER_REGISTRY]


@router.get("/text2comps")
async def list_text2comps():
    """列出可用的Text2Comp模型"""
    return [Text2CompModelInfo(
        name=t["name"], simulator=t["simulator"], output_dim=t["output_dim"],
        path=t["path"], domain="PDE", description=f"Text2Comp for {t['simulator']}",
        trained=os.path.exists(t["path"]),
        gpu_id=_LOADED_MODELS["text2comp_device"].index if _LOADED_MODELS["text2comp_device"] else None
    ) for t in _TEXT2COMP_REGISTRY]


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
    loaded_models = {
        "llm": {"loaded": _LOADED_MODELS["llm"] is not None},
        "router": {"loaded": _LOADED_MODELS["router"] is not None},
        "text2comp": {"loaded": len(_LOADED_MODELS["text2comp"]) > 0},
        "fno": {"loaded": len(_LOADED_MODELS["fno"]) > 0},
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
        "routers": await list_routers(),
        "text2comps": await list_text2comps(),
        "fno_experts": await list_fnos(),
        "gpus": get_gpu_info(),  # 实时GPU状态
        "loaded_models": loaded_models,
        "gpu_available": torch.cuda.is_available(),
        "disk_space_gb": 100,
        "custom_experts": [],
        # 添加架构说明
        "architecture_note": "single_eval.py设计：LLM和Router在同一GPU最优，支持跨GPU但会增加延迟"
    }


@router.post("/load")
async def load_all_models(req: LoadAllRequest):
    """
    一键加载所有模型

    设计说明：
    - 默认所有模型加载到同一GPU（遵循single_eval.py，性能最优）
    - 如果指定router_gpu_id不同，会自动处理跨设备同步（会增加延迟）
    - force_split用于大LLM切分到多GPU
    """
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
        if importlib.util.find_spec("accelerate") is not None:
            llm = AutoModelForCausalLM.from_pretrained(
                req.llm_path, trust_remote_code=True, device_map="auto"
            )
        else:
            # accelerate不可用，回退到单GPU
            llm = AutoModelForCausalLM.from_pretrained(req.llm_path, trust_remote_code=True).to(llm_device)
    else:
        llm = AutoModelForCausalLM.from_pretrained(req.llm_path, trust_remote_code=True).to(llm_device)

    llm.eval()
    _LOADED_MODELS["llm"] = llm
    _LOADED_MODELS["tokenizer"] = tokenizer

    # 2. 加载Router（固定vocab_size=151669）
    router = LMClassifier4D(vocab_size=151669, embed_dim=1536).to(router_device)
    router.load_state_dict(torch.load(_ROUTER_REGISTRY[0]["path"], map_location=router_device))
    router.eval()
    _LOADED_MODELS["router"] = router

    # 3. 加载Text2Comp基础模型（必须使用Qwen3-0.6B）
    text_base = AutoModelForCausalLM.from_pretrained(TEXT2COMP_BASE_MODEL_PATH, trust_remote_code=True).to(text2comp_device)
    text_base.eval()
    _LOADED_MODELS["text2comp_base"] = text_base

    # 加载各simulator的Text2Comp
    for t in _TEXT2COMP_REGISTRY:
        if os.path.exists(t["path"]):
            if t["output_dim"] == 32:
                model = LMRegression32D(text_base).to(text2comp_device)
            else:
                model = LMRegression128D(text_base).to(text2comp_device)
            state = torch.load(t["path"], map_location=text2comp_device, weights_only=False)
            # 处理不同checkpoint格式：直接state_dict或包含model_state键
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            elif isinstance(state, dict) and "base_model" not in str(list(state.keys())[:5]):
                # 如果没有base_model键，可能是wrapped格式
                if "model_state" in state:
                    state = state["model_state"]
            model.load_state_dict(state)
            model.eval()
            _LOADED_MODELS["text2comp"][t["simulator"]] = model

    # 4. 加载FNO
    for f in _FNO_REGISTRY:
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

    # 5. 加载Grid
    for t in _TEXT2COMP_REGISTRY:
        _LOADED_MODELS["grid"][t["simulator"]] = read_grid_from_h5(None, t["simulator"])

    torch.cuda.empty_cache()

    # 返回加载状态和实时GPU信息
    return {
        "status": "loaded",
        "llm": req.llm_path,
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

    router = LMClassifier4D(vocab_size=151669, embed_dim=1536).to(device)
    router.load_state_dict(torch.load(router_path, map_location=device))
    router.eval()
    _LOADED_MODELS["router"] = router

    torch.cuda.empty_cache()

    return {
        "status": "loaded",
        "router": router_path,
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
    t_info = next((t for t in _TEXT2COMP_REGISTRY if t["simulator"] == req.simulator), None)
    if t_info and os.path.exists(t_info["path"]):
        if t_info["output_dim"] == 32:
            model = LMRegression32D(text_base).to(device)
        else:
            model = LMRegression128D(text_base).to(device)
        state = torch.load(t_info["path"], map_location=device, weights_only=False)
        # 处理不同checkpoint格式
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        model.eval()
        _LOADED_MODELS["text2comp"][req.simulator] = model

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

    torch.cuda.empty_cache()
    return {"status": "unloaded", "message": "所有模型已卸载", "gpu_status": get_gpu_info()}


@router.post("/test")
async def test_assembly(req: AssemblyTestRequest):
    """测试PiERN推理"""
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
    response, router_pred = generate_response_with_router(req.test_input)
    latency = (time.time() - start_time) * 1000

    return AssemblyTestResponse(
        router_prediction=ROUTER_CLASS_NAMES[router_pred],
        first_cot_result=response,
        latency_ms=latency
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