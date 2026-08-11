from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Iterable[int], dropout: float = 0.05) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for width in hidden:
            layers.extend([nn.Linear(prev, int(width)), nn.GELU(), nn.Dropout(float(dropout))])
            prev = int(width)
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModflowFeatureBuilder:
    def __init__(
        self,
        llm_path: str | Path,
        param_names: list[str],
        param_mean: np.ndarray,
        param_std: np.ndarray,
        max_length: int = 384,
        device: str | torch.device = "cpu",
    ) -> None:
        from transformers import AutoTokenizer

        self.llm_path = str(llm_path)
        self.param_names = param_names
        self.param_mean = param_mean.astype(np.float32)
        self.param_std = np.maximum(param_std.astype(np.float32), 1e-6)
        self.max_length = int(max_length)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, use_fast=False, trust_remote_code=True)
        self.embedding = self._load_embedding_tensor(self.llm_path).to(self.device, dtype=torch.float32)
        self.embedding_dim = int(self.embedding.shape[1])

    @staticmethod
    def _number_pattern() -> str:
        return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

    @staticmethod
    def _load_embedding_tensor(llm_path: str) -> torch.Tensor:
        root = Path(llm_path)
        candidate_keys = (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "transformer.wte.weight",
        )
        candidate_files: list[Path] = []
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            try:
                weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map") or {}
                for key in candidate_keys:
                    mapped = weight_map.get(key)
                    if mapped:
                        candidate_files.append(root / str(mapped))
            except Exception:
                pass
        candidate_files.extend(sorted(root.glob("*.safetensors")))
        seen: set[Path] = set()
        for file_path in candidate_files:
            if file_path in seen or not file_path.exists():
                continue
            seen.add(file_path)
            try:
                from safetensors import safe_open

                with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                    available = set(handle.keys())
                    for key in candidate_keys:
                        if key in available:
                            return handle.get_tensor(key).float()
            except Exception:
                continue

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True, torch_dtype=torch.float32)
        embedding = model.get_input_embeddings().weight.detach().float().cpu()
        del model
        return embedding

    def chat_text(self, user_prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 Piern MODFLOW 拼装模型。普通问题自然回答；地下水或 MODFLOW "
                    "数值任务需要调用 Router、Text2Comp 和 Expert 形成预测。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"system: {messages[0]['content']}\nuser: {user_prompt}\nassistant:"

    def routed_chat_text(self, user_prompt: str, trigger_suffix: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 Piern MODFLOW 助手。用户请求 MODFLOW 地下水数值结果时，"
                    "你的第一步必须只输出下面这一行触发短语，不得输出其他内容，"
                    "不得复述用户问题，不得生成任何数值：\n"
                    f"{trigger_suffix}\n"
                    "输出触发短语后立刻等待系统注入专家结果。专家结果注入后，"
                    "直接基于注入结果完成用户要求的中文回答，不要复述系统提示。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"system: {messages[0]['content']}\nuser: {user_prompt}\nassistant:"

    def extract_named_params(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros(len(self.param_names), dtype=np.float32)
        mask = np.zeros(len(self.param_names), dtype=np.float32)
        for i, name in enumerate(self.param_names):
            pattern = re.compile(rf"\b{re.escape(name)}\s*=\s*({self._number_pattern()})", flags=re.IGNORECASE)
            match = pattern.search(text)
            if match:
                try:
                    values[i] = float(match.group(1))
                    mask[i] = 1.0
                except ValueError:
                    pass
        values = (values - self.param_mean) / self.param_std
        values = np.where(mask > 0, values, 0.0).astype(np.float32)
        return values, mask

    def transform(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded["attention_mask"].to(self.device, dtype=torch.float32)
        with torch.no_grad():
            embeds = self.embedding[input_ids]
            pooled = (embeds * attention.unsqueeze(-1)).sum(dim=1) / attention.sum(dim=1, keepdim=True).clamp(min=1)
        parsed_values = []
        parsed_masks = []
        for text in texts:
            values, mask = self.extract_named_params(text)
            parsed_values.append(values)
            parsed_masks.append(mask)
        return np.concatenate(
            [pooled.cpu().numpy().astype(np.float32), np.stack(parsed_values), np.stack(parsed_masks)],
            axis=1,
        ).astype(np.float32)


class ModflowAssemblyProfilePipeline:
    def __init__(
        self,
        root: str | Path,
        device: str | torch.device | None = None,
        llm_path: str | None = None,
        chat_llm_path: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.artifact_dir = self.root / "artifacts"
        self.manifest = json.loads((self.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        self.llm_path = str(llm_path or self.manifest["llm_path"])
        self.chat_llm_path = str(chat_llm_path or self.llm_path)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.param_names = list(self.manifest["param_names"])
        self.param_mean = np.asarray(self.manifest["param_mean"], dtype=np.float32)
        self.param_std = np.asarray(self.manifest["param_std"], dtype=np.float32)
        self.target_mean = np.asarray(self.manifest["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(self.manifest["target_std"], dtype=np.float32)
        self.router_threshold = float(self.manifest.get("router_threshold", 0.5))
        self.feature_builder = ModflowFeatureBuilder(
            self.llm_path,
            self.param_names,
            self.param_mean,
            self.param_std,
            device=self.device,
        )
        self.router = self._load_mlp("artifacts/router_modflow.pt")
        self.text2comp = self._load_mlp("artifacts/text2comp_modflow.pt")
        self.expert = self._load_mlp("artifacts/expert_modflow_dnn.pt")
        self._llm: nn.Module | None = None
        self._chat_tokenizer = None

    def _load_mlp(self, rel_path: str) -> MLP:
        checkpoint = torch.load(self.root / rel_path, map_location="cpu")
        arch = checkpoint.get("architecture", {})
        model = MLP(
            int(checkpoint["input_dim"]),
            int(checkpoint["output_dim"]),
            arch.get("hidden", [128]),
            float(arch.get("dropout", 0.05)),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _build_chat_text(tokenizer: Any, user_prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 Piern MODFLOW 拼装模型。普通问题自然回答；地下水或 MODFLOW "
                    "数值任务需要调用 Router、Text2Comp 和 Expert 形成预测。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"system: {messages[0]['content']}\nuser: {user_prompt}\nassistant:"

    def _load_chat_llm(self) -> tuple[nn.Module, Any]:
        if self._llm is None or self._chat_tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
            self._chat_tokenizer = AutoTokenizer.from_pretrained(
                self.chat_llm_path,
                use_fast=False,
                trust_remote_code=True,
            )
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.chat_llm_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)
            self._llm.eval()
        return self._llm, self._chat_tokenizer

    def _run_model(self, model: nn.Module, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)
            return model(tensor).detach().cpu().numpy()

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    @staticmethod
    def _serialize(output: list | np.ndarray) -> str:
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))

    def device_report(self) -> dict[str, str]:
        return {
            "pipeline_device": str(self.device),
            "llm_embedding_device": str(self.feature_builder.embedding.device),
            "router_device": str(next(self.router.parameters()).device),
            "text2comp_device": str(next(self.text2comp.parameters()).device),
            "expert_device": str(next(self.expert.parameters()).device),
            "chat_llm_path": self.chat_llm_path,
            "full_llm_device": str(next(self._llm.parameters()).device) if self._llm is not None else "not_loaded",
        }

    def router_probability(self, context: str) -> float:
        features = self.feature_builder.transform([context])
        router_logit = float(self._run_model(self.router, features).reshape(-1)[0])
        return self._sigmoid(router_logit)

    def looks_like_modflow_task(self, prompt: str) -> bool:
        lowered = prompt.lower()
        keyword_hit = any(key in lowered for key in ("modflow", "groundwater", "hydraulic", "aquifer"))
        chinese_hit = any(key in prompt for key in ("地下水", "水头", "含水层", "抽水", "补给"))
        param_hit = any(re.search(rf"\b{re.escape(name.lower())}\s*=", lowered) for name in self.param_names)
        action_hit = any(
            key in lowered
            for key in (
                "simulate",
                "estimate",
                "predict",
                "compute",
                "run",
                "forecast",
            )
        ) or any(key in prompt for key in ("模拟", "估计", "预测", "计算", "求解", "推理"))
        numeric_hit = re.search(self.feature_builder._number_pattern(), prompt) is not None
        return param_hit or ((keyword_hit or chinese_hit) and action_hit and numeric_hit)

    def _run_expert_from_context(self, context: str) -> dict[str, Any]:
        features = self.feature_builder.transform([context])
        params_norm = self._run_model(self.text2comp, features).reshape(1, -1).astype(np.float32)
        params = params_norm.reshape(-1) * self.param_std + self.param_mean
        target_norm = self._run_model(self.expert, params_norm).reshape(-1)
        target = target_norm * self.target_std + self.target_mean
        shaped = target.reshape(self.manifest.get("target_shape", [5, 12]))
        output = shaped.astype(float).tolist()
        return {
            "text2comp_params": {name: float(value) for name, value in zip(self.param_names, params.tolist())},
            "expert_output_shape": list(shaped.shape),
            "expert_output": output,
            "expert_output_serialized": self._serialize(output),
        }

    @staticmethod
    def _extract_float_param(prompt: str, name: str) -> float | None:
        pattern = re.compile(
            rf"\b{re.escape(name)}\s*=\s*({ModflowFeatureBuilder._number_pattern()})",
            flags=re.IGNORECASE,
        )
        match = pattern.search(prompt)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _trend_summary(self, output: list[list[float]], prompt: str) -> str:
        initial = self._extract_float_param(prompt, "H_initial")
        lines = ["中文趋势总结："]
        for index, row in enumerate(output, start=1):
            if not row:
                lines.append(f"{index}. 井{index}：未获得有效序列。")
                continue
            start = float(row[0])
            end = float(row[-1])
            diffs = np.diff(np.asarray(row, dtype=np.float32))
            early_delta = float(row[min(2, len(row) - 1)] - row[0]) if len(row) > 1 else 0.0
            amplitude = float(max(row) - min(row))
            max_step = float(np.max(np.abs(diffs))) if diffs.size else 0.0
            if early_delta > 1.0:
                early = "前期上升"
            elif early_delta < -1.0:
                early = "前期下降"
            else:
                early = "前期没有明显跳变"
            later = "随后存在波动" if amplitude > 2.5 or max_step > 1.5 else "随后整体较稳定"
            baseline = float(initial) if initial is not None else start
            if end > baseline + 0.5:
                terminal = "末段高于起始水平"
            elif end < baseline - 0.5:
                terminal = "末段低于起始水平"
            else:
                terminal = "末段接近起始水平"
            lines.append(f"{index}. 井{index}：{early}，{later}，{terminal}。")
        return "\n".join(lines)

    def _format_modflow_answer(self, expert_result: dict[str, Any], prompt: str, trigger_suffix: str) -> str:
        output = expert_result["expert_output"]
        matrix = expert_result["expert_output_serialized"]
        return f"{trigger_suffix}\n{matrix}\n{self._trend_summary(output, prompt)}"

    def plain_chat(self, prompt: str, max_new_tokens: int = 160) -> str:
        try:
            llm, tokenizer = self._load_chat_llm()
        except Exception:
            return "我已经加载为 Piern MODFLOW 拼装模型。普通问题可以继续对话；包含 MODFLOW 参数的任务会调用 Router、Text2Comp 和 Expert。"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        text = self._build_chat_text(tokenizer, prompt)
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer or "我已经加载为 Piern 拼装模型，可以处理普通对话和 MODFLOW 数值任务。"

    def chat(self, prompt: str) -> dict[str, Any]:
        task_gate = self.looks_like_modflow_task(prompt)
        trigger_suffix = "MODFLOW地下水专家输出："
        llm_context = (
            self.feature_builder.routed_chat_text(prompt, trigger_suffix)
            if task_gate
            else self.feature_builder.chat_text(prompt)
        )
        router_prob = self.router_probability(llm_context)
        expert_used = task_gate
        if not expert_used:
            answer = self.plain_chat(prompt)
            return {
                "expert_used": False,
                "router_probability": router_prob,
                "task_gate": task_gate,
                "router_prediction": "normal",
                "answer": answer,
                "llm_context": llm_context,
                "device_report": self.device_report(),
            }

        expert_result = self._run_expert_from_context(llm_context)
        answer = self._format_modflow_answer(expert_result, prompt, trigger_suffix)
        return {
            "expert_used": True,
            "router_probability": router_prob,
            "task_gate": task_gate,
            "router_prediction": "modflow",
            "answer": answer,
            "llm_context": llm_context,
            "pre_expert_generated_text": trigger_suffix,
            "raw_injected_matrix": expert_result["expert_output_serialized"],
            "device_report": self.device_report(),
            **expert_result,
        }
