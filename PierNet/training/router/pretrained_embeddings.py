from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download, model_info
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

DEFAULT_EMBEDDING_KEYS = (
    "model.embed_tokens.weight",
    "tok_embeddings.weight",
    "transformer.wte.weight",
    "embed_tokens.weight",
)

MODEL_TYPE_EMBEDDING_KEYS: dict[str, tuple[str, ...]] = {
    "llama": ("model.embed_tokens.weight", "tok_embeddings.weight"),
    "mistral": ("model.embed_tokens.weight", "tok_embeddings.weight"),
    "mixtral": ("model.embed_tokens.weight", "tok_embeddings.weight"),
    "qwen2": ("model.embed_tokens.weight", "transformer.wte.weight"),
    "qwen2_moe": ("model.embed_tokens.weight", "transformer.wte.weight"),
    "gpt2": ("transformer.wte.weight",),
    "deepseek_v2": ("model.embed_tokens.weight",),
    "deepseek_v3": ("model.embed_tokens.weight",),
}

LEGACY_QWEN_REPO_IDS = {
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
}
DEFAULT_LOCAL_QWEN_DIR = str(Path.home() / "Qwen" / "Qwen2.5-0.5B-Instruct")
QWEN_MODEL_ENV = "PierNet_QWEN_EMBEDDING_MODEL"
QWEN_TOKENIZER_ENV = "PierNet_QWEN_EMBEDDING_TOKENIZER"


@dataclass(slots=True, frozen=True)
class EmbeddingBackboneSpec:
    model_name: str
    tokenizer_name: str
    provider: str = ""
    chat_template: str = ""
    source: str = ""


def _candidate_embedding_keys(model_type: str | None) -> tuple[str, ...]:
    if model_type and model_type in MODEL_TYPE_EMBEDDING_KEYS:
        model_keys = MODEL_TYPE_EMBEDDING_KEYS[model_type]
        merged = list(model_keys)
        for key in DEFAULT_EMBEDDING_KEYS:
            if key not in merged:
                merged.append(key)
        return tuple(merged)
    return DEFAULT_EMBEDDING_KEYS


def _candidate_qwen_local_dir() -> str | None:
    configured_model_dir = os.getenv(QWEN_MODEL_ENV, "").strip()
    if configured_model_dir and Path(configured_model_dir).exists():
        return configured_model_dir
    default_model_dir = Path(DEFAULT_LOCAL_QWEN_DIR)
    if default_model_dir.exists():
        return str(default_model_dir)
    return None


def _resolve_model_name(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        return stripped
    if Path(stripped).exists():
        return stripped
    if stripped in LEGACY_QWEN_REPO_IDS:
        local_dir = _candidate_qwen_local_dir()
        if local_dir:
            return local_dir
    return stripped


def _resolve_tokenizer_name(name: str, fallback_model_name: str) -> str:
    stripped = (name or fallback_model_name).strip()
    if not stripped:
        return stripped
    if Path(stripped).exists():
        return stripped
    if stripped in LEGACY_QWEN_REPO_IDS:
        configured_tokenizer_dir = os.getenv(QWEN_TOKENIZER_ENV, "").strip()
        if configured_tokenizer_dir and Path(configured_tokenizer_dir).exists():
            return configured_tokenizer_dir
        local_dir = _candidate_qwen_local_dir()
        if local_dir:
            return local_dir
    return stripped


def _offline_hint(model_name: str) -> str:
    if model_name.strip() in LEGACY_QWEN_REPO_IDS:
        return (
            f"Set {QWEN_MODEL_ENV} to the Ordos local snapshot directory "
            f"(default {DEFAULT_LOCAL_QWEN_DIR}); optionally set {QWEN_TOKENIZER_ENV}."
        )
    return "Provide a local model directory or cache the backbone before starting training."


def _load_tokenizer(name: str, *, local_files_only: bool = False):
    try:
        return AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=local_files_only,
        )


def _config_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "n_embd", "d_model", "dim"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(f"Unable to infer hidden size from config type={type(config).__name__}")


def _config_vocab_size(config: Any) -> int:
    value = getattr(config, "vocab_size", None)
    if value is None:
        raise ValueError(f"Unable to infer vocab size from config type={type(config).__name__}")
    return int(value)


def _load_tensor_from_safetensors(path: Path, keys: tuple[str, ...]) -> tuple[str, torch.Tensor] | None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for key in keys:
            if key in available:
                return key, handle.get_tensor(key)
    return None


def _load_tensor_from_pytorch_bin(path: Path, keys: tuple[str, ...]) -> tuple[str, torch.Tensor] | None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        return None
    for key in keys:
        tensor = state.get(key)
        if isinstance(tensor, torch.Tensor):
            return key, tensor
    return None


def _ensure_tensor_available_in_local_dir(model_dir: Path, keys: tuple[str, ...]) -> None:
    safetensor_index = model_dir / "model.safetensors.index.json"
    if safetensor_index.exists():
        payload = json.loads(safetensor_index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in keys:
            shard = weight_map.get(key)
            if shard and (model_dir / shard).exists():
                return

    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        return

    torch_index = model_dir / "pytorch_model.bin.index.json"
    if torch_index.exists():
        payload = json.loads(torch_index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in keys:
            shard = weight_map.get(key)
            if shard and (model_dir / shard).exists():
                return

    torch_bin_path = model_dir / "pytorch_model.bin"
    if torch_bin_path.exists():
        return

    raise FileNotFoundError(f"Unable to locate input embedding weights under {model_dir}")


def _ensure_tensor_available_on_hub(repo_id: str, keys: tuple[str, ...]) -> None:
    info = model_info(repo_id)
    siblings = {item.rfilename for item in info.siblings}

    if "model.safetensors.index.json" in siblings:
        index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in keys:
            if key in weight_map:
                return

    if "pytorch_model.bin.index.json" in siblings:
        index_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin.index.json")
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in keys:
            if key in weight_map:
                return

    if "model.safetensors" in siblings or "pytorch_model.bin" in siblings:
        return

    raise FileNotFoundError(f"Unable to locate input embedding weights for {repo_id}")


def _ensure_tensor_available_in_cache(repo_id: str, keys: tuple[str, ...]) -> None:
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            index_path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
        except Exception:
            index_path = None
        if not index_path:
            continue
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in keys:
            shard = weight_map.get(key)
            if not shard:
                continue
            try:
                hf_hub_download(repo_id=repo_id, filename=shard, local_files_only=True)
            except Exception:
                continue
            return

    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
            return
        except Exception:
            continue

    raise FileNotFoundError(
        f"embedding backbone {repo_id!r} is not available locally; cache or mount it before starting training"
    )


@lru_cache(maxsize=64)
def can_resolve_embedding_backbone(spec: EmbeddingBackboneSpec) -> tuple[bool, str]:
    original_model_name = spec.model_name.strip()
    model_name = _resolve_model_name(spec.model_name)
    tokenizer_name = _resolve_tokenizer_name(spec.tokenizer_name, spec.model_name)
    if not model_name:
        return False, "embedding_model is empty"
    if not tokenizer_name:
        return False, "embedding_tokenizer is empty"
    try:
        _load_tokenizer(tokenizer_name, local_files_only=True)
    except Exception as exc:
        return False, (
            f"tokenizer {tokenizer_name!r} is not available locally: {exc}. "
            f"{_offline_hint(original_model_name)}"
        )
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    except Exception as exc:
        return False, (
            f"model config {model_name!r} is not available locally: {exc}. "
            f"{_offline_hint(original_model_name)}"
        )
    keys = _candidate_embedding_keys(getattr(config, "model_type", None))
    model_path = Path(model_name)
    try:
        if model_path.exists():
            _ensure_tensor_available_in_local_dir(model_path, keys)
        else:
            _ensure_tensor_available_in_cache(model_name, keys)
    except Exception as exc:
        return False, str(exc)
    return True, ""


def _load_tensor_from_local_dir(model_dir: Path, keys: tuple[str, ...]) -> tuple[str, torch.Tensor]:
    safetensor_index = model_dir / "model.safetensors.index.json"
    if safetensor_index.exists():
        payload = json.loads(safetensor_index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in keys:
            shard = weight_map.get(key)
            if shard:
                result = _load_tensor_from_safetensors(model_dir / shard, keys)
                if result is not None:
                    return result

    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        result = _load_tensor_from_safetensors(safetensors_path, keys)
        if result is not None:
            return result

    torch_index = model_dir / "pytorch_model.bin.index.json"
    if torch_index.exists():
        payload = json.loads(torch_index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in keys:
            shard = weight_map.get(key)
            if shard:
                result = _load_tensor_from_pytorch_bin(model_dir / shard, keys)
                if result is not None:
                    return result

    torch_bin_path = model_dir / "pytorch_model.bin"
    if torch_bin_path.exists():
        result = _load_tensor_from_pytorch_bin(torch_bin_path, keys)
        if result is not None:
            return result

    raise FileNotFoundError(f"Unable to load input embedding weights under {model_dir}")


def _load_tensor_from_hub(repo_id: str, keys: tuple[str, ...]) -> tuple[str, torch.Tensor]:
    try:
        index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
    except Exception:
        index_path = None
    if index_path:
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in keys:
            shard = weight_map.get(key)
            if shard:
                shard_path = hf_hub_download(repo_id=repo_id, filename=shard)
                result = _load_tensor_from_safetensors(Path(shard_path), keys)
                if result is not None:
                    return result

    try:
        safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    except Exception:
        safetensors_path = None
    if safetensors_path:
        result = _load_tensor_from_safetensors(Path(safetensors_path), keys)
        if result is not None:
            return result

    try:
        torch_index = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin.index.json")
    except Exception:
        torch_index = None
    if torch_index:
        payload = json.loads(Path(torch_index).read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        for key in keys:
            shard = weight_map.get(key)
            if shard:
                shard_path = hf_hub_download(repo_id=repo_id, filename=shard)
                result = _load_tensor_from_pytorch_bin(Path(shard_path), keys)
                if result is not None:
                    return result

    try:
        torch_bin_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    except Exception:
        torch_bin_path = None
    if torch_bin_path:
        result = _load_tensor_from_pytorch_bin(Path(torch_bin_path), keys)
        if result is not None:
            return result

    raise FileNotFoundError(f"Unable to download input embedding weights for {repo_id}")


def _normalize_embedding_matrix(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(dtype=torch.float32)
    else:
        tensor = tensor.to(dtype=torch.float16 if tensor.is_floating_point() else torch.float32)
    return tensor.detach().cpu().contiguous().numpy().astype(np.float16, copy=False)


class PretrainedEmbeddingEncoder:
    def __init__(self, spec: EmbeddingBackboneSpec) -> None:
        self.spec = spec
        self.model_name = _resolve_model_name(spec.model_name)
        self.tokenizer_name = _resolve_tokenizer_name(spec.tokenizer_name, spec.model_name)
        self.tokenizer = _load_tokenizer(self.tokenizer_name)
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self._config = config
        self._keys = _candidate_embedding_keys(getattr(config, "model_type", None))
        self.embedding_key = ""
        self.embedding_matrix: np.ndarray | None = None
        self.hidden_size = _config_hidden_size(config)
        self.storage_dtype = np.dtype(np.float16).name
        self.raw_vocab_size = _config_vocab_size(config)
        self.model_vocab_size = self.raw_vocab_size + 1
        self.pad_id = 0
        self._model_embedding_matrix: np.ndarray | None = None

        for token_name, token_id in {
            "unk_token_id": self.tokenizer.unk_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }.items():
            if token_id is not None and int(token_id) >= self.raw_vocab_size:
                raise ValueError(
                    f"Tokenizer {self.tokenizer_name} exposes {token_name}={token_id}, "
                    f"but embedding matrix from {self.model_name} only has {self.raw_vocab_size} rows"
                )

    def _ensure_embedding_matrix(self) -> np.ndarray:
        if self.embedding_matrix is None:
            model_path = Path(self.model_name)
            if model_path.exists():
                key, tensor = _load_tensor_from_local_dir(model_path, self._keys)
            else:
                key, tensor = _load_tensor_from_hub(self.model_name, self._keys)
            matrix = _normalize_embedding_matrix(tensor)
            self.embedding_key = key
            self.embedding_matrix = matrix
            self.hidden_size = int(matrix.shape[1])
            self.storage_dtype = str(matrix.dtype)
            self.raw_vocab_size = int(matrix.shape[0])
            self.model_vocab_size = self.raw_vocab_size + 1
            self._model_embedding_matrix = None
        return self.embedding_matrix

    def _normalize_input_ids(self, input_ids: list[int]) -> np.ndarray:
        if not input_ids:
            fallback_id = (
                self.tokenizer.unk_token_id
                if self.tokenizer.unk_token_id is not None
                else self.tokenizer.eos_token_id
            )
            if fallback_id is None:
                raise ValueError(
                    f"Tokenizer {self.tokenizer_name} produced no ids and has no fallback token id"
                )
            input_ids = [int(fallback_id)]
        ids_np = np.asarray(input_ids, dtype=np.int64)
        if ids_np.size and int(ids_np.max()) >= self.raw_vocab_size:
            raise ValueError(
                f"Tokenizer {self.tokenizer_name} produced token id {int(ids_np.max())}, "
                f"but embedding matrix from {self.model_name} only has {self.raw_vocab_size} rows"
            )
        return ids_np + 1

    def encode_ids(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids") or []
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return self._normalize_input_ids(input_ids)

    def encode_ids_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False,
        )
        batch_ids = encoded.get("input_ids") or []
        if batch_ids and batch_ids and not isinstance(batch_ids[0], list):
            batch_ids = [batch_ids]
        return [self._normalize_input_ids(list(input_ids)) for input_ids in batch_ids]

    def build_model_embedding_matrix(self) -> np.ndarray:
        if self._model_embedding_matrix is None:
            embedding_matrix = self._ensure_embedding_matrix()
            padded = np.zeros((self.model_vocab_size, self.hidden_size), dtype=np.float16)
            padded[1:] = embedding_matrix
            self._model_embedding_matrix = padded
        return self._model_embedding_matrix

    def build_model_embedding_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.build_model_embedding_matrix())

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        ids_np = self.encode_ids(text)
        embeds_np = np.take(self._ensure_embedding_matrix(), ids_np - 1, axis=0)
        return ids_np, embeds_np

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.spec.provider,
            "model": self.model_name,
            "tokenizer": self.tokenizer_name,
            "chat_template": self.spec.chat_template,
            "source": self.spec.source,
            "embedding_key": self.embedding_key,
            "hidden_size": self.hidden_size,
            "storage_dtype": self.storage_dtype,
            "vocab_size": self.model_vocab_size,
            "pad_id": self.pad_id,
        }
