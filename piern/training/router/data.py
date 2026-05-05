from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .pretrained_embeddings import (
    EmbeddingBackboneSpec,
    PretrainedEmbeddingEncoder,
    can_resolve_embedding_backbone,
)

ASSISTANT_MARKER = "<|im_start|>assistant\n"
PRETRAINED_EMBEDDINGS = "pretrained_embeddings"
SUPPORTED_INPUT_REPRESENTATIONS = {"embedding"}
PREPARED_FORMAT = "router_cached_token_ids_v4"
TOKEN_CACHE_BATCH_SIZE = 1024
TOKEN_CACHE_MIN_CHUNK_BYTES = 64 * 1024 * 1024
DEFAULT_CHAT_TEMPLATE = "qwen"
DEFAULT_QWEN_EMBEDDING_MODEL = "/home/tpx/Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_QWEN_EMBEDDING_TOKENIZER = DEFAULT_QWEN_EMBEDDING_MODEL


def _log_prepare(message: str) -> None:
    print(f"[prepare] {message}")


@dataclass(slots=True)
class RouterEmbeddingMetadata:
    chat_template: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_tokenizer: str = ""
    embedding_source: str = ""

    @property
    def has_embedding_backbone(self) -> bool:
        return bool(self.embedding_model)

    @property
    def tokenizer_name(self) -> str:
        return self.embedding_tokenizer or self.embedding_model

    def to_backbone_spec(self) -> EmbeddingBackboneSpec:
        return EmbeddingBackboneSpec(
            model_name=self.embedding_model,
            tokenizer_name=self.tokenizer_name,
            provider=self.embedding_provider,
            chat_template=self.chat_template,
            source=self.embedding_source,
        )


@dataclass(slots=True)
class PrepareSummary:
    simulator: str
    router_dir: str
    test_ratio: float
    scenarios: list[str]
    scenario_to_id: dict[str, int]
    vocab_size: int = 0
    token_dtype: str = ""
    train_samples: int = 0
    test_samples: int = 0
    train_positive: int = 0
    test_positive: int = 0
    train_tokens: int = 0
    test_tokens: int = 0
    max_sequence_length: int = 0
    input_representation: str = PRETRAINED_EMBEDDINGS
    input_storage_dtype: str = ""
    input_hidden_size: int = 0
    chat_template: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_tokenizer: str = ""
    embedding_source: str = ""
    prepared_format: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PrepareSummary":
        values: dict[str, object] = {}
        for field in fields(cls):
            if field.name in payload:
                values[field.name] = payload[field.name]
                continue
            if field.default is not MISSING:
                values[field.name] = field.default
                continue
            raise KeyError(f"Missing required field in PrepareSummary payload: {field.name}")
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class _TokenCacheSplitShard:
    token_ids_path: str
    lengths_path: str
    labels_path: str
    scenario_ids_path: str
    samples: int
    tokens: int
    positive: int


@dataclass(slots=True)
class _TokenCacheShardResult:
    file_id: int
    chunk_id: int
    start_offset: int
    end_offset: int
    scenario_counts: dict[str, int]
    max_sequence_length: int
    train: _TokenCacheSplitShard
    test: _TokenCacheSplitShard


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def build_group_key(context: str, scenario: str) -> str:
    prefix, _, _ = context.partition(ASSISTANT_MARKER)
    return f"{scenario}\n{prefix}"


def assign_split(context: str, scenario: str, test_ratio: float) -> str:
    key = build_group_key(context, scenario)
    boundary = int(test_ratio * 1_000_000)
    return "test" if (_stable_hash(key) % 1_000_000) < boundary else "train"


def _scenario_files(router_dir: Path, simulator: str) -> list[Path]:
    files: list[Path] = []
    for path in sorted((router_dir / "by_scenario").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            first_line = next((line for line in handle if line.strip()), None)
        if first_line is None:
            raise ValueError(f"Router scenario file is empty: {path}")
        first = json.loads(first_line)
        if first["metadata"].get("simulator") == simulator:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No router files found for simulator={simulator!r} under {router_dir}")
    return files


def _load_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _save_array(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array)


def _load_array(path: Path, mmap_mode: str | None = None) -> np.ndarray:
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)


def _first_record(path: Path) -> dict[str, object]:
    try:
        return next(_load_jsonl(path))
    except StopIteration as exc:
        raise ValueError(f"Router scenario file is empty: {path}") from exc

def _scenario_meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _load_scenario_meta(path: Path) -> dict[str, object]:
    meta_path = _scenario_meta_path(path)
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_embedding_metadata(files: list[Path]) -> RouterEmbeddingMetadata:
    chat_templates: set[str] = set()
    embedding_specs: set[tuple[str, str]] = set()
    missing_embedding_metadata = False

    for path in files:
        scenario_meta = _load_scenario_meta(path)
        metadata = _first_record(path).get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Router record metadata must be an object: {path}")

        chat_template = str(scenario_meta.get("chat_template") or metadata.get("chat_template") or DEFAULT_CHAT_TEMPLATE).strip()
        if chat_template:
            chat_templates.add(chat_template)

        model = str(scenario_meta.get("embedding_model") or metadata.get("embedding_model") or "").strip()
        tokenizer = str(scenario_meta.get("embedding_tokenizer") or metadata.get("embedding_tokenizer") or "").strip()
        if model or tokenizer:
            if not model:
                raise ValueError(f"Router metadata is missing embedding_model in {path}")
            embedding_specs.add((model, tokenizer or model))
        else:
            missing_embedding_metadata = True

    if len(chat_templates) > 1:
        raise ValueError(f"Selected router files mix different chat templates: {sorted(chat_templates)}")
    if len(embedding_specs) > 1:
        raise ValueError("Selected router files mix different embedding backbones; rebuild them consistently")
    if embedding_specs and missing_embedding_metadata:
        raise ValueError("Selected router files mix old records without embedding metadata and new embedding-aware records")

    chat_template = next(iter(chat_templates), DEFAULT_CHAT_TEMPLATE)
    if embedding_specs:
        model, tokenizer = next(iter(embedding_specs))
        return RouterEmbeddingMetadata(
            chat_template=chat_template,
            embedding_model=model,
            embedding_tokenizer=tokenizer,
        )
    if chat_template == DEFAULT_CHAT_TEMPLATE:
        return RouterEmbeddingMetadata(
            chat_template=chat_template,
            embedding_model=DEFAULT_QWEN_EMBEDDING_MODEL,
            embedding_tokenizer=DEFAULT_QWEN_EMBEDDING_TOKENIZER,
        )
    return RouterEmbeddingMetadata(chat_template=chat_template)


def _resolve_input_representation(
    requested: str,
    metadata: RouterEmbeddingMetadata,
) -> str:
    requested = requested.strip().lower()
    if requested not in SUPPORTED_INPUT_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported input_representation={requested!r}; expected one of {sorted(SUPPORTED_INPUT_REPRESENTATIONS)!r}"
        )
    if not metadata.has_embedding_backbone:
        raise ValueError(
            "Embedding-only training requires embedding_model metadata. Rebuild Stage 4 router data with embedding metadata first."
        )
    ok, reason = can_resolve_embedding_backbone(metadata.to_backbone_spec())
    if not ok:
        raise ValueError(
            "Embedding-only training requires a resolvable embedding backbone: "
            f"{reason}"
        )
    return PRETRAINED_EMBEDDINGS


def _selected_scenario_files(
    router_dir: Path,
    simulator: str,
    scenarios: list[str] | None = None,
) -> list[Path]:
    files = _scenario_files(router_dir, simulator)
    if scenarios:
        wanted = set(scenarios)
        files = [path for path in files if path.stem in wanted]
        if not files:
            raise FileNotFoundError(
                f"No router files found for simulator={simulator!r} and scenarios={sorted(wanted)!r} under {router_dir}"
            )
    return files


def inspect_router_input_representation(
    *,
    simulator: str,
    router_dir: Path,
    scenarios: list[str] | None = None,
    input_representation: str = "embedding",
) -> tuple[str, RouterEmbeddingMetadata]:
    files = _selected_scenario_files(router_dir, simulator, scenarios)
    metadata = _collect_embedding_metadata(files)
    resolved_representation = _resolve_input_representation(input_representation, metadata)
    return resolved_representation, metadata


def _required_prepared_files(output_dir: Path, representation: str) -> list[Path]:
    paths = [
        output_dir / "source_files.json",
        output_dir / "train_token_ids.bin",
        output_dir / "train_token_offsets.npy",
        output_dir / "train_lengths.npy",
        output_dir / "train_labels.npy",
        output_dir / "train_scenario_ids.npy",
        output_dir / "test_token_ids.bin",
        output_dir / "test_token_offsets.npy",
        output_dir / "test_lengths.npy",
        output_dir / "test_labels.npy",
        output_dir / "test_scenario_ids.npy",
    ]
    return paths


def _prepared_summary_matches(
    summary: PrepareSummary,
    *,
    resolved_representation: str,
    output_dir: Path,
    router_dir: Path,
    test_ratio: float,
    selected_scenarios: list[str],
    metadata: RouterEmbeddingMetadata,
) -> bool:
    if summary.prepared_format != PREPARED_FORMAT:
        return False
    if summary.router_dir != str(router_dir):
        return False
    if summary.input_representation != resolved_representation:
        return False
    if abs(float(summary.test_ratio) - float(test_ratio)) > 1e-9:
        return False
    if sorted(summary.scenarios) != sorted(selected_scenarios):
        return False
    if summary.chat_template != metadata.chat_template:
        return False
    if summary.embedding_model != metadata.embedding_model:
        return False
    if summary.embedding_tokenizer != metadata.tokenizer_name:
        return False
    if summary.embedding_provider != metadata.embedding_provider:
        return False
    if summary.embedding_source != metadata.embedding_source:
        return False
    if summary.vocab_size <= 0 or not summary.token_dtype:
        return False
    if resolved_representation == PRETRAINED_EMBEDDINGS and summary.input_hidden_size <= 0:
        return False
    for path in _required_prepared_files(output_dir, resolved_representation):
        if not path.exists():
            return False
    return True


def _cleanup_prepared_dir(output_dir: Path) -> None:
    for name in (
        "meta.json",
        "source_files.json",
        "train_token_ids.bin",
        "train_token_offsets.npy",
        "train_file_ids.npy",
        "train_offsets.npy",
        "train_lengths.npy",
        "train_labels.npy",
        "train_scenario_ids.npy",
        "test_token_ids.bin",
        "test_token_offsets.npy",
        "test_file_ids.npy",
        "test_offsets.npy",
        "test_lengths.npy",
        "test_labels.npy",
        "test_scenario_ids.npy",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def _scan_router_files(
    files: list[Path],
    *,
    test_ratio: float,
) -> tuple[Counter[str], int, int, int, int]:
    scenario_counts: Counter[str] = Counter()
    train_samples = 0
    test_samples = 0
    train_positive = 0
    test_positive = 0

    for path in files:
        for idx, record in enumerate(_load_jsonl(path), start=1):
            text = str(record["context"])
            label = int(record["label"])
            scenario = str(record["metadata"]["scenario"])
            split = assign_split(text, scenario, test_ratio)
            scenario_counts[scenario] += 1
            if split == "train":
                train_samples += 1
                train_positive += label
            else:
                test_samples += 1
                test_positive += label
            if idx % 200_000 == 0:
                print(f"[prepare:scan] {path.name}: {idx} records")

    return scenario_counts, train_samples, test_samples, train_positive, test_positive


def _iter_jsonl_with_offsets(path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                yield offset, payload


def _scenario_names_from_files(files: list[Path]) -> list[str]:
    names: list[str] = []
    for path in files:
        metadata = _first_record(path).get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Router record metadata must be an object: {path}")
        scenario = str(metadata.get("scenario") or path.stem)
        names.append(scenario)
    return sorted(set(names))


def _resolve_prepare_workers(prepare_workers: int | None, chunk_count: int) -> int:
    if chunk_count <= 1:
        return 1
    if prepare_workers is None:
        return 1
    if prepare_workers <= 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, chunk_count))
    return max(1, min(int(prepare_workers), chunk_count))


def _build_token_cache_chunks(files: list[Path], prepare_workers: int | None) -> list[tuple[int, int, int, int, Path]]:
    total_bytes = sum(path.stat().st_size for path in files)
    requested_workers = prepare_workers if prepare_workers and prepare_workers > 0 else (os.cpu_count() or 1)
    target_chunk_count = max(len(files), int(requested_workers) * 4)
    target_bytes = max(TOKEN_CACHE_MIN_CHUNK_BYTES, total_bytes // max(target_chunk_count, 1))
    chunks: list[tuple[int, int, int, int, Path]] = []
    for file_id, path in enumerate(files):
        size = path.stat().st_size
        if size <= target_bytes:
            chunks.append((file_id, 0, 0, size, path))
            continue
        start = 0
        chunk_id = 0
        while start < size:
            end = min(size, start + target_bytes)
            chunks.append((file_id, chunk_id, start, end, path))
            start = end
            chunk_id += 1
    return chunks


def _encode_ids_batch(encoder: PretrainedEmbeddingEncoder, texts: list[str]) -> list[np.ndarray]:
    encode_batch = getattr(encoder, "encode_ids_batch", None)
    if callable(encode_batch):
        return encode_batch(texts)
    return [encoder.encode_ids(text) for text in texts]


def _empty_split_shard(
    *,
    shard_dir: Path,
    file_id: int,
    chunk_id: int,
    split: str,
    scenario_id_dtype: np.dtype,
) -> tuple[_TokenCacheSplitShard, BinaryIO]:
    prefix = shard_dir / f"file{file_id:04d}_chunk{chunk_id:05d}_{split}"
    token_path = prefix.with_suffix(".tokens.bin")
    lengths_path = prefix.with_suffix(".lengths.npy")
    labels_path = prefix.with_suffix(".labels.npy")
    scenario_ids_path = prefix.with_suffix(".scenario_ids.npy")
    token_handle = token_path.open("wb")
    _save_array(lengths_path, np.empty((0,), dtype=np.uint32))
    _save_array(labels_path, np.empty((0,), dtype=np.uint8))
    _save_array(scenario_ids_path, np.empty((0,), dtype=scenario_id_dtype))
    return (
        _TokenCacheSplitShard(
            token_ids_path=str(token_path),
            lengths_path=str(lengths_path),
            labels_path=str(labels_path),
            scenario_ids_path=str(scenario_ids_path),
            samples=0,
            tokens=0,
            positive=0,
        ),
        token_handle,
    )


def _prepare_token_cache_chunk(
    *,
    file_id: int,
    chunk_id: int,
    path: str,
    start_offset: int,
    end_offset: int,
    test_ratio: float,
    scenario_to_id: dict[str, int],
    backbone_spec: EmbeddingBackboneSpec,
    token_dtype_name: str,
    scenario_id_dtype_name: str,
    shard_dir: str,
    batch_size: int,
) -> _TokenCacheShardResult:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    path_obj = Path(path)
    shard_dir_obj = Path(shard_dir)
    token_dtype = np.dtype(token_dtype_name)
    scenario_id_dtype = np.dtype(scenario_id_dtype_name)
    encoder = PretrainedEmbeddingEncoder(backbone_spec)

    split_shards: dict[str, _TokenCacheSplitShard] = {}
    token_handles: dict[str, BinaryIO] = {}
    split_buffers: dict[str, dict[str, list[int]]] = {
        "train": {"lengths": [], "labels": [], "scenario_ids": []},
        "test": {"lengths": [], "labels": [], "scenario_ids": []},
    }
    split_tokens = {"train": 0, "test": 0}
    split_positive = {"train": 0, "test": 0}
    scenario_counts: Counter[str] = Counter()
    max_sequence_length = 0

    for split in ("train", "test"):
        shard, handle = _empty_split_shard(
            shard_dir=shard_dir_obj,
            file_id=file_id,
            chunk_id=chunk_id,
            split=split,
            scenario_id_dtype=scenario_id_dtype,
        )
        split_shards[split] = shard
        token_handles[split] = handle

    pending_texts: list[str] = []
    pending_labels: list[int] = []
    pending_scenarios: list[str] = []
    pending_splits: list[str] = []

    def flush_pending() -> None:
        nonlocal max_sequence_length
        if not pending_texts:
            return
        encoded_batch = _encode_ids_batch(encoder, pending_texts)
        if len(encoded_batch) != len(pending_texts):
            raise RuntimeError(
                f"Tokenizer returned {len(encoded_batch)} sequences for {len(pending_texts)} inputs"
            )
        for token_ids, label, scenario, split in zip(
            encoded_batch,
            pending_labels,
            pending_scenarios,
            pending_splits,
            strict=True,
        ):
            ids = np.asarray(token_ids, dtype=token_dtype)
            length = int(ids.shape[0])
            max_sequence_length = max(max_sequence_length, length)
            token_handles[split].write(ids.tobytes(order="C"))
            split_buffers[split]["lengths"].append(length)
            split_buffers[split]["labels"].append(label)
            split_buffers[split]["scenario_ids"].append(scenario_to_id[scenario])
            split_tokens[split] += length
            split_positive[split] += label
        pending_texts.clear()
        pending_labels.clear()
        pending_scenarios.clear()
        pending_splits.clear()

    processed = 0
    with path_obj.open("rb") as handle:
        if start_offset > 0:
            handle.seek(start_offset - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        else:
            handle.seek(0)
        while True:
            offset = handle.tell()
            if offset >= end_offset:
                break
            raw = handle.readline()
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"Router record metadata must be an object: {path_obj}")
            text = str(record["context"])
            label = int(record["label"])
            scenario = str(metadata["scenario"])
            if scenario not in scenario_to_id:
                raise ValueError(f"Unknown scenario={scenario!r} in {path_obj}")
            split = assign_split(text, scenario, test_ratio)
            scenario_counts[scenario] += 1
            pending_texts.append(text)
            pending_labels.append(label)
            pending_scenarios.append(scenario)
            pending_splits.append(split)
            processed += 1
            if len(pending_texts) >= batch_size:
                flush_pending()
            if processed % 50_000 == 0:
                print(
                    f"[prepare:embed] {path_obj.name} chunk={chunk_id} "
                    f"records={processed} offset={offset}"
                )
    flush_pending()

    for split, handle in token_handles.items():
        handle.close()
        lengths = np.asarray(split_buffers[split]["lengths"], dtype=np.uint32)
        labels = np.asarray(split_buffers[split]["labels"], dtype=np.uint8)
        scenario_ids = np.asarray(split_buffers[split]["scenario_ids"], dtype=scenario_id_dtype)
        shard = split_shards[split]
        _save_array(Path(shard.lengths_path), lengths)
        _save_array(Path(shard.labels_path), labels)
        _save_array(Path(shard.scenario_ids_path), scenario_ids)
        split_shards[split] = _TokenCacheSplitShard(
            token_ids_path=shard.token_ids_path,
            lengths_path=shard.lengths_path,
            labels_path=shard.labels_path,
            scenario_ids_path=shard.scenario_ids_path,
            samples=int(lengths.shape[0]),
            tokens=int(split_tokens[split]),
            positive=int(split_positive[split]),
        )

    return _TokenCacheShardResult(
        file_id=file_id,
        chunk_id=chunk_id,
        start_offset=start_offset,
        end_offset=end_offset,
        scenario_counts=dict(scenario_counts),
        max_sequence_length=max_sequence_length,
        train=split_shards["train"],
        test=split_shards["test"],
    )


def _concat_arrays(arrays: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not arrays:
        return np.empty((0,), dtype=dtype)
    return np.concatenate(arrays).astype(dtype, copy=False)


def _merge_token_cache_split(
    *,
    output_dir: Path,
    split: str,
    shards: list[_TokenCacheShardResult],
    scenario_id_dtype: np.dtype,
) -> tuple[int, int, int]:
    final_token_path = output_dir / f"{split}_token_ids.bin"
    offset_arrays: list[np.ndarray] = []
    length_arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []
    scenario_arrays: list[np.ndarray] = []
    total_tokens = 0
    total_samples = 0
    total_positive = 0

    with final_token_path.open("wb") as output_handle:
        for shard in shards:
            split_shard = getattr(shard, split)
            lengths = _load_array(Path(split_shard.lengths_path))
            labels = _load_array(Path(split_shard.labels_path))
            scenario_ids = _load_array(Path(split_shard.scenario_ids_path))
            if lengths.size:
                starts = np.empty(lengths.shape[0], dtype=np.uint64)
                starts[0] = total_tokens
                if lengths.shape[0] > 1:
                    starts[1:] = total_tokens + np.cumsum(lengths[:-1], dtype=np.uint64)
            else:
                starts = np.empty((0,), dtype=np.uint64)
            offset_arrays.append(starts)
            length_arrays.append(lengths.astype(np.uint32, copy=False))
            label_arrays.append(labels.astype(np.uint8, copy=False))
            scenario_arrays.append(scenario_ids.astype(scenario_id_dtype, copy=False))
            total_tokens += int(lengths.astype(np.uint64, copy=False).sum())
            total_samples += int(lengths.shape[0])
            total_positive += int(labels.astype(np.uint64, copy=False).sum())
            with Path(split_shard.token_ids_path).open("rb") as input_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024 * 8)

    _save_array(output_dir / f"{split}_token_offsets.npy", _concat_arrays(offset_arrays, np.dtype(np.uint64)))
    _save_array(output_dir / f"{split}_lengths.npy", _concat_arrays(length_arrays, np.dtype(np.uint32)))
    _save_array(output_dir / f"{split}_labels.npy", _concat_arrays(label_arrays, np.dtype(np.uint8)))
    _save_array(output_dir / f"{split}_scenario_ids.npy", _concat_arrays(scenario_arrays, scenario_id_dtype))
    return total_samples, total_tokens, total_positive


def _prepare_embedding_router_dataset(
    *,
    simulator: str,
    files: list[Path],
    router_dir: Path,
    output_dir: Path,
    test_ratio: float,
    metadata: RouterEmbeddingMetadata,
    prepare_workers: int | None,
) -> PrepareSummary:
    _log_prepare(
        "starting dataset preparation "
        f"simulator={simulator} files={len(files)} test_ratio={test_ratio:.2f}"
    )
    scenario_names = _scenario_names_from_files(files)
    scenario_to_id = {name: idx for idx, name in enumerate(scenario_names)}
    scenario_id_dtype = np.uint8 if len(scenario_to_id) <= (np.iinfo(np.uint8).max + 1) else np.uint16
    _log_prepare(
        "loading tokenizer metadata "
        f"chat_template={metadata.chat_template or DEFAULT_CHAT_TEMPLATE} "
        f"embedding_model={metadata.embedding_model} tokenizer={metadata.tokenizer_name}"
    )
    encoder = PretrainedEmbeddingEncoder(metadata.to_backbone_spec())
    _log_prepare(
        f"tokenizer ready vocab_size={encoder.model_vocab_size} hidden_size={encoder.hidden_size}"
    )
    token_dtype = np.uint16 if encoder.model_vocab_size <= (np.iinfo(np.uint16).max + 1) else np.uint32
    chunks = _build_token_cache_chunks(files, prepare_workers)
    workers = _resolve_prepare_workers(prepare_workers, len(chunks))
    _log_prepare(
        "token cache build "
        f"workers={workers} chunks={len(chunks)} batch_size={TOKEN_CACHE_BATCH_SIZE} "
        f"token_dtype={np.dtype(token_dtype).name}"
    )

    (output_dir / "source_files.json").write_text(
        json.dumps([str(path) for path in files], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shard_dir = Path(tempfile.mkdtemp(prefix="token-cache-shards-", dir=str(output_dir)))
    results: list[_TokenCacheShardResult] = []
    try:
        worker_kwargs = [
            {
                "file_id": file_id,
                "chunk_id": chunk_id,
                "path": str(path),
                "start_offset": start,
                "end_offset": end,
                "test_ratio": test_ratio,
                "scenario_to_id": scenario_to_id,
                "backbone_spec": metadata.to_backbone_spec(),
                "token_dtype_name": np.dtype(token_dtype).name,
                "scenario_id_dtype_name": np.dtype(scenario_id_dtype).name,
                "shard_dir": str(shard_dir),
                "batch_size": TOKEN_CACHE_BATCH_SIZE,
            }
            for file_id, chunk_id, start, end, path in chunks
        ]
        if workers <= 1:
            for kwargs in worker_kwargs:
                result = _prepare_token_cache_chunk(**kwargs)
                results.append(result)
                _log_prepare(
                    f"chunk complete file={result.file_id + 1}/{len(files)} "
                    f"chunk={result.chunk_id} train={result.train.samples} test={result.test.samples}"
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_prepare_token_cache_chunk, **kwargs) for kwargs in worker_kwargs]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    _log_prepare(
                        f"chunk complete file={result.file_id + 1}/{len(files)} "
                        f"chunk={result.chunk_id} train={result.train.samples} test={result.test.samples}"
                    )

        results.sort(key=lambda item: (item.file_id, item.start_offset, item.chunk_id))
        scenario_counts: Counter[str] = Counter()
        max_sequence_length = 0
        for result in results:
            scenario_counts.update(result.scenario_counts)
            max_sequence_length = max(max_sequence_length, result.max_sequence_length)

        train_samples, train_tokens, train_positive = _merge_token_cache_split(
            output_dir=output_dir,
            split="train",
            shards=results,
            scenario_id_dtype=np.dtype(scenario_id_dtype),
        )
        test_samples, test_tokens, test_positive = _merge_token_cache_split(
            output_dir=output_dir,
            split="test",
            shards=results,
            scenario_id_dtype=np.dtype(scenario_id_dtype),
        )
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)

    _log_prepare(
        "prepared token cache written "
        f"output_dir={output_dir} max_sequence_length={max_sequence_length} "
        f"train_tokens={train_tokens} test_tokens={test_tokens}"
    )

    return PrepareSummary(
        simulator=simulator,
        router_dir=str(router_dir),
        test_ratio=test_ratio,
        scenarios=sorted(scenario_counts),
        scenario_to_id=scenario_to_id,
        vocab_size=encoder.model_vocab_size,
        token_dtype=np.dtype(token_dtype).name,
        train_samples=train_samples,
        test_samples=test_samples,
        train_positive=train_positive,
        test_positive=test_positive,
        train_tokens=train_tokens,
        test_tokens=test_tokens,
        max_sequence_length=max_sequence_length,
        input_representation=PRETRAINED_EMBEDDINGS,
        input_storage_dtype="",
        input_hidden_size=encoder.hidden_size,
        chat_template=metadata.chat_template,
        embedding_provider=metadata.embedding_provider,
        embedding_model=metadata.embedding_model,
        embedding_tokenizer=metadata.tokenizer_name,
        embedding_source=metadata.embedding_source,
        prepared_format=PREPARED_FORMAT,
    )


def prepare_router_dataset(
    *,
    simulator: str,
    router_dir: Path,
    output_dir: Path,
    test_ratio: float,
    scenarios: list[str] | None = None,
    force: bool = False,
    input_representation: str = "embedding",
    prepare_workers: int | None = None,
) -> PrepareSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "meta.json"

    files = _selected_scenario_files(router_dir, simulator, scenarios)
    metadata = _collect_embedding_metadata(files)
    resolved_representation = _resolve_input_representation(input_representation, metadata)
    _log_prepare(
        "requested dataset "
        f"simulator={simulator} scenarios={','.join(path.stem for path in files)} "
        f"representation={resolved_representation} output_dir={output_dir}"
    )

    if summary_path.exists() and not force:
        cached = PrepareSummary.from_dict(json.loads(summary_path.read_text(encoding="utf-8")))
        if _prepared_summary_matches(
            cached,
            resolved_representation=resolved_representation,
            output_dir=output_dir,
            router_dir=router_dir,
            test_ratio=test_ratio,
            selected_scenarios=[path.stem for path in files],
            metadata=metadata,
        ):
            _log_prepare(
                "reusing cached prepared dataset "
                f"train_samples={cached.train_samples} test_samples={cached.test_samples} "
                f"prepared_format={cached.prepared_format}"
            )
            return cached

    _log_prepare(f"rebuilding prepared dataset in {output_dir}")
    _cleanup_prepared_dir(output_dir)
    summary = _prepare_embedding_router_dataset(
        simulator=simulator,
        files=files,
        router_dir=router_dir,
        output_dir=output_dir,
        test_ratio=test_ratio,
        metadata=metadata,
        prepare_workers=prepare_workers,
    )

    summary_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    _log_prepare(
        "prepared dataset ready "
        f"train_samples={summary.train_samples} test_samples={summary.test_samples} "
        f"max_sequence_length={summary.max_sequence_length}"
    )
    return summary


class PackedSequenceDataset(Dataset):
    def __init__(
        self,
        *,
        prepared_dir: Path,
        split: str,
        summary: PrepareSummary,
        pad_id: int,
        max_samples: int | None = None,
    ) -> None:
        self.prepared_dir = prepared_dir
        self.split = split
        self.summary = summary
        self.pad_id = pad_id
        self.input_representation = summary.input_representation
        self.token_dtype = np.dtype(summary.token_dtype) if summary.token_dtype else None
        self.source_files = json.loads((prepared_dir / "source_files.json").read_text(encoding="utf-8"))
        self.has_token_cache = (prepared_dir / f"{split}_token_ids.bin").exists() and (
            prepared_dir / f"{split}_token_offsets.npy"
        ).exists()
        self.token_offsets: np.ndarray | None = None
        self.token_ids: np.memmap | None = None
        self.file_ids: np.ndarray | None = None
        self.offsets: np.ndarray | None = None
        self.lengths = _load_array(prepared_dir / f"{split}_lengths.npy", mmap_mode="r")
        self.labels = _load_array(prepared_dir / f"{split}_labels.npy", mmap_mode="r")
        self.scenario_ids = _load_array(prepared_dir / f"{split}_scenario_ids.npy", mmap_mode="r")
        if self.has_token_cache:
            if self.token_dtype is None:
                raise ValueError("Prepared token cache requires summary.token_dtype")
            self.token_offsets = _load_array(prepared_dir / f"{split}_token_offsets.npy", mmap_mode="r")
            total_tokens = int(self.token_offsets[-1] + self.lengths[-1]) if self.lengths.shape[0] else 0
            self.token_ids = np.memmap(
                prepared_dir / f"{split}_token_ids.bin",
                dtype=self.token_dtype,
                mode="r",
                shape=(total_tokens,),
            )
        else:
            self.file_ids = _load_array(prepared_dir / f"{split}_file_ids.npy", mmap_mode="r")
            self.offsets = _load_array(prepared_dir / f"{split}_offsets.npy", mmap_mode="r")
        if max_samples is not None:
            if self.file_ids is not None:
                self.file_ids = self.file_ids[:max_samples]
            if self.offsets is not None:
                self.offsets = self.offsets[:max_samples]
            if self.token_offsets is not None:
                self.token_offsets = self.token_offsets[:max_samples]
            self.lengths = self.lengths[:max_samples]
            self.labels = self.labels[:max_samples]
            self.scenario_ids = self.scenario_ids[:max_samples]
        self._handles: dict[int, BinaryIO] = {}
        self._encoder: PretrainedEmbeddingEncoder | None = None

    def _get_handle(self, file_id: int) -> BinaryIO:
        handle = self._handles.get(file_id)
        if handle is None:
            handle = Path(self.source_files[file_id]).open("rb")
            self._handles[file_id] = handle
        return handle

    def _get_encoder(self) -> PretrainedEmbeddingEncoder:
        if self._encoder is None:
            self._encoder = PretrainedEmbeddingEncoder(
                EmbeddingBackboneSpec(
                    model_name=self.summary.embedding_model,
                    tokenizer_name=self.summary.embedding_tokenizer or self.summary.embedding_model,
                    provider=self.summary.embedding_provider,
                    chat_template=self.summary.chat_template,
                    source=self.summary.embedding_source,
                )
            )
        return self._encoder

    def _load_record(self, file_id: int, offset: int) -> dict[str, object]:
        handle = self._get_handle(file_id)
        handle.seek(offset)
        raw = handle.readline()
        if not raw:
            raise RuntimeError(f"missing router record at file_id={file_id}, offset={offset}")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"router record at file_id={file_id}, offset={offset} is not an object")
        return payload

    def __len__(self) -> int:
        return int(self.lengths.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        length = int(self.lengths[index])
        if self.has_token_cache:
            if self.token_offsets is None or self.token_ids is None:
                raise RuntimeError(f"missing token cache for split={self.split}")
            start = int(self.token_offsets[index])
            token_ids = np.asarray(self.token_ids[start : start + length]).astype(np.int64, copy=True)
        else:
            if self.file_ids is None or self.offsets is None:
                raise RuntimeError(f"missing dynamic record index for split={self.split}")
            file_id = int(self.file_ids[index])
            offset = int(self.offsets[index])
            record = self._load_record(file_id, offset)
            text = str(record["context"])
            token_ids = self._get_encoder().encode_ids(text)
            if int(token_ids.shape[0]) != length:
                raise RuntimeError(
                    f"dynamic token length mismatch at file_id={file_id}, offset={offset}: "
                    f"expected {length}, got {int(token_ids.shape[0])}"
                )
        return {
            "length": length,
            "label": int(self.labels[index]),
            "scenario_id": int(self.scenario_ids[index]),
            "input_ids": torch.from_numpy(token_ids),
        }

    def __del__(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        order = np.argsort(lengths, kind="stable")
        self.batches = [order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size)]
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        indices = list(range(len(self.batches)))
        if self.shuffle:
            generator = np.random.default_rng(self.seed + self._epoch)
            generator.shuffle(indices)
        for idx in indices:
            batch = self.batches[idx].copy()
            if self.shuffle:
                generator = np.random.default_rng(self.seed + self._epoch + idx + 1)
                generator.shuffle(batch)
            yield batch


def collate_batch(batch: list[dict[str, object]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(item["length"]) for item in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.float32)
    scenario_ids = torch.empty(batch_size, dtype=torch.long)

    for row, item in enumerate(batch):
        ids = item["input_ids"]
        length = int(item["length"])
        input_ids[row, :length] = ids
        attention_mask[row, :length] = True
        labels[row] = float(item["label"])
        scenario_ids[row] = int(item["scenario_id"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "scenario_ids": scenario_ids,
    }
