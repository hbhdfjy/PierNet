from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .pretrained_embeddings import (
    EmbeddingBackboneSpec,
    PretrainedEmbeddingEncoder,
    can_resolve_embedding_backbone,
)
from .tokenizer import CharTokenizer

ASSISTANT_MARKER = "<|im_start|>assistant\n"
CHAR_TOKENS = "char_tokens"
PRETRAINED_EMBEDDINGS = "pretrained_embeddings"
SUPPORTED_INPUT_REPRESENTATIONS = {"auto", "char", "embedding"}
PREPARED_FORMAT = "router_token_ids_v2"


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
    input_representation: str = CHAR_TOKENS
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


def _collect_embedding_metadata(files: list[Path]) -> RouterEmbeddingMetadata:
    chat_templates: set[str] = set()
    embedding_specs: set[tuple[str, str, str, str]] = set()
    missing_embedding_metadata = False

    for path in files:
        metadata = _first_record(path).get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Router record metadata must be an object: {path}")
        chat_template = str(metadata.get("chat_template") or "").strip()
        if chat_template:
            chat_templates.add(chat_template)

        provider = str(metadata.get("embedding_provider") or "").strip()
        model = str(metadata.get("embedding_model") or "").strip()
        tokenizer = str(metadata.get("embedding_tokenizer") or "").strip()
        source = str(metadata.get("embedding_source") or "").strip()
        if model or tokenizer or provider or source:
            if not model:
                raise ValueError(f"Router metadata is missing embedding_model in {path}")
            embedding_specs.add((provider, model, tokenizer or model, source))
        else:
            missing_embedding_metadata = True

    if len(chat_templates) > 1:
        raise ValueError(f"Selected router files mix different chat templates: {sorted(chat_templates)}")
    if len(embedding_specs) > 1:
        raise ValueError("Selected router files mix different embedding backbones; rebuild them consistently")
    if embedding_specs and missing_embedding_metadata:
        raise ValueError("Selected router files mix old records without embedding metadata and new embedding-aware records")

    if embedding_specs:
        provider, model, tokenizer, source = next(iter(embedding_specs))
        return RouterEmbeddingMetadata(
            chat_template=next(iter(chat_templates), ""),
            embedding_provider=provider,
            embedding_model=model,
            embedding_tokenizer=tokenizer,
            embedding_source=source,
        )
    return RouterEmbeddingMetadata(chat_template=next(iter(chat_templates), ""))


def _resolve_input_representation(
    requested: str,
    metadata: RouterEmbeddingMetadata,
) -> str:
    requested = requested.strip().lower()
    if requested not in SUPPORTED_INPUT_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported input_representation={requested!r}; "
            f"expected one of {sorted(SUPPORTED_INPUT_REPRESENTATIONS)!r}"
        )
    if requested == "char":
        return CHAR_TOKENS
    if requested == "auto":
        if not metadata.has_embedding_backbone:
            return CHAR_TOKENS
        ok, reason = can_resolve_embedding_backbone(metadata.to_backbone_spec())
        if ok:
            return PRETRAINED_EMBEDDINGS
        print(
            "[prepare] embedding backbone could not be resolved; "
            f"falling back to char tokens ({reason})"
        )
        return CHAR_TOKENS
    if not metadata.has_embedding_backbone:
        raise ValueError(
            "Embedding mode requested, but router metadata does not include embedding_model / embedding_tokenizer. "
            "Rebuild Stage 4 router data with embedding metadata first."
        )
    ok, reason = can_resolve_embedding_backbone(metadata.to_backbone_spec())
    if not ok:
        raise ValueError(
            "Embedding mode requested, but embedding backbone cannot be resolved: "
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
    input_representation: str = "auto",
) -> tuple[str, RouterEmbeddingMetadata]:
    files = _selected_scenario_files(router_dir, simulator, scenarios)
    metadata = _collect_embedding_metadata(files)
    resolved_representation = _resolve_input_representation(input_representation, metadata)
    return resolved_representation, metadata


def _required_prepared_files(output_dir: Path, representation: str) -> list[Path]:
    paths = [
        output_dir / "train_tokens.bin",
        output_dir / "test_tokens.bin",
        output_dir / "train_offsets.npy",
        output_dir / "train_lengths.npy",
        output_dir / "train_labels.npy",
        output_dir / "train_scenario_ids.npy",
        output_dir / "test_offsets.npy",
        output_dir / "test_lengths.npy",
        output_dir / "test_labels.npy",
        output_dir / "test_scenario_ids.npy",
    ]
    if representation == CHAR_TOKENS:
        paths.append(output_dir / "vocab.json")
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
        "vocab.json",
        "train_tokens.bin",
        "test_tokens.bin",
        "train_embeddings.bin",
        "test_embeddings.bin",
        "train_offsets.npy",
        "train_lengths.npy",
        "train_labels.npy",
        "train_scenario_ids.npy",
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


def _prepare_char_router_dataset(
    *,
    simulator: str,
    files: list[Path],
    router_dir: Path,
    output_dir: Path,
    test_ratio: float,
    metadata: RouterEmbeddingMetadata,
) -> PrepareSummary:
    train_counter: Counter[str] = Counter()
    max_sequence_length = 0
    train_samples = 0
    test_samples = 0
    train_positive = 0
    test_positive = 0
    scenario_counts: Counter[str] = Counter()

    for path in files:
        for idx, record in enumerate(_load_jsonl(path), start=1):
            text = str(record["context"])
            label = int(record["label"])
            scenario = str(record["metadata"]["scenario"])
            split = assign_split(text, scenario, test_ratio)
            scenario_counts[scenario] += 1
            max_sequence_length = max(max_sequence_length, len(text))
            if split == "train":
                train_counter.update(text)
                train_samples += 1
                train_positive += label
            else:
                test_samples += 1
                test_positive += label
            if idx % 200_000 == 0:
                print(f"[prepare:scan] {path.name}: {idx} records")

    vocab_path = output_dir / "vocab.json"
    tokenizer = CharTokenizer.from_counter(train_counter)
    tokenizer.save(vocab_path)
    token_dtype = np.uint16 if tokenizer.vocab_size <= (np.iinfo(np.uint16).max + 1) else np.uint32
    scenario_to_id = {name: idx for idx, name in enumerate(sorted(scenario_counts))}
    scenario_id_dtype = np.uint8 if len(scenario_to_id) <= (np.iinfo(np.uint8).max + 1) else np.uint16
    writers = {
        "train": (output_dir / "train_tokens.bin").open("wb"),
        "test": (output_dir / "test_tokens.bin").open("wb"),
    }
    buffers: dict[str, defaultdict[str, list[int]]] = {
        "train": defaultdict(list),
        "test": defaultdict(list),
    }
    token_totals = {"train": 0, "test": 0}

    try:
        for path in files:
            for idx, record in enumerate(_load_jsonl(path), start=1):
                text = str(record["context"])
                label = int(record["label"])
                scenario = str(record["metadata"]["scenario"])
                split = assign_split(text, scenario, test_ratio)
                token_ids = tokenizer.encode(text)
                token_array = np.asarray(token_ids, dtype=token_dtype)
                buffers[split]["offsets"].append(token_totals[split])
                buffers[split]["lengths"].append(len(token_ids))
                buffers[split]["labels"].append(label)
                buffers[split]["scenario_ids"].append(scenario_to_id[scenario])
                writers[split].write(token_array.tobytes())
                token_totals[split] += len(token_ids)
                if idx % 200_000 == 0:
                    print(f"[prepare:write] {path.name}: {idx} records")
    finally:
        for handle in writers.values():
            handle.close()

    for split in ("train", "test"):
        _save_array(output_dir / f"{split}_offsets.npy", np.asarray(buffers[split]["offsets"], dtype=np.uint64))
        _save_array(output_dir / f"{split}_lengths.npy", np.asarray(buffers[split]["lengths"], dtype=np.uint32))
        _save_array(output_dir / f"{split}_labels.npy", np.asarray(buffers[split]["labels"], dtype=np.uint8))
        _save_array(output_dir / f"{split}_scenario_ids.npy", np.asarray(buffers[split]["scenario_ids"], dtype=scenario_id_dtype))

    return PrepareSummary(
        simulator=simulator,
        router_dir=str(router_dir),
        test_ratio=test_ratio,
        scenarios=sorted(scenario_counts),
        scenario_to_id=scenario_to_id,
        vocab_size=tokenizer.vocab_size,
        token_dtype=np.dtype(token_dtype).name,
        train_samples=train_samples,
        test_samples=test_samples,
        train_positive=train_positive,
        test_positive=test_positive,
        train_tokens=token_totals["train"],
        test_tokens=token_totals["test"],
        max_sequence_length=max_sequence_length,
        input_representation=CHAR_TOKENS,
        input_storage_dtype=np.dtype(token_dtype).name,
        input_hidden_size=0,
        chat_template=metadata.chat_template,
        embedding_provider=metadata.embedding_provider,
        embedding_model=metadata.embedding_model,
        embedding_tokenizer=metadata.tokenizer_name,
        embedding_source=metadata.embedding_source,
        prepared_format=PREPARED_FORMAT,
    )


def _prepare_embedding_router_dataset(
    *,
    simulator: str,
    files: list[Path],
    router_dir: Path,
    output_dir: Path,
    test_ratio: float,
    metadata: RouterEmbeddingMetadata,
) -> PrepareSummary:
    scenario_counts, train_samples, test_samples, train_positive, test_positive = _scan_router_files(
        files,
        test_ratio=test_ratio,
    )
    scenario_to_id = {name: idx for idx, name in enumerate(sorted(scenario_counts))}
    scenario_id_dtype = np.uint8 if len(scenario_to_id) <= (np.iinfo(np.uint8).max + 1) else np.uint16
    encoder = PretrainedEmbeddingEncoder(metadata.to_backbone_spec())
    token_dtype = np.uint16 if encoder.model_vocab_size <= (np.iinfo(np.uint16).max + 1) else np.uint32
    writers = {
        "train": (output_dir / "train_tokens.bin").open("wb"),
        "test": (output_dir / "test_tokens.bin").open("wb"),
    }
    buffers: dict[str, defaultdict[str, list[int]]] = {
        "train": defaultdict(list),
        "test": defaultdict(list),
    }
    token_totals = {"train": 0, "test": 0}
    max_sequence_length = 0

    try:
        for path in files:
            for idx, record in enumerate(_load_jsonl(path), start=1):
                text = str(record["context"])
                label = int(record["label"])
                scenario = str(record["metadata"]["scenario"])
                split = assign_split(text, scenario, test_ratio)
                token_ids = encoder.encode_ids(text)
                length = int(token_ids.shape[0])
                max_sequence_length = max(max_sequence_length, length)
                token_array = np.asarray(token_ids, dtype=token_dtype)
                buffers[split]["offsets"].append(token_totals[split])
                buffers[split]["lengths"].append(length)
                buffers[split]["labels"].append(label)
                buffers[split]["scenario_ids"].append(scenario_to_id[scenario])
                writers[split].write(token_array.tobytes())
                token_totals[split] += length
                if idx % 50_000 == 0:
                    print(f"[prepare:embed] {path.name}: {idx} records")
    finally:
        for handle in writers.values():
            handle.close()

    for split in ("train", "test"):
        _save_array(output_dir / f"{split}_offsets.npy", np.asarray(buffers[split]["offsets"], dtype=np.uint64))
        _save_array(output_dir / f"{split}_lengths.npy", np.asarray(buffers[split]["lengths"], dtype=np.uint32))
        _save_array(output_dir / f"{split}_labels.npy", np.asarray(buffers[split]["labels"], dtype=np.uint8))
        _save_array(output_dir / f"{split}_scenario_ids.npy", np.asarray(buffers[split]["scenario_ids"], dtype=scenario_id_dtype))

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
        train_tokens=token_totals["train"],
        test_tokens=token_totals["test"],
        max_sequence_length=max_sequence_length,
        input_representation=PRETRAINED_EMBEDDINGS,
        input_storage_dtype=np.dtype(token_dtype).name,
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
    input_representation: str = "auto",
) -> PrepareSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "meta.json"

    files = _selected_scenario_files(router_dir, simulator, scenarios)
    metadata = _collect_embedding_metadata(files)
    resolved_representation = _resolve_input_representation(input_representation, metadata)

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
            return cached

    _cleanup_prepared_dir(output_dir)
    if resolved_representation == PRETRAINED_EMBEDDINGS:
        summary = _prepare_embedding_router_dataset(
            simulator=simulator,
            files=files,
            router_dir=router_dir,
            output_dir=output_dir,
            test_ratio=test_ratio,
            metadata=metadata,
        )
    else:
        summary = _prepare_char_router_dataset(
            simulator=simulator,
            files=files,
            router_dir=router_dir,
            output_dir=output_dir,
            test_ratio=test_ratio,
            metadata=metadata,
        )

    summary_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
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
        self.input_hidden_size = int(summary.input_hidden_size)
        self.total_tokens = int(summary.train_tokens if split == "train" else summary.test_tokens)
        self.offsets = _load_array(prepared_dir / f"{split}_offsets.npy")
        self.lengths = _load_array(prepared_dir / f"{split}_lengths.npy")
        self.labels = _load_array(prepared_dir / f"{split}_labels.npy")
        self.scenario_ids = _load_array(prepared_dir / f"{split}_scenario_ids.npy")
        if max_samples is not None:
            self.offsets = self.offsets[:max_samples]
            self.lengths = self.lengths[:max_samples]
            self.labels = self.labels[:max_samples]
            self.scenario_ids = self.scenario_ids[:max_samples]
        self._tokens: np.memmap | None = None

    def _token_buffer(self) -> np.memmap:
        if self._tokens is None:
            if self.token_dtype is None:
                raise RuntimeError("Token buffer requested for dataset without token_dtype")
            self._tokens = np.memmap(
                self.prepared_dir / f"{self.split}_tokens.bin",
                dtype=self.token_dtype,
                mode="r",
            )
        return self._tokens

    def __len__(self) -> int:
        return int(self.lengths.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        start = int(self.offsets[index])
        length = int(self.lengths[index])
        token_buffer = self._token_buffer()
        token_ids = np.asarray(token_buffer[start : start + length], dtype=np.int64)
        return {
            "length": length,
            "label": int(self.labels[index]),
            "scenario_id": int(self.scenario_ids[index]),
            "input_ids": torch.from_numpy(token_ids),
        }


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
