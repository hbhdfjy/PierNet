from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
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
PREPARED_FORMAT = "router_dynamic_tokens_v3"
DEFAULT_CHAT_TEMPLATE = "qwen"
DEFAULT_QWEN_EMBEDDING_MODEL = "/data/models/Qwen/Qwen2.5-0.5B-Instruct"
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
        output_dir / "train_file_ids.npy",
        output_dir / "train_offsets.npy",
        output_dir / "train_lengths.npy",
        output_dir / "train_labels.npy",
        output_dir / "train_scenario_ids.npy",
        output_dir / "test_file_ids.npy",
        output_dir / "test_offsets.npy",
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
        "train_file_ids.npy",
        "train_offsets.npy",
        "train_lengths.npy",
        "train_labels.npy",
        "train_scenario_ids.npy",
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


def _prepare_embedding_router_dataset(
    *,
    simulator: str,
    files: list[Path],
    router_dir: Path,
    output_dir: Path,
    test_ratio: float,
    metadata: RouterEmbeddingMetadata,
) -> PrepareSummary:
    _log_prepare(
        "starting dataset preparation "
        f"simulator={simulator} files={len(files)} test_ratio={test_ratio:.2f}"
    )
    scenario_counts, train_samples, test_samples, train_positive, test_positive = _scan_router_files(
        files,
        test_ratio=test_ratio,
    )
    _log_prepare(
        "scan complete "
        f"train_samples={train_samples} test_samples={test_samples} "
        f"train_positive={train_positive} test_positive={test_positive}"
    )
    scenario_to_id = {name: idx for idx, name in enumerate(sorted(scenario_counts))}
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
    file_id_dtype = np.uint16 if len(files) <= (np.iinfo(np.uint16).max + 1) else np.uint32
    buffers: dict[str, defaultdict[str, list[int]]] = {
        "train": defaultdict(list),
        "test": defaultdict(list),
    }
    token_totals = {"train": 0, "test": 0}
    max_sequence_length = 0

    for file_id, path in enumerate(files):
        _log_prepare(f"indexing router file {file_id + 1}/{len(files)}: {path.name}")
        for idx, (offset, record) in enumerate(_iter_jsonl_with_offsets(path), start=1):
            text = str(record["context"])
            label = int(record["label"])
            scenario = str(record["metadata"]["scenario"])
            split = assign_split(text, scenario, test_ratio)
            token_ids = encoder.encode_ids(text)
            length = int(token_ids.shape[0])
            max_sequence_length = max(max_sequence_length, length)
            buffers[split]["file_ids"].append(file_id)
            buffers[split]["lengths"].append(length)
            buffers[split]["labels"].append(label)
            buffers[split]["scenario_ids"].append(scenario_to_id[scenario])
            buffers[split]["line_offsets"].append(offset)
            token_totals[split] += length
            if idx % 50_000 == 0:
                print(f"[prepare:embed] {path.name}: {idx} records")

    (output_dir / "source_files.json").write_text(
        json.dumps([str(path) for path in files], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for split in ("train", "test"):
        _save_array(output_dir / f"{split}_file_ids.npy", np.asarray(buffers[split]["file_ids"], dtype=file_id_dtype))
        _save_array(output_dir / f"{split}_offsets.npy", np.asarray(buffers[split]["line_offsets"], dtype=np.uint64))
        _save_array(output_dir / f"{split}_lengths.npy", np.asarray(buffers[split]["lengths"], dtype=np.uint32))
        _save_array(output_dir / f"{split}_labels.npy", np.asarray(buffers[split]["labels"], dtype=np.uint8))
        _save_array(output_dir / f"{split}_scenario_ids.npy", np.asarray(buffers[split]["scenario_ids"], dtype=scenario_id_dtype))

    _log_prepare(
        "prepared arrays written "
        f"output_dir={output_dir} max_sequence_length={max_sequence_length} "
        f"train_tokens={token_totals['train']} test_tokens={token_totals['test']}"
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
        train_tokens=token_totals["train"],
        test_tokens=token_totals["test"],
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
        self.file_ids = _load_array(prepared_dir / f"{split}_file_ids.npy")
        self.offsets = _load_array(prepared_dir / f"{split}_offsets.npy")
        self.lengths = _load_array(prepared_dir / f"{split}_lengths.npy")
        self.labels = _load_array(prepared_dir / f"{split}_labels.npy")
        self.scenario_ids = _load_array(prepared_dir / f"{split}_scenario_ids.npy")
        if max_samples is not None:
            self.file_ids = self.file_ids[:max_samples]
            self.offsets = self.offsets[:max_samples]
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
        file_id = int(self.file_ids[index])
        offset = int(self.offsets[index])
        length = int(self.lengths[index])
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


