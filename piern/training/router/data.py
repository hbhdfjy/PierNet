from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .tokenizer import CharTokenizer

ASSISTANT_MARKER = "<|im_start|>assistant\n"


@dataclass(slots=True)
class PrepareSummary:
    simulator: str
    router_dir: str
    test_ratio: float
    scenarios: list[str]
    scenario_to_id: dict[str, int]
    vocab_size: int
    token_dtype: str
    train_samples: int
    test_samples: int
    train_positive: int
    test_positive: int
    train_tokens: int
    test_tokens: int
    max_sequence_length: int

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


def prepare_router_dataset(
    *,
    simulator: str,
    router_dir: Path,
    output_dir: Path,
    test_ratio: float,
    scenarios: list[str] | None = None,
    force: bool = False,
) -> PrepareSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "meta.json"
    vocab_path = output_dir / "vocab.json"
    if summary_path.exists() and vocab_path.exists() and not force:
        return PrepareSummary(**json.loads(summary_path.read_text(encoding="utf-8")))

    files = _scenario_files(router_dir, simulator)
    if scenarios:
        wanted = set(scenarios)
        files = [path for path in files if path.stem in wanted]
        if not files:
            raise FileNotFoundError(
                f"No router files found for simulator={simulator!r} and scenarios={sorted(wanted)!r} under {router_dir}"
            )
    train_counter: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    max_sequence_length = 0
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

    summary = PrepareSummary(
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
    )
    summary_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class PackedSequenceDataset(Dataset):
    def __init__(
        self,
        *,
        prepared_dir: Path,
        split: str,
        token_dtype: str,
        pad_id: int,
        max_samples: int | None = None,
    ) -> None:
        self.prepared_dir = prepared_dir
        self.split = split
        self.token_dtype = np.dtype(token_dtype)
        self.pad_id = pad_id
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
            "input_ids": torch.from_numpy(token_ids),
            "length": length,
            "label": int(self.labels[index]),
            "scenario_id": int(self.scenario_ids[index]),
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
