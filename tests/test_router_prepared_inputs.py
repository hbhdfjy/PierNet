from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from piern.shared.storage import portable
from piern.training.router.data import (
    DEFAULT_QWEN_EMBEDDING_MODEL,
    PREPARED_FORMAT,
    PRETRAINED_EMBEDDINGS,
    PackedSequenceDataset,
    collate_batch,
    inspect_router_input_representation,
    prepare_router_dataset,
)


def _write_router_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_router_meta(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_prepare_router_dataset_embedding_mode_indexes_router_records(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    scenario_path = router_dir / "by_scenario" / "coastal_seawater.jsonl"
    records = [
        {
            "context": "<|im_start|>user\nhello one<|im_end|>\n<|im_start|>assistant\nanswer",
            "label": 1,
            "metadata": {
                "simulator": "modflow",
                "scenario": "coastal_seawater",
                "language": "en",
            },
        },
        {
            "context": "<|im_start|>user\nhello two<|im_end|>\n<|im_start|>assistant\nans",
            "label": 0,
            "metadata": {
                "simulator": "modflow",
                "scenario": "coastal_seawater",
                "language": "en",
            },
        },
    ]
    _write_router_jsonl(scenario_path, records)
    _write_router_meta(
        scenario_path.with_suffix(".meta.json"),
        {
            "scenario": "coastal_seawater",
            "chat_template": "qwen",
            "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
            "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
            "output_count": len(records),
        },
    )

    class FakeEncoder:
        hidden_size = 3
        model_vocab_size = 17

        def __init__(self, spec):
            self.spec = spec

        def encode_ids(self, text: str):
            length = 2 if "one" in text else 4
            return np.arange(1, length + 1, dtype=np.int64)

        def encode_ids_batch(self, texts: list[str]):
            return [self.encode_ids(text) for text in texts]

    monkeypatch.setattr("piern.training.router.data.PretrainedEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr("piern.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    prepared_dir = tmp_path / "prepared"
    summary = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=prepared_dir,
        test_ratio=0.0,
        force=True,
        input_representation="embedding",
    )

    assert summary.input_representation == PRETRAINED_EMBEDDINGS
    assert summary.input_hidden_size == 3
    assert summary.embedding_model == DEFAULT_QWEN_EMBEDDING_MODEL
    assert summary.vocab_size == 17
    assert (prepared_dir / "source_files.json").exists()
    assert (prepared_dir / "train_token_ids.bin").exists()
    assert (prepared_dir / "train_token_offsets.npy").exists()
    assert not (prepared_dir / "train_file_ids.npy").exists()
    assert not (prepared_dir / "train_embeddings.bin").exists()

    class FailingEncoder:
        def __init__(self, spec):
            raise AssertionError("token cache dataset should not initialize tokenizer")

    monkeypatch.setattr("piern.training.router.data.PretrainedEmbeddingEncoder", FailingEncoder)
    dataset = PackedSequenceDataset(
        prepared_dir=prepared_dir,
        split="train",
        summary=summary,
        pad_id=0,
    )
    first = dataset[0]
    assert "input_ids" in first
    batch = collate_batch([dataset[0], dataset[1]], pad_id=0)
    assert "input_ids" in batch
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].dtype == torch.bool


def test_prepare_router_dataset_embedding_defaults_to_qwen_backbone(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    scenario_path = router_dir / "by_scenario" / "coastal_seawater.jsonl"
    _write_router_jsonl(
        scenario_path,
        [
            {
                "context": "plain text",
                "label": 1,
                "metadata": {
                    "simulator": "modflow",
                    "scenario": "coastal_seawater",
                },
            }
        ],
    )

    class FakeEncoder:
        hidden_size = 3
        model_vocab_size = 17

        def __init__(self, spec):
            self.spec = spec

        def encode_ids(self, text: str):
            return np.arange(1, 3, dtype=np.int64)

        def encode_ids_batch(self, texts: list[str]):
            return [self.encode_ids(text) for text in texts]

    monkeypatch.setattr("piern.training.router.data.PretrainedEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr("piern.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    summary = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=tmp_path / "prepared",
        test_ratio=0.0,
        force=True,
        input_representation="embedding",
    )

    assert summary.input_representation == PRETRAINED_EMBEDDINGS
    assert summary.embedding_model == DEFAULT_QWEN_EMBEDDING_MODEL


def test_prepare_router_dataset_parallel_token_cache_preserves_records(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    scenario_path = router_dir / "by_scenario" / "coastal_seawater.jsonl"
    records = [
        {
            "context": f"<|im_start|>user\nsample {idx}<|im_end|>\n<|im_start|>assistant\nanswer {idx}",
            "label": idx % 2,
            "metadata": {
                "simulator": "modflow",
                "scenario": "coastal_seawater",
                "language": "en",
            },
        }
        for idx in range(24)
    ]
    _write_router_jsonl(scenario_path, records)
    _write_router_meta(
        scenario_path.with_suffix(".meta.json"),
        {
            "scenario": "coastal_seawater",
            "chat_template": "qwen",
            "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
            "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
            "output_count": len(records),
        },
    )

    class FakeEncoder:
        hidden_size = 3
        model_vocab_size = 128

        def __init__(self, spec):
            self.spec = spec

        def encode_ids(self, text: str):
            idx = int(text.rsplit(" ", 1)[-1])
            return np.full((idx % 4) + 1, idx + 1, dtype=np.int64)

        def encode_ids_batch(self, texts: list[str]):
            return [self.encode_ids(text) for text in texts]

    monkeypatch.setattr("piern.training.router.data.TOKEN_CACHE_MIN_CHUNK_BYTES", 256)
    monkeypatch.setattr("piern.training.router.data.PretrainedEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr("piern.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    prepared_dir = tmp_path / "prepared"
    summary = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=prepared_dir,
        test_ratio=0.0,
        force=True,
        input_representation="embedding",
        prepare_workers=2,
    )

    dataset = PackedSequenceDataset(
        prepared_dir=prepared_dir,
        split="train",
        summary=summary,
        pad_id=0,
    )
    assert len(dataset) == len(records)
    assert summary.train_tokens == sum((idx % 4) + 1 for idx in range(len(records)))
    for idx in range(len(records)):
        item = dataset[idx]
        assert item["input_ids"].tolist() == [idx + 1] * ((idx % 4) + 1)
        assert item["label"] == idx % 2


def test_prepare_router_dataset_embedding_requires_resolvable_backbone(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    scenario_path = router_dir / "by_scenario" / "coastal_seawater.jsonl"
    _write_router_jsonl(
        scenario_path,
        [
            {
                "context": "plain text",
                "label": 1,
                "metadata": {
                    "simulator": "modflow",
                    "scenario": "coastal_seawater",
                },
            }
        ],
    )
    _write_router_meta(
        scenario_path.with_suffix(".meta.json"),
        {
            "scenario": "coastal_seawater",
            "chat_template": "qwen",
            "output_count": 1,
        },
    )
    monkeypatch.setattr(
        "piern.training.router.data.can_resolve_embedding_backbone",
        lambda spec: (False, "unreachable backbone"),
    )

    with pytest.raises(ValueError, match="resolvable embedding backbone"):
        prepare_router_dataset(
            simulator="modflow",
            router_dir=router_dir,
            output_dir=tmp_path / "prepared",
            test_ratio=0.0,
            force=True,
            input_representation="embedding",
        )


def _write_minimal_prepared_cache(prepared_dir, *, router_dir, scenarios):
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / "source_files.json").write_text("[]", encoding="utf-8")
    for name in ("train_token_ids.bin", "test_token_ids.bin"):
        (prepared_dir / name).write_bytes(b"")
    for name in (
        "train_token_offsets.npy",
        "train_lengths.npy",
        "train_labels.npy",
        "train_scenario_ids.npy",
        "test_token_offsets.npy",
        "test_lengths.npy",
        "test_labels.npy",
        "test_scenario_ids.npy",
    ):
        np.save(prepared_dir / name, np.zeros(1, dtype=np.int64))
    (prepared_dir / "meta.json").write_text(
        json.dumps(
            {
                "simulator": "modflow",
                "router_dir": str(router_dir),
                "test_ratio": 0.1,
                "scenarios": scenarios,
                "scenario_to_id": {scenario: idx for idx, scenario in enumerate(scenarios)},
                "vocab_size": 17,
                "token_dtype": "uint16",
                "train_samples": 1,
                "test_samples": 1,
                "train_positive": 0,
                "test_positive": 0,
                "train_tokens": 0,
                "test_tokens": 0,
                "max_sequence_length": 4,
                "input_representation": PRETRAINED_EMBEDDINGS,
                "input_storage_dtype": "",
                "input_hidden_size": 3,
                "chat_template": "qwen",
                "embedding_provider": "",
                "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
                "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
                "embedding_source": "",
                "prepared_format": PREPARED_FORMAT,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _router_partition(tmp_path, scenario="coastal_seawater", metadata=None):
    partition_dir = tmp_path / "router_parquet" / "simulator=modflow" / f"scenario={scenario}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    return portable.PartitionInfo(
        kind="router",
        simulator="modflow",
        scenario=scenario,
        path=partition_dir,
        row_count=10,
        file_size_bytes=100,
        mtime=1.0,
        metadata=metadata or {
            "chat_template": "qwen",
            "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
            "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
        },
    )


def test_inspect_router_input_representation_reads_parquet_manifest_without_jsonl_export(tmp_path, monkeypatch):
    partition = _router_partition(tmp_path)
    monkeypatch.setattr("piern.training.router.data.portable.discover_partitions", lambda kind: [partition])
    monkeypatch.setattr(
        "piern.training.router.data.portable.export_records_to_jsonl",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not export parquet for inspection")),
    )
    monkeypatch.setattr("piern.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    representation, metadata = inspect_router_input_representation(
        simulator="modflow",
        router_dir=tmp_path / "router",
        scenarios=["coastal_seawater"],
        input_representation="embedding",
    )

    assert representation == PRETRAINED_EMBEDDINGS
    assert metadata.embedding_model == DEFAULT_QWEN_EMBEDDING_MODEL


def test_prepare_router_dataset_reuses_cached_parquet_summary_without_jsonl_export(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    prepared_dir = tmp_path / "prepared"
    scenarios = ["coastal_seawater"]
    _write_minimal_prepared_cache(prepared_dir, router_dir=router_dir, scenarios=scenarios)
    partition = _router_partition(tmp_path, scenario="coastal_seawater")
    monkeypatch.setattr("piern.training.router.data.portable.discover_partitions", lambda kind: [partition])
    monkeypatch.setattr(
        "piern.training.router.data.portable.export_records_to_jsonl",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cached parquet path should not export JSONL")),
    )
    monkeypatch.setattr("piern.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    summary = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=prepared_dir,
        test_ratio=0.1,
        scenarios=scenarios,
        force=False,
        input_representation="embedding",
    )

    assert summary.input_representation == PRETRAINED_EMBEDDINGS
    assert summary.scenarios == scenarios
    assert not (router_dir / ".parquet_jsonl_cache").exists()
