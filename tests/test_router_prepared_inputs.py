from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from piern.training.router.data import (
    DEFAULT_QWEN_EMBEDDING_MODEL,
    PRETRAINED_EMBEDDINGS,
    PackedSequenceDataset,
    collate_batch,
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
