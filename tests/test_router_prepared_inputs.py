from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from piern.training.router.data import (
    CHAR_TOKENS,
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


def test_prepare_router_dataset_embedding_mode_writes_token_ids(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    scenario_path = router_dir / "by_scenario" / "coastal_seawater.jsonl"
    records = [
        {
            "context": "<|im_start|>user\nhello one<|im_end|>\n<|im_start|>assistant\nanswer",
            "label": 1,
            "metadata": {
                "simulator": "modflow",
                "scenario": "coastal_seawater",
                "chat_template": "qwen",
                "embedding_provider": "siliconflow",
                "embedding_model": "Qwen/Qwen2.5-7B-Instruct",
                "embedding_tokenizer": "Qwen/Qwen2.5-7B-Instruct",
                "embedding_source": "llm_config",
            },
        },
        {
            "context": "<|im_start|>user\nhello two<|im_end|>\n<|im_start|>assistant\nans",
            "label": 0,
            "metadata": {
                "simulator": "modflow",
                "scenario": "coastal_seawater",
                "chat_template": "qwen",
                "embedding_provider": "siliconflow",
                "embedding_model": "Qwen/Qwen2.5-7B-Instruct",
                "embedding_tokenizer": "Qwen/Qwen2.5-7B-Instruct",
                "embedding_source": "llm_config",
            },
        },
    ]
    _write_router_jsonl(scenario_path, records)

    class FakeEncoder:
        hidden_size = 3
        model_vocab_size = 17

        def __init__(self, spec):
            self.spec = spec

        def encode_ids(self, text: str):
            length = 2 if "one" in text else 4
            return np.arange(1, length + 1, dtype=np.int64)

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
    assert summary.embedding_model == "Qwen/Qwen2.5-7B-Instruct"
    assert summary.vocab_size == 17
    assert (prepared_dir / "train_tokens.bin").exists()
    assert not (prepared_dir / "train_embeddings.bin").exists()

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


def test_prepare_router_dataset_auto_falls_back_to_char_tokens(tmp_path):
    router_dir = tmp_path / "router"
    scenario_path = router_dir / "by_scenario" / "coastal_seawater.jsonl"
    records = [
        {
            "context": "<|im_start|>user\nhello one<|im_end|>\n<|im_start|>assistant\nanswer",
            "label": 1,
            "metadata": {
                "simulator": "modflow",
                "scenario": "coastal_seawater",
                "chat_template": "qwen",
            },
        }
    ]
    _write_router_jsonl(scenario_path, records)

    prepared_dir = tmp_path / "prepared"
    summary = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=prepared_dir,
        test_ratio=0.0,
        force=True,
        input_representation="auto",
    )

    assert summary.input_representation == CHAR_TOKENS
    assert (prepared_dir / "vocab.json").exists()


def test_prepare_router_dataset_auto_falls_back_when_backbone_unresolvable(tmp_path, monkeypatch):
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
                    "chat_template": "qwen",
                    "embedding_provider": "siliconflow",
                    "embedding_model": "not-a-real-model",
                    "embedding_tokenizer": "not-a-real-tokenizer",
                    "embedding_source": "llm_config",
                },
            }
        ],
    )
    monkeypatch.setattr(
        "piern.training.router.data.can_resolve_embedding_backbone",
        lambda spec: (False, "unreachable backbone"),
    )

    summary = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=tmp_path / "prepared",
        test_ratio=0.0,
        force=True,
        input_representation="auto",
    )

    assert summary.input_representation == CHAR_TOKENS


def test_prepare_router_dataset_forced_embedding_requires_metadata(tmp_path):
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

    with pytest.raises(ValueError, match="embedding_model"):
        prepare_router_dataset(
            simulator="modflow",
            router_dir=router_dir,
            output_dir=tmp_path / "prepared",
            test_ratio=0.0,
            force=True,
            input_representation="embedding",
        )
