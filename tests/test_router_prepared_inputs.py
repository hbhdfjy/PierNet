from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from PierNet.shared.storage import portable
from PierNet.training.router.data import (
    DEFAULT_QWEN_EMBEDDING_MODEL,
    PREPARED_FORMAT,
    PRETRAINED_EMBEDDINGS,
    PackedSequenceDataset,
    RouterDatasetPreparationCancelled,
    collate_batch,
    inspect_router_input_representation,
    _materialize_parquet_router_files,
    _router_source_fingerprint,
    prepare_router_dataset,
)


@pytest.fixture(autouse=True)
def _isolate_router_jsonl_cache_env(monkeypatch):
    monkeypatch.delenv("PierNet_ROUTER_JSONL_CACHE_DIR", raising=False)


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

    monkeypatch.setattr("PierNet.training.router.data.PretrainedEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

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

    monkeypatch.setattr("PierNet.training.router.data.PretrainedEmbeddingEncoder", FailingEncoder)
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

    monkeypatch.setattr("PierNet.training.router.data.PretrainedEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

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


def test_prepare_router_dataset_stop_cleans_incomplete_cache(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    scenario_path = router_dir / "by_scenario" / "coastal_seawater.jsonl"
    stop_file = tmp_path / "stop.json"
    _write_router_jsonl(
        scenario_path,
        [
            {
                "context": "stop while tokenizing",
                "label": 1,
                "metadata": {
                    "simulator": "modflow",
                    "scenario": "coastal_seawater",
                    "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
                    "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
                },
            }
        ],
    )

    class StoppingEncoder:
        hidden_size = 3
        model_vocab_size = 17

        def __init__(self, spec):
            self.spec = spec

        def encode_ids_batch(self, texts: list[str]):
            stop_file.write_text("stop", encoding="utf-8")
            return [np.arange(1, 3, dtype=np.int64) for _ in texts]

    monkeypatch.setattr("PierNet.training.router.data.PretrainedEmbeddingEncoder", StoppingEncoder)
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    prepared_dir = tmp_path / "prepared"
    with pytest.raises(RouterDatasetPreparationCancelled):
        prepare_router_dataset(
            simulator="modflow",
            router_dir=router_dir,
            output_dir=prepared_dir,
            test_ratio=0.0,
            force=True,
            input_representation="embedding",
            prepare_workers=1,
            stop_file=stop_file,
        )

    assert prepared_dir.exists()
    assert not (prepared_dir / "meta.json").exists()
    assert not (prepared_dir / "source_files.json").exists()
    assert not (prepared_dir / "train_token_ids.bin").exists()


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

    monkeypatch.setattr("PierNet.training.router.data.TOKEN_CACHE_MIN_CHUNK_BYTES", 256)
    monkeypatch.setattr("PierNet.training.router.data.PretrainedEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

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
        "PierNet.training.router.data.can_resolve_embedding_backbone",
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


def _write_minimal_prepared_cache(prepared_dir, *, router_dir, scenarios, source_fingerprint=""):
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
                "source_fingerprint": source_fingerprint,
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
    monkeypatch.setattr("PierNet.training.router.data._router_dir_uses_default_parquet", lambda router_dir: True)
    monkeypatch.setattr("PierNet.training.router.data.portable.discover_partitions", lambda kind: [partition])
    monkeypatch.setattr(
        "PierNet.training.router.data.portable.export_records_to_jsonl",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not export parquet for inspection")),
    )
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    representation, metadata = inspect_router_input_representation(
        simulator="modflow",
        router_dir=tmp_path / "router",
        scenarios=["coastal_seawater"],
        input_representation="embedding",
    )

    assert representation == PRETRAINED_EMBEDDINGS
    assert metadata.embedding_model == DEFAULT_QWEN_EMBEDDING_MODEL


def test_materialize_parquet_router_files_rebuilds_when_cache_meta_counts_are_invalid(
    tmp_path,
    monkeypatch,
):
    router_dir = tmp_path / "router"
    partition = _router_partition(tmp_path, scenario="coastal_seawater")
    cache_root = tmp_path / "router-cache"
    monkeypatch.setenv("PierNet_ROUTER_JSONL_CACHE_DIR", str(cache_root))
    cache_dir = cache_root / "modflow"
    cache_dir.mkdir(parents=True)
    cached_path = cache_dir / "coastal_seawater.jsonl"
    cached_path.write_text("{}\n", encoding="utf-8")
    cached_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "source_path": str(partition.path),
                "source_mtime": "bad-mtime",
                "row_count": "bad-count",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    exports: list[tuple[str | None, str | None]] = []

    def fake_export_records_to_jsonl(kind, output_path, *, simulator=None, scenario=None, **kwargs):
        exports.append((simulator, scenario))
        _write_router_jsonl(
            output_path,
            [
                {
                    "context": "fresh export",
                    "label": 1,
                    "metadata": {"simulator": simulator, "scenario": scenario},
                }
            ],
        )
        return partition.row_count

    monkeypatch.setattr("PierNet.training.router.data.portable.export_records_to_jsonl", fake_export_records_to_jsonl)

    files = _materialize_parquet_router_files(router_dir, "modflow", partitions=[partition])

    assert files == [cached_path]
    assert exports == [("modflow", "coastal_seawater")]
    cache_meta = json.loads(cached_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert cache_meta["row_count"] == partition.row_count
    assert cache_meta["source_mtime"] == partition.mtime


def test_materialize_parquet_router_files_uses_safe_cache_name_for_special_scenario(
    tmp_path,
    monkeypatch,
):
    router_dir = tmp_path / "router"
    scenario = "case/a"
    partition = _router_partition(tmp_path, scenario=scenario)
    cache_root = tmp_path / "router-cache"
    monkeypatch.setenv("PierNet_ROUTER_JSONL_CACHE_DIR", str(cache_root))
    exports: list[Path] = []

    def fake_export_records_to_jsonl(kind, output_path, *, simulator=None, scenario=None, **kwargs):
        exports.append(Path(output_path))
        _write_router_jsonl(
            output_path,
            [
                {
                    "context": "fresh export",
                    "label": 1,
                    "metadata": {"simulator": simulator, "scenario": scenario},
                }
            ],
        )
        return partition.row_count

    monkeypatch.setattr("PierNet.training.router.data.portable.export_records_to_jsonl", fake_export_records_to_jsonl)

    files = _materialize_parquet_router_files(router_dir, "modflow", partitions=[partition])

    cached_path = cache_root / "modflow" / f"{portable.safe_partition_value(scenario)}.jsonl"
    assert files == [cached_path]
    assert exports == [cached_path]
    assert not (cache_root / "modflow" / "case").exists()
    cache_meta = json.loads(cached_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert cache_meta["scenario"] == scenario


def test_prepare_router_dataset_reuses_cached_parquet_summary_without_jsonl_export(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    prepared_dir = tmp_path / "prepared"
    scenarios = ["coastal_seawater"]
    partition = _router_partition(tmp_path, scenario="coastal_seawater")
    source_fingerprint = _router_source_fingerprint(jsonl_files=[], parquet_partitions=[partition])
    _write_minimal_prepared_cache(
        prepared_dir,
        router_dir=router_dir,
        scenarios=scenarios,
        source_fingerprint=source_fingerprint,
    )
    monkeypatch.setattr("PierNet.training.router.data._router_dir_uses_default_parquet", lambda router_dir: True)
    monkeypatch.setattr("PierNet.training.router.data.portable.discover_partitions", lambda kind: [partition])
    monkeypatch.setattr(
        "PierNet.training.router.data.portable.export_records_to_jsonl",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cached parquet path should not export JSONL")),
    )
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

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



def test_prepare_router_dataset_rebuilds_when_jsonl_source_changes(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    scenario_path = router_dir / "by_scenario" / "coastal_seawater.jsonl"
    base_metadata = {
        "simulator": "modflow",
        "scenario": "coastal_seawater",
        "language": "en",
        "chat_template": "qwen",
        "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
        "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
    }

    def write_records(count: int) -> None:
        _write_router_jsonl(
            scenario_path,
            [
                {
                    "context": f"source version {idx}",
                    "label": idx % 2,
                    "metadata": base_metadata,
                }
                for idx in range(count)
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

    monkeypatch.setattr("PierNet.training.router.data.PretrainedEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    prepared_dir = tmp_path / "prepared"
    write_records(1)
    first = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=prepared_dir,
        test_ratio=0.0,
        scenarios=["coastal_seawater"],
        force=True,
        input_representation="embedding",
    )

    write_records(2)
    second = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=prepared_dir,
        test_ratio=0.0,
        scenarios=["coastal_seawater"],
        force=False,
        input_representation="embedding",
    )

    assert first.train_samples == 1
    assert second.train_samples == 2
    assert second.source_fingerprint != first.source_fingerprint



def test_inspect_router_input_representation_ignores_unrelated_corrupt_jsonl(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    valid_path = router_dir / "by_scenario" / "modflow_valid.jsonl"
    corrupt_path = router_dir / "by_scenario" / "simpeg_corrupt.jsonl"
    _write_router_jsonl(
        valid_path,
        [
            {
                "context": "valid router sample",
                "label": 1,
                "metadata": {
                    "simulator": "modflow",
                    "scenario": "valid",
                    "chat_template": "qwen",
                    "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
                    "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
                },
            }
        ],
    )
    corrupt_path.write_text("{bad json\n", encoding="utf-8")
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    representation, metadata = inspect_router_input_representation(
        simulator="modflow",
        router_dir=router_dir,
        scenarios=["valid"],
        input_representation="embedding",
    )

    assert representation == PRETRAINED_EMBEDDINGS
    assert metadata.embedding_model == DEFAULT_QWEN_EMBEDDING_MODEL


def test_prepare_router_dataset_selects_jsonl_by_metadata_scenario(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    scenario_path = router_dir / "by_scenario" / "modflow_shared.jsonl"
    _write_router_jsonl(
        scenario_path,
        [
            {
                "context": "metadata scenario",
                "label": 1,
                "metadata": {
                    "simulator": "modflow",
                    "scenario": "shared",
                    "chat_template": "qwen",
                    "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
                    "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
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

    monkeypatch.setattr("PierNet.training.router.data.PretrainedEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    prepared_dir = tmp_path / "prepared"
    summary = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=prepared_dir,
        test_ratio=0.0,
        scenarios=["shared"],
        force=True,
        input_representation="embedding",
    )

    source_files = json.loads((prepared_dir / "source_files.json").read_text(encoding="utf-8"))
    assert [Path(path).name for path in source_files] == ["modflow_shared.jsonl"]
    assert summary.scenarios == ["shared"]
    assert summary.train_samples == 1


def test_prepare_router_dataset_combines_jsonl_and_parquet_selected_scenarios(tmp_path, monkeypatch):
    router_dir = tmp_path / "router"
    jsonl_path = router_dir / "by_scenario" / "jsonl_only.jsonl"
    metadata = {
        "simulator": "modflow",
        "language": "en",
        "chat_template": "qwen",
        "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
        "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
    }
    _write_router_jsonl(
        jsonl_path,
        [
            {
                "context": "jsonl source",
                "label": 1,
                "metadata": {**metadata, "scenario": "jsonl_only"},
            }
        ],
    )
    _write_router_meta(
        jsonl_path.with_suffix(".meta.json"),
        {
            "scenario": "jsonl_only",
            "chat_template": "qwen",
            "embedding_model": DEFAULT_QWEN_EMBEDDING_MODEL,
            "embedding_tokenizer": DEFAULT_QWEN_EMBEDDING_MODEL,
            "output_count": 1,
        },
    )
    partition = _router_partition(tmp_path, scenario="parquet_only")
    exported: list[str] = []
    export_paths: list[Path] = []

    def fake_export_records_to_jsonl(kind, output_path, *, simulator=None, scenario=None, **kwargs):
        exported.append(str(scenario))
        export_paths.append(Path(output_path))
        _write_router_jsonl(
            output_path,
            [
                {
                    "context": "parquet source",
                    "label": 0,
                    "metadata": {**metadata, "scenario": scenario},
                }
            ],
        )
        return 1

    class FakeEncoder:
        hidden_size = 3
        model_vocab_size = 17

        def __init__(self, spec):
            self.spec = spec

        def encode_ids(self, text: str):
            if "jsonl" in text:
                return np.arange(1, 3, dtype=np.int64)
            return np.arange(1, 4, dtype=np.int64)

        def encode_ids_batch(self, texts: list[str]):
            return [self.encode_ids(text) for text in texts]

    monkeypatch.setattr("PierNet.training.router.data._router_dir_uses_default_parquet", lambda router_dir: True)
    monkeypatch.setattr("PierNet.training.router.data.portable.discover_partitions", lambda kind: [partition])
    monkeypatch.setattr("PierNet.training.router.data.portable.export_records_to_jsonl", fake_export_records_to_jsonl)
    monkeypatch.setattr("PierNet.training.router.data.PretrainedEmbeddingEncoder", FakeEncoder)
    monkeypatch.setattr("PierNet.training.router.data.can_resolve_embedding_backbone", lambda spec: (True, ""))

    prepared_dir = tmp_path / "prepared"
    summary = prepare_router_dataset(
        simulator="modflow",
        router_dir=router_dir,
        output_dir=prepared_dir,
        test_ratio=0.0,
        scenarios=["jsonl_only", "parquet_only"],
        force=True,
        input_representation="embedding",
    )

    source_files = json.loads((prepared_dir / "source_files.json").read_text(encoding="utf-8"))
    assert exported == ["parquet_only"]
    assert all(path.is_relative_to(tmp_path) for path in export_paths)
    assert {Path(path).stem for path in source_files} == {"jsonl_only", "parquet_only"}
    assert summary.scenarios == ["jsonl_only", "parquet_only"]
    assert summary.train_samples == 2
    assert summary.train_tokens == 5
