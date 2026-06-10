from __future__ import annotations

import json

from PierNet.training.services import cache_cleanup


DAY = cache_cleanup.SECONDS_PER_DAY


def _write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_router_jsonl_cache_dry_run_reports_reclaimable_space(tmp_path):
    now = 10_000_000.0
    cache_root = tmp_path / "data" / "router" / ".parquet_jsonl_cache"
    cache_path = cache_root / "modflow" / "case_a.jsonl"
    _write(cache_path, b"abc")
    _write_json(
        cache_path.with_suffix(".meta.json"),
        {
            "simulator": "modflow",
            "scenario": "case_a",
            "last_used_at": now - 8 * DAY,
        },
    )

    result = cache_cleanup.cleanup_training_cache(
        router_jsonl_cache_dir=cache_root,
        training_artifact_root=tmp_path / "artifacts" / "token_router",
        router_jsonl_ttl_days=7,
        dry_run=True,
        jobs=[],
        now=now,
    )

    assert result.dry_run is True
    assert result.reclaimable_bytes > 0
    assert len(result.candidates) == 1
    assert result.candidates[0].kind == cache_cleanup.ROUTER_JSONL_CACHE_KIND
    assert cache_path.exists()


def test_execute_deletes_expired_router_jsonl_cache_and_meta(tmp_path):
    now = 10_000_000.0
    cache_root = tmp_path / "data" / "router" / ".parquet_jsonl_cache"
    cache_path = cache_root / "modflow" / "case_a.jsonl"
    meta_path = cache_path.with_suffix(".meta.json")
    _write(cache_path, b"abc")
    _write_json(meta_path, {"last_used_at": now - 8 * DAY})

    result = cache_cleanup.cleanup_training_cache(
        router_jsonl_cache_dir=cache_root,
        training_artifact_root=tmp_path / "artifacts" / "token_router",
        router_jsonl_ttl_days=7,
        dry_run=False,
        jobs=[],
        now=now,
    )

    assert len(result.deleted) == 1
    assert result.deleted_bytes > 0
    assert not cache_path.exists()
    assert not meta_path.exists()


def test_active_training_job_protects_prepared_cache(tmp_path):
    now = 10_000_000.0
    artifact_root = tmp_path / "artifacts" / "token_router"
    prepared_dir = artifact_root / "modflow" / "prepared" / "modflow-abcd12"
    _write(prepared_dir / "train_token_ids.bin", b"abc")
    _write_json(prepared_dir / "meta.json", {"last_used_at": now - 30 * DAY})

    result = cache_cleanup.cleanup_training_cache(
        router_jsonl_cache_dir=tmp_path / "data" / "router" / ".parquet_jsonl_cache",
        training_artifact_root=artifact_root,
        training_prepared_ttl_days=7,
        dry_run=False,
        jobs=[
            {
                "job_id": "train-active",
                "status": "running",
                "simulator": "modflow",
                "prepared_name": "modflow-abcd12",
            }
        ],
        now=now,
    )

    assert prepared_dir.exists()
    assert not result.deleted
    protected = next(item for item in result.skipped if item.path == str(prepared_dir))
    assert protected.reason == "active_training_job"
    assert protected.active_job_ids == ["train-active"]


def test_execute_deletes_expired_inactive_prepared_cache(tmp_path):
    now = 10_000_000.0
    artifact_root = tmp_path / "artifacts" / "token_router"
    prepared_dir = artifact_root / "modflow" / "prepared" / "modflow-abcd12"
    _write(prepared_dir / "train_token_ids.bin", b"abc")
    _write_json(prepared_dir / "meta.json", {"last_used_at": now - 30 * DAY})

    result = cache_cleanup.cleanup_training_cache(
        router_jsonl_cache_dir=tmp_path / "data" / "router" / ".parquet_jsonl_cache",
        training_artifact_root=artifact_root,
        training_prepared_ttl_days=7,
        dry_run=False,
        jobs=[],
        now=now,
    )

    assert len(result.deleted) == 1
    assert result.deleted[0].kind == cache_cleanup.TRAINING_PREPARED_CACHE_KIND
    assert result.deleted_bytes > 0
    assert not prepared_dir.exists()


def test_cleanup_does_not_touch_core_data_or_training_runs(tmp_path):
    now = 10_000_000.0
    router_parquet = tmp_path / "data" / "router_parquet" / "simulator=modflow" / "scenario=case_a" / "part.parquet"
    text2comp_parquet = tmp_path / "data" / "text2comp_parquet" / "simulator=modflow" / "scenario=case_a" / "part.parquet"
    run_file = tmp_path / "artifacts" / "token_router" / "modflow" / "runs" / "run1" / "router_final.pt"
    hdf5_file = tmp_path / "data" / "modflow" / "modflow_case_a.h5"
    for path in (router_parquet, text2comp_parquet, run_file, hdf5_file):
        _write(path, b"keep")

    result = cache_cleanup.cleanup_training_cache(
        router_jsonl_cache_dir=tmp_path / "data" / "router" / ".parquet_jsonl_cache",
        training_artifact_root=tmp_path / "artifacts" / "token_router",
        dry_run=False,
        jobs=[],
        now=now,
    )

    assert not result.candidates
    assert router_parquet.exists()
    assert text2comp_parquet.exists()
    assert run_file.exists()
    assert hdf5_file.exists()


def test_max_delete_budget_limits_real_deletion(tmp_path):
    now = 10_000_000.0
    artifact_root = tmp_path / "artifacts" / "token_router"
    first = artifact_root / "modflow" / "prepared" / "old_a"
    second = artifact_root / "modflow" / "prepared" / "old_b"
    for index, prepared_dir in enumerate((first, second)):
        _write(prepared_dir / "train_token_ids.bin", b"12345")
        _write_json(prepared_dir / "meta.json", {"last_used_at": now - (30 - index) * DAY})

    result = cache_cleanup.cleanup_training_cache(
        router_jsonl_cache_dir=tmp_path / "data" / "router" / ".parquet_jsonl_cache",
        training_artifact_root=artifact_root,
        training_prepared_ttl_days=7,
        max_delete_bytes=cache_cleanup._dir_size(first),
        dry_run=False,
        jobs=[],
        now=now,
    )

    assert len(result.candidates) == 2
    assert len(result.deleted) == 1
    assert not first.exists()
    assert second.exists()
    assert any(item.reason == "max_delete_bytes_exceeded" for item in result.skipped)


def test_touch_prepared_cache_meta_refreshes_last_used_without_losing_summary(tmp_path):
    prepared_dir = tmp_path / "artifacts" / "token_router" / "modflow" / "prepared" / "modflow-abcd12"
    _write(prepared_dir / "train_token_ids.bin", b"abc")
    _write_json(prepared_dir / "meta.json", {"train_samples": 3, "test_samples": 1})

    meta = cache_cleanup.touch_prepared_cache_meta(
        prepared_dir,
        simulator="modflow",
        prepared_name="modflow-abcd12",
        now=123.0,
    )

    assert meta["train_samples"] == 3
    assert meta["cache_kind"] == cache_cleanup.TRAINING_PREPARED_CACHE_KIND
    assert meta["last_used_at"] == 123.0
    assert meta["last_built_at"] == 123.0
    assert meta["size_bytes"] > 0
