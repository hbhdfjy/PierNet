"""Training cache metadata and cleanup utilities."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PierNet.training.services import job_store as training_job_store
from PierNet.shared.tasks.state import ACTIVE_STATUSES

ROUTER_JSONL_CACHE_KIND = "router_jsonl_cache"
TRAINING_PREPARED_CACHE_KIND = "training_prepared"
CLEANUP_KINDS = frozenset({ROUTER_JSONL_CACHE_KIND, TRAINING_PREPARED_CACHE_KIND})
DEFAULT_ROUTER_JSONL_TTL_DAYS = 7.0
DEFAULT_TRAINING_PREPARED_TTL_DAYS = 7.0
DEFAULT_MAX_DELETE_GB = 1024.0
SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(slots=True)
class CacheCleanupItem:
    kind: str
    path: str
    size_bytes: int
    last_used_at: float | None
    age_days: float | None
    simulator: str = ""
    name: str = ""
    meta_path: str | None = None
    last_used_source: str = ""
    reason: str = ""
    active_job_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CacheCleanupResult:
    dry_run: bool
    max_delete_bytes: int
    reclaimable_bytes: int = 0
    deleted_bytes: int = 0
    candidates: list[CacheCleanupItem] = field(default_factory=list)
    deleted: list[CacheCleanupItem] = field(default_factory=list)
    skipped: list[CacheCleanupItem] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "max_delete_bytes": self.max_delete_bytes,
            "reclaimable_bytes": self.reclaimable_bytes,
            "deleted_bytes": self.deleted_bytes,
            "candidate_count": len(self.candidates),
            "deleted_count": len(self.deleted),
            "skipped_count": len(self.skipped),
            "candidates": [item.to_dict() for item in self.candidates],
            "deleted": [item.to_dict() for item in self.deleted],
            "skipped": [item.to_dict() for item in self.skipped],
            "errors": self.errors,
        }


def _now() -> float:
    return time.time()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_symlink():
            continue
        if item.is_file():
            total += _safe_file_size(item)
    return total


def _path_size(path: Path, meta_path: Path | None = None) -> int:
    total = _dir_size(path) if path.is_dir() else _safe_file_size(path)
    if meta_path and meta_path.exists() and meta_path != path:
        total += _safe_file_size(meta_path)
    return total


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _last_used(meta: dict[str, Any], fallback_path: Path) -> tuple[float | None, str]:
    for key in ("last_used_at", "last_built_at"):
        value = _coerce_float(meta.get(key))
        if value is not None:
            return value, key
    try:
        return float(fallback_path.stat().st_mtime), "legacy_mtime"
    except OSError:
        return None, "missing"


def _age_days(last_used_at: float | None, now: float) -> float | None:
    if last_used_at is None:
        return None
    return max(0.0, (now - last_used_at) / SECONDS_PER_DAY)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _is_expired(last_used_at: float | None, ttl_days: float, now: float) -> bool:
    if last_used_at is None:
        return False
    return now - last_used_at >= max(0.0, ttl_days) * SECONDS_PER_DAY


def _active_prepared_refs(jobs: Iterable[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[str]]]:
    refs: dict[tuple[str, str], list[str]] = {}
    unknown_by_simulator: dict[str, list[str]] = {}
    for job in jobs:
        status = str(job.get("status") or "").strip().lower()
        if status not in ACTIVE_STATUSES:
            continue
        simulator = str(job.get("simulator") or "").strip()
        if not simulator:
            continue
        job_id = str(job.get("job_id") or "").strip() or "unknown"
        prepared_name = str(job.get("prepared_name") or "").strip()
        if prepared_name:
            refs.setdefault((simulator, prepared_name), []).append(job_id)
        else:
            unknown_by_simulator.setdefault(simulator, []).append(job_id)
    return refs, unknown_by_simulator


def _load_training_jobs(jobs: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if jobs is not None:
        return [dict(job) for job in jobs]
    try:
        return training_job_store.list_job_snapshots()
    except Exception:
        return []


def touch_router_jsonl_cache_meta(
    cache_path: Path,
    *,
    meta_path: Path | None = None,
    source_path: str | None = None,
    source_mtime: float | None = None,
    row_count: int | None = None,
    simulator: str | None = None,
    scenario: str | None = None,
    built: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Refresh explicit usage metadata for a materialized Router JSONL cache."""

    cache_path = Path(cache_path)
    meta_path = Path(meta_path) if meta_path else cache_path.with_suffix(".meta.json")
    payload = _read_json_object(meta_path)
    ts = _now() if now is None else float(now)
    payload["cache_kind"] = ROUTER_JSONL_CACHE_KIND
    payload["cache_path"] = str(cache_path)
    payload["size_bytes"] = _safe_file_size(cache_path)
    payload["last_used_at"] = ts
    if built or _coerce_float(payload.get("last_built_at")) is None:
        payload["last_built_at"] = ts
    if source_path is not None:
        payload["source_path"] = source_path
    if source_mtime is not None:
        payload["source_mtime"] = source_mtime
    if row_count is not None:
        payload["row_count"] = int(row_count)
    if simulator:
        payload["simulator"] = simulator
    if scenario:
        payload["scenario"] = scenario
    _write_json_object(meta_path, payload)
    return payload


def touch_prepared_cache_meta(
    prepared_dir: Path,
    *,
    simulator: str | None = None,
    prepared_name: str | None = None,
    built: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Refresh explicit usage metadata for a Token Router prepared cache."""

    prepared_dir = Path(prepared_dir)
    meta_path = prepared_dir / "meta.json"
    payload = _read_json_object(meta_path)
    ts = _now() if now is None else float(now)
    payload["cache_kind"] = TRAINING_PREPARED_CACHE_KIND
    payload["cache_path"] = str(prepared_dir)
    payload["size_bytes"] = _dir_size(prepared_dir)
    payload["last_used_at"] = ts
    if built or _coerce_float(payload.get("last_built_at")) is None:
        payload["last_built_at"] = ts
    if simulator:
        payload.setdefault("simulator", simulator)
    payload["prepared_name"] = prepared_name or payload.get("prepared_name") or prepared_dir.name
    _write_json_object(meta_path, payload)
    return payload


def _router_jsonl_items(root: Path, ttl_days: float, now: float) -> tuple[list[CacheCleanupItem], list[CacheCleanupItem]]:
    candidates: list[CacheCleanupItem] = []
    skipped: list[CacheCleanupItem] = []
    root = Path(root)
    if not root.exists():
        return candidates, skipped
    if root.name != ".parquet_jsonl_cache":
        skipped.append(
            CacheCleanupItem(
                kind=ROUTER_JSONL_CACHE_KIND,
                path=str(root),
                size_bytes=0,
                last_used_at=None,
                age_days=None,
                reason="router_cache_root_must_be_dot_parquet_jsonl_cache",
            )
        )
        return candidates, skipped
    for path in sorted(root.glob("*/*.jsonl")):
        if path.is_symlink() or not path.is_file() or not _is_under(path, root):
            skipped.append(
                CacheCleanupItem(
                    kind=ROUTER_JSONL_CACHE_KIND,
                    path=str(path),
                    size_bytes=0,
                    last_used_at=None,
                    age_days=None,
                    reason="unsafe_router_cache_path",
                )
            )
            continue
        meta_path = path.with_suffix(".meta.json")
        meta = _read_json_object(meta_path)
        last_used_at, source = _last_used(meta, meta_path if meta_path.exists() else path)
        age = _age_days(last_used_at, now)
        item = CacheCleanupItem(
            kind=ROUTER_JSONL_CACHE_KIND,
            path=str(path),
            meta_path=str(meta_path),
            size_bytes=_path_size(path, meta_path),
            last_used_at=last_used_at,
            age_days=age,
            simulator=str(meta.get("simulator") or path.parent.name),
            name=str(meta.get("scenario") or path.stem),
            last_used_source=source,
        )
        if _is_expired(last_used_at, ttl_days, now):
            candidates.append(item)
        else:
            item.reason = "ttl_not_expired"
            skipped.append(item)
    return candidates, skipped


def _prepared_items(
    artifact_root: Path,
    ttl_days: float,
    now: float,
    *,
    active_refs: dict[tuple[str, str], list[str]],
    active_unknown_simulators: dict[str, list[str]],
) -> tuple[list[CacheCleanupItem], list[CacheCleanupItem]]:
    candidates: list[CacheCleanupItem] = []
    skipped: list[CacheCleanupItem] = []
    artifact_root = Path(artifact_root)
    if not artifact_root.exists():
        return candidates, skipped
    for prepared_parent in sorted(artifact_root.glob("*/prepared")):
        if prepared_parent.is_symlink() or not prepared_parent.is_dir():
            continue
        simulator = prepared_parent.parent.name
        for path in sorted(prepared_parent.iterdir()):
            if path.is_symlink() or not path.is_dir() or not _is_under(path, artifact_root):
                skipped.append(
                    CacheCleanupItem(
                        kind=TRAINING_PREPARED_CACHE_KIND,
                        path=str(path),
                        size_bytes=0,
                        last_used_at=None,
                        age_days=None,
                        simulator=simulator,
                        name=path.name,
                        reason="unsafe_prepared_cache_path",
                    )
                )
                continue
            prepared_name = path.name
            active_job_ids = active_refs.get((simulator, prepared_name), []) + active_unknown_simulators.get(simulator, [])
            meta_path = path / "meta.json"
            meta = _read_json_object(meta_path)
            last_used_at, source = _last_used(meta, meta_path if meta_path.exists() else path)
            item = CacheCleanupItem(
                kind=TRAINING_PREPARED_CACHE_KIND,
                path=str(path),
                meta_path=str(meta_path),
                size_bytes=_path_size(path),
                last_used_at=last_used_at,
                age_days=_age_days(last_used_at, now),
                simulator=simulator,
                name=prepared_name,
                last_used_source=source,
                active_job_ids=active_job_ids,
            )
            if active_job_ids:
                item.reason = "active_training_job"
                skipped.append(item)
                continue
            if _is_expired(last_used_at, ttl_days, now):
                candidates.append(item)
            else:
                item.reason = "ttl_not_expired"
                skipped.append(item)
    return candidates, skipped


def _delete_item(item: CacheCleanupItem) -> None:
    path = Path(item.path)
    if item.kind == ROUTER_JSONL_CACHE_KIND:
        meta_path = Path(item.meta_path) if item.meta_path else path.with_suffix(".meta.json")
        if path.exists():
            path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        return
    if item.kind == TRAINING_PREPARED_CACHE_KIND:
        if path.exists():
            shutil.rmtree(path)
        return
    raise ValueError(f"unsupported cache cleanup kind: {item.kind}")


def cleanup_training_cache(
    *,
    router_jsonl_cache_dir: Path,
    training_artifact_root: Path,
    router_jsonl_ttl_days: float = DEFAULT_ROUTER_JSONL_TTL_DAYS,
    training_prepared_ttl_days: float = DEFAULT_TRAINING_PREPARED_TTL_DAYS,
    max_delete_bytes: int | None = None,
    dry_run: bool = True,
    kinds: Iterable[str] | None = None,
    jobs: Iterable[dict[str, Any]] | None = None,
    now: float | None = None,
) -> CacheCleanupResult:
    """Collect and optionally delete expired training caches."""

    selected = set(kinds or CLEANUP_KINDS)
    invalid = selected - CLEANUP_KINDS
    if invalid:
        raise ValueError(f"Unsupported cache cleanup kind(s): {sorted(invalid)!r}")
    now_value = _now() if now is None else float(now)
    max_bytes = int(DEFAULT_MAX_DELETE_GB * 1024**3) if max_delete_bytes is None else max(0, int(max_delete_bytes))
    result = CacheCleanupResult(dry_run=dry_run, max_delete_bytes=max_bytes)

    training_jobs = _load_training_jobs(jobs)
    active_refs, active_unknown_simulators = _active_prepared_refs(training_jobs)

    candidates: list[CacheCleanupItem] = []
    skipped: list[CacheCleanupItem] = []
    if ROUTER_JSONL_CACHE_KIND in selected:
        found, ignored = _router_jsonl_items(Path(router_jsonl_cache_dir), router_jsonl_ttl_days, now_value)
        candidates.extend(found)
        skipped.extend(ignored)
    if TRAINING_PREPARED_CACHE_KIND in selected:
        found, ignored = _prepared_items(
            Path(training_artifact_root),
            training_prepared_ttl_days,
            now_value,
            active_refs=active_refs,
            active_unknown_simulators=active_unknown_simulators,
        )
        candidates.extend(found)
        skipped.extend(ignored)

    candidates.sort(key=lambda item: (item.last_used_at if item.last_used_at is not None else 0.0, item.kind, item.path))
    result.candidates = candidates
    result.skipped = skipped
    result.reclaimable_bytes = sum(item.size_bytes for item in candidates)

    budget_used = 0
    for item in candidates:
        if budget_used + item.size_bytes > max_bytes:
            limited = CacheCleanupItem(**item.to_dict())
            limited.reason = "max_delete_bytes_exceeded"
            result.skipped.append(limited)
            continue
        if dry_run:
            budget_used += item.size_bytes
            continue
        try:
            _delete_item(item)
        except Exception as exc:  # pragma: no cover - defensive reporting
            result.errors.append({"path": item.path, "kind": item.kind, "error": str(exc)})
            continue
        budget_used += item.size_bytes
        result.deleted_bytes += item.size_bytes
        result.deleted.append(item)
    return result


def result_as_json(result: CacheCleanupResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
