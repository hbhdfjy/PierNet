"""文件管理服务：列举模板并管理样本文件。"""

from __future__ import annotations

import json
from pathlib import Path

from piern.shared.runtime.paths import DATA_DIR, TEMPLATES_DIR
from piern.synth.api.schemas.jobs import TemplateFileInfo
from . import jsonl_filter_index, jsonl_index, manifest_store


def _rewrite_merged_samples() -> None:
    if not DATA_DIR.exists():
        return

    merged_path = DATA_DIR / "all_training_data.jsonl"
    scenario_files = [
        path
        for path in sorted(DATA_DIR.glob("*.jsonl"))
        if path.name != merged_path.name
    ]

    with open(merged_path, "w", encoding="utf-8") as fout:
        for path in scenario_files:
            try:
                with open(path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            fout.write(line + "\n")
            except Exception:
                continue


def list_template_files() -> list[TemplateFileInfo]:
    """返回模板文件信息，优先走 manifest。"""
    try:
        manifest = manifest_store.ensure_template_manifest()
        return [
            TemplateFileInfo(
                scenario=item["scenario"],
                simulator=item.get("simulator") or None,
                template_count=item.get("template_count", 0),
                file_size_bytes=item.get("file_size_bytes", 0),
                mtime=item.get("mtime", 0),
                path=item.get("path", ""),
            )
            for item in manifest.get("items", [])
        ]
    except Exception:
        return _legacy_list_template_files()

def _jsonl_identity(path: Path) -> tuple[str | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                metadata = json.loads(line).get("metadata", {})
                if not isinstance(metadata, dict):
                    return None, None
                scenario = str(metadata.get("scenario") or "").strip() or None
                simulator = str(metadata.get("simulator") or "").strip() or None
                return scenario, simulator
    except Exception:
        return None, None
    return None, None


def _is_safe_name_component(value: str | None) -> bool:
    component = str(value or "")
    return (
        bool(component)
        and component not in {".", ".."}
        and "\x00" not in component
        and Path(component).name == component
        and "\\" not in component
    )


def _record_matches_identity(record: dict, fallback_scenario: str, scenario: str, simulator: str | None) -> bool:
    metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    record_scenario = str(metadata.get("scenario") or fallback_scenario).strip()
    record_simulator = str(metadata.get("simulator") or "").strip() or None
    return record_scenario == scenario and (simulator is None or record_simulator in {None, simulator})


def _jsonl_contains_identity(path: Path, scenario: str, simulator: str | None = None) -> bool:
    saw_record = False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                saw_record = True
                if _record_matches_identity(record, path.stem, scenario, simulator):
                    return True
    except OSError:
        return False
    return not saw_record and path.stem == scenario


def _rewrite_jsonl_without_identity(path: Path, scenario: str, simulator: str | None = None) -> tuple[bool, int]:
    kept: list[str] = []
    removed = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    kept.append(line if line.endswith("\n") else line + "\n")
                    continue
                if isinstance(record, dict) and _record_matches_identity(record, path.stem, scenario, simulator):
                    removed += 1
                    continue
                kept.append(line if line.endswith("\n") else line + "\n")
    except OSError:
        return False, 0

    if removed == 0:
        return False, len(kept)
    _delete_sample_indexes(path)
    if kept:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("".join(kept), encoding="utf-8")
        tmp_path.replace(path)
    else:
        path.unlink(missing_ok=True)
    return True, len(kept)


def resolve_sample_file(scenario: str, simulator: str | None = None) -> Path | None:
    if not _is_safe_name_component(scenario) or (simulator is not None and not _is_safe_name_component(simulator)):
        return None
    direct = DATA_DIR / f"{scenario}.jsonl"
    matches: list[Path] = []
    if direct.exists() and _jsonl_contains_identity(direct, scenario, simulator=simulator):
        matches.append(direct)

    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.glob("*.jsonl")):
            if path.name == "all_training_data.jsonl" or path == direct:
                continue
            if _jsonl_contains_identity(path, scenario, simulator=simulator):
                matches.append(path)

    if len(matches) > 1:
        target = f"{simulator}/{scenario}" if simulator else scenario
        raise ValueError(f"ambiguous sample files for {target}")
    return matches[0] if matches else None


def _delete_sample_indexes(path: Path) -> None:
    index_paths = [
        jsonl_index.get_index_path(path),
        jsonl_filter_index.get_filter_index_path(path, "sample_language_style"),
    ]
    for index_path in index_paths:
        try:
            index_path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            continue


def delete_sample_file(scenario: str, simulator: str | None = None) -> bool:
    """删除指定场景样本文件并重建聚合文件。"""
    path = resolve_sample_file(scenario, simulator=simulator)
    if path is None or not path.exists():
        return False
    deleted, _remaining = _rewrite_jsonl_without_identity(path, scenario, simulator=simulator)
    if not deleted:
        return False
    _rewrite_merged_samples()
    _delete_sample_indexes(DATA_DIR / "all_training_data.jsonl")
    _safe_refresh_samples()
    _safe_invalidate_text2comp_scenarios_cache()
    return True


def clear_all_samples() -> int:
    """删除所有样本文件并重建聚合文件，返回删除数量。"""
    if not DATA_DIR.exists():
        return 0

    count = 0
    for path in DATA_DIR.glob("*.jsonl"):
        if path.name == "all_training_data.jsonl":
            continue
        _delete_sample_indexes(path)
        path.unlink()
        count += 1

    _rewrite_merged_samples()
    _delete_sample_indexes(DATA_DIR / "all_training_data.jsonl")
    _safe_refresh_samples()
    _safe_invalidate_text2comp_scenarios_cache()
    return count



def _safe_refresh_samples() -> None:
    try:
        manifest_store.rebuild_sample_manifest()
    except Exception:
        return


def _safe_invalidate_text2comp_scenarios_cache() -> None:
    try:
        from piern.synth.api.routers.config import invalidate_text2comp_scenarios_cache

        invalidate_text2comp_scenarios_cache()
    except Exception:
        return


def _legacy_list_template_files() -> list[TemplateFileInfo]:
    if not TEMPLATES_DIR.exists():
        return []

    results = []
    for path in sorted(TEMPLATES_DIR.glob("*_templates.jsonl")):
        scenario = path.stem.removesuffix("_templates")
        stat = path.stat()
        results.append(
            TemplateFileInfo(
                scenario=scenario,
                simulator=None,
                template_count=_count_lines(path),
                file_size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                path=str(path),
            )
        )
    return results


def _count_lines(path: Path) -> int:
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    return count
