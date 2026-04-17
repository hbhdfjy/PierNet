"""文件管理服务：列举和删除模板/样本文件。"""

from __future__ import annotations

from pathlib import Path

from piern.api.deps import DATA_DIR, TEMPLATES_DIR
from piern.api.schemas.jobs import SampleFileInfo, TemplateFileInfo
from piern.api.services import manifest_store


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
                template_count=item.get("template_count", 0),
                file_size_bytes=item.get("file_size_bytes", 0),
                mtime=item.get("mtime", 0),
                path=item.get("path", ""),
            )
            for item in manifest.get("items", [])
        ]
    except Exception:
        return _legacy_list_template_files()


def list_sample_files() -> list[SampleFileInfo]:
    """返回样本文件信息，优先走 manifest。"""
    try:
        manifest = manifest_store.ensure_sample_manifest()
        return [
            SampleFileInfo(
                scenario=item["scenario"],
                sample_count=item.get("sample_count", 0),
                file_size_bytes=item.get("file_size_bytes", 0),
                mtime=item.get("mtime", 0),
                path=item.get("path", ""),
            )
            for item in manifest.get("items", [])
        ]
    except Exception:
        return _legacy_list_sample_files()


def delete_template_file(scenario: str) -> bool:
    """删除指定场景模板文件。"""
    path = TEMPLATES_DIR / f"{scenario}_templates.jsonl"
    if not path.exists():
        return False
    path.unlink()
    _safe_refresh_templates()
    return True


def delete_sample_file(scenario: str) -> bool:
    """删除指定场景样本文件并重建聚合文件。"""
    path = DATA_DIR / f"{scenario}.jsonl"
    if not path.exists():
        return False
    path.unlink()
    _rewrite_merged_samples()
    _safe_refresh_samples()
    return True


def clear_all_templates() -> int:
    """删除所有模板文件，返回删除数量。"""
    if not TEMPLATES_DIR.exists():
        return 0

    count = 0
    for path in TEMPLATES_DIR.glob("*_templates.jsonl"):
        path.unlink()
        count += 1

    _safe_refresh_templates()
    return count


def clear_all_samples() -> int:
    """删除所有样本文件并重建聚合文件，返回删除数量。"""
    if not DATA_DIR.exists():
        return 0

    count = 0
    for path in DATA_DIR.glob("*.jsonl"):
        if path.name == "all_training_data.jsonl":
            continue
        path.unlink()
        count += 1

    _rewrite_merged_samples()
    _safe_refresh_samples()
    return count


def _safe_refresh_templates() -> None:
    try:
        manifest_store.rebuild_template_manifest()
    except Exception:
        return


def _safe_refresh_samples() -> None:
    try:
        manifest_store.rebuild_sample_manifest()
    except Exception:
        return


def _legacy_list_template_files() -> list[TemplateFileInfo]:
    if not TEMPLATES_DIR.exists():
        return []

    results = []
    for path in sorted(TEMPLATES_DIR.glob("*_templates.jsonl")):
        scenario = path.stem.replace("_templates", "")
        stat = path.stat()
        results.append(
            TemplateFileInfo(
                scenario=scenario,
                template_count=_count_lines(path),
                file_size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                path=str(path),
            )
        )
    return results


def _legacy_list_sample_files() -> list[SampleFileInfo]:
    if not DATA_DIR.exists():
        return []

    results = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        if path.name == "all_training_data.jsonl":
            continue
        stat = path.stat()
        results.append(
            SampleFileInfo(
                scenario=path.stem,
                sample_count=_count_lines(path),
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
