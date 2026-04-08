"""文件管理服务：列举和删除模板/样本文件。"""

from pathlib import Path
from piern.api.deps import TEMPLATES_DIR, DATA_DIR
from piern.api.schemas.jobs import TemplateFileInfo, SampleFileInfo


def list_template_files() -> list[TemplateFileInfo]:
    """扫描 data/templates/ 目录，返回各场景的模板文件信息。"""
    if not TEMPLATES_DIR.exists():
        return []
    results = []
    for f in sorted(TEMPLATES_DIR.glob("*_templates.jsonl")):
        scenario = f.stem.replace("_templates", "")
        stat = f.stat()
        template_count = _count_lines(f)
        results.append(TemplateFileInfo(
            scenario=scenario,
            template_count=template_count,
            file_size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            path=str(f),
        ))
    return results


def list_sample_files() -> list[SampleFileInfo]:
    """扫描 data/text2comp/ 目录，返回各场景的样本文件信息（排除 all_training_data.jsonl）。"""
    if not DATA_DIR.exists():
        return []
    results = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        if f.name == "all_training_data.jsonl":
            continue
        stat = f.stat()
        sample_count = _count_lines(f)
        results.append(SampleFileInfo(
            scenario=f.stem,
            sample_count=sample_count,
            file_size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            path=str(f),
        ))
    return results


def delete_template_file(scenario: str) -> bool:
    """删除指定场景的模板文件，返回是否成功。"""
    f = TEMPLATES_DIR / f"{scenario}_templates.jsonl"
    if not f.exists():
        return False
    f.unlink()
    return True


def delete_sample_file(scenario: str) -> bool:
    """删除指定场景的样本文件，返回是否成功。"""
    f = DATA_DIR / f"{scenario}.jsonl"
    if not f.exists():
        return False
    f.unlink()
    return True


def clear_all_templates() -> int:
    """删除所有模板文件，返回删除数量。"""
    if not TEMPLATES_DIR.exists():
        return 0
    count = 0
    for f in TEMPLATES_DIR.glob("*_templates.jsonl"):
        f.unlink()
        count += 1
    return count


def clear_all_samples() -> int:
    """删除所有样本文件（排除 all_training_data.jsonl），返回删除数量。"""
    if not DATA_DIR.exists():
        return 0
    count = 0
    for f in DATA_DIR.glob("*.jsonl"):
        if f.name == "all_training_data.jsonl":
            continue
        f.unlink()
        count += 1
    return count


def _count_lines(path: Path) -> int:
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        pass
    return count
