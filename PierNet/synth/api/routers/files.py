"""模板浏览辅助路由：/api/files/templates。"""

import json

from fastapi import APIRouter, HTTPException, Query

from PierNet.shared.runtime.paths import TEMPLATES_DIR
from PierNet.synth.services import file_manager, jsonl_filter_index, jsonl_index, manifest_store
from PierNet.synth.api.schemas.jobs import TemplateFileInfo

router = APIRouter()


def _template_path_for_scenario(scenario: str):
    if not file_manager._is_safe_name_component(scenario):
        raise HTTPException(status_code=400, detail="scenario must be a file name component")
    return TEMPLATES_DIR / f"{scenario}_templates.jsonl"


@router.get("/files/templates", response_model=list[TemplateFileInfo])
def list_template_files():
    """列出所有模板文件。"""
    return file_manager.list_template_files()


@router.get("/files/templates/{scenario}/items")
def get_template_items(
    scenario: str,
    page: int = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
    language: str = Query(None),
    style: str = Query(None),
):
    """分页读取指定场景的模板条目（TemplateRecord）。"""
    path = _template_path_for_scenario(scenario)
    if not path.exists():
        raise HTTPException(404, f"场景 {scenario} 的模板文件不存在")

    has_filter = bool(language or style)
    start = page * page_size
    end = start + page_size
    try:
        if not has_filter:
            total_hint = _template_total_from_manifest(scenario)
            try:
                total, items = jsonl_index.read_page(
                    path,
                    page=page,
                    page_size=page_size,
                    total_rows=total_hint,
                )
                return {"total": total, "page": page, "page_size": page_size, "items": items}
            except Exception:
                with open(path, "rb") as fb:
                    content = fb.read()
                    total = content.count(b"\n")
                    if content and not content.endswith(b"\n"):
                        total += 1
                items = []
                with open(path, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if idx >= end:
                            break
                        if idx < start:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return {"total": total, "page": page, "page_size": page_size, "items": items}

        filter_key = _template_filter_key(language=language, style=style)
        if filter_key:
            try:
                total, items = jsonl_filter_index.read_filtered_page(
                    path,
                    profile="template_language_style",
                    key=filter_key,
                    page=page,
                    page_size=page_size,
                )
                return {"total": total, "page": page, "page_size": page_size, "items": items}
            except Exception:
                pass

        total = 0
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if language and record.get("language") != language:
                        continue
                    if style and record.get("style") != style:
                        continue
                    if start <= total < end:
                        items.append(record)
                    total += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        raise HTTPException(500, f"读取模板文件失败: {e}")

    return {"total": total, "page": page, "page_size": page_size, "items": items}


def _template_total_from_manifest(scenario: str) -> int | None:
    try:
        manifest = manifest_store.ensure_template_manifest()
    except Exception:
        return None

    for item in manifest.get("items", []):
        if item.get("scenario") == scenario:
            return int(item.get("template_count", 0))
    return None


def _template_filter_key(language: str | None, style: str | None) -> str | None:
    if language and style:
        return f"language={language}|style={style}"
    if language:
        return f"language={language}"
    if style:
        return f"style={style}"
    return None
