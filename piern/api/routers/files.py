"""文件管理路由：/api/files/templates, /api/files/samples。"""

import json

from fastapi import APIRouter, HTTPException, Query

from piern.api.deps import TEMPLATES_DIR
from piern.api.services import file_manager
from piern.api.schemas.jobs import TemplateFileInfo, SampleFileInfo

router = APIRouter()


@router.get("/files/templates", response_model=list[TemplateFileInfo])
def list_template_files():
    """列出所有模板文件。"""
    return file_manager.list_template_files()


@router.get("/files/samples", response_model=list[SampleFileInfo])
def list_sample_files():
    """列出所有样本文件。"""
    return file_manager.list_sample_files()


@router.get("/files/templates/{scenario}/items")
def get_template_items(
    scenario: str,
    page: int = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
    language: str = Query(None),
    style: str = Query(None),
):
    """分页读取指定场景的模板条目（TemplateRecord）。"""
    path = TEMPLATES_DIR / f"{scenario}_templates.jsonl"
    if not path.exists():
        raise HTTPException(404, f"场景 {scenario} 的模板文件不存在")

    items = []
    try:
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
                    items.append(record)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        raise HTTPException(500, f"读取模板文件失败: {e}")

    total = len(items)
    start = page * page_size
    end = start + page_size
    return {"total": total, "page": page, "page_size": page_size, "items": items[start:end]}


@router.delete("/files/templates/{scenario}")
def delete_template_file(scenario: str):
    """删除指定场景的模板文件。"""
    if not file_manager.delete_template_file(scenario):
        raise HTTPException(404, f"场景 {scenario} 的模板文件不存在")
    return {"ok": True, "scenario": scenario}


@router.delete("/files/samples/{scenario}")
def delete_sample_file(scenario: str):
    """删除指定场景的样本文件。"""
    if not file_manager.delete_sample_file(scenario):
        raise HTTPException(404, f"场景 {scenario} 的样本文件不存在")
    return {"ok": True, "scenario": scenario}


@router.delete("/files/templates")
def clear_all_templates():
    """清空所有模板文件。"""
    count = file_manager.clear_all_templates()
    return {"ok": True, "deleted": count}


@router.delete("/files/samples")
def clear_all_samples():
    """清空所有样本文件（保留 all_training_data.jsonl）。"""
    count = file_manager.clear_all_samples()
    return {"ok": True, "deleted": count}
