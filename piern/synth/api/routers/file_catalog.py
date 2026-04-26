
"""Unified file-management API under /api/files/catalog."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from piern.synth.services import file_catalog

router = APIRouter()


@router.get("/files/catalog")
def get_file_catalog():
    return file_catalog.get_file_catalog()


@router.delete("/files/catalog/assets/{asset_id}")
def delete_file_asset(asset_id: str):
    try:
        return file_catalog.delete_asset(asset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/files/catalog/groups/{kind}")
def clear_file_group(kind: str):
    try:
        return file_catalog.clear_group(kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/files/catalog/rebuild")
def rebuild_file_catalog_indexes(scope: str = Query("all")):
    if scope not in {"all", "templates", "samples", "router"}:
        raise HTTPException(status_code=400, detail="scope must be one of: all, templates, samples, router")
    return file_catalog.rebuild_indexes(scope=scope)
