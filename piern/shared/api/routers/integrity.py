"""Data integrity API."""

from __future__ import annotations

from fastapi import APIRouter

from piern.shared.storage import integrity

router = APIRouter(prefix="/storage/integrity", tags=["storage"])


@router.get("")
def get_integrity_status():
    return integrity.status()


@router.post("/manifest")
def rebuild_integrity_manifest():
    manifest = integrity.write_manifest()
    return {
        "ok": True,
        "manifest_path": str(integrity.DEFAULT_MANIFEST),
        "entries": len(manifest.get("entries", [])),
        "generated_at": manifest.get("generated_at"),
    }
