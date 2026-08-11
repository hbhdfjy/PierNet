"""System-level uploaded expert model registry API."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from PierNet.synth.services import expert_models

router = APIRouter(prefix="/expert-models", tags=["expert-models"])


class ExpertModelUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=160)
    status: Literal["active", "disabled", "invalid"] | None = None
    domain: str | None = Field(None, max_length=160)
    simulator: str | None = Field(None, max_length=160)
    demo_prompt: str | None = Field(None, max_length=4000)
    demo_prompt_label: str | None = Field(None, max_length=80)
    assembly_enabled: bool | None = None
    data_generation_enabled: bool | None = None


async def _read_upload_body(request: Request) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total > expert_models.MAX_MODEL_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"专家模型上传体不能超过 {expert_models.MAX_MODEL_BYTES_LABEL}",
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"上传写入失败: {exc}") from exc
    return b"".join(chunks)


def _constraints_response(models: list[dict[str, Any]]) -> dict[str, Any]:
    return {**expert_models.describe_constraints(), "models": models}


@router.get("")
def list_expert_models() -> dict[str, Any]:
    return _constraints_response(expert_models.list_models())


@router.get("/{model_id}")
def get_expert_model(model_id: str) -> dict[str, Any]:
    try:
        return expert_models.get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"专家模型不存在: {model_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/upload")
async def upload_expert_model(
    request: Request,
    name: str = Query(..., min_length=1, max_length=180),
) -> dict[str, Any]:
    content = await _read_upload_body(request)
    try:
        model = expert_models.upload_model(name, content)
    except expert_models.ExpertModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"专家模型加载失败: {exc}") from exc
    return {"ok": True, **expert_models.describe_constraints(), "model": model}


@router.post("/{model_id}/validate")
def validate_expert_model(model_id: str) -> dict[str, Any]:
    try:
        model = expert_models.revalidate_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"专家模型不存在: {model_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": model.get("status") != "invalid", "model": model, "error": model.get("last_error")}


@router.patch("/{model_id}")
def update_expert_model(model_id: str, req: ExpertModelUpdateRequest) -> dict[str, Any]:
    payload = req.model_dump(exclude_unset=True)
    try:
        model = expert_models.update_model(model_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"专家模型不存在: {model_id}") from exc
    except (ValueError, expert_models.ExpertModelError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "model": model}


@router.delete("/{model_id}", status_code=204)
def delete_expert_model(model_id: str) -> Response:
    try:
        expert_models.delete_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"专家模型不存在: {model_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(status_code=204)
