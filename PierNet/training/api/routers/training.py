from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from PierNet.training.api.schemas.training import (
    GPUInfo,
    TrainingCurvesResponse,
    TrainingDatasetInfo,
    TrainingJobCreateRequest,
    TrainingJobDetail,
    TrainingJobSummary,
    TrainingLogResponse,
    TrainingOverviewResponse,
)
from PierNet.training.services import training_manager

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/overview", response_model=TrainingOverviewResponse)
def get_training_overview():
    return TrainingOverviewResponse(**training_manager.get_overview())


@router.get("/datasets", response_model=list[TrainingDatasetInfo])
def get_training_datasets():
    return [TrainingDatasetInfo(**item) for item in training_manager.list_datasets()]


@router.get("/gpus", response_model=list[GPUInfo])
def get_training_gpus():
    return [GPUInfo(**item) for item in training_manager.get_gpu_inventory()]


@router.get("/jobs", response_model=list[TrainingJobSummary])
def get_training_jobs():
    return [TrainingJobSummary(**item) for item in training_manager.list_jobs(refresh=True)]


@router.post("/jobs", response_model=TrainingJobSummary)
def create_training_job(req: TrainingJobCreateRequest):
    try:
        return TrainingJobSummary(**training_manager.create_job(req.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=TrainingJobDetail)
def get_training_job(job_id: str):
    try:
        return TrainingJobDetail(**training_manager.get_job(job_id, refresh=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}") from exc


@router.post("/jobs/{job_id}/stop", response_model=TrainingJobSummary)
def stop_training_job(job_id: str):
    try:
        return TrainingJobSummary(**training_manager.stop_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/jobs/{job_id}", status_code=204)
def delete_training_job(job_id: str):
    try:
        training_manager.delete_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


@router.get("/jobs/{job_id}/curves", response_model=TrainingCurvesResponse)
def get_training_curves(job_id: str, max_points: int = Query(default=2000, ge=100, le=10000)):
    try:
        return TrainingCurvesResponse(**training_manager.get_curves(job_id, max_points=max_points))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}") from exc


@router.get("/jobs/{job_id}/logs", response_model=TrainingLogResponse)
def get_training_logs(job_id: str, limit: int = Query(default=300, ge=20, le=2000)):
    try:
        return TrainingLogResponse(job_id=job_id, lines=training_manager.get_job_logs(job_id, limit=limit))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}") from exc
