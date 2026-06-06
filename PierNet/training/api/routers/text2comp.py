"""
文生计算模块训练 API router
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from PierNet.training.api.schemas.text2comp import (
    ExpertModelSummary,
    Text2CompCurvesResponse,
    Text2CompDatasetInfo,
    Text2CompGPUInfo,
    Text2CompJobCreateRequest,
    Text2CompJobDetail,
    Text2CompJobSummary,
    Text2CompLogResponse,
    Text2CompOverviewResponse,
    Text2CompTrainRequest,
    Text2CompTrainResponse,
)
from PierNet.training.text2comp import text2comp_manager

router = APIRouter(prefix="/text2comp", tags=["text2comp"])


@router.get("/status", response_model=Text2CompOverviewResponse)
def get_text2comp_status():
    """获取文生计算模块训练状态（兼容前端API）"""
    return Text2CompOverviewResponse(**text2comp_manager.get_overview())


@router.post("/train", response_model=Text2CompTrainResponse)
def start_text2comp_train(req: Text2CompTrainRequest):
    """启动Text2Comp训练任务（兼容前端API）"""
    try:
        result = text2comp_manager.create_job(req.model_dump())
        return Text2CompTrainResponse(
            ok=True,
            job_id=result.get("job_id", ""),
            model_path=result.get("artifact_root", ""),
            config=result.get("config", {}),
        )
    except ValueError as exc:
        return Text2CompTrainResponse(ok=False, error=str(exc))
    except Exception as exc:
        return Text2CompTrainResponse(ok=False, error=str(exc))


@router.get("/overview", response_model=Text2CompOverviewResponse)
def get_text2comp_overview():
    """获取文生计算模块训练总览"""
    return Text2CompOverviewResponse(**text2comp_manager.get_overview())


@router.get("/experts", response_model=list[ExpertModelSummary])
def get_expert_models():
    """获取可用的专家模型列表"""
    return [ExpertModelSummary(**item) for item in text2comp_manager.list_simulators()]


@router.get("/datasets", response_model=list[Text2CompDatasetInfo])
def get_text2comp_datasets():
    """获取可用的数据集列表"""
    return [Text2CompDatasetInfo(**item) for item in text2comp_manager.list_datasets()]


@router.get("/gpus", response_model=list[Text2CompGPUInfo])
def get_text2comp_gpus():
    """获取GPU资源状态"""
    return [Text2CompGPUInfo(**item) for item in text2comp_manager.get_gpu_inventory()]


@router.get("/jobs", response_model=list[Text2CompJobSummary])
def get_text2comp_jobs():
    """获取所有训练任务列表"""
    return [Text2CompJobSummary(**item) for item in text2comp_manager.list_jobs(refresh=True)]


@router.post("/jobs", response_model=Text2CompJobSummary)
def create_text2comp_job(req: Text2CompJobCreateRequest):
    """创建新的训练任务"""
    try:
        return Text2CompJobSummary(**text2comp_manager.create_job(req.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=Text2CompJobDetail)
def get_text2comp_job(job_id: str):
    """获取单个训练任务详情"""
    try:
        return Text2CompJobDetail(**text2comp_manager.get_job(job_id, refresh=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Text2Comp training job not found: {job_id}") from exc


@router.post("/jobs/{job_id}/stop", response_model=Text2CompJobSummary)
def stop_text2comp_job(job_id: str):
    """停止正在运行的训练任务"""
    try:
        return Text2CompJobSummary(**text2comp_manager.stop_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Text2Comp training job not found: {job_id}") from exc


@router.delete("/jobs/{job_id}", status_code=204)
def delete_text2comp_job(job_id: str):
    """删除训练任务"""
    try:
        text2comp_manager.delete_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Text2Comp training job not found: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


@router.get("/jobs/{job_id}/curves", response_model=Text2CompCurvesResponse)
def get_text2comp_curves(job_id: str, max_points: int = Query(default=2000, ge=100, le=10000)):
    """获取训练曲线数据"""
    try:
        return Text2CompCurvesResponse(**text2comp_manager.get_curves(job_id, max_points=max_points))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Text2Comp training job not found: {job_id}") from exc


@router.get("/jobs/{job_id}/logs", response_model=Text2CompLogResponse)
def get_text2comp_logs(job_id: str, limit: int = Query(default=300, ge=20, le=2000)):
    """获取训练日志"""
    try:
        return Text2CompLogResponse(job_id=job_id, lines=text2comp_manager.get_job_logs(job_id, limit=limit))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Text2Comp training job not found: {job_id}") from exc


@router.post("/jobs/{job_id}/validate")
def validate_job_data(job_id: str):
    """验证训练数据是否有效"""
    try:
        job = text2comp_manager.get_job(job_id)
        train_data_path = job.get("train_data_path")
        if not train_data_path:
            return {"job_id": job_id, "is_valid": False, "message": "No training data path in job"}
        expert_model = job.get("simulator")
        expert_info = text2comp_manager.EXPERT_MODEL_LIBRARY.get(expert_model, {})
        expected_dim = expert_info.get("output_dim", 128)
        validation = text2comp_manager.validate_training_data(train_data_path, expected_dim)
        return {"job_id": job_id, "is_valid": validation.get("is_valid", False), "message": validation}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Text2Comp training job not found: {job_id}") from exc

@router.get("/models")
def get_trained_models():
    """获取已训练的模型文件列表"""

    models_dir = text2comp_manager.ARTIFACTS_ROOT

    models = []
    if models_dir.exists():
        for simulator_dir in models_dir.iterdir():
            if simulator_dir.is_dir():
                runs_dir = simulator_dir / "runs"
                if runs_dir.exists():
                    for run_dir in runs_dir.iterdir():
                        if run_dir.is_dir():
                            final_model = run_dir / "final_model.pt"
                            config_file = run_dir / "config.json"
                            if final_model.exists():
                                # Get file info
                                stat = final_model.stat()
                                models.append({
                                    "name": run_dir.name,
                                    "simulator": simulator_dir.name,
                                    "path": str(final_model),
                                    "size_mb": round(stat.st_size / (1024*1024), 2),
                                    "mtime": stat.st_mtime,
                                    "has_config": config_file.exists(),
                                })
    return {"models": models, "total": len(models)}
