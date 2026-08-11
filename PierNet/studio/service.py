from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from PierNet.studio import store
from PierNet.studio.data_io import (
    DataInspectionError,
    canonicalize_data,
    discover_data_file,
    inspect_source_data,
)
from PierNet.studio.expert import (
    ExpertValidationError,
    check_compatibility,
    prepare_expert_package,
)
from PierNet.studio.paths import project_paths
from PierNet.studio.training import (
    DEFAULT_BASE_MODEL,
    prepare_training_data,
    run_project_inference,
    train_project_models,
)

_THREADS: dict[str, threading.Thread] = {}
_MODEL_LEASES: dict[str, threading.RLock] = {}
_DELETING_PROJECTS: set[str] = set()
_THREAD_LOCK = threading.RLock()
_PIPELINE_STAGES = ("preparation", "training", "assembly", "validation")
MAX_CONCURRENT_RUNS = max(1, int(os.getenv("PIERN_STUDIO_MAX_CONCURRENT_RUNS", "1")))
MAX_RUN_SECONDS = max(60, int(os.getenv("PIERN_STUDIO_MAX_RUN_SECONDS", "1800")))
MIN_FREE_DISK_BYTES = max(0, int(os.getenv("PIERN_STUDIO_MIN_FREE_DISK_BYTES", str(2 * 1024**3))))


class StudioError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def initialize() -> None:
    store.init_store()


def presets() -> dict[str, Any]:
    return {
        "data": {
            "extensions": [".h5", ".hdf5", ".npz", ".csv", ".parquet"],
            "archives": [".zip", ".tar.gz", ".tgz"],
            "minimum_samples": 4,
        },
        "expert": {
            "extensions": [".py", ".zip", ".tar.gz", ".tgz"],
            "call": "predict(inputs) -> outputs",
            "runtime": "python",
            "extra_environment_install": False,
        },
        "limits": {
            "upload_bytes": 1024**3,
            "max_concurrent_runs": MAX_CONCURRENT_RUNS,
            "max_run_seconds": MAX_RUN_SECONDS,
        },
        "outputs": ["scalar", "vector", "timeseries", "field", "json"],
    }


def _public_summary(project: dict[str, Any]) -> dict[str, Any]:
    return {
        key: project[key]
        for key in (
            "project_id",
            "name",
            "goal",
            "status",
            "current_stage",
            "created_at",
            "updated_at",
        )
    }


def project_summary(project: dict[str, Any]) -> dict[str, Any]:
    return _public_summary(project)


def project_snapshot(project: dict[str, Any]) -> dict[str, Any]:
    stages = project["stages"] or {}
    ordered_stages = [stages[stage_id] for stage_id, _ in store.STAGE_DEFINITIONS if stage_id in stages]
    compatibility = project.get("compatibility") or {}
    artifacts = project.get("artifacts") or {}
    result = project.get("result") or {}
    return {
        **_public_summary(project),
        "stages": ordered_stages,
        "data": _sanitize_resource(project.get("data")),
        "expert": _sanitize_resource(project.get("expert")),
        "inspection": project.get("inspection"),
        "compatibility": compatibility or None,
        "artifacts": _sanitize_artifacts(artifacts) if artifacts else None,
        "result": result or None,
        "error": project.get("error"),
        "recommended_prompt": artifacts.get("recommended_prompt") or result.get("message"),
        "can_run": bool(compatibility.get("compatible")) and project["status"] != "running",
        "can_chat": project["status"] == "ready" and bool(artifacts.get("manifest_path")),
    }


def _sanitize_resource(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    hidden = {"source_path", "data_path", "canonical_path", "root", "manifest_path"}
    return {key: item for key, item in value.items() if key not in hidden}


def _sanitize_artifacts(value: dict[str, Any]) -> dict[str, Any]:
    hidden = {"manifest_path", "router_path", "text2comp_path", "training_data_path"}
    return {key: item for key, item in value.items() if key not in hidden}


def list_projects(owner_id: str) -> list[dict[str, Any]]:
    return [project_summary(project) for project in store.list_projects(owner_id)]


def get_project(owner_id: str, project_id: str) -> dict[str, Any]:
    return project_snapshot(store.get_project(owner_id, project_id))


def create_project(owner_id: str, name: str, goal: str) -> dict[str, Any]:
    project = store.create_project(owner_id, name, goal)
    project_paths(project["project_id"], create=True)
    store.append_audit(
        owner_id,
        "project_created",
        project_id=project["project_id"],
        payload={"name": project["name"]},
    )
    return project_snapshot(project)


def delete_project(owner_id: str, project_id: str) -> dict[str, Any]:
    project = store.get_project(owner_id, project_id)
    _ensure_mutable(project)
    with _THREAD_LOCK:
        thread = _THREADS.get(project_id)
        if thread and thread.is_alive():
            raise StudioError(
                "project_running",
                "Demo 正在构建，请先停止任务再删除项目。",
            )
        if project_id in _DELETING_PROJECTS:
            raise StudioError("project_deleting", "项目正在删除，请稍候。")
        _DELETING_PROJECTS.add(project_id)
    try:
        with _project_model_lease(project_id):
            project = store.get_project(owner_id, project_id)
            _ensure_mutable(project, allow_deleting=True)
            paths = project_paths(project_id, create=False)
            targets = (paths.data_root, paths.artifact_root, paths.logs)
            try:
                for target in targets:
                    if not target.exists():
                        continue
                    if target.is_symlink() or target.resolve().parent != target.parent.resolve():
                        raise StudioError(
                            "unsafe_project_path",
                            "项目存储路径异常，系统已停止删除，请联系管理员。",
                        )
                    shutil.rmtree(target)
            except StudioError:
                raise
            except OSError as exc:
                store.append_audit(
                    owner_id,
                    "project_delete_failed",
                    project_id=project_id,
                    payload={"error_type": type(exc).__name__},
                )
                raise StudioError(
                    "project_cleanup_failed",
                    "项目文件暂时无法清理，请稍后重试。",
                ) from exc
            store.delete_project(owner_id, project_id)
            store.append_audit(
                owner_id,
                "project_deleted",
                project_id=project_id,
                payload={"name": project["name"]},
            )
    finally:
        with _THREAD_LOCK:
            _DELETING_PROJECTS.discard(project_id)
            _THREADS.pop(project_id, None)
            _MODEL_LEASES.pop(project_id, None)
    return {
        "project_id": project_id,
        "deleted": True,
        "message": "项目及其资源已删除",
    }


def _ensure_mutable(
    project: dict[str, Any],
    *,
    allow_deleting: bool = False,
) -> None:
    with _THREAD_LOCK:
        deleting = project["project_id"] in _DELETING_PROJECTS
    if deleting and not allow_deleting:
        raise StudioError("project_deleting", "项目正在删除，请稍候。")
    if project["status"] == "running":
        raise StudioError("project_running", "Demo 正在构建，请完成或停止后再更换资源")


def _reset_stages(project_id: str, stage_ids: tuple[str, ...]) -> None:
    project = store.get_project_unscoped(project_id)
    stages = project["stages"]
    for stage_id in stage_ids:
        stage = stages[stage_id]
        stage.update(
            {
                "status": "waiting",
                "progress": None,
                "message": "等待开始",
                "retryable": False,
                "started_at": None,
                "finished_at": None,
            }
        )
    store.update_project(project_id, stages=stages)


def _update_resources_stage(project_id: str) -> None:
    project = store.get_project_unscoped(project_id)
    has_data = bool(project.get("data"))
    has_expert = bool(project.get("expert"))
    if has_data and has_expert:
        message = "科学计算数据和计算模型均已上传"
        status = "succeeded"
        progress = 1.0
    elif has_data:
        message = "数据已上传，请继续上传计算模型"
        status = "running"
        progress = 0.5
    elif has_expert:
        message = "计算模型已上传，请继续上传科学计算数据"
        status = "running"
        progress = 0.5
    else:
        message = "请上传科学计算数据和计算模型"
        status = "waiting"
        progress = 0.0
    store.update_stage(
        project_id,
        "resources",
        status=status,
        progress=progress,
        message=message,
    )


def attach_data(
    owner_id: str,
    project_id: str,
    source_path: Path,
    original_name: str,
) -> dict[str, Any]:
    project = store.get_project(owner_id, project_id)
    _ensure_mutable(project)
    paths = project_paths(project_id)
    try:
        data_path = discover_data_file(source_path, paths.source / "extracted")
        inspection = inspect_source_data(data_path)
        data_record: dict[str, Any] = {
            "filename": original_name,
            "format": data_path.suffix.lower().removeprefix("."),
            "source_path": str(source_path),
            "data_path": str(data_path),
            "size_bytes": source_path.stat().st_size,
            "needs_mapping": bool(inspection.get("needs_mapping")),
        }
        if not data_record["needs_mapping"]:
            metadata = canonicalize_data(data_path, paths.canonical / "data.npz")
            data_record.update(metadata)
        store.update_project(
            project_id,
            data=data_record,
            inspection={"data": inspection},
            compatibility=None,
            artifacts=None,
            result=None,
            error=None,
            status="draft",
        )
        _reset_stages(
            project_id,
            ("inspection", "compatibility", *_PIPELINE_STAGES),
        )
        _update_resources_stage(project_id)
        store.update_stage(
            project_id,
            "inspection",
            status="running" if data_record["needs_mapping"] else "succeeded",
            progress=0.5 if data_record["needs_mapping"] else 1.0,
            message=("请选择数据中的输入列和输出列" if data_record["needs_mapping"] else "数据结构和数值范围检查通过"),
        )
        store.append_event(project_id, "data_uploaded", {"message": "科学计算数据已上传"})
        store.append_audit(
            owner_id,
            "data_uploaded",
            project_id=project_id,
            payload={
                "filename": original_name,
                "size_bytes": data_record["size_bytes"],
            },
        )
    except (DataInspectionError, OSError, ValueError) as exc:
        raise StudioError("invalid_data", str(exc)) from exc
    return get_project(owner_id, project_id)


def apply_mapping(
    owner_id: str,
    project_id: str,
    input_fields: list[str],
    output_fields: list[str],
) -> dict[str, Any]:
    project = store.get_project(owner_id, project_id)
    _ensure_mutable(project)
    data = dict(project.get("data") or {})
    if not data.get("data_path"):
        raise StudioError("missing_data", "请先上传科学计算数据")
    paths = project_paths(project_id)
    try:
        metadata = canonicalize_data(
            Path(data["data_path"]),
            paths.canonical / "data.npz",
            input_fields=input_fields,
            output_fields=output_fields,
        )
    except (DataInspectionError, OSError, ValueError) as exc:
        raise StudioError("invalid_mapping", str(exc)) from exc
    data.update(metadata)
    data["needs_mapping"] = False
    data["input_fields"] = input_fields
    data["output_fields"] = output_fields
    store.update_project(
        project_id,
        data=data,
        compatibility=None,
        artifacts=None,
        result=None,
        error=None,
        status="draft",
    )
    _reset_stages(project_id, ("compatibility", *_PIPELINE_STAGES))
    store.update_stage(
        project_id,
        "inspection",
        status="succeeded",
        progress=1.0,
        message="输入和输出字段已经确认",
    )
    store.append_event(project_id, "mapping_applied", {"message": "数据字段已经确认"})
    store.append_audit(
        owner_id,
        "mapping_applied",
        project_id=project_id,
        payload={
            "input_field_count": len(input_fields),
            "output_field_count": len(output_fields),
        },
    )
    return get_project(owner_id, project_id)


def attach_expert(
    owner_id: str,
    project_id: str,
    source_path: Path,
    original_name: str,
) -> dict[str, Any]:
    project = store.get_project(owner_id, project_id)
    _ensure_mutable(project)
    paths = project_paths(project_id)
    try:
        expert = prepare_expert_package(source_path, paths.expert / "package")
    except (ExpertValidationError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise StudioError("invalid_expert", str(exc)) from exc
    expert.update(
        {
            "filename": original_name,
            "size_bytes": source_path.stat().st_size,
            "validated": False,
        }
    )
    store.update_project(
        project_id,
        expert=expert,
        compatibility=None,
        artifacts=None,
        result=None,
        error=None,
        status="draft",
    )
    _reset_stages(project_id, ("compatibility", *_PIPELINE_STAGES))
    _update_resources_stage(project_id)
    store.append_event(project_id, "expert_uploaded", {"message": "计算模型已上传"})
    store.append_audit(
        owner_id,
        "expert_uploaded",
        project_id=project_id,
        payload={
            "filename": original_name,
            "size_bytes": expert["size_bytes"],
        },
    )
    return get_project(owner_id, project_id)


def inspect_resources(owner_id: str, project_id: str) -> dict[str, Any]:
    project = store.get_project(owner_id, project_id)
    _ensure_mutable(project)
    preserved_stage = project["current_stage"] if project["status"] == "ready" else None
    data = project.get("data") or {}
    expert = project.get("expert") or {}
    if not data:
        raise StudioError("missing_data", "请先上传科学计算数据")
    if data.get("needs_mapping"):
        raise StudioError("mapping_required", "请先确认输入列和输出列")
    if not expert:
        raise StudioError("missing_expert", "请先上传计算模型")
    canonical_path = Path(str(data.get("canonical_path") or ""))
    manifest_path = Path(str(expert.get("manifest_path") or ""))
    if not canonical_path.is_file():
        raise StudioError("missing_canonical_data", "标准数据文件不存在，请重新上传数据")
    if not manifest_path.is_file():
        raise StudioError("missing_expert", "计算模型入口不存在，请重新上传模型")
    store.update_stage(
        project_id,
        "inspection",
        status="succeeded",
        progress=1.0,
        message="数据结构和计算模型接口均已识别",
    )
    if preserved_stage is not None:
        store.update_project(project_id, current_stage=preserved_stage)
    store.append_event(
        project_id,
        "inspection_finished",
        {"message": "数据结构和计算模型接口检查通过"},
    )
    return get_project(owner_id, project_id)


def inspect_and_check(owner_id: str, project_id: str) -> dict[str, Any]:
    inspect_resources(owner_id, project_id)
    project = store.get_project(owner_id, project_id)
    _ensure_mutable(project)
    data = project.get("data") or {}
    expert = project.get("expert") or {}
    if not data:
        raise StudioError("missing_data", "请先上传科学计算数据")
    if data.get("needs_mapping"):
        raise StudioError("mapping_required", "请先确认输入列和输出列")
    if not expert:
        raise StudioError("missing_expert", "请先上传计算模型")
    canonical_path = Path(str(data.get("canonical_path") or ""))
    if not canonical_path.exists():
        raise StudioError("missing_canonical_data", "标准数据文件不存在，请重新上传数据")
    paths = project_paths(project_id)
    store.update_stage(
        project_id,
        "compatibility",
        status="running",
        progress=None,
        message="正在使用真实数据检查计算模型",
    )
    store.append_event(project_id, "compatibility_started", {"message": "开始检查资源匹配"})
    try:
        report = check_compatibility(canonical_path, expert, work_dir=paths.logs)
    except (ExpertValidationError, OSError, ValueError) as exc:
        store.update_stage(
            project_id,
            "compatibility",
            status="failed",
            progress=1.0,
            message=str(exc),
            retryable=True,
        )
        store.update_project(
            project_id,
            status="failed",
            error={"code": "expert_execution_failed", "message": str(exc)},
        )
        raise StudioError("expert_execution_failed", str(exc)) from exc
    expert["validated"] = bool(report["compatible"])
    store.update_project(
        project_id,
        expert=expert,
        compatibility=report,
        status="draft" if report["compatible"] else "failed",
        error=(
            None
            if report["compatible"]
            else {
                "code": "incompatible_resources",
                "message": report.get("message") or "数据与计算模型不匹配",
            }
        ),
    )
    store.update_stage(
        project_id,
        "compatibility",
        status="succeeded" if report["compatible"] else "failed",
        progress=1.0,
        message=(
            "数据与计算模型匹配，可以开始构建 Demo"
            if report["compatible"]
            else report.get("message", "数据与计算模型不匹配")
        ),
        retryable=not report["compatible"],
    )
    store.append_event(
        project_id,
        "compatibility_finished",
        {
            "message": ("数据与计算模型匹配" if report["compatible"] else "数据与计算模型不匹配"),
            "compatible": bool(report["compatible"]),
        },
    )
    store.append_audit(
        owner_id,
        "compatibility_finished",
        project_id=project_id,
        payload={
            "compatible": bool(report["compatible"]),
            "input_shape": report.get("input_shape"),
            "output_shape": report.get("actual_output_shape"),
        },
    )
    return get_project(owner_id, project_id)


def _cancel_requested(project_id: str) -> bool:
    return bool(store.get_project_unscoped(project_id).get("cancel_requested"))


def _run_should_cancel(project_id: str, deadline: float) -> bool:
    if time.monotonic() >= deadline:
        raise TimeoutError("Demo 构建已达到运行时长上限")
    return _cancel_requested(project_id)


def _enforce_run_capacity(project_id: str) -> None:
    active = sum(1 for active_project, thread in _THREADS.items() if active_project != project_id and thread.is_alive())
    if active >= MAX_CONCURRENT_RUNS:
        raise StudioError(
            "run_quota_reached",
            "当前已有 Demo 正在构建，请等待完成后再开始。",
        )


def _preflight_run(project_id: str) -> None:
    paths = project_paths(project_id)
    free_bytes = shutil.disk_usage(paths.data_root).free
    if free_bytes < MIN_FREE_DISK_BYTES:
        required_gib = MIN_FREE_DISK_BYTES / 1024**3
        raise StudioError(
            "insufficient_storage",
            f"可用存储空间不足，请至少保留 {required_gib:.0f} GB 后重试。",
        )
    if not Path(DEFAULT_BASE_MODEL).is_dir():
        raise StudioError("base_model_unavailable", "训练基础模型暂不可用，请联系管理员。")


@contextmanager
def _project_model_lease(project_id: str) -> Iterator[None]:
    with _THREAD_LOCK:
        lease = _MODEL_LEASES.setdefault(project_id, threading.RLock())
    with lease:
        yield


def _load_project_manifest(
    project_id: str,
    artifacts: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(str(artifacts.get("manifest_path") or "")).resolve()
    assembly_root = project_paths(project_id).assembly.resolve()
    if assembly_root not in manifest_path.parents or not manifest_path.is_file():
        raise StudioError("missing_manifest", "Demo 文件不存在，请重新构建")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("project_id") != project_id:
        raise StudioError(
            "project_model_mismatch",
            "当前 Demo 与项目不一致，请重新构建后再试。",
        )
    return manifest_path, manifest


def _progress(project_id: str, stage_id: str, value: float, message: str) -> None:
    store.update_stage(
        project_id,
        stage_id,
        status="running",
        progress=value,
        message=message,
    )
    store.append_event(
        project_id,
        "progress",
        {"stage": stage_id, "progress": value, "message": message},
    )


def _run_pipeline(project_id: str) -> None:
    paths = project_paths(project_id)
    deadline = time.monotonic() + MAX_RUN_SECONDS

    def should_cancel() -> bool:
        return _run_should_cancel(project_id, deadline)

    try:
        project = store.get_project_unscoped(project_id)
        data = project["data"]
        expert = project["expert"]
        store.update_stage(
            project_id,
            "preparation",
            status="running",
            progress=0.0,
            message="正在根据你的数据准备训练内容",
        )
        metadata = prepare_training_data(
            Path(data["canonical_path"]),
            paths.training_data,
            goal=project["goal"],
        )
        if should_cancel():
            raise InterruptedError("用户取消了任务")
        store.update_stage(
            project_id,
            "preparation",
            status="succeeded",
            progress=1.0,
            message=f"已准备 {metadata['prompt_count']} 条训练语句",
        )
        store.append_event(
            project_id,
            "preparation_finished",
            {"message": "训练内容已经准备完成"},
        )

        store.update_stage(
            project_id,
            "training",
            status="running",
            progress=0.0,
            message="正在训练 Demo",
        )
        router_path = paths.router / "router.pt"
        text2comp_path = paths.text2comp / "text2comp.pt"
        metrics = train_project_models(
            Path(metadata["training_data_path"]),
            router_path,
            text2comp_path,
            base_model=DEFAULT_BASE_MODEL,
            progress=lambda value, message: _progress(project_id, "training", value, message),
            cancel=should_cancel,
        )
        store.update_stage(
            project_id,
            "training",
            status="succeeded",
            progress=1.0,
            message="Demo 训练完成",
        )
        if should_cancel():
            raise InterruptedError("用户取消了任务")

        store.update_stage(
            project_id,
            "assembly",
            status="running",
            progress=0.4,
            message="正在连接训练结果与计算模型",
        )
        manifest = {
            "version": 1,
            "project_id": project_id,
            "base_model": DEFAULT_BASE_MODEL,
            "router_path": str(router_path),
            "text2comp_path": str(text2comp_path),
            "expert": expert,
            "input_shape": data["input_shape"],
            "output_shape": data["output_shape"],
            "input_names": data["input_names"],
            "output_names": data["output_names"],
            "metrics": metrics,
            "recommended_prompt": metadata["recommended_prompt"],
            "created_at": time.time(),
        }
        manifest_path = paths.assembly / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts = {
            "manifest_path": str(manifest_path),
            "router_path": str(router_path),
            "text2comp_path": str(text2comp_path),
            "training_data_path": metadata["training_data_path"],
            "recommended_prompt": metadata["recommended_prompt"],
            "metrics": metrics,
        }
        store.update_project(project_id, artifacts=artifacts)
        store.update_stage(
            project_id,
            "assembly",
            status="succeeded",
            progress=1.0,
            message="可对话 Demo 已创建",
        )

        store.update_stage(
            project_id,
            "validation",
            status="running",
            progress=None,
            message="正在使用推荐问题验证计算结果",
        )
        with _project_model_lease(project_id):
            result = run_project_inference(
                message=metadata["recommended_prompt"],
                manifest=manifest,
                expert=expert,
                work_dir=paths.logs,
            )
        store.update_project(
            project_id,
            result=result,
            status="ready",
            current_stage="validation",
            error=None,
            cancel_requested=False,
        )
        store.update_stage(
            project_id,
            "validation",
            status="succeeded",
            progress=1.0,
            message="端到端计算验证通过",
        )
        store.append_event(
            project_id,
            "project_ready",
            {"message": "你的 Demo 已经可以使用"},
        )
        store.append_audit(
            project["owner_id"],
            "project_ready",
            project_id=project_id,
            payload={"metrics": metrics},
        )
    except InterruptedError as exc:
        project = store.get_project_unscoped(project_id)
        current = project["current_stage"]
        store.update_stage(
            project_id,
            current,
            status="cancelled",
            progress=project["stages"][current].get("progress"),
            message=str(exc),
            retryable=True,
        )
        store.update_project(
            project_id,
            status="cancelled",
            error={"code": "cancelled", "message": str(exc)},
            cancel_requested=False,
        )
        store.append_event(project_id, "cancelled", {"message": str(exc)})
        store.append_audit(
            project["owner_id"],
            "run_cancelled",
            project_id=project_id,
            payload={"stage": current},
        )
    except Exception as exc:
        project = store.get_project_unscoped(project_id)
        current = project["current_stage"]
        public_message = _friendly_pipeline_error(exc)
        store.update_stage(
            project_id,
            current,
            status="failed",
            progress=project["stages"][current].get("progress"),
            message=public_message,
            retryable=True,
        )
        store.update_project(
            project_id,
            status="failed",
            error={
                "code": "pipeline_failed",
                "message": public_message,
                "technical": f"{type(exc).__name__}: {exc}",
            },
            cancel_requested=False,
        )
        (paths.logs / "pipeline_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        store.append_event(
            project_id,
            "failed",
            {"message": public_message, "stage": current},
        )
        store.append_audit(
            project["owner_id"],
            "run_failed",
            project_id=project_id,
            payload={
                "stage": current,
                "error_type": type(exc).__name__,
            },
        )
    finally:
        with _THREAD_LOCK:
            _THREADS.pop(project_id, None)


def _friendly_pipeline_error(exc: Exception) -> str:
    text = str(exc).strip()
    if isinstance(exc, FileNotFoundError):
        return "运行所需文件不存在，请重新上传资源后重试。"
    if "out of memory" in text.lower():
        return "当前计算资源不足，系统已安全停止。请稍后重试。"
    if isinstance(exc, ExpertValidationError):
        return text
    if isinstance(exc, TimeoutError):
        return "Demo 构建超过运行时长上限，系统已安全停止。请缩小数据后重试。"
    return "Demo 构建没有完成。系统已保留现有进度，可以安全重试。"


def start_run(owner_id: str, project_id: str) -> dict[str, Any]:
    project = store.get_project(owner_id, project_id)
    if not (project.get("compatibility") or {}).get("compatible"):
        raise StudioError("compatibility_required", "请先完成数据与计算模型匹配检查")
    with _THREAD_LOCK:
        if project_id in _DELETING_PROJECTS:
            raise StudioError("project_deleting", "项目正在删除，请稍候。")
        existing = _THREADS.get(project_id)
        if existing and existing.is_alive():
            return {
                "project_id": project_id,
                "status": "running",
                "message": "Demo 正在构建中",
            }
        _enforce_run_capacity(project_id)
        _preflight_run(project_id)
        _reset_stages(project_id, _PIPELINE_STAGES)
        store.update_project(
            project_id,
            status="running",
            current_stage="preparation",
            cancel_requested=False,
            error=None,
        )
        thread = threading.Thread(
            target=_run_pipeline,
            args=(project_id,),
            daemon=True,
            name=f"studio-{project_id}",
        )
        _THREADS[project_id] = thread
        thread.start()
    store.append_audit(
        owner_id,
        "run_started",
        project_id=project_id,
        payload={"max_run_seconds": MAX_RUN_SECONDS},
    )
    return {
        "project_id": project_id,
        "status": "running",
        "message": "Demo 构建已经开始",
    }


def cancel_run(owner_id: str, project_id: str) -> dict[str, Any]:
    project = store.get_project(owner_id, project_id)
    if project["status"] != "running":
        raise StudioError("not_running", "当前项目没有运行中的任务")
    store.update_project(project_id, cancel_requested=True)
    store.append_event(project_id, "cancel_requested", {"message": "正在安全停止"})
    store.append_audit(
        owner_id,
        "cancel_requested",
        project_id=project_id,
    )
    return {
        "project_id": project_id,
        "status": "running",
        "message": "已发送停止请求",
    }


def chat(owner_id: str, project_id: str, message: str) -> dict[str, Any]:
    paths = project_paths(project_id)
    with _project_model_lease(project_id):
        project = store.get_project(owner_id, project_id)
        _ensure_mutable(project)
        if project["status"] != "ready":
            raise StudioError("project_not_ready", "请先完成 Demo 构建")
        artifacts = project.get("artifacts") or {}
        _, manifest = _load_project_manifest(project_id, artifacts)
        result = run_project_inference(
            message=message,
            manifest=manifest,
            expert=project["expert"],
            work_dir=paths.logs,
        )
    created_at = time.time()
    chat_id = store.save_chat(project_id, {"message": message}, result)
    store.append_event(project_id, "chat_completed", {"message": "计算完成"})
    store.append_audit(
        owner_id,
        "chat_completed",
        project_id=project_id,
        payload={
            "chat_id": chat_id,
            "latency_ms": result.get("latency_ms"),
            "output_shape": (result.get("chart") or {}).get("shape"),
        },
    )
    return {
        "chat_id": chat_id,
        "project_id": project_id,
        **result,
        "created_at": created_at,
    }
