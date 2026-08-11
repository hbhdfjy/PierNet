from __future__ import annotations

import asyncio

import numpy as np

from PierNet.training.api.routers import assembly


def _restore_loaded_models(snapshot: dict) -> None:
    assembly._LOADED_MODELS.clear()
    assembly._LOADED_MODELS.update(snapshot)


def test_assembly_test_without_profile_id_does_not_use_loaded_profile(monkeypatch) -> None:
    snapshot = dict(assembly._LOADED_MODELS)
    try:
        assembly._LOADED_MODELS["assembly_profile"] = object()
        assembly._LOADED_MODELS["assembly_profile_info"] = {"model_id": "profile-loaded"}
        assembly._LOADED_MODELS["llm"] = object()

        def fail_profile_call(req):
            raise AssertionError("stale assembly profile should not handle a standard-chain test request")

        monkeypatch.setattr(assembly, "_test_assembly_profile", fail_profile_call)
        monkeypatch.setattr(
            assembly,
            "generate_response_with_router",
            lambda prompt: ("LLM answer", 0, "standard answer", "expert output", {"mode": "standard"}),
        )

        result = asyncio.run(
            assembly.test_assembly(
                assembly.AssemblyTestRequest(
                    config={"main_llm_path": "/models/qwen", "gpu_config": {"llm_gpu_ids": [0]}},
                    test_input="standard-chain request",
                )
            )
        )

        assert result.router_prediction == "normal"
        assert result.final_answer == "LLM answer\n\nstandard answer"
        assert result.debug_info == {"mode": "standard"}
    finally:
        _restore_loaded_models(snapshot)


def test_assembly_test_with_matching_profile_id_uses_loaded_profile(monkeypatch) -> None:
    snapshot = dict(assembly._LOADED_MODELS)
    try:
        assembly._LOADED_MODELS["assembly_profile"] = object()
        assembly._LOADED_MODELS["assembly_profile_info"] = {"model_id": "profile-loaded"}

        def profile_call(req):
            return assembly.AssemblyTestResponse(
                router_prediction="modflow",
                first_cot_result="profile",
                final_answer="profile answer",
                expert_used=True,
                latency_ms=1.0,
            )

        monkeypatch.setattr(assembly, "_test_assembly_profile", profile_call)

        result = asyncio.run(
            assembly.test_assembly(
                assembly.AssemblyTestRequest(
                    config={"assembly_profile_id": "profile-loaded", "main_llm_path": "/models/qwen"},
                    test_input="profile request",
                )
            )
        )

        assert result.final_answer == "profile answer"
        assert result.expert_used is True
    finally:
        _restore_loaded_models(snapshot)


def test_explicit_gcam_request_selects_loaded_uploaded_expert() -> None:
    snapshot = dict(assembly._LOADED_MODELS)
    try:
        assembly._LOADED_MODELS["expert_executor"] = "uploaded"
        assembly._LOADED_MODELS["uploaded_expert_model"] = {"simulator": "gcam"}

        assert assembly._explicit_uploaded_expert_simulator("请使用 GCAM 完成预测") == "gcam"
        assert assembly._explicit_uploaded_expert_simulator("你好") == ""
        assert assembly._expert_router_trigger_prefill("请使用 GCAM 完成预测")
        assert assembly.map_router_prediction_for_assembly(1) == (1, "gcam")
    finally:
        _restore_loaded_models(snapshot)


def test_gcam_prediction_formats_five_trajectories() -> None:
    answer = assembly._format_expert_prediction_answer(
        "gcam",
        np.arange(80, dtype=np.float32),
    )

    assert "2025—2100 年预测" in answer
    assert "煤电占比：2025: 0.00000" in answer
    assert "可再生能源占比：2025: 16.00000" in answer
    assert "全球平均气温异常：2025: 64.00000" in answer
    assert "2100: 79.00000" in answer


def test_training_job_profile_routes_only_complete_prediction_requests() -> None:
    snapshot = dict(assembly._LOADED_MODELS)
    try:
        assembly._LOADED_MODELS["assembly_profile_info"] = {
            "executor": "training_job_profile",
            "prediction_keywords": ["预测", "predict"],
            "task_keywords": ["modflow", "地下水"],
            "expert_input_dim": 3,
            "min_user_numeric_values": 3,
        }
        assert assembly._router_enabled_for_input("你好") is False
        assert assembly._router_enabled_for_input("介绍地下水") is False
        assert assembly._router_enabled_for_input("预测参数 [35.3, 5.6] 的结果") is False
        assert assembly._router_enabled_for_input("预测参数 [35.3, 5.6, 2.0] 的结果") is True
        assert assembly._training_profile_input_validation("1")["state"] == "ambiguous_input"
    finally:
        _restore_loaded_models(snapshot)
