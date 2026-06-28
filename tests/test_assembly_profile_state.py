from __future__ import annotations

import asyncio

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
