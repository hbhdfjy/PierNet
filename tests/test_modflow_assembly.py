from PierNet.training.api.modflow_assembly import ModflowAssemblyProfilePipeline, ModflowFeatureBuilder


def _pipeline_for_gate() -> ModflowAssemblyProfilePipeline:
    pipeline = object.__new__(ModflowAssemblyProfilePipeline)
    pipeline.param_names = ["recharge", "hydraulic_conductivity", "storage"]
    pipeline.feature_builder = ModflowFeatureBuilder
    return pipeline


def test_modflow_task_gate_requires_numerical_task_intent() -> None:
    pipeline = _pipeline_for_gate()

    assert pipeline.looks_like_modflow_task("什么是地下水？请用一句话解释。") is False
    assert pipeline.looks_like_modflow_task("使用 MODFLOW 估计 recharge=0.001 的补给场景。") is True
    assert pipeline.looks_like_modflow_task("请预测含水层水头，初始水位为 12.5。") is True
