"""
Stage 2: Text-to-Computation 训练数据生成。

使用完全 LLM 生成方案（零模板依赖）。
"""

from PierNet.synth.text2comp.generator import LLMTextGenerator

__all__ = [
    "LLMTextGenerator",
]
