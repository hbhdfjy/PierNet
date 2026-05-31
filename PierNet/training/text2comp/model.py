"""
文生计算模型定义

架构：LLM embedding (冻结或微调) + MLP回归头

输入：文本描述 (prompt)
输出：物理数值预测 (regression)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


class RegressionHead(nn.Module):
    """MLP回归头"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int] = [128, 256, 512, 1024],
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Text2CompModel(nn.Module):
    """文生计算模型：文本 → 物理数值"""

    def __init__(
        self,
        base_model_path: str,
        output_dim: int,
        hidden_layers: list[int] = [128, 256, 512, 1024],
        activation: str = "relu",
        freeze_base: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_cache=False,
        )

        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        self.hidden_size = getattr(config, "hidden_size", None)
        if self.hidden_size is None:
            self.hidden_size = self.base_model.model.embed_tokens.embedding_dim

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.gradient_checkpointing_enable()

        self.head = RegressionHead(
            input_dim=self.hidden_size,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
        )

        self.freeze_base = freeze_base
        self.output_dim = output_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = last_hidden * mask
        sum_mask = mask.sum(dim=1).clamp(min=1.0)
        pooled = masked_hidden.sum(dim=1) / sum_mask

        predictions = self.head(pooled)
        return predictions

    def get_num_parameters(self) -> dict[str, int]:
        base_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_base = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters())

        return {
            "base_model_total": base_params,
            "base_model_trainable": trainable_base,
            "head_total": head_params,
            "total_trainable": trainable_base + head_params,
            "total": base_params + head_params,
        }


def create_text2comp_model(config) -> Text2CompModel:
    """从配置创建模型"""
    return Text2CompModel(
        base_model_path=config.base_model_path,
        output_dim=config.output_dim,
        hidden_layers=config.head_layers,
        activation=config.head_activation,
        freeze_base=config.freeze_base_model,
    )