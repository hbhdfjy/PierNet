from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDepthwiseConvBlock(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            padding=0,
            bias=False,
        )
        self.pointwise_in = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.pointwise_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.left_padding, 0))
        x = self.depthwise(x)
        x = self.pointwise_in(x)
        x = F.glu(x, dim=1)
        x = self.pointwise_out(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return residual + x


class FullSeqDilatedConvRouter(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_scenarios: int,
        max_sequence_length: int,
        embedding_dim: int = 192,
        model_dim: int = 256,
        scene_dim: int = 16,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.input_proj = nn.Linear(embedding_dim, model_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, model_dim)
        self.blocks = nn.ModuleList(
            [
                CausalDepthwiseConvBlock(
                    channels=model_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.scene_embedding = nn.Embedding(num_scenarios, scene_dim)
        hidden_dim = model_dim * 3 + scene_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scenario_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = self.input_norm(x)
        x = self.input_proj(x)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)

        mask = attention_mask.unsqueeze(-1)
        mask_f = mask.to(dtype=x.dtype)
        mean_pool = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        max_fill = torch.full_like(x, -1e4)
        max_pool = torch.where(mask, x, max_fill).amax(dim=1)
        last_index = attention_mask.sum(dim=1).clamp(min=1) - 1
        last_token = x[torch.arange(x.size(0), device=x.device), last_index]
        scene = self.scene_embedding(scenario_ids)
        features = torch.cat([mean_pool, max_pool, last_token, scene], dim=-1)
        return self.head(features).squeeze(-1)
