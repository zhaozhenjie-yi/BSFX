from __future__ import annotations

import torch
from torch import Tensor, nn


def build_structural_bias(
    depth_ids: Tensor,
    sibling_ids: Tensor,
    segment_ids: Tensor,
    *,
    lambda_d: float = 1.0,
    lambda_o: float = 0.25,
    ref_bias_scale: float = 1.0,
    ref_matrix: Tensor | None = None,
) -> Tensor:
    depth_diff = (depth_ids.unsqueeze(-1) - depth_ids.unsqueeze(-2)).abs().float()
    sibling_diff = (sibling_ids.unsqueeze(-1) - sibling_ids.unsqueeze(-2)).abs().float()

    bias = -lambda_d * depth_diff - lambda_o * sibling_diff
    same_segment = segment_ids.unsqueeze(-1).eq(segment_ids.unsqueeze(-2))
    bias = bias + same_segment.float() * 2.0
    if ref_matrix is not None:
        bias = bias + ref_matrix.float() * ref_bias_scale
    return bias


class StructuralTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        *,
        padding_mask: Tensor | None = None,
        structural_bias: Tensor | None = None,
    ) -> Tensor:
        attn_mask = None
        key_padding_mask = padding_mask
        if structural_bias is not None:
            attn_mask = structural_bias
            if padding_mask is not None:
                pad_bias = padding_mask.unsqueeze(1).expand(-1, x.size(1), -1).float() * -1e9
                attn_mask = attn_mask + pad_bias
                key_padding_mask = None
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class StructuralTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                StructuralTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        *,
        padding_mask: Tensor | None = None,
        structural_bias: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask, structural_bias=structural_bias)
        return x
