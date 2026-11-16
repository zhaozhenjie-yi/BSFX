from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def binary_depth_encoding(depth_ids: Tensor, width: int) -> Tensor:
    shifts = torch.arange(width, device=depth_ids.device, dtype=depth_ids.dtype)
    bits = (depth_ids.unsqueeze(-1) >> shifts) & 1
    return bits.float()


def sinusoidal_encoding(indices: Tensor, dim: int) -> Tensor:
    device = indices.device
    dtype = torch.float32
    position = indices.float().unsqueeze(-1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    embedding = torch.zeros(*indices.shape, dim, device=device, dtype=dtype)
    embedding[..., 0::2] = torch.sin(position * div_term)
    embedding[..., 1::2] = torch.cos(position * div_term)
    return embedding


class HierarchicalPositionalEncoder(nn.Module):
    """
    First-stage approximation of the thesis HPE module.

    It fuses semantic embeddings with depth, sibling-order and path-position
    signals, then uses a second gate to combine semantic and structural views.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        word_dim: int = 128,
        struct_dim: int = 64,
        output_dim: int = 128,
        global_pos_dim: int = 16,
        local_pos_dim: int = 16,
        depth_dim: int = 32,
        sibling_dim: int = 16,
        alpha: float = 0.3,
        beta: float = 0.5,
        max_depth_bits: int = 8,
        max_global_pos: int = 4096,
        max_local_pos: int = 128,
        padding_idx: int = 0,
        ref_cue_scale: float = 1.0,
        ref_cue_gate_reg_weight: float = 0.0,
        zero_delta_identity: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.max_depth_bits = max_depth_bits
        self.ref_cue_scale = float(ref_cue_scale)
        self.ref_cue_gate_reg_weight = float(ref_cue_gate_reg_weight)
        self.zero_delta_identity = bool(zero_delta_identity)

        self.word_embedding = nn.Embedding(vocab_size, word_dim, padding_idx=padding_idx)
        self.global_pos_embedding = nn.Embedding(max_global_pos, global_pos_dim, padding_idx=0)
        self.local_pos_embedding = nn.Embedding(max_local_pos, local_pos_dim, padding_idx=0)
        self.delta_seg_projection = nn.Linear(1, global_pos_dim)

        self.depth_projection = nn.Linear(max_depth_bits, depth_dim)

        self.depth_to_struct = nn.Linear(depth_dim, struct_dim)
        self.sibling_to_struct = nn.Linear(sibling_dim, struct_dim)
        self.pos_to_struct = nn.Linear(global_pos_dim, struct_dim)
        self.struct_gate = nn.Linear(depth_dim + sibling_dim + global_pos_dim, 3)

        self.semantic_projection = nn.Linear(word_dim, output_dim)
        self.struct_projection = nn.Linear(struct_dim, output_dim)
        self.semantic_struct_gate = nn.Linear(word_dim + struct_dim, 2)
        self.ref_cue_embedding = nn.Embedding(4, struct_dim, padding_idx=0)
        self.ref_cue_projection = nn.Linear(struct_dim, output_dim, bias=False)
        self.ref_cue_gate = nn.Linear(word_dim + struct_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(output_dim)
        self._last_aux_losses: dict[str, Tensor] = {}

    def auxiliary_losses(self) -> dict[str, Tensor]:
        return dict(self._last_aux_losses)

    def forward(
        self,
        token_ids: Tensor,
        depth_ids: Tensor,
        sibling_ids: Tensor,
        global_pos_ids: Tensor,
        local_pos_ids: Tensor,
        *,
        delta_seg: Tensor | None = None,
        ref_cue_ids: Tensor | None = None,
    ) -> Tensor:
        self._last_aux_losses = {}
        token_ids = token_ids.long()
        depth_ids = depth_ids.long().clamp(min=0)
        sibling_ids = sibling_ids.long().clamp(min=0)
        global_pos_ids = global_pos_ids.long().clamp(min=0)
        local_pos_ids = local_pos_ids.long().clamp(min=0)
        if ref_cue_ids is None:
            ref_cue_ids = torch.zeros_like(token_ids)
        ref_cue_ids = ref_cue_ids.long().clamp(min=0, max=3)

        semantic = self.word_embedding(token_ids)

        global_pos = self.global_pos_embedding(global_pos_ids)
        local_pos = self.local_pos_embedding(local_pos_ids)
        if local_pos.shape[-1] != global_pos.shape[-1]:
            raise ValueError("local_pos_dim must match global_pos_dim in the current implementation")
        position = global_pos + self.alpha * local_pos

        if delta_seg is not None:
            delta_seg = delta_seg.float().unsqueeze(-1)
            if self.zero_delta_identity:
                delta_effect = F.linear(delta_seg, self.delta_seg_projection.weight, bias=None)
            else:
                delta_effect = self.delta_seg_projection(delta_seg)
            position = position + self.beta * delta_effect

        depth_bits = binary_depth_encoding(depth_ids, self.max_depth_bits)
        depth_embedding = self.depth_projection(depth_bits)
        sibling_embedding = sinusoidal_encoding(sibling_ids, self.sibling_to_struct.in_features)

        struct_gate = torch.sigmoid(
            self.struct_gate(torch.cat([depth_embedding, sibling_embedding, position], dim=-1))
        )
        struct = (
            struct_gate[..., 0:1] * self.depth_to_struct(depth_embedding)
            + struct_gate[..., 1:2] * self.sibling_to_struct(sibling_embedding)
            + struct_gate[..., 2:3] * self.pos_to_struct(position)
        )

        semantic_gate = torch.softmax(
            self.semantic_struct_gate(torch.cat([semantic, struct], dim=-1)),
            dim=-1,
        )
        fused = (
            semantic_gate[..., 0:1] * self.semantic_projection(semantic)
            + semantic_gate[..., 1:2] * self.struct_projection(struct)
        )
        ref_cue = self.ref_cue_embedding(ref_cue_ids)
        ref_gate = torch.sigmoid(self.ref_cue_gate(torch.cat([semantic, struct, ref_cue], dim=-1)))
        fused = fused + (self.ref_cue_scale * ref_gate) * self.ref_cue_projection(ref_cue)
        if self.training and self.ref_cue_gate_reg_weight > 0.0:
            active_mask = (ref_cue_ids > 0).unsqueeze(-1)
            if active_mask.any():
                active_gate = ref_gate.masked_select(active_mask)
                self._last_aux_losses["ref_cue_gate_reg_loss"] = (
                    active_gate.square().mean() * self.ref_cue_gate_reg_weight
                )
        return self.layer_norm(fused)
