from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .hpe import binary_depth_encoding, sinusoidal_encoding


HPE_MODE_CURRENT_V1 = "current_hpe_approx_v1"
HPE_MODE_H1_V1 = "formula_aligned_hpe_v1"
HPE_MODES = frozenset({HPE_MODE_CURRENT_V1, HPE_MODE_H1_V1})
H1_REF_IDENTITY_BUCKETS = 65536


class FormulaAlignedHierarchicalPositionalEncoder(nn.Module):
    """Task 28 H1 implementation of the frozen formula/shape contract.

    Contract (all dimensions are fixed by ``formula_aligned_hpe_v1``):

    * semantic: word embedding 128 -> linear projection 64;
    * structural: depth/ordering/position mixture in 64 dimensions;
    * REF: identity embedding 32 concatenated with a 128 -> 32
      definition-end-node projection, added once to the structural view;
    * fusion: a 64-dimensional sigmoid gate over [semantic64; struct64];
    * padding: padded output positions are explicitly zeroed.

    The delta term implements formula (4-13) once.  Its projection has no
    bias, so an all-zero delta tensor is exactly equivalent to omitting it.
    No second formula-(4-16) delta injection is performed.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        word_dim: int = 128,
        struct_dim: int = 64,
        output_dim: int = 64,
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
        ref_identity_buckets: int = H1_REF_IDENTITY_BUCKETS,
        ref_identity_dim: int = 32,
        ref_definition_dim: int = 32,
    ) -> None:
        super().__init__()
        if word_dim != 128 or struct_dim != 64 or output_dim != 64:
            raise ValueError(
                "formula_aligned_hpe_v1 requires word_dim=128, struct_dim=64, output_dim=64"
            )
        if global_pos_dim != local_pos_dim:
            raise ValueError("formula_aligned_hpe_v1 requires global_pos_dim == local_pos_dim")
        if ref_identity_dim + ref_definition_dim != struct_dim:
            raise ValueError("REF identity and definition projections must concatenate to struct_dim=64")
        if ref_identity_buckets < 2:
            raise ValueError("ref_identity_buckets must reserve index 0 plus at least one identity")

        self.output_dim = output_dim
        self.padding_idx = int(padding_idx)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_depth_bits = int(max_depth_bits)
        self.ref_identity_buckets = int(ref_identity_buckets)

        self.word_embedding = nn.Embedding(vocab_size, word_dim, padding_idx=padding_idx)
        self.global_pos_embedding = nn.Embedding(max_global_pos, global_pos_dim, padding_idx=0)
        self.local_pos_embedding = nn.Embedding(max_local_pos, local_pos_dim, padding_idx=0)
        self.delta_seg_projection = nn.Linear(1, global_pos_dim, bias=False)

        self.depth_projection = nn.Linear(max_depth_bits, depth_dim)
        self.depth_to_struct = nn.Linear(depth_dim, struct_dim)
        self.sibling_to_struct = nn.Linear(sibling_dim, struct_dim)
        self.pos_to_struct = nn.Linear(global_pos_dim, struct_dim)
        self.struct_gate = nn.Linear(depth_dim + sibling_dim + global_pos_dim, 3)

        self.ref_identity_embedding = nn.Embedding(
            ref_identity_buckets,
            ref_identity_dim,
            padding_idx=0,
        )
        self.ref_definition_projection = nn.Linear(word_dim, ref_definition_dim, bias=False)

        self.semantic_projection = nn.Linear(word_dim, output_dim)
        self.vector_gate = nn.Linear(output_dim + struct_dim, output_dim)
        self._last_observability: dict[str, Tensor] = {}

    def auxiliary_losses(self) -> dict[str, Tensor]:
        return {}

    def observability(self) -> dict[str, Tensor]:
        """Return detached tensors from the latest forward pass."""

        return dict(self._last_observability)

    @staticmethod
    def _require_shape(name: str, value: Tensor, shape: torch.Size) -> None:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}")

    def forward(
        self,
        token_ids: Tensor,
        depth_ids: Tensor,
        sibling_ids: Tensor,
        global_pos_ids: Tensor,
        local_pos_ids: Tensor,
        *,
        delta_seg: Tensor | None = None,
        ref_identity_ids: Tensor | None = None,
        ref_definition_token_ids: Tensor | None = None,
        padding_mask: Tensor | None = None,
        ref_cue_ids: Tensor | None = None,
    ) -> Tensor:
        del ref_cue_ids  # H1 REF is formula (4-9), never the historical role cue.
        shape = token_ids.shape
        if token_ids.ndim != 2:
            raise ValueError("formula_aligned_hpe_v1 expects [batch, token] inputs")
        for name, value in (
            ("depth_ids", depth_ids),
            ("sibling_ids", sibling_ids),
            ("global_pos_ids", global_pos_ids),
            ("local_pos_ids", local_pos_ids),
        ):
            self._require_shape(name, value, shape)

        token_ids = token_ids.long()
        depth_ids = depth_ids.long().clamp(min=0)
        sibling_ids = sibling_ids.long().clamp(min=0)
        global_pos_ids = global_pos_ids.long().clamp(min=0)
        local_pos_ids = local_pos_ids.long().clamp(min=0)
        if padding_mask is None:
            padding_mask = token_ids.eq(self.padding_idx)
        self._require_shape("padding_mask", padding_mask, shape)
        padding_mask = padding_mask.bool()

        if delta_seg is None:
            delta_seg = torch.zeros(shape, device=token_ids.device, dtype=torch.float32)
        self._require_shape("delta_seg", delta_seg, shape)
        delta_effect = F.linear(
            delta_seg.float().unsqueeze(-1),
            self.delta_seg_projection.weight,
            bias=None,
        )

        semantic = self.word_embedding(token_ids)
        semantic_projected = self.semantic_projection(semantic)

        global_pos = self.global_pos_embedding(global_pos_ids)
        local_pos = self.local_pos_embedding(local_pos_ids)
        position = global_pos + self.alpha * local_pos + self.beta * delta_effect

        depth_bits = binary_depth_encoding(depth_ids, self.max_depth_bits)
        depth_embedding = self.depth_projection(depth_bits)
        sibling_embedding = sinusoidal_encoding(sibling_ids, self.sibling_to_struct.in_features)
        structure_gate = torch.sigmoid(
            self.struct_gate(torch.cat([depth_embedding, sibling_embedding, position], dim=-1))
        )
        structural_base = (
            structure_gate[..., 0:1] * self.depth_to_struct(depth_embedding)
            + structure_gate[..., 1:2] * self.sibling_to_struct(sibling_embedding)
            + structure_gate[..., 2:3] * self.pos_to_struct(position)
        )

        if ref_identity_ids is None:
            ref_identity_ids = torch.zeros(shape, device=token_ids.device, dtype=torch.long)
        if ref_definition_token_ids is None:
            ref_definition_token_ids = torch.zeros(shape, device=token_ids.device, dtype=torch.long)
        self._require_shape("ref_identity_ids", ref_identity_ids, shape)
        self._require_shape("ref_definition_token_ids", ref_definition_token_ids, shape)
        ref_identity_ids = ref_identity_ids.long()
        ref_definition_token_ids = ref_definition_token_ids.long()
        if torch.any(ref_identity_ids < 0) or torch.any(ref_identity_ids >= self.ref_identity_buckets):
            raise ValueError("ref_identity_ids exceed the configured H1 identity table")
        if torch.any(ref_definition_token_ids < 0) or torch.any(
            ref_definition_token_ids >= self.word_embedding.num_embeddings
        ):
            raise ValueError("ref_definition_token_ids exceed the shared word embedding table")

        ref_identity = self.ref_identity_embedding(ref_identity_ids)
        ref_definition = self.ref_definition_projection(
            self.word_embedding(ref_definition_token_ids)
        )
        ref_active = (ref_identity_ids > 0) & (ref_definition_token_ids > 0) & (~padding_mask)
        ref_vector = torch.cat([ref_identity, ref_definition], dim=-1)
        ref_vector = ref_vector * ref_active.unsqueeze(-1).to(ref_vector.dtype)
        structural_with_ref = structural_base + ref_vector

        vector_gate = torch.sigmoid(
            self.vector_gate(torch.cat([semantic_projected, structural_with_ref], dim=-1))
        )
        output = vector_gate * semantic_projected + (1.0 - vector_gate) * structural_with_ref
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        self._last_observability = {
            "semantic_projected": semantic_projected.detach(),
            "structural_base": structural_base.detach(),
            "delta_effect": delta_effect.detach(),
            "ref_active_mask": ref_active.detach(),
            "ref_vector": ref_vector.detach(),
            "structural_with_ref": structural_with_ref.detach(),
            "vector_gate": vector_gate.detach(),
            "output": output.detach(),
        }
        return output
