from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn


RESIDUAL_STANDARD = "standard_residual"
RESIDUAL_PAPER_ALIGNED = "paper_aligned_residual"

SPECIAL_TOKEN_NONE = 0
SPECIAL_TOKEN_SEG_START = 1
SPECIAL_TOKEN_SEG_END = 2
SPECIAL_TOKEN_PAD = 3
SPECIAL_TOKEN_TYPE_COUNT = 4

REF_RELATION_NONE = 0
REF_RELATION_DEF_TO_USE = 1
REF_RELATION_USE_TO_DEF = 2
REF_RELATION_BIDIRECTIONAL = 3
REF_RELATION_COUNT = 4


@dataclass(frozen=True)
class StructureSensitiveTransformerConfig:
    """Frozen Task 31 encoder contract.

    ST0 and ST1 construct exactly the same parameters.  The only execution
    difference is ``structural_bias_enabled``.
    """

    profile_id: str
    structural_bias_enabled: bool
    num_layers: int = 6
    num_heads: int = 8
    d_model: int = 512
    dim_feedforward: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_depth_delta: int = 16
    max_sibling_delta: int = 16
    residual_protocol: str = RESIDUAL_STANDARD
    layer_norm_eps: float = 1e-5
    residual_branch_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.residual_branch_scale) or self.residual_branch_scale <= 0:
            raise ValueError("residual_branch_scale must be finite and > 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError("attention_dropout must be in [0,1)")
        if self.max_depth_delta <= 0 or self.max_sibling_delta <= 0:
            raise ValueError("relation clipping radii must be positive")
        if self.residual_protocol == RESIDUAL_PAPER_ALIGNED:
            raise ValueError(
                "paper_aligned_residual is fail-closed: the chapter states that "
                "boundary residuals are disabled but does not uniquely define the "
                "affected sublayer or operator"
            )
        if self.residual_protocol != RESIDUAL_STANDARD:
            raise ValueError(f"unsupported residual_protocol={self.residual_protocol!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


@dataclass
class StructuralTransformerInput:
    """Explicit, token-aligned input for Task 31.

    Shapes use ``B`` for batch, ``T`` for the flattened multi-segment path
    length and ``D`` for ``d_model``.

    ``token_embeddings``: ``[B,T,D]`` floating point, gradient-bearing.
    ``attention_mask``: optional ``[B,T,T]`` bool, True means the pair is
    allowed; missing means all non-padding pairs are allowed.
    ``padding_mask``: ``[B,T]`` bool, True means PAD.
    ``depth_ids`` / ``sibling_order_ids``: ``[B,T]`` int64, non-negative for
    valid tokens and zero on PAD.
    ``sibling_parent_ids``: ``[B,T]`` int64, a real AST-parent identity or -1
    when no parent exists; -1 on PAD and special boundary tokens.
    ``ast_node_identity_ids``: ``[B,T]`` int64, a stable identity for the
    underlying AST node.  Repeated context-window copies share an identity;
    boundary/PAD tokens use -1.  This prevents a repeated node from being
    misclassified as its own sibling merely because it appears twice.
    ``segment_ids``: ``[B,T]`` int64, non-negative for valid tokens and -1 on
    PAD.
    ``ref_link_matrix``: ``[B,T,T]`` int64 relation codes 0..3, zero on every
    padded row/column.  Row is the attention query/source and column is the
    key/target.
    ``special_token_ids``: ``[B,T]`` int64 using NONE/START/END/PAD = 0/1/2/3.

    Only ``token_embeddings`` may participate in gradient computation.  Every
    structural field is required for ST1; no label, split or dataset identity
    is consulted.
    """

    token_embeddings: Tensor
    attention_mask: Tensor | None
    padding_mask: Tensor
    depth_ids: Tensor
    sibling_order_ids: Tensor
    sibling_parent_ids: Tensor
    ast_node_identity_ids: Tensor
    segment_ids: Tensor
    ref_link_matrix: Tensor
    special_token_ids: Tensor

    def validate(self, *, d_model: int) -> None:
        if self.token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must have shape [B,T,D]")
        batch, tokens, width = self.token_embeddings.shape
        if width != d_model:
            raise ValueError(f"token embedding width {width} != d_model {d_model}")
        if not self.token_embeddings.is_floating_point():
            raise TypeError("token_embeddings must be floating point")

        expected_token_shape = (batch, tokens)
        integer_fields = {
            "depth_ids": self.depth_ids,
            "sibling_order_ids": self.sibling_order_ids,
            "sibling_parent_ids": self.sibling_parent_ids,
            "ast_node_identity_ids": self.ast_node_identity_ids,
            "segment_ids": self.segment_ids,
            "special_token_ids": self.special_token_ids,
        }
        for name, value in integer_fields.items():
            if tuple(value.shape) != expected_token_shape:
                raise ValueError(f"{name} must have shape [B,T]")
            if value.dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.long")
            if value.requires_grad:
                raise ValueError(f"{name} must not require gradients")

        if tuple(self.padding_mask.shape) != expected_token_shape:
            raise ValueError("padding_mask must have shape [B,T]")
        if self.padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must have dtype torch.bool")
        if self.attention_mask is not None:
            if tuple(self.attention_mask.shape) != (batch, tokens, tokens):
                raise ValueError("attention_mask must have shape [B,T,T]")
            if self.attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must have dtype torch.bool")
        if tuple(self.ref_link_matrix.shape) != (batch, tokens, tokens):
            raise ValueError("ref_link_matrix must have shape [B,T,T]")
        if self.ref_link_matrix.dtype != torch.long:
            raise TypeError("ref_link_matrix must have dtype torch.long")
        if self.ref_link_matrix.requires_grad:
            raise ValueError("ref_link_matrix must not require gradients")

        tensors = [
            self.padding_mask,
            self.depth_ids,
            self.sibling_order_ids,
            self.sibling_parent_ids,
            self.ast_node_identity_ids,
            self.segment_ids,
            self.ref_link_matrix,
            self.special_token_ids,
        ]
        if self.attention_mask is not None:
            tensors.append(self.attention_mask)
        if any(value.device != self.token_embeddings.device for value in tensors):
            raise ValueError("all structural inputs must be on the token embedding device")

        valid = ~self.padding_mask
        if (self.depth_ids[valid] < 0).any():
            raise ValueError("valid depth_ids must be non-negative")
        if (self.sibling_order_ids[valid] < 0).any():
            raise ValueError("valid sibling_order_ids must be non-negative")
        if (self.sibling_parent_ids < -1).any():
            raise ValueError("sibling_parent_ids must be -1 or a non-negative identity")
        if (self.ast_node_identity_ids < -1).any():
            raise ValueError("ast_node_identity_ids must be -1 or a non-negative identity")
        if (self.segment_ids[valid] < 0).any():
            raise ValueError("valid segment_ids must be non-negative")
        if (self.special_token_ids < SPECIAL_TOKEN_NONE).any() or (
            self.special_token_ids >= SPECIAL_TOKEN_TYPE_COUNT
        ).any():
            raise ValueError("special_token_ids contain an unknown type")
        if (self.ref_link_matrix < REF_RELATION_NONE).any() or (
            self.ref_link_matrix >= REF_RELATION_COUNT
        ).any():
            raise ValueError("ref_link_matrix contains an unknown relation code")

        content_tokens = valid & self.special_token_ids.eq(SPECIAL_TOKEN_NONE)
        non_content_tokens = ~content_tokens
        if (self.ast_node_identity_ids[content_tokens] < 0).any():
            raise ValueError("every ordinary valid token requires a stable AST node identity")
        if (self.ast_node_identity_ids[non_content_tokens] != -1).any():
            raise ValueError("boundary and PAD tokens must use AST node identity -1")

        if self.padding_mask.any():
            padding = self.padding_mask
            if (self.depth_ids[padding] != 0).any():
                raise ValueError("PAD depth must be zero")
            if (self.sibling_order_ids[padding] != 0).any():
                raise ValueError("PAD sibling order must be zero")
            if (self.sibling_parent_ids[padding] != -1).any():
                raise ValueError("PAD parent identity must be -1")
            if (self.ast_node_identity_ids[padding] != -1).any():
                raise ValueError("PAD AST node identity must be -1")
            if (self.segment_ids[padding] != -1).any():
                raise ValueError("PAD segment identity must be -1")
            if (self.special_token_ids[padding] != SPECIAL_TOKEN_PAD).any():
                raise ValueError("PAD tokens must use SPECIAL_TOKEN_PAD")
            valid_pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
            if (self.ref_link_matrix.masked_select(~valid_pair) != REF_RELATION_NONE).any():
                raise ValueError("REF relations must be zero on padded rows and columns")
        if (self.special_token_ids[valid] == SPECIAL_TOKEN_PAD).any():
            raise ValueError("valid tokens must not use SPECIAL_TOKEN_PAD")

    def valid_pair_mask(self) -> Tensor:
        valid_tokens = ~self.padding_mask
        pair_mask = valid_tokens.unsqueeze(-1) & valid_tokens.unsqueeze(-2)
        if self.attention_mask is not None:
            pair_mask = pair_mask & self.attention_mask
        return pair_mask


@dataclass
class StructuralTransformerOutput:
    last_hidden_state: Tensor
    attention_audit: dict[str, dict[str, Tensor]]


@dataclass
class Task31ClassifierOutput:
    logits: Tensor
    last_hidden_state: Tensor
    attention_audit: dict[str, dict[str, Tensor]]


def _masked_softmax(scores: Tensor, valid_mask: Tensor, *, dim: int = -1) -> Tensor:
    if valid_mask.shape != scores.shape:
        valid_mask = valid_mask.expand_as(scores)
    # Complete masking and normalization in FP32.  In FP16, 1e-8 rounds to
    # zero; casting before the denominator guard would therefore turn an
    # all-masked row into 0/0.  The explicit where keeps that row exactly zero.
    float_scores = scores.to(dtype=torch.float32)
    masked = float_scores.masked_fill(~valid_mask, torch.finfo(torch.float32).min)
    probabilities = torch.softmax(masked, dim=dim)
    probabilities = probabilities * valid_mask.to(dtype=torch.float32)
    denominator = probabilities.sum(dim=dim, keepdim=True)
    normalized = torch.where(
        denominator > 0,
        probabilities / denominator.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(probabilities),
    )
    return normalized.to(dtype=scores.dtype)


class StructuralRelationBias(nn.Module):
    """Per-head learnable relation tables for one Transformer layer."""

    def __init__(
        self,
        *,
        num_heads: int,
        max_depth_delta: int,
        max_sibling_delta: int,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_depth_delta = int(max_depth_delta)
        self.max_sibling_delta = int(max_sibling_delta)
        self.depth_bias_embedding = nn.Embedding(
            2 * self.max_depth_delta + 1,
            self.num_heads,
        )
        # The final entry is the dedicated non-sibling bucket.
        self.sibling_non_relation_bucket = 2 * self.max_sibling_delta + 1
        self.sibling_bias_embedding = nn.Embedding(
            self.sibling_non_relation_bucket + 1,
            self.num_heads,
        )
        self.same_segment_bias_embedding = nn.Embedding(2, self.num_heads)
        self.ref_bias_embedding = nn.Embedding(REF_RELATION_COUNT, self.num_heads)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.depth_bias_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.sibling_bias_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            # Formula (4-37) supplies useful non-degenerate initial values.  All
            # entries remain trainable and independent per head/layer.
            self.same_segment_bias_embedding.weight.zero_()
            self.same_segment_bias_embedding.weight[1].fill_(2.0)
            self.ref_bias_embedding.weight.zero_()
            self.ref_bias_embedding.weight[REF_RELATION_DEF_TO_USE].fill_(1.0)
            self.ref_bias_embedding.weight[REF_RELATION_USE_TO_DEF].fill_(0.75)
            self.ref_bias_embedding.weight[REF_RELATION_BIDIRECTIONAL].fill_(1.25)

    @staticmethod
    def _per_head(embedded: Tensor) -> Tensor:
        # Embedding result [B,T,T,H] -> attention layout [B,H,T,T].
        return embedded.permute(0, 3, 1, 2).contiguous()

    def depth_bucket_ids(self, depth_ids: Tensor) -> Tensor:
        signed_difference = depth_ids.unsqueeze(-1) - depth_ids.unsqueeze(-2)
        clipped = signed_difference.clamp(-self.max_depth_delta, self.max_depth_delta)
        return clipped + self.max_depth_delta

    def sibling_bucket_ids(
        self,
        sibling_order_ids: Tensor,
        sibling_parent_ids: Tensor,
        ast_node_identity_ids: Tensor,
    ) -> Tensor:
        batch, tokens = sibling_order_ids.shape
        signed_difference = sibling_order_ids.unsqueeze(-1) - sibling_order_ids.unsqueeze(-2)
        clipped = signed_difference.clamp(-self.max_sibling_delta, self.max_sibling_delta)
        related_bucket = clipped + self.max_sibling_delta
        identity = torch.eye(tokens, dtype=torch.bool, device=sibling_order_ids.device).unsqueeze(0)
        same_ast_node = (
            ast_node_identity_ids.unsqueeze(-1).eq(ast_node_identity_ids.unsqueeze(-2))
            & ast_node_identity_ids.unsqueeze(-1).ge(0)
        )
        same_real_parent = (
            sibling_parent_ids.unsqueeze(-1).eq(sibling_parent_ids.unsqueeze(-2))
            & sibling_parent_ids.unsqueeze(-1).ge(0)
            & ~same_ast_node
        )
        true_sibling_or_self = identity.expand(batch, -1, -1) | same_ast_node | same_real_parent
        # Context windows may contain the same AST node more than once.  That
        # relation is "same node", never left/right sibling, even if malformed
        # input copies disagree about their sibling-order field.
        related_bucket = torch.where(
            same_ast_node,
            torch.full_like(related_bucket, self.max_sibling_delta),
            related_bucket,
        )
        non_relation = torch.full_like(related_bucket, self.sibling_non_relation_bucket)
        return torch.where(true_sibling_or_self, related_bucket, non_relation)

    def forward(self, structural_input: StructuralTransformerInput) -> dict[str, Tensor]:
        valid_pair = structural_input.valid_pair_mask()
        valid_pair_heads = valid_pair.unsqueeze(1)

        depth_bucket = self.depth_bucket_ids(structural_input.depth_ids)
        depth_bias = self._per_head(self.depth_bias_embedding(depth_bucket))

        sibling_bucket = self.sibling_bucket_ids(
            structural_input.sibling_order_ids,
            structural_input.sibling_parent_ids,
            structural_input.ast_node_identity_ids,
        )
        sibling_bias = self._per_head(self.sibling_bias_embedding(sibling_bucket))

        same_segment = structural_input.segment_ids.unsqueeze(-1).eq(
            structural_input.segment_ids.unsqueeze(-2)
        )
        same_segment_bias = self._per_head(
            self.same_segment_bias_embedding(same_segment.long())
        )
        ref_bias = self._per_head(
            self.ref_bias_embedding(structural_input.ref_link_matrix)
        )

        result = {
            "depth_bias": depth_bias,
            "sibling_bias": sibling_bias,
            "same_segment_bias": same_segment_bias,
            "ref_bias": ref_bias,
        }
        return {
            name: value.masked_fill(~valid_pair_heads, 0.0)
            for name, value in result.items()
        }

    def parameter_count(self) -> dict[str, int]:
        return {
            "depth_bias": self.depth_bias_embedding.weight.numel(),
            "sibling_bias": self.sibling_bias_embedding.weight.numel(),
            "same_segment_bias": self.same_segment_bias_embedding.weight.numel(),
            "ref_bias": self.ref_bias_embedding.weight.numel(),
        }


def _normalize_head_indices(indices: Iterable[int] | None, *, num_heads: int) -> list[int]:
    if indices is None:
        return [0]
    normalized = sorted({int(index) for index in indices})
    if not normalized:
        raise ValueError("audit_head_indices must not be empty")
    if normalized[0] < 0 or normalized[-1] >= num_heads:
        raise ValueError("audit head index out of range")
    return normalized


class StructureSensitiveSelfAttention(nn.Module):
    def __init__(self, config: StructureSensitiveTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv_projection = nn.Linear(config.d_model, config.d_model * 3)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.relation_bias = StructuralRelationBias(
            num_heads=config.num_heads,
            max_depth_delta=config.max_depth_delta,
            max_sibling_delta=config.max_sibling_delta,
        )

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, tokens, _ = value.shape
        return value.view(
            batch,
            tokens,
            self.config.num_heads,
            self.config.head_dim,
        ).transpose(1, 2)

    @staticmethod
    def _audit_slice(
        value: Tensor,
        *,
        max_samples: int,
        head_indices: list[int],
    ) -> Tensor:
        return value[:max_samples, head_indices].detach().to(device="cpu")

    def forward(
        self,
        hidden_states: Tensor,
        *,
        structural_input: StructuralTransformerInput,
        return_attention_audit: bool,
        audit_max_samples: int,
        audit_head_indices: Iterable[int] | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, tokens, width = hidden_states.shape
        if width != self.config.d_model:
            raise ValueError("hidden state width does not match d_model")
        qkv = self.qkv_projection(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query_heads = self._split_heads(query)
        key_heads = self._split_heads(key)
        value_heads = self._split_heads(value)
        raw_qk_logits = torch.matmul(query_heads, key_heads.transpose(-2, -1))
        raw_qk_logits = raw_qk_logits / math.sqrt(self.config.head_dim)

        if self.config.structural_bias_enabled:
            bias_parts = self.relation_bias(structural_input)
        else:
            zeros = torch.zeros_like(raw_qk_logits)
            bias_parts = {
                "depth_bias": zeros,
                "sibling_bias": zeros,
                "same_segment_bias": zeros,
                "ref_bias": zeros,
            }
        combined_bias = sum(bias_parts.values())
        biased_logits = raw_qk_logits + combined_bias
        valid_pair = structural_input.valid_pair_mask().unsqueeze(1)
        masked_logits = biased_logits.masked_fill(
            ~valid_pair,
            torch.finfo(biased_logits.dtype).min,
        )
        attention_probabilities = _masked_softmax(biased_logits, valid_pair, dim=-1)
        dropped_probabilities = self.attention_dropout(attention_probabilities)
        context = torch.matmul(dropped_probabilities, value_heads)
        context = context.transpose(1, 2).contiguous().view(batch, tokens, width)
        output = self.output_projection(context)
        output = output.masked_fill(structural_input.padding_mask.unsqueeze(-1), 0.0)

        audit: dict[str, Tensor] = {}
        if return_attention_audit:
            if audit_max_samples <= 0:
                raise ValueError("audit_max_samples must be positive")
            heads = _normalize_head_indices(
                audit_head_indices,
                num_heads=self.config.num_heads,
            )
            values = {
                "raw_qk_logits": raw_qk_logits,
                **bias_parts,
                "combined_bias": combined_bias,
                "masked_logits": masked_logits,
                "attention_probabilities": attention_probabilities,
            }
            audit = {
                name: self._audit_slice(
                    tensor,
                    max_samples=min(audit_max_samples, batch),
                    head_indices=heads,
                )
                for name, tensor in values.items()
            }
        return output, audit


class StructureSensitiveTransformerLayer(nn.Module):
    def __init__(self, config: StructureSensitiveTransformerConfig) -> None:
        super().__init__()
        self.self_attention = StructureSensitiveSelfAttention(config)
        self.attention_output_dropout = nn.Dropout(config.dropout)
        self.attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
        )
        self.feed_forward_output_dropout = nn.Dropout(config.dropout)
        self.feed_forward_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.residual_protocol = config.residual_protocol
        self.residual_branch_scale = float(config.residual_branch_scale)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        structural_input: StructuralTransformerInput,
        return_attention_audit: bool,
        audit_max_samples: int,
        audit_head_indices: Iterable[int] | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.residual_protocol != RESIDUAL_STANDARD:
            raise RuntimeError("only standard_residual is executable in Task 31 v1")
        attention_output, audit = self.self_attention(
            hidden_states,
            structural_input=structural_input,
            return_attention_audit=return_attention_audit,
            audit_max_samples=audit_max_samples,
            audit_head_indices=audit_head_indices,
        )
        hidden_states = self.attention_norm(
            hidden_states + self.residual_branch_scale * self.attention_output_dropout(attention_output)
        )
        hidden_states = hidden_states.masked_fill(
            structural_input.padding_mask.unsqueeze(-1),
            0.0,
        )
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_norm(
            hidden_states + self.residual_branch_scale * self.feed_forward_output_dropout(feed_forward_output)
        )
        hidden_states = hidden_states.masked_fill(
            structural_input.padding_mask.unsqueeze(-1),
            0.0,
        )
        return hidden_states, audit


class StructureSensitiveTransformerEncoder(nn.Module):
    def __init__(self, config: StructureSensitiveTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [StructureSensitiveTransformerLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        structural_input: StructuralTransformerInput,
        *,
        return_attention_audit: bool = False,
        audit_layer_indices: Iterable[int] | None = None,
        audit_head_indices: Iterable[int] | None = None,
        audit_max_samples: int = 1,
    ) -> StructuralTransformerOutput:
        structural_input.validate(d_model=self.config.d_model)
        if audit_layer_indices is None:
            audited_layers = {0}
        else:
            audited_layers = {int(index) for index in audit_layer_indices}
            if not audited_layers:
                raise ValueError("audit_layer_indices must not be empty")
            if min(audited_layers) < 0 or max(audited_layers) >= len(self.layers):
                raise ValueError("audit layer index out of range")

        hidden_states = structural_input.token_embeddings.masked_fill(
            structural_input.padding_mask.unsqueeze(-1),
            0.0,
        )
        audit: dict[str, dict[str, Tensor]] = {}
        for layer_index, layer in enumerate(self.layers):
            capture = return_attention_audit and layer_index in audited_layers
            hidden_states, layer_audit = layer(
                hidden_states,
                structural_input=structural_input,
                return_attention_audit=capture,
                audit_max_samples=audit_max_samples,
                audit_head_indices=audit_head_indices,
            )
            if capture:
                audit[f"layer_{layer_index}"] = layer_audit
        return StructuralTransformerOutput(
            last_hidden_state=hidden_states,
            attention_audit=audit,
        )

    def parameter_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        first_layer = self.layers[0]
        attention = sum(
            parameter.numel()
            for name, parameter in first_layer.named_parameters()
            if name.startswith("self_attention.qkv_projection")
            or name.startswith("self_attention.output_projection")
        )
        feed_forward = sum(
            parameter.numel()
            for name, parameter in first_layer.named_parameters()
            if name.startswith("feed_forward")
        )
        norms = sum(
            parameter.numel()
            for name, parameter in first_layer.named_parameters()
            if "_norm." in name
        )
        bias_per_layer = first_layer.self_attention.relation_bias.parameter_count()
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "d_model": self.config.d_model,
            "dim_feedforward": self.config.dim_feedforward,
            "head_dim": self.config.head_dim,
            "per_layer": {
                "attention_projection_parameters": attention,
                "feed_forward_parameters": feed_forward,
                "layer_norm_parameters": norms,
                "structural_bias_parameters": sum(bias_per_layer.values()),
                "structural_bias_breakdown": bias_per_layer,
            },
            "all_layers_structural_bias_parameters": sum(bias_per_layer.values())
            * self.config.num_layers,
        }


class Task31StructureSensitiveClassifier(nn.Module):
    """Minimal executable ST0/ST1 wrapper used by acceptance and smoke.

    The token-type embedding is deliberately separate from the vocabulary
    embedding.  START and END therefore have both independent vocabulary rows
    and independent type rows.  PAD is zeroed by both embeddings and the mask.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        padding_idx: int,
        config: StructureSensitiveTransformerConfig,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = int(padding_idx)
        self.token_embedding = nn.Embedding(
            vocab_size,
            config.d_model,
            padding_idx=self.padding_idx,
        )
        self.special_token_embedding = nn.Embedding(
            SPECIAL_TOKEN_TYPE_COUNT,
            config.d_model,
            padding_idx=SPECIAL_TOKEN_PAD,
        )
        self.encoder = StructureSensitiveTransformerEncoder(config)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def forward(
        self,
        *,
        token_ids: Tensor,
        attention_mask: Tensor | None,
        padding_mask: Tensor,
        depth_ids: Tensor,
        sibling_order_ids: Tensor,
        sibling_parent_ids: Tensor,
        ast_node_identity_ids: Tensor,
        segment_ids: Tensor,
        ref_link_matrix: Tensor,
        special_token_ids: Tensor,
        return_attention_audit: bool = False,
        audit_layer_indices: Iterable[int] | None = None,
        audit_head_indices: Iterable[int] | None = None,
        audit_max_samples: int = 1,
    ) -> Task31ClassifierOutput:
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must have dtype torch.long")
        if token_ids.shape != padding_mask.shape:
            raise ValueError("token_ids and padding_mask must have shape [B,T]")
        token_embeddings = self.token_embedding(token_ids)
        token_embeddings = token_embeddings + self.special_token_embedding(special_token_ids)
        structural_input = StructuralTransformerInput(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            depth_ids=depth_ids,
            sibling_order_ids=sibling_order_ids,
            sibling_parent_ids=sibling_parent_ids,
            ast_node_identity_ids=ast_node_identity_ids,
            segment_ids=segment_ids,
            ref_link_matrix=ref_link_matrix,
            special_token_ids=special_token_ids,
        )
        encoded = self.encoder(
            structural_input,
            return_attention_audit=return_attention_audit,
            audit_layer_indices=audit_layer_indices,
            audit_head_indices=audit_head_indices,
            audit_max_samples=audit_max_samples,
        )
        masked = encoded.last_hidden_state.masked_fill(
            padding_mask.unsqueeze(-1),
            float("-inf"),
        )
        pooled = masked.max(dim=1).values
        pooled = pooled.masked_fill(~pooled.isfinite(), 0.0)
        logits = self.classifier(pooled)
        return Task31ClassifierOutput(
            logits=logits,
            last_hidden_state=encoded.last_hidden_state,
            attention_audit=encoded.attention_audit,
        )

    def parameter_report(self) -> dict[str, Any]:
        encoder = self.encoder.parameter_report()
        return {
            "model_total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "token_embedding_parameters": self.token_embedding.weight.numel(),
            "special_token_embedding_parameters": self.special_token_embedding.weight.numel(),
            "classifier_parameters": sum(parameter.numel() for parameter in self.classifier.parameters()),
            "transformer": encoder,
        }


class Task31PostDWAFTransformerHead(nn.Module):
    """Versioned production seam: accepted D1 fused tokens -> Task31 encoder.

    Task 29 freezes D1 at width 128, while the paper-aligned Transformer is
    width 512.  This head makes the 128->512 bridge explicit and learnable;
    it does not recreate or alter HPE, GRU, CNN or D1.  Callers must flatten
    the per-segment D1 outputs into the same multi-segment token order used by
    the structural fields.  The lightweight classifier above remains only a
    zero-update smoke wrapper for raw train examples.
    """

    architecture_id = "task31-post-dwaf-structure-transformer-v1"

    def __init__(
        self,
        *,
        upstream_dim: int,
        config: StructureSensitiveTransformerConfig,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if upstream_dim <= 0:
            raise ValueError("upstream_dim must be positive")
        self.upstream_dim = int(upstream_dim)
        self.config = config
        self.input_projection = nn.Linear(self.upstream_dim, config.d_model)
        self.special_token_embedding = nn.Embedding(
            SPECIAL_TOKEN_TYPE_COUNT,
            config.d_model,
            padding_idx=SPECIAL_TOKEN_PAD,
        )
        self.encoder = StructureSensitiveTransformerEncoder(config)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def forward(
        self,
        *,
        dwaf_fused_token_embeddings: Tensor,
        attention_mask: Tensor | None,
        padding_mask: Tensor,
        depth_ids: Tensor,
        sibling_order_ids: Tensor,
        sibling_parent_ids: Tensor,
        ast_node_identity_ids: Tensor,
        segment_ids: Tensor,
        ref_link_matrix: Tensor,
        special_token_ids: Tensor,
        return_attention_audit: bool = False,
        audit_layer_indices: Iterable[int] | None = None,
        audit_head_indices: Iterable[int] | None = None,
        audit_max_samples: int = 1,
    ) -> Task31ClassifierOutput:
        if dwaf_fused_token_embeddings.ndim != 3:
            raise ValueError("dwaf_fused_token_embeddings must have shape [B,T,H_D1]")
        if dwaf_fused_token_embeddings.shape[-1] != self.upstream_dim:
            raise ValueError("D1 fused-token width does not match upstream_dim")
        if dwaf_fused_token_embeddings.shape[:2] != padding_mask.shape:
            raise ValueError("D1 fused-token batch/time dimensions must match padding_mask")
        token_embeddings = self.input_projection(dwaf_fused_token_embeddings)
        token_embeddings = token_embeddings + self.special_token_embedding(special_token_ids)
        structural_input = StructuralTransformerInput(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            depth_ids=depth_ids,
            sibling_order_ids=sibling_order_ids,
            sibling_parent_ids=sibling_parent_ids,
            ast_node_identity_ids=ast_node_identity_ids,
            segment_ids=segment_ids,
            ref_link_matrix=ref_link_matrix,
            special_token_ids=special_token_ids,
        )
        encoded = self.encoder(
            structural_input,
            return_attention_audit=return_attention_audit,
            audit_layer_indices=audit_layer_indices,
            audit_head_indices=audit_head_indices,
            audit_max_samples=audit_max_samples,
        )
        masked = encoded.last_hidden_state.masked_fill(
            padding_mask.unsqueeze(-1),
            float("-inf"),
        )
        pooled = masked.max(dim=1).values
        pooled = pooled.masked_fill(~pooled.isfinite(), 0.0)
        return Task31ClassifierOutput(
            logits=self.classifier(pooled),
            last_hidden_state=encoded.last_hidden_state,
            attention_audit=encoded.attention_audit,
        )

    def load_versioned_checkpoint(self, checkpoint: dict[str, Any]) -> Any:
        """Strict-only loader; legacy or partial checkpoints fail closed."""

        architecture_id = checkpoint.get("architecture_id")
        if architecture_id != self.architecture_id:
            raise ValueError(
                f"checkpoint architecture_id={architecture_id!r} is incompatible with "
                f"{self.architecture_id!r}"
            )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Task31 checkpoint requires a state_dict mapping")
        return self.load_state_dict(state_dict, strict=True)

    def parameter_report(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "model_total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "upstream_dim": self.upstream_dim,
            "input_projection_parameters": sum(
                parameter.numel() for parameter in self.input_projection.parameters()
            ),
            "special_token_embedding_parameters": self.special_token_embedding.weight.numel(),
            "classifier_parameters": sum(
                parameter.numel() for parameter in self.classifier.parameters()
            ),
            "transformer": self.encoder.parameter_report(),
        }
