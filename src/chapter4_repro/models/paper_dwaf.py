from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .full_model import Chapter4BaseModel
from .hierarchical_path_fusion import (
    PATH_FUSION_MODE_FLAT_V1,
    PATH_FUSION_MODE_PF1_V1,
    PATH_FUSION_MODES,
    HierarchicalPathFusionAggregator,
)
from .path_fusion import SegmentAttentionAggregator, apply_ref_cue_segment_filter


DWAF_MODE_D1_V1 = "token_three_source_mha8_dwaf_v1"
DWAF_MODE_D0_ROLLBACK_V1 = "post_transformer_segment_gate_v1"


def _validate_right_padding(sequence: Tensor, padding_mask: Tensor) -> Tensor:
    if padding_mask.shape != sequence.shape[:2]:
        raise ValueError("padding_mask must match token batch/time dimensions")
    lengths = (~padding_mask).sum(dim=1)
    positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
    expected = positions >= lengths.unsqueeze(1)
    if not torch.equal(padding_mask, expected):
        raise ValueError("D1 requires a right-padded token mask")
    return lengths


def _masked_softmax(scores: Tensor, valid_mask: Tensor, *, dim: int) -> Tensor:
    """Finite masked softmax with an exact-zero all-masked result."""
    if scores.shape != valid_mask.shape:
        valid_mask = valid_mask.expand_as(scores)
    masked = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim, dtype=torch.float32).to(dtype=scores.dtype)
    weights = weights * valid_mask.to(dtype=weights.dtype)
    denominator = weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return weights / denominator


class PaperGRUTokenBranch(nn.Module):
    """Formula (4-20)--(4-25), retaining an attention-weighted token sequence.

    The paper writes ``v_gru`` as a pooled vector in (4-25), then requires an
    ``n x d_f`` input in (4-31).  D1 v1 freezes the sequence interpretation as
    ``alpha_t * h_t``; its sum is the formula (4-25) pooled diagnostic.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_scorer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        sequence: Tensor,
        *,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        lengths = _validate_right_padding(sequence, padding_mask)
        masked_sequence = sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        packed = pack_padded_sequence(
            masked_sequence,
            lengths.clamp_min(1).to(device="cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.gru(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        outputs = self.dropout(outputs).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        scores = self.attention_scorer(outputs).squeeze(-1)
        token_attention = _masked_softmax(scores, ~padding_mask, dim=-1)
        weighted_sequence = outputs * token_attention.unsqueeze(-1)
        pooled = weighted_sequence.sum(dim=1)
        return weighted_sequence, token_attention, pooled


class PaperCNNChannelBranch(nn.Module):
    """Formula (4-26)--(4-29) with a frozen N,T,H <-> N,H,T order.

    Contract order: zero masked tokens -> Conv1d(N,H,T) -> ReLU -> zero mask
    -> masked temporal GAP -> sigmoid channel gate -> channel-wise multiply.
    Formula (4-29)'s channel reduction is interpreted as channel-wise gating,
    because (4-32) requires a token sequence with feature width ``d_f``.
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("D1 CNN kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.activation = nn.ReLU()
        self.channel_gate = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor, *, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        _validate_right_padding(sequence, padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        masked_sequence = sequence.masked_fill(~valid, 0.0)
        conv_tokens = self.conv(masked_sequence.transpose(1, 2)).transpose(1, 2)
        conv_tokens = self.activation(conv_tokens).masked_fill(~valid, 0.0)
        valid_count = valid.sum(dim=1).clamp_min(1).to(dtype=conv_tokens.dtype)
        channel_summary = conv_tokens.sum(dim=1) / valid_count
        channel_attention = torch.sigmoid(self.channel_gate(channel_summary))
        gated_tokens = self.dropout(conv_tokens * channel_attention.unsqueeze(1))
        gated_tokens = gated_tokens.masked_fill(~valid, 0.0)
        return gated_tokens, channel_attention


class PaperLikeDWAF(nn.Module):
    """D1 three-source Q/K/V, eight-head token attention and 256->128 closure."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads != 8:
            raise ValueError("D1 v1 freezes DWAF num_heads=8")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by eight DWAF heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.q_projection = nn.Linear(hidden_dim, hidden_dim)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        # Formula (4-35) produces 2*d_f although the downstream Transformer is
        # d_f wide. D1 v1 freezes the minimal explicit dimension closure.
        self.concat_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.capture_observability = False
        self._last_observability: dict[str, Tensor] = {}

    def set_observability(self, enabled: bool) -> None:
        self.capture_observability = bool(enabled)
        if not self.capture_observability:
            self._last_observability = {}

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        original_tokens: Tensor,
        gru_tokens: Tensor,
        cnn_tokens: Tensor,
        *,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not (original_tokens.shape == gru_tokens.shape == cnn_tokens.shape):
            raise ValueError("D1 Q/K/V sources must share [N,T,H]")
        if original_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("D1 source width does not match hidden_dim")
        _validate_right_padding(original_tokens, padding_mask)
        valid_tokens = ~padding_mask
        valid_feature = valid_tokens.unsqueeze(-1)

        q = self.q_projection(original_tokens).masked_fill(~valid_feature, 0.0)
        k = self.k_projection(gru_tokens).masked_fill(~valid_feature, 0.0)
        v = self.v_projection(cnn_tokens).masked_fill(~valid_feature, 0.0)
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)
        pair_mask = valid_tokens[:, None, :, None] & valid_tokens[:, None, None, :]
        attention = _masked_softmax(scores, pair_mask, dim=-1)
        attention = self.attention_dropout(attention)
        context = torch.matmul(attention, v_heads)
        context = context.transpose(1, 2).contiguous().view_as(q)
        context = self.output_projection(context).masked_fill(~valid_feature, 0.0)
        concatenated = torch.cat([original_tokens, context], dim=-1)
        fused = self.concat_projection(concatenated).masked_fill(~valid_feature, 0.0)

        if self.capture_observability:
            self._last_observability = {
                "q": q.detach(),
                "k": k.detach(),
                "v": v.detach(),
                "attention": attention.detach(),
                "context": context.detach(),
                "concatenated": concatenated.detach(),
                "fused": fused.detach(),
            }
        return fused, attention

    def observability(self) -> dict[str, Tensor]:
        return dict(self._last_observability)


class Chapter4PaperDWAFPathModel(nn.Module):
    """HPE -> GRU/CNN -> token D1 -> Transformer -> path/sample aggregation."""

    def __init__(
        self,
        backbone: Chapter4BaseModel,
        *,
        cnn_kernel_size: int = 3,
        dwaf_num_heads: int = 8,
        branch_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        path_fusion_mode: str = PATH_FUSION_MODE_FLAT_V1,
    ) -> None:
        super().__init__()
        if path_fusion_mode not in PATH_FUSION_MODES:
            raise ValueError(f"Unsupported path_fusion_mode={path_fusion_mode!r}")
        self.backbone = backbone
        hidden_dim = backbone.classifier.output.in_features
        num_classes = backbone.classifier.output.out_features
        classifier_dropout = backbone.classifier.dropout.p
        self.dwaf_mode = DWAF_MODE_D1_V1
        self.gru_branch = PaperGRUTokenBranch(hidden_dim, dropout=branch_dropout)
        self.cnn_branch = PaperCNNChannelBranch(
            hidden_dim,
            kernel_size=cnn_kernel_size,
            dropout=branch_dropout,
        )
        self.dwaf = PaperLikeDWAF(
            hidden_dim=hidden_dim,
            num_heads=dwaf_num_heads,
            attention_dropout=attention_dropout,
        )
        self.path_fusion_mode = path_fusion_mode
        self.aggregator = (
            SegmentAttentionAggregator(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=classifier_dropout,
            )
            if path_fusion_mode == PATH_FUSION_MODE_FLAT_V1
            else None
        )
        self.pf1_aggregator = (
            HierarchicalPathFusionAggregator(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=classifier_dropout,
            )
            if path_fusion_mode == PATH_FUSION_MODE_PF1_V1
            else None
        )
        self._last_observability: dict[str, Tensor] = {}
        self.capture_observability = False

    def auxiliary_losses(self) -> dict[str, Tensor]:
        return self.backbone.auxiliary_losses()

    def observability(self) -> dict[str, Tensor]:
        observed = dict(self._last_observability)
        observed.update(
            {f"dwaf_core_{key}": value for key, value in self.dwaf.observability().items()}
        )
        return observed

    def set_observability(self, enabled: bool) -> None:
        """Opt in to full diagnostic tensors without retaining them during training."""
        self.capture_observability = bool(enabled)
        self.dwaf.set_observability(enabled)
        if not self.capture_observability:
            self._last_observability = {}

    def forward(
        self,
        *,
        token_ids: Tensor,
        depth_ids: Tensor,
        sibling_ids: Tensor,
        global_pos_ids: Tensor,
        local_pos_ids: Tensor,
        segment_ids: Tensor,
        token_padding_mask: Tensor,
        segment_padding_mask: Tensor,
        delta_seg: Tensor | None = None,
        ref_cue_ids: Tensor | None = None,
        ref_identity_ids: Tensor | None = None,
        ref_definition_token_ids: Tensor | None = None,
        ref_matrix: Tensor | None = None,
        path_kind_ids: Tensor | None = None,
        path_group_ids: Tensor | None = None,
        path_duplicate_weights: Tensor | None = None,
        segment_shell_flags: Tensor | None = None,
        segment_union_target_flags: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del labels  # D1 has no label-conditioned mechanism.
        batch_size, max_segments, max_tokens = token_ids.shape

        def flat(value: Tensor) -> Tensor:
            return value.reshape(batch_size * max_segments, max_tokens)

        flat_token_mask = flat(token_padding_mask)
        filtered_ref_cue_ids = apply_ref_cue_segment_filter(
            ref_cue_ids,
            path_kind_ids=path_kind_ids,
            segment_shell_flags=segment_shell_flags,
            segment_union_target_flags=segment_union_target_flags,
            segment_scope=getattr(self, "ref_cue_segment_scope", "all"),
            path_kind_filter_ids=tuple(getattr(self, "ref_cue_path_kind_ids", ())),
        )
        hpe_tokens = self.backbone.encode_hpe_tokens(
            token_ids=flat(token_ids),
            depth_ids=flat(depth_ids),
            sibling_ids=flat(sibling_ids),
            global_pos_ids=flat(global_pos_ids),
            local_pos_ids=flat(local_pos_ids),
            padding_mask=flat_token_mask,
            delta_seg=flat(delta_seg) if delta_seg is not None else None,
            ref_cue_ids=flat(filtered_ref_cue_ids) if filtered_ref_cue_ids is not None else None,
            ref_identity_ids=flat(ref_identity_ids) if ref_identity_ids is not None else None,
            ref_definition_token_ids=(
                flat(ref_definition_token_ids) if ref_definition_token_ids is not None else None
            ),
            explicit_padding_zero=True,
        )
        gru_tokens, gru_attention, gru_pooled = self.gru_branch(
            hpe_tokens,
            padding_mask=flat_token_mask,
        )
        cnn_tokens, cnn_channel_attention = self.cnn_branch(
            hpe_tokens,
            padding_mask=flat_token_mask,
        )
        fused_tokens, dwaf_attention = self.dwaf(
            hpe_tokens,
            gru_tokens,
            cnn_tokens,
            padding_mask=flat_token_mask,
        )
        flat_ref_matrix = (
            ref_matrix.reshape(batch_size * max_segments, max_tokens, max_tokens)
            if ref_matrix is not None
            else None
        )
        transformed = self.backbone.transform_tokens(
            fused_tokens,
            depth_ids=flat(depth_ids),
            sibling_ids=flat(sibling_ids),
            segment_ids=flat(segment_ids),
            padding_mask=flat_token_mask,
            ref_matrix=flat_ref_matrix,
        )
        segment_embeddings = self.backbone.pool_segments(
            transformed,
            padding_mask=flat_token_mask,
        ).view(batch_size, max_segments, -1)

        if self.path_fusion_mode == PATH_FUSION_MODE_PF1_V1:
            if path_group_ids is None or path_duplicate_weights is None:
                raise ValueError("PF1 requires path_group_ids and path_duplicate_weights")
            assert self.pf1_aggregator is not None
            logits, segment_attention = self.pf1_aggregator(
                segment_embeddings,
                segment_padding_mask=segment_padding_mask,
                path_group_ids=path_group_ids,
                path_duplicate_weights=path_duplicate_weights,
            )
        else:
            assert self.aggregator is not None
            logits, segment_attention = self.aggregator(
                segment_embeddings,
                segment_mask=segment_padding_mask,
            )

        if self.capture_observability:
            self._last_observability = {
                "gru_token_attention": gru_attention.detach().view(
                    batch_size, max_segments, max_tokens
                ),
                "gru_pooled": gru_pooled.detach().view(batch_size, max_segments, -1),
                "cnn_channel_attention": cnn_channel_attention.detach().view(
                    batch_size, max_segments, -1
                ),
                "dwaf_attention": dwaf_attention.detach().view(
                    batch_size,
                    max_segments,
                    self.dwaf.num_heads,
                    max_tokens,
                    max_tokens,
                ),
                "segment_attention": segment_attention.detach(),
            }
        return logits, segment_attention
