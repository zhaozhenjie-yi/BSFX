from __future__ import annotations

import torch
from torch import Tensor, nn

from .full_model import Chapter4BaseModel
from .hierarchical_path_fusion import (
    PATH_FUSION_MODE_FLAT_V1,
    PATH_FUSION_MODE_PF1_V1,
    PATH_FUSION_MODES,
    HierarchicalPathFusionAggregator,
)


VALID_REF_CUE_SEGMENT_SCOPES = {
    "all",
    "shell_only",
    "non_shell",
    "union_target_only",
    "non_union_target",
}


def apply_ref_cue_segment_filter(
    ref_cue_ids: Tensor | None,
    *,
    path_kind_ids: Tensor | None,
    segment_shell_flags: Tensor | None,
    segment_union_target_flags: Tensor | None,
    segment_scope: str = "all",
    path_kind_filter_ids: tuple[int, ...] = (),
) -> Tensor | None:
    if ref_cue_ids is None:
        return None
    if segment_scope not in VALID_REF_CUE_SEGMENT_SCOPES:
        raise ValueError(f"Unsupported ref cue segment scope: {segment_scope}")

    segment_mask = torch.ones(ref_cue_ids.shape[:2], dtype=torch.bool, device=ref_cue_ids.device)
    if segment_scope == "shell_only" and segment_shell_flags is not None:
        segment_mask &= segment_shell_flags.bool()
    elif segment_scope == "non_shell" and segment_shell_flags is not None:
        segment_mask &= ~segment_shell_flags.bool()
    elif segment_scope == "union_target_only" and segment_union_target_flags is not None:
        segment_mask &= segment_union_target_flags.bool()
    elif segment_scope == "non_union_target" and segment_union_target_flags is not None:
        segment_mask &= ~segment_union_target_flags.bool()

    if path_kind_filter_ids and path_kind_ids is not None:
        path_mask = torch.zeros_like(segment_mask)
        for path_kind_id in path_kind_filter_ids:
            path_mask |= path_kind_ids == int(path_kind_id)
        segment_mask &= path_mask

    return ref_cue_ids * segment_mask.unsqueeze(-1).to(dtype=ref_cue_ids.dtype)


class SegmentAttentionAggregator(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, segment_embeddings: Tensor, segment_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        scores = self.scorer(segment_embeddings).squeeze(-1)
        if segment_mask is not None:
            scores = scores.masked_fill(segment_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if segment_mask is not None:
            weights = weights.masked_fill(segment_mask, 0.0)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = weights / denom
        fused = torch.sum(segment_embeddings * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(fused))
        return logits, weights


class Chapter4PathModel(nn.Module):
    """
    Sample-level model:
    segment encoder -> per-segment pooling -> attention aggregation -> sample logits.
    """

    def __init__(
        self,
        backbone: Chapter4BaseModel,
        *,
        path_fusion_mode: str = PATH_FUSION_MODE_FLAT_V1,
    ) -> None:
        super().__init__()
        if path_fusion_mode not in PATH_FUSION_MODES:
            raise ValueError(f"Unsupported path_fusion_mode={path_fusion_mode!r}")
        self.backbone = backbone
        hidden_dim = backbone.classifier.output.in_features
        num_classes = backbone.classifier.output.out_features
        dropout = backbone.classifier.dropout.p
        self.path_fusion_mode = path_fusion_mode
        self.aggregator = (
            SegmentAttentionAggregator(hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
            if path_fusion_mode == PATH_FUSION_MODE_FLAT_V1
            else None
        )
        self.pf1_aggregator = (
            HierarchicalPathFusionAggregator(hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
            if path_fusion_mode == PATH_FUSION_MODE_PF1_V1
            else None
        )

    def auxiliary_losses(self) -> dict[str, Tensor]:
        return self.backbone.auxiliary_losses()

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
        batch_size, max_segments, max_tokens = token_ids.shape
        flat = lambda x: x.reshape(batch_size * max_segments, max_tokens)
        filtered_ref_cue_ids = apply_ref_cue_segment_filter(
            ref_cue_ids,
            path_kind_ids=path_kind_ids,
            segment_shell_flags=segment_shell_flags,
            segment_union_target_flags=segment_union_target_flags,
            segment_scope=getattr(self, "ref_cue_segment_scope", "all"),
            path_kind_filter_ids=tuple(getattr(self, "ref_cue_path_kind_ids", ())),
        )

        encoded = self.backbone.encode_segments(
            token_ids=flat(token_ids),
            depth_ids=flat(depth_ids),
            sibling_ids=flat(sibling_ids),
            global_pos_ids=flat(global_pos_ids),
            local_pos_ids=flat(local_pos_ids),
            segment_ids=flat(segment_ids),
            padding_mask=flat(token_padding_mask),
            delta_seg=flat(delta_seg) if delta_seg is not None else None,
            ref_cue_ids=flat(filtered_ref_cue_ids) if filtered_ref_cue_ids is not None else None,
            ref_identity_ids=flat(ref_identity_ids) if ref_identity_ids is not None else None,
            ref_definition_token_ids=flat(ref_definition_token_ids)
            if ref_definition_token_ids is not None
            else None,
            ref_matrix=ref_matrix.reshape(batch_size * max_segments, max_tokens, max_tokens)
            if ref_matrix is not None
            else None,
        )
        pooled = self.backbone.pool_segments(encoded, padding_mask=flat(token_padding_mask))
        pooled = pooled.view(batch_size, max_segments, -1)
        if self.path_fusion_mode == PATH_FUSION_MODE_PF1_V1:
            if path_group_ids is None or path_duplicate_weights is None:
                raise ValueError("PF1 requires path_group_ids and path_duplicate_weights")
            assert self.pf1_aggregator is not None
            return self.pf1_aggregator(
                pooled,
                segment_padding_mask=segment_padding_mask,
                path_group_ids=path_group_ids,
                path_duplicate_weights=path_duplicate_weights,
            )
        assert self.aggregator is not None
        return self.aggregator(pooled, segment_mask=segment_padding_mask)
