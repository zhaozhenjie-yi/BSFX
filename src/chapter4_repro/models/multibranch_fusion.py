from __future__ import annotations

from collections.abc import Sequence

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


class TokenAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        scores = self.scorer(sequence).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask, 0.0)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = weights / denom
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class AttentiveGRUBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        *,
        pivot_dims: Sequence[int] | None = None,
        pivot_clip_value: float | None = None,
        pivot_reg_weight: float = 0.0,
        padding_safe: bool = False,
    ) -> None:
        super().__init__()
        gru_hidden = max(hidden_dim // 2, 1)
        self.gru = nn.GRU(
            input_dim,
            gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        gru_output_dim = gru_hidden * 2
        self.output_projection = (
            nn.Identity() if gru_output_dim == hidden_dim else nn.Linear(gru_output_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = TokenAttentionPool(hidden_dim)
        normalized_dims = tuple(
            sorted({int(dim) for dim in (pivot_dims or []) if 0 <= int(dim) < int(hidden_dim)})
        )
        self.pivot_dims = normalized_dims
        self.pivot_clip_value = float(pivot_clip_value) if pivot_clip_value is not None else None
        self.pivot_reg_weight = float(pivot_reg_weight)
        self.padding_safe = bool(padding_safe)
        if normalized_dims:
            self.register_buffer(
                "pivot_index",
                torch.tensor(normalized_dims, dtype=torch.long),
                persistent=False,
            )
        else:
            self.pivot_index = None
        self._last_aux_losses: dict[str, Tensor] = {}

    def _apply_pivot_controls(self, outputs: Tensor) -> Tensor:
        self._last_aux_losses = {}
        if self.pivot_index is None:
            return outputs

        controlled = outputs
        if self.pivot_clip_value is not None:
            controlled = controlled.clone()
            controlled[..., self.pivot_index] = controlled[..., self.pivot_index].clamp(
                min=-self.pivot_clip_value,
                max=self.pivot_clip_value,
            )
        if self.training and self.pivot_reg_weight > 0.0:
            pivot_slice = controlled[..., self.pivot_index]
            self._last_aux_losses["pivot_reg_loss"] = pivot_slice.square().mean() * self.pivot_reg_weight
        return controlled

    def auxiliary_losses(self) -> dict[str, Tensor]:
        return dict(self._last_aux_losses)

    def forward(self, sequence: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.padding_safe and padding_mask is not None:
            if padding_mask.shape != sequence.shape[:2]:
                raise ValueError("padding_mask must match the GRU sequence batch/time dimensions")
            valid_lengths = (~padding_mask).sum(dim=1)
            positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
            expected_padding = positions >= valid_lengths.unsqueeze(1)
            if not torch.equal(padding_mask, expected_padding):
                raise ValueError("padding-safe GRU requires a right-padded token mask")
            masked_sequence = sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            packed = pack_padded_sequence(
                masked_sequence,
                valid_lengths.clamp_min(1).to(device="cpu"),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, _ = self.gru(packed)
            outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=sequence.shape[1],
            )
            outputs = outputs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        else:
            outputs, _ = self.gru(sequence)
        outputs = self.output_projection(self.dropout(outputs))
        if self.padding_safe and padding_mask is not None:
            outputs = outputs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        outputs = self._apply_pivot_controls(outputs)
        return self.pool(outputs, padding_mask=padding_mask)


class AttentiveCNNBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_sizes: Sequence[int] = (3, 5),
        dropout: float = 0.1,
        padding_safe: bool = False,
    ) -> None:
        super().__init__()
        kernels = tuple(dict.fromkeys(int(size) for size in kernel_sizes if int(size) > 0))
        if not kernels:
            kernels = (3, 5)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in kernels
            ]
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pool = TokenAttentionPool(hidden_dim)
        self.padding_safe = bool(padding_safe)

    def forward(self, sequence: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.padding_safe and padding_mask is not None:
            if padding_mask.shape != sequence.shape[:2]:
                raise ValueError("padding_mask must match the CNN sequence batch/time dimensions")
            sequence = sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = sequence.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_x = conv(x).transpose(1, 2)
            conv_outputs.append(self.activation(conv_x))
        merged = torch.stack(conv_outputs, dim=0).mean(dim=0)
        merged = self.dropout(merged)
        if self.padding_safe and padding_mask is not None:
            merged = merged.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.pool(merged, padding_mask=padding_mask)


class DynamicWeightFusion(nn.Module):
    def __init__(self, hidden_dim: int, fusion_hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        inner_dim = fusion_hidden_dim or hidden_dim
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, 2),
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gru_features: Tensor, cnn_features: Tensor) -> tuple[Tensor, Tensor]:
        gate_logits = self.gate(torch.cat([gru_features, cnn_features], dim=-1))
        weights = torch.softmax(gate_logits, dim=-1)
        fused = weights[..., :1] * gru_features + weights[..., 1:] * cnn_features
        fused = self.output_projection(self.dropout(fused))
        return fused, weights


class Chapter4MultiBranchPathModel(nn.Module):
    """
    Second-stage approximation of the thesis path model:
    structural transformer -> GRU branch + CNN branch -> dynamic weight fusion -> sample aggregation.
    """

    def __init__(
        self,
        backbone: Chapter4BaseModel,
        *,
        gru_hidden_dim: int | None = None,
        cnn_kernel_sizes: Sequence[int] = (3, 5),
        fusion_hidden_dim: int | None = None,
        gru_pivot_dims: Sequence[int] | None = None,
        gru_pivot_clip_value: float | None = None,
        gru_pivot_reg_weight: float = 0.0,
        residual_keynode_reg_weight: float = 0.0,
        residual_keynode_topk: int = 0,
        ref_free_callpath_reg_weight: float = 0.0,
        ref_free_callpath_topk: int = 0,
        padding_safe_branches: bool = False,
        path_fusion_mode: str = PATH_FUSION_MODE_FLAT_V1,
    ) -> None:
        super().__init__()
        if path_fusion_mode not in PATH_FUSION_MODES:
            raise ValueError(f"Unsupported path_fusion_mode={path_fusion_mode!r}")
        self.backbone = backbone
        hidden_dim = backbone.classifier.output.in_features
        num_classes = backbone.classifier.output.out_features
        dropout = backbone.classifier.dropout.p
        branch_hidden_dim = gru_hidden_dim or hidden_dim

        self.gru_branch = AttentiveGRUBranch(
            input_dim=hidden_dim,
            hidden_dim=branch_hidden_dim,
            dropout=dropout,
            pivot_dims=gru_pivot_dims,
            pivot_clip_value=gru_pivot_clip_value,
            pivot_reg_weight=gru_pivot_reg_weight,
            padding_safe=padding_safe_branches,
        )
        self.cnn_branch = AttentiveCNNBranch(
            input_dim=hidden_dim,
            hidden_dim=branch_hidden_dim,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
            padding_safe=padding_safe_branches,
        )
        self.fusion = DynamicWeightFusion(
            hidden_dim=branch_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )
        self.path_fusion_mode = path_fusion_mode
        self.aggregator = (
            SegmentAttentionAggregator(hidden_dim=branch_hidden_dim, num_classes=num_classes, dropout=dropout)
            if path_fusion_mode == PATH_FUSION_MODE_FLAT_V1
            else None
        )
        self.pf1_aggregator = (
            HierarchicalPathFusionAggregator(
                hidden_dim=branch_hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
            if path_fusion_mode == PATH_FUSION_MODE_PF1_V1
            else None
        )
        self.gru_pivot_dims = tuple(int(dim) for dim in (gru_pivot_dims or ()))
        self.gru_pivot_clip_value = float(gru_pivot_clip_value) if gru_pivot_clip_value is not None else None
        self.gru_pivot_reg_weight = float(gru_pivot_reg_weight)
        self.residual_keynode_reg_weight = float(residual_keynode_reg_weight)
        self.residual_keynode_topk = int(residual_keynode_topk)
        self.ref_free_callpath_reg_weight = float(ref_free_callpath_reg_weight)
        self.ref_free_callpath_topk = int(ref_free_callpath_topk)
        self.padding_safe_branches = bool(padding_safe_branches)
        self._last_aux_losses: dict[str, Tensor] = {}

    def _active_classifier(self) -> nn.Linear:
        if self.path_fusion_mode == PATH_FUSION_MODE_PF1_V1:
            assert self.pf1_aggregator is not None
            return self.pf1_aggregator.classifier
        assert self.aggregator is not None
        return self.aggregator.classifier

    def auxiliary_losses(self) -> dict[str, Tensor]:
        losses = {}
        losses.update(self.backbone.auxiliary_losses())
        for name, value in self.gru_branch.auxiliary_losses().items():
            losses[f"gru_{name}"] = value
        losses.update(self._last_aux_losses)
        return losses

    def _compute_residual_keynode_reg_loss(
        self,
        *,
        fused_segments: Tensor,
        segment_attention: Tensor,
        segment_padding_mask: Tensor,
        path_kind_ids: Tensor | None,
        segment_shell_flags: Tensor | None,
        segment_union_target_flags: Tensor | None,
        labels: Tensor | None,
    ) -> None:
        if (
            not self.training
            or self.residual_keynode_reg_weight <= 0.0
            or self.residual_keynode_topk <= 0
            or path_kind_ids is None
            or segment_shell_flags is None
            or segment_union_target_flags is None
            or labels is None
        ):
            return

        key_node_path_id = 2
        branch_path_id = 3
        call_path_id = 4
        valid_segment_mask = ~segment_padding_mask
        has_branch = ((path_kind_ids == branch_path_id) & valid_segment_mask).any(dim=1)
        has_call = ((path_kind_ids == call_path_id) & valid_segment_mask).any(dim=1)
        has_shell = (segment_shell_flags & valid_segment_mask).any(dim=1)
        eligible_samples = (labels == 0) & has_branch & has_call & has_shell
        if not eligible_samples.any():
            return

        classifier = self._active_classifier()
        margin_vector = classifier.weight[1] - classifier.weight[0]
        segment_margin_raw = torch.einsum("bsh,h->bs", fused_segments, margin_vector)
        segment_margin_contrib = segment_attention * segment_margin_raw
        positive_contrib = torch.relu(segment_margin_contrib)

        per_sample_losses = []
        for sample_index in torch.nonzero(eligible_samples, as_tuple=False).flatten().tolist():
            candidate_mask = (
                valid_segment_mask[sample_index]
                & (path_kind_ids[sample_index] == key_node_path_id)
                & (~segment_union_target_flags[sample_index])
            )
            if not candidate_mask.any():
                continue
            candidate_scores = positive_contrib[sample_index][candidate_mask]
            if candidate_scores.numel() == 0:
                continue
            topk = min(self.residual_keynode_topk, int(candidate_scores.numel()))
            top_values, _ = torch.topk(candidate_scores, k=topk, largest=True)
            if top_values.numel() == 0:
                continue
            per_sample_losses.append(top_values.mean())

        if not per_sample_losses:
            return
        reg_loss = torch.stack(per_sample_losses).mean() * self.residual_keynode_reg_weight
        self._last_aux_losses["residual_keynode_reg_loss"] = reg_loss

    def _compute_ref_free_callpath_reg_loss(
        self,
        *,
        fused_segments: Tensor,
        segment_attention: Tensor,
        segment_padding_mask: Tensor,
        path_kind_ids: Tensor | None,
        ref_matrix: Tensor | None,
        ref_cue_ids: Tensor | None,
        labels: Tensor | None,
    ) -> None:
        if (
            not self.training
            or self.ref_free_callpath_reg_weight <= 0.0
            or self.ref_free_callpath_topk <= 0
            or path_kind_ids is None
            or labels is None
        ):
            return

        call_path_id = 4
        valid_segment_mask = ~segment_padding_mask
        candidate_mask = valid_segment_mask & (path_kind_ids == call_path_id) & (labels.unsqueeze(1) == 0)
        if ref_matrix is not None:
            has_ref = ref_matrix.abs().sum(dim=(-1, -2)) > 0
            candidate_mask &= ~has_ref
        elif ref_cue_ids is not None:
            has_ref = ref_cue_ids.abs().sum(dim=-1) > 0
            candidate_mask &= ~has_ref
        if not candidate_mask.any():
            return

        classifier = self._active_classifier()
        margin_vector = classifier.weight[1] - classifier.weight[0]
        segment_margin_raw = torch.einsum("bsh,h->bs", fused_segments, margin_vector)
        segment_margin_contrib = segment_attention * segment_margin_raw
        positive_contrib = torch.relu(segment_margin_contrib)

        per_sample_losses = []
        for sample_index in torch.nonzero(candidate_mask.any(dim=1), as_tuple=False).flatten().tolist():
            candidate_scores = positive_contrib[sample_index][candidate_mask[sample_index]]
            if candidate_scores.numel() == 0:
                continue
            topk = min(self.ref_free_callpath_topk, int(candidate_scores.numel()))
            top_values, _ = torch.topk(candidate_scores, k=topk, largest=True)
            if top_values.numel() == 0:
                continue
            per_sample_losses.append(top_values.mean())

        if not per_sample_losses:
            return
        reg_loss = torch.stack(per_sample_losses).mean() * self.ref_free_callpath_reg_weight
        self._last_aux_losses["ref_free_callpath_reg_loss"] = reg_loss

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
        gru_features, _ = self.gru_branch(encoded, padding_mask=flat(token_padding_mask))
        cnn_features, _ = self.cnn_branch(encoded, padding_mask=flat(token_padding_mask))
        fused_features, _ = self.fusion(gru_features, cnn_features)
        fused_segments = fused_features.view(batch_size, max_segments, -1)

        if self.path_fusion_mode == PATH_FUSION_MODE_PF1_V1:
            if path_group_ids is None or path_duplicate_weights is None:
                raise ValueError("PF1 requires path_group_ids and path_duplicate_weights")
            assert self.pf1_aggregator is not None
            logits, segment_attention = self.pf1_aggregator(
                fused_segments,
                segment_padding_mask=segment_padding_mask,
                path_group_ids=path_group_ids,
                path_duplicate_weights=path_duplicate_weights,
            )
        else:
            assert self.aggregator is not None
            logits, segment_attention = self.aggregator(
                fused_segments,
                segment_mask=segment_padding_mask,
            )

        self._last_aux_losses = {}
        self._compute_residual_keynode_reg_loss(
            fused_segments=fused_segments,
            segment_attention=segment_attention,
            segment_padding_mask=segment_padding_mask,
            path_kind_ids=path_kind_ids,
            segment_shell_flags=segment_shell_flags,
            segment_union_target_flags=segment_union_target_flags,
            labels=labels,
        )
        self._compute_ref_free_callpath_reg_loss(
            fused_segments=fused_segments,
            segment_attention=segment_attention,
            segment_padding_mask=segment_padding_mask,
            path_kind_ids=path_kind_ids,
            ref_matrix=ref_matrix,
            ref_cue_ids=filtered_ref_cue_ids,
            labels=labels,
        )
        return logits, segment_attention
