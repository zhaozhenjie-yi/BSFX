from __future__ import annotations

import torch
from torch import Tensor, nn


PATH_FUSION_MODE_FLAT_V1 = "flat_segment_v1"
PATH_FUSION_MODE_PF1_V1 = "hierarchical_pf1_v1"
PATH_FUSION_MODES = {PATH_FUSION_MODE_FLAT_V1, PATH_FUSION_MODE_PF1_V1}


def _masked_softmax(scores: Tensor, padding_mask: Tensor) -> Tensor:
    """Masked softmax with an exactly-zero, finite all-padding result."""

    if scores.shape != padding_mask.shape:
        raise ValueError("scores and padding_mask must have the same shape")
    masked_scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=-1)
    weights = weights.masked_fill(padding_mask, 0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class HierarchicalPathFusionAggregator(nn.Module):
    """PF1 segment-to-path-to-sample masked attention.

    Exact duplicates remain visible to the model. Their inverse-count weights
    are applied only at path-to-sample fusion so each duplicate equivalence
    class contributes total prior mass one.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.segment_scorer = nn.Linear(hidden_dim, 1)
        self.path_scorer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.last_details: dict[str, Tensor] = {}

    def forward(
        self,
        segment_embeddings: Tensor,
        *,
        segment_padding_mask: Tensor,
        path_group_ids: Tensor,
        path_duplicate_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if segment_embeddings.ndim != 3:
            raise ValueError("segment_embeddings must have shape [batch, segments, hidden]")
        batch_size, max_segments, hidden_dim = segment_embeddings.shape
        expected_shape = (batch_size, max_segments)
        if segment_padding_mask.shape != expected_shape:
            raise ValueError("segment_padding_mask must match [batch, segments]")
        if path_group_ids.shape != expected_shape:
            raise ValueError("path_group_ids must match [batch, segments]")
        if path_duplicate_weights.shape != expected_shape:
            raise ValueError("path_duplicate_weights must match [batch, segments]")
        if torch.any(path_group_ids[segment_padding_mask] != -1):
            raise ValueError("padding segments must use path_group_id=-1")
        real_mask = ~segment_padding_mask
        if torch.any(path_group_ids[real_mask] < 0):
            raise ValueError("real segments require non-negative path_group_ids")
        if torch.any(path_duplicate_weights[real_mask] <= 0):
            raise ValueError("real segments require positive duplicate weights")
        if torch.any(path_duplicate_weights[segment_padding_mask] != 0):
            raise ValueError("padding segments must use duplicate weight zero")

        path_counts = [int(torch.unique(path_group_ids[row][real_mask[row]]).numel()) for row in range(batch_size)]
        max_paths = max(max(path_counts, default=0), 1)
        # Keep every normalization and probability in FP32.  CUDA autocast
        # may emit FP16 scorer outputs while softmax promotes to FP32; using
        # the scorer dtype for the attention buffers therefore causes either
        # an index-put dtype error or a silent loss of normalization accuracy.
        segment_scores = self.segment_scorer(segment_embeddings).squeeze(-1).float()
        attention_dtype = torch.float32
        path_embeddings = segment_embeddings.new_zeros((batch_size, max_paths, hidden_dim))
        path_padding_mask = torch.ones((batch_size, max_paths), dtype=torch.bool, device=segment_embeddings.device)
        path_prior_weights = torch.zeros(
            (batch_size, max_paths),
            dtype=attention_dtype,
            device=segment_embeddings.device,
        )
        within_path_attention = torch.zeros(
            (batch_size, max_segments),
            dtype=attention_dtype,
            device=segment_embeddings.device,
        )

        path_segment_indices: list[list[Tensor]] = []
        for batch_index in range(batch_size):
            row_indices: list[Tensor] = []
            group_values = torch.unique(path_group_ids[batch_index][real_mask[batch_index]], sorted=True)
            for path_index, group_value in enumerate(group_values):
                indices = torch.nonzero(
                    real_mask[batch_index] & (path_group_ids[batch_index] == group_value),
                    as_tuple=False,
                ).flatten()
                row_indices.append(indices)
                duplicate_values = path_duplicate_weights[batch_index, indices]
                if not torch.allclose(
                    duplicate_values,
                    duplicate_values[0].expand_as(duplicate_values),
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise ValueError("all segments in one path must share the same duplicate weight")
                local_weights = torch.softmax(segment_scores[batch_index, indices], dim=0)
                within_path_attention[batch_index, indices] = local_weights
                path_embeddings[batch_index, path_index] = torch.sum(
                    segment_embeddings[batch_index, indices]
                    * local_weights.to(dtype=segment_embeddings.dtype).unsqueeze(-1),
                    dim=0,
                )
                path_padding_mask[batch_index, path_index] = False
                path_prior_weights[batch_index, path_index] = duplicate_values[0].to(dtype=attention_dtype)
            path_segment_indices.append(row_indices)

        path_scores = self.path_scorer(path_embeddings).squeeze(-1).float()
        path_attention = _masked_softmax(path_scores, path_padding_mask)
        path_attention = path_attention * path_prior_weights
        path_attention = path_attention / path_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        path_attention = path_attention.masked_fill(path_padding_mask, 0.0)

        effective_segment_attention = torch.zeros(
            (batch_size, max_segments),
            dtype=path_attention.dtype,
            device=segment_embeddings.device,
        )
        for batch_index, row_indices in enumerate(path_segment_indices):
            for path_index, indices in enumerate(row_indices):
                effective_segment_attention[batch_index, indices] = (
                    within_path_attention[batch_index, indices] * path_attention[batch_index, path_index]
                )

        sample_embedding = torch.sum(path_embeddings.float() * path_attention.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(sample_embedding))
        self.last_details = {
            "within_path_attention": within_path_attention.detach(),
            "path_attention": path_attention.detach(),
            "path_padding_mask": path_padding_mask.detach(),
            "path_prior_weights": path_prior_weights.detach(),
            "effective_segment_attention": effective_segment_attention.detach(),
        }
        return logits, effective_segment_attention
