from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


TASK32_ARCHITECTURE_ID = "task32-multilevel-path-segment-classifier-v1"
TASK32_OUTPUT_PROTOCOL = "two_logits"
TASK32_POSITIVE_CLASS_INDEX = 1
PF1_TASK32_COVERAGE = "partial"
PF1_TASK32_COVERAGE_EVIDENCE = {
    "masked_hierarchical_embedding_attention": True,
    "path_level_token_max_pooling": False,
    "segment_independent_prediction": False,
    "segment_probability_attention": False,
    "final_weighted_probability": False,
}


def _masked_max(
    values: Tensor,
    valid_mask: Tensor,
    *,
    dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Max-pool valid entries without allowing zero-filled PAD to win.

    The returned validity mask has the pooled dimension removed.  Invalid
    groups are represented by an exact zero only *after* the maximum has been
    computed with ``-inf``.  ``winning_indices`` are per-hidden-dimension and
    use ``-1`` for invalid groups.
    """

    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    expected = values.shape[:-1]
    if tuple(valid_mask.shape) != tuple(expected):
        raise ValueError("valid_mask must match every values dimension except hidden")
    if dim < 0:
        dim += values.ndim
    if dim < 0 or dim >= values.ndim - 1:
        raise ValueError("pooling dim must be a non-hidden values dimension")

    masked = values.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
    pooled, winning_indices = masked.max(dim=dim)
    group_valid = valid_mask.any(dim=dim)
    pooled = pooled.masked_fill(~group_valid.unsqueeze(-1), 0.0)
    winning_indices = winning_indices.masked_fill(~group_valid.unsqueeze(-1), -1)
    return pooled, group_valid, winning_indices


def _masked_softmax(scores: Tensor, valid_mask: Tensor) -> Tensor:
    """FP32 masked softmax with an exactly-zero all-invalid result."""

    if scores.shape != valid_mask.shape:
        raise ValueError("scores and valid_mask must have the same shape")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    scores_fp32 = scores.float()
    masked = scores_fp32.masked_fill(~valid_mask, torch.finfo(torch.float32).min)
    shifted = masked - masked.max(dim=-1, keepdim=True).values
    numerators = torch.exp(shifted).masked_fill(~valid_mask, 0.0)
    denominator = numerators.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > 0,
        numerators / denominator.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(numerators),
    )


@dataclass
class MultilevelClassifierOutput:
    """Task 32 output; all live tensors retain gradients except ``audit``."""

    final_logits: Tensor
    final_probability: Tensor
    segment_logits: Tensor
    segment_probabilities: Tensor
    segment_attention_scores: Tensor
    segment_attention_weights: Tensor
    segment_valid_mask: Tensor
    path_valid_mask: Tensor
    audit: dict[str, Any] | None = None


class Task32GlobalClassifier(nn.Module):
    """C0 rollback control: one masked global max followed by two logits."""

    architecture_id = "task32-c0-global-classifier-v1"

    def __init__(self, hidden_dim: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, token_representations: Tensor, token_valid_mask: Tensor) -> Tensor:
        if token_representations.ndim != 5:
            raise ValueError("token_representations must have shape [B,S,P,T,D]")
        if token_valid_mask.shape != token_representations.shape[:-1]:
            raise ValueError("token_valid_mask must have shape [B,S,P,T]")
        if token_valid_mask.dtype != torch.bool:
            raise TypeError("token_valid_mask must be boolean")
        batch, segments, paths, tokens, hidden = token_representations.shape
        flat_values = token_representations.reshape(batch, segments * paths * tokens, hidden)
        flat_mask = token_valid_mask.reshape(batch, segments * paths * tokens)
        pooled, sample_valid, _ = _masked_max(flat_values, flat_mask, dim=1)
        if not bool(sample_valid.all().item()):
            raise ValueError("every sample requires at least one valid token")
        return self.classifier(self.dropout(pooled))


class Task32MultilevelClassifier(nn.Module):
    """C1 path/segment hierarchy with probability-driven segment attention.

    Frozen shapes:

    * tokens: ``[B,S,P,T,D]``
    * paths: ``[B,S,P,D]`` (masked max over token ``T``)
    * segments: ``[B,S,D]`` (masked max over path ``P``)
    * per-segment logits: ``[B,S,2]``
    * per-segment positive probability and final probability: ``[B,S]`` / ``[B]``

    The paper's formula (4-45) is instantiated permutation-equivariantly as
    ``score_s = w_a * tanh(p_s) + b_a`` with a shared learnable scalar.  This
    is the only variable-length interpretation that does not encode an
    arbitrary segment slot.  Formula (4-46) is exact:
    ``p_final = sum_s softmax(score)_s * p_s``.
    """

    architecture_id = TASK32_ARCHITECTURE_ID
    output_protocol = TASK32_OUTPUT_PROTOCOL
    positive_class_index = TASK32_POSITIVE_CLASS_INDEX

    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        probability_epsilon: float = 1e-7,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 < probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must be in (0,0.5)")
        self.hidden_dim = int(hidden_dim)
        self.probability_epsilon = float(probability_epsilon)
        self.dropout = nn.Dropout(dropout)
        self.segment_classifier = nn.Linear(self.hidden_dim, 2)
        self.probability_attention_weight = nn.Parameter(torch.ones(()))
        self.probability_attention_bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def function_loss(final_logits: Tensor, labels: Tensor) -> Tensor:
        """Function-level supervision only; no synthetic segment labels."""

        if final_logits.ndim != 2 or final_logits.shape[-1] != 2:
            raise ValueError("final_logits must have shape [B,2]")
        if labels.shape != final_logits.shape[:1] or labels.dtype != torch.long:
            raise ValueError("labels must be int64 with shape [B]")
        return torch.nn.functional.cross_entropy(final_logits, labels)

    def forward(
        self,
        token_representations: Tensor,
        token_valid_mask: Tensor,
        *,
        path_valid_mask: Tensor | None = None,
        segment_valid_mask: Tensor | None = None,
        return_multilevel_audit: bool = False,
        audit_max_samples: int = 1,
    ) -> MultilevelClassifierOutput:
        if token_representations.ndim != 5:
            raise ValueError("token_representations must have shape [B,S,P,T,D]")
        batch, segments, paths, _, hidden = token_representations.shape
        if hidden != self.hidden_dim:
            raise ValueError("token representation width does not match hidden_dim")
        if token_valid_mask.shape != token_representations.shape[:-1]:
            raise ValueError("token_valid_mask must have shape [B,S,P,T]")
        if token_valid_mask.dtype != torch.bool:
            raise TypeError("token_valid_mask must be boolean")
        if path_valid_mask is not None:
            if path_valid_mask.shape != (batch, segments, paths):
                raise ValueError("path_valid_mask must have shape [B,S,P]")
            if path_valid_mask.dtype != torch.bool:
                raise TypeError("path_valid_mask must be boolean")
        if segment_valid_mask is not None:
            if segment_valid_mask.shape != (batch, segments):
                raise ValueError("segment_valid_mask must have shape [B,S]")
            if segment_valid_mask.dtype != torch.bool:
                raise TypeError("segment_valid_mask must be boolean")

        path_representations, nonempty_paths, winning_token_indices = _masked_max(
            token_representations,
            token_valid_mask,
            dim=3,
        )
        actual_path_mask = nonempty_paths
        if path_valid_mask is not None:
            actual_path_mask = actual_path_mask & path_valid_mask
        path_representations = path_representations.masked_fill(
            ~actual_path_mask.unsqueeze(-1),
            0.0,
        )
        winning_token_indices = winning_token_indices.masked_fill(
            ~actual_path_mask.unsqueeze(-1),
            -1,
        )

        segment_representations, nonempty_segments, winning_path_indices = _masked_max(
            path_representations,
            actual_path_mask,
            dim=2,
        )
        actual_segment_mask = nonempty_segments
        if segment_valid_mask is not None:
            actual_segment_mask = actual_segment_mask & segment_valid_mask
        segment_representations = segment_representations.masked_fill(
            ~actual_segment_mask.unsqueeze(-1),
            0.0,
        )
        winning_path_indices = winning_path_indices.masked_fill(
            ~actual_segment_mask.unsqueeze(-1),
            -1,
        )
        if not bool(actual_segment_mask.any(dim=1).all().item()):
            raise ValueError("every sample requires at least one valid segment")

        segment_logits = self.segment_classifier(self.dropout(segment_representations))
        segment_logits = segment_logits.masked_fill(~actual_segment_mask.unsqueeze(-1), 0.0)
        segment_probabilities = torch.softmax(segment_logits.float(), dim=-1)[..., 1]
        segment_probabilities = segment_probabilities.masked_fill(~actual_segment_mask, 0.0)
        raw_attention_scores = (
            self.probability_attention_weight.float() * torch.tanh(segment_probabilities)
            + self.probability_attention_bias.float()
        )
        attention_weights = _masked_softmax(raw_attention_scores, actual_segment_mask)
        attention_scores = raw_attention_scores.masked_fill(~actual_segment_mask, 0.0)
        final_probability = torch.sum(attention_weights * segment_probabilities, dim=-1)
        safe_probability = final_probability.clamp(
            min=self.probability_epsilon,
            max=1.0 - self.probability_epsilon,
        )
        # Two-logit compatibility: softmax(log([1-p,p])) recovers p exactly up
        # to floating-point roundoff and remains valid for CrossEntropy/Focal.
        final_logits = torch.stack(
            (torch.log1p(-safe_probability), torch.log(safe_probability)),
            dim=-1,
        )

        audit = None
        if return_multilevel_audit:
            if audit_max_samples <= 0:
                raise ValueError("audit_max_samples must be positive")
            limit = min(batch, int(audit_max_samples))

            def snapshot(value: Tensor) -> Tensor:
                return value[:limit].detach().cpu()

            audit = {
                "path_representations": snapshot(path_representations),
                "path_valid_mask": snapshot(actual_path_mask),
                "valid_path_count": snapshot(actual_path_mask.sum(dim=2)),
                "segment_representations": snapshot(segment_representations),
                "segment_logits": snapshot(segment_logits),
                "segment_probabilities": snapshot(segment_probabilities),
                "segment_attention_scores": snapshot(attention_scores),
                "segment_attention_weights": snapshot(attention_weights),
                "segment_valid_mask": snapshot(actual_segment_mask),
                "final_probability": snapshot(final_probability),
                "winning_token_index_per_path": snapshot(winning_token_indices),
                "winning_path_index_per_segment": snapshot(winning_path_indices),
            }

        return MultilevelClassifierOutput(
            final_logits=final_logits,
            final_probability=final_probability,
            segment_logits=segment_logits,
            segment_probabilities=segment_probabilities,
            segment_attention_scores=attention_scores,
            segment_attention_weights=attention_weights,
            segment_valid_mask=actual_segment_mask,
            path_valid_mask=actual_path_mask,
            audit=audit,
        )


def task31_hidden_to_task32_hierarchy(
    last_hidden_state: Tensor,
    *,
    segment_ids: Tensor,
    padding_mask: Tensor,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Restore Task31's flattened real path to Task32 ``[B,S,1,T,D]``.

    Task31 accepts one source path per batch row and flattens that path's
    segments.  Consequently every restored segment contains one real path
    slot (``P=1``); the generic Task32 head still supports ``P>1`` for future
    adapters without changing classifier semantics.
    """

    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [B,T,D]")
    batch, _, hidden = last_hidden_state.shape
    if segment_ids.shape != last_hidden_state.shape[:2]:
        raise ValueError("segment_ids must have shape [B,T]")
    if padding_mask.shape != segment_ids.shape or padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be boolean with shape [B,T]")

    row_segment_values: list[list[int]] = []
    max_segments = 0
    max_tokens = 0
    for row in range(batch):
        valid = ~padding_mask[row]
        values = sorted(int(value) for value in torch.unique(segment_ids[row][valid]).tolist())
        if not values or values != list(range(len(values))):
            raise ValueError("valid Task31 segment IDs must be contiguous from zero")
        row_segment_values.append(values)
        max_segments = max(max_segments, len(values))
        for value in values:
            max_tokens = max(max_tokens, int((valid & segment_ids[row].eq(value)).sum().item()))

    hierarchy = last_hidden_state.new_zeros((batch, max_segments, 1, max_tokens, hidden))
    token_valid = torch.zeros(
        (batch, max_segments, 1, max_tokens),
        dtype=torch.bool,
        device=last_hidden_state.device,
    )
    lengths: list[list[int]] = []
    for row, values in enumerate(row_segment_values):
        row_lengths = []
        valid = ~padding_mask[row]
        for output_segment, source_segment in enumerate(values):
            indices = valid & segment_ids[row].eq(source_segment)
            segment_tokens = last_hidden_state[row, indices]
            length = int(segment_tokens.shape[0])
            hierarchy[row, output_segment, 0, :length] = segment_tokens
            token_valid[row, output_segment, 0, :length] = True
            row_lengths.append(length)
        lengths.append(row_lengths)
    return hierarchy, token_valid, {
        "contract": "task31-flattened-path-to-task32-hierarchy-v1",
        "shape": list(hierarchy.shape),
        "segment_counts": [len(values) for values in row_segment_values],
        "segment_token_lengths": lengths,
        "path_slots_per_segment": 1,
    }
