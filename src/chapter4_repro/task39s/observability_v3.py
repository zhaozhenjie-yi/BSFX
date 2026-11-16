"""Additive, non-mutating observability for the frozen Task34 graph.

The formal Task39/Task39R memory-efficient path invokes Task34's component
modules directly and therefore bypasses :meth:`Task34IntegratedModel.forward`.
This module deliberately uses removable forward hooks so one observation
session covers both schedules without changing either legacy implementation.

Only detached aggregate statistics are retained.  In particular, the
classifier pooling reconstruction below is a diagnostic recomputation; its
derived weighted function representation is *not* an input to either frozen
classifier and is labelled accordingly in every record.

Recommended lifetime::

    observation = Task34ObservabilitySession(model, batch=batch)
    with observation:                 # hooks exist for forward only
        output = model(batch)          # or memory_efficient_forward(...)
    loss = ...
    loss.backward()                    # checkpoint replay cannot duplicate hooks
    observation.record_gradients()

The observer never changes module modes, parameters, buffers, RNG state, or
the tensors returned by a hooked module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn


OBSERVABILITY_SCHEMA_VERSION = "task39s_protocol_v3_observability_schema_v1"
PROTOCOL_ID = "TASK39S_PROTOCOL_V3"
PROTOCOL_VERSION = "V3"
IMPLEMENTATION_REVISION = "task39s_protocol_v3_freeze_v1"
REQUIRED_TRANSFORMER_LAYER_COUNT = 6


class ObservabilityContractError(RuntimeError):
    """Raised when a frozen Task34 shape cannot be observed unambiguously."""


def _to_cpu(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu")


def _tensor_sha256(value: Tensor) -> str:
    detached = _to_cpu(value).contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(str(tuple(detached.shape)).encode("ascii"))
    digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _expanded_mask(value: Tensor, valid_mask: Tensor | None) -> Tensor | None:
    if valid_mask is None:
        return None
    mask = valid_mask.detach().bool()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(mask, value.shape)
    except RuntimeError as exc:
        raise ObservabilityContractError(
            f"mask {tuple(valid_mask.shape)} cannot cover tensor {tuple(value.shape)}"
        ) from exc


def tensor_statistics(
    value: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> dict[str, Any]:
    """Return JSON-safe numerical and mask-aware statistics for ``value``.

    Statistics are computed from detached CPU float64 values.  Non-finite
    values are counted and excluded from finite moments, rather than causing
    the remaining useful diagnostics to disappear.
    """

    if not torch.is_tensor(value):
        raise TypeError("tensor_statistics requires a Tensor")
    original = value.detach()
    data = original.to(device="cpu", dtype=torch.float64)
    expanded = _expanded_mask(original, valid_mask)
    if expanded is None:
        selected = data.reshape(-1)
        valid_count = int(data.numel())
    else:
        selected = data[_to_cpu(expanded)]
        valid_count = int(expanded.sum().item())
    finite = torch.isfinite(selected)
    finite_values = selected[finite]
    selected_count = int(selected.numel())
    # Global anomaly counts intentionally include invalid/padded locations.
    # A NaN hidden behind a mask is still a numerical defect worth surfacing.
    nan_count = int(torch.isnan(data).sum().item())
    inf_count = int(torch.isinf(data).sum().item())
    selected_nan_count = int(torch.isnan(selected).sum().item())
    selected_inf_count = int(torch.isinf(selected).sum().item())
    result: dict[str, Any] = {
        "shape": list(original.shape),
        "dtype": str(original.dtype),
        "device": str(original.device),
        "element_count": int(original.numel()),
        "valid_element_count": valid_count,
        "padding_or_invalid_element_count": int(original.numel()) - valid_count,
        "valid_mask_count": valid_count,
        "padding_count": int(original.numel()) - valid_count,
        "selected_count": selected_count,
        "finite_count": int(finite.sum().item()),
        "finite_ratio": (
            1.0 if selected_count == 0 else float(finite.sum().item()) / selected_count
        ),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "selected_nan_count": selected_nan_count,
        "selected_inf_count": selected_inf_count,
    }
    if finite_values.numel() == 0:
        result.update(
            {
                "minimum": None,
                "maximum": None,
                "mean": None,
                "standard_deviation": None,
                "variance": None,
                "l1_norm": None,
                "l2_norm": None,
                "zero_count": 0,
                "zero_ratio": None,
            }
        )
        return result
    result.update(
        {
            "minimum": float(finite_values.min().item()),
            "maximum": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "standard_deviation": float(
                finite_values.std(unbiased=False).item()
            ),
            "variance": float(finite_values.var(unbiased=False).item()),
            "l1_norm": float(finite_values.abs().sum().item()),
            "l2_norm": float(torch.linalg.vector_norm(finite_values).item()),
            "zero_count": int(finite_values.eq(0).sum().item()),
            "zero_ratio": float(finite_values.eq(0).sum().item())
            / int(finite_values.numel()),
        }
    )
    return result


def mask_statistics(mask: Tensor, *, semantics: str) -> dict[str, Any]:
    """Return counts that preserve the semantic role of a boolean mask."""

    if not torch.is_tensor(mask):
        raise TypeError("mask_statistics requires a Tensor")
    boolean = mask.detach().bool().to(device="cpu")
    count = int(boolean.numel())
    true_count = int(boolean.sum().item())
    if boolean.ndim == 0:
        all_true_groups = int(boolean.item())
        all_false_groups = int(not boolean.item())
    else:
        rows = boolean.reshape(-1, boolean.shape[-1])
        all_true_groups = int(rows.all(dim=-1).sum().item())
        all_false_groups = int((~rows).all(dim=-1).sum().item())
    return {
        "semantics": semantics,
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "device": str(mask.device),
        "element_count": count,
        "true_count": true_count,
        "false_count": count - true_count,
        "true_ratio": 1.0 if count == 0 else true_count / count,
        "all_true_last_axis_group_count": all_true_groups,
        "all_false_last_axis_group_count": all_false_groups,
    }


def _row_representations(
    value: Tensor,
    valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Reduce every leading row to a last-dimension feature centroid."""

    detached = value.detach().to(device="cpu", dtype=torch.float64)
    if detached.ndim == 0:
        return detached.reshape(1, 1), torch.ones(1, dtype=torch.bool)
    if detached.ndim == 1:
        return detached.reshape(-1, 1), torch.ones(
            detached.shape[0], dtype=torch.bool
        )
    rows = detached.shape[0]
    features = detached.shape[-1]
    flattened = detached.reshape(rows, -1, features)
    expanded = _expanded_mask(value.detach(), valid_mask)
    if expanded is None:
        weights = torch.ones_like(flattened, dtype=torch.bool)
    else:
        weights = _to_cpu(expanded).reshape(rows, -1, features)
    finite = torch.isfinite(flattened) & weights
    safe = flattened.masked_fill(~finite, 0.0)
    counts = finite.sum(dim=1)
    centroids = safe.sum(dim=1) / counts.clamp_min(1)
    row_valid = counts.gt(0).any(dim=-1)
    return centroids, row_valid


def class_conditioned_statistics(
    value: Tensor,
    labels: Tensor,
    *,
    sample_indices: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> dict[str, Any]:
    """Compare detached representations by the function-level class label."""

    if labels.ndim != 1:
        raise ObservabilityContractError("labels must have shape [B]")
    rows = 1 if value.ndim == 0 else int(value.shape[0])
    if sample_indices is None:
        if rows != labels.shape[0]:
            raise ObservabilityContractError(
                "sample_indices are required when tensor rows are packed paths"
            )
        sample_indices = torch.arange(rows, device=labels.device)
    if sample_indices.ndim != 1 or sample_indices.shape[0] != rows:
        raise ObservabilityContractError("sample_indices must have one entry per row")
    parents = sample_indices.detach().long().to(device="cpu")
    cpu_labels = labels.detach().long().to(device="cpu")
    if parents.numel() and (
        int(parents.min().item()) < 0
        or int(parents.max().item()) >= cpu_labels.shape[0]
    ):
        raise ObservabilityContractError("sample_indices exceed labels")
    centroids, row_valid = _row_representations(value, valid_mask)

    # Packed-path tensors can contain many rows for one function.  Aggregate
    # those rows within their parent function first so functions with more
    # paths cannot dominate a class statistic.
    sample_vectors: list[Tensor] = []
    sample_labels: list[int] = []
    for parent in sorted(set(int(value) for value in parents.tolist())):
        selected = parents.eq(parent) & row_valid
        if not bool(selected.any().item()):
            continue
        sample_vectors.append(centroids[selected].mean(dim=0))
        sample_labels.append(int(cpu_labels[parent].item()))
    if sample_vectors:
        centroids = torch.stack(sample_vectors, dim=0)
        row_labels = torch.tensor(sample_labels, dtype=torch.long)
        row_valid = torch.ones(len(sample_vectors), dtype=torch.bool)
    else:
        centroids = centroids.new_empty((0, centroids.shape[-1]))
        row_labels = torch.empty(0, dtype=torch.long)
        row_valid = torch.empty(0, dtype=torch.bool)

    class_rows: dict[str, Any] = {}
    class_centroids: dict[int, Tensor] = {}
    for class_id in sorted(int(item) for item in torch.unique(cpu_labels).tolist()):
        selected = row_labels.eq(class_id) & row_valid
        vectors = centroids[selected]
        if vectors.numel() == 0:
            class_rows[str(class_id)] = {
                "row_count": 0,
                "centroid_l2_norm": None,
                "within_class_variance": None,
                "mean_row_l2_norm": None,
            }
            continue
        centroid = vectors.mean(dim=0)
        class_centroids[class_id] = centroid
        centered = vectors - centroid
        class_rows[str(class_id)] = {
            "row_count": int(vectors.shape[0]),
            "parent_sample_count": int(vectors.shape[0]),
            "representation_mean": float(vectors.mean().item()),
            "representation_standard_deviation": float(
                vectors.std(unbiased=False).item()
            ),
            "centroid_l2_norm": float(torch.linalg.vector_norm(centroid).item()),
            "within_class_variance": float(centered.square().mean().item()),
            "mean_row_l2_norm": float(
                torch.linalg.vector_norm(vectors, dim=-1).mean().item()
            ),
        }
    separation: dict[str, float] = {}
    ordered = sorted(class_centroids)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            distance = torch.linalg.vector_norm(
                class_centroids[left] - class_centroids[right]
            )
            separation[f"{left}_vs_{right}_centroid_l2"] = float(distance.item())
    zero_vs_one = separation.get("0_vs_1_centroid_l2")
    class_0_count = int(class_rows.get("0", {}).get("parent_sample_count", 0))
    class_1_count = int(class_rows.get("1", {}).get("parent_sample_count", 0))
    class_0_variance = class_rows.get("0", {}).get("within_class_variance")
    class_1_variance = class_rows.get("1", {}).get("within_class_variance")
    pooled_denominator = class_0_count + class_1_count
    pooled_variance = (
        None
        if pooled_denominator == 0
        or class_0_variance is None
        or class_1_variance is None
        else (
            class_0_count * float(class_0_variance)
            + class_1_count * float(class_1_variance)
        )
        / pooled_denominator
    )
    return {
        "classes": class_rows,
        "centroid_separation": separation,
        "negative_representation_mean": class_rows.get("0", {}).get(
            "representation_mean"
        ),
        "positive_representation_mean": class_rows.get("1", {}).get(
            "representation_mean"
        ),
        "negative_within_class_variance": class_rows.get("0", {}).get(
            "within_class_variance"
        ),
        "positive_within_class_variance": class_rows.get("1", {}).get(
            "within_class_variance"
        ),
        "positive_negative_centroid_l2_distance": zero_vs_one,
        "class_0_parent_sample_count": class_0_count,
        "class_1_parent_sample_count": class_1_count,
        "class_0_representation_mean": class_rows.get("0", {}).get(
            "representation_mean"
        ),
        "class_1_representation_mean": class_rows.get("1", {}).get(
            "representation_mean"
        ),
        "between_class_centroid_l2": zero_vs_one,
        "class_0_within_class_variance": class_0_variance,
        "class_1_within_class_variance": class_1_variance,
        "pooled_within_class_variance": pooled_variance,
        "adjacent_stage_centroid_distance_retention": None,
        "adjacent_stage_within_class_variance_retention": None,
        "source": "aggregate_within_parent_sample_before_class_aggregation",
    }


def attention_statistics(
    probabilities: Tensor,
    *,
    valid_pair_mask: Tensor,
    scores: Tensor | None = None,
) -> dict[str, Any]:
    """Check support, normalization, entropy, and score/probability ranges."""

    if probabilities.ndim < 2:
        raise ObservabilityContractError("attention probabilities need a key axis")
    expanded = _expanded_mask(probabilities, valid_pair_mask)
    assert expanded is not None
    valid = expanded.detach().bool()
    probs = probabilities.detach().to(dtype=torch.float64)
    row_valid = valid.any(dim=-1)
    supported = probs.masked_fill(~valid, 0.0)
    sums = supported.sum(dim=-1)
    finite_rows = row_valid & torch.isfinite(sums)
    if bool(finite_rows.any().item()):
        normalization_error = (sums[finite_rows] - 1.0).abs()
        max_error = float(normalization_error.max().item())
        mean_error = float(normalization_error.mean().item())
    else:
        max_error = None
        mean_error = None
    masked_values = probs.masked_select(~valid)
    finite_masked_values = masked_values[torch.isfinite(masked_values)]
    masked_mass = (
        None
        if finite_masked_values.numel() != masked_values.numel()
        else float(finite_masked_values.abs().sum().item())
    )
    invalid_query_values = probs.masked_select(
        (~row_valid).unsqueeze(-1).expand_as(probs)
    )
    finite_invalid_query_values = invalid_query_values[
        torch.isfinite(invalid_query_values)
    ]
    invalid_query_mass = (
        None
        if finite_invalid_query_values.numel() != invalid_query_values.numel()
        else float(finite_invalid_query_values.abs().sum().item())
    )
    finite_supported = torch.isfinite(supported).all(dim=-1)
    entropy_rows = row_valid & finite_supported
    safe = supported.clamp_min(torch.finfo(torch.float64).tiny)
    entropy = -(supported * safe.log()).sum(dim=-1)
    entropy_values = entropy[entropy_rows]
    per_head: list[dict[str, Any]] = []
    if valid.ndim >= 4:
        # Frozen attention layouts are [..., H, Q, K].  Collapse any leading
        # batch/chunk axes while retaining one auditable row per head.
        head_count = int(valid.shape[-3])
        key_counts = valid.reshape(-1, head_count, valid.shape[-2], valid.shape[-1]).sum(
            dim=-1
        )
        for head in range(head_count):
            counts = key_counts[:, head].reshape(-1).to(dtype=torch.float64)
            per_head.append(
                {
                    "head_index": head,
                    "minimum_valid_key_count": int(counts.min().item()),
                    "maximum_valid_key_count": int(counts.max().item()),
                    "mean_valid_key_count": float(counts.mean().item()),
                    "fully_masked_query_count": int(counts.eq(0).sum().item()),
                }
            )
    valid_probabilities = probs.masked_select(valid)
    finite_valid_probabilities = valid_probabilities[
        torch.isfinite(valid_probabilities)
    ]
    all_valid_probabilities_finite = (
        finite_valid_probabilities.numel() == valid_probabilities.numel()
    )
    minimum_attention_weight = (
        None
        if finite_valid_probabilities.numel() == 0
        or not all_valid_probabilities_finite
        else float(finite_valid_probabilities.min().item())
    )
    maximum_attention_weight = (
        None
        if finite_valid_probabilities.numel() == 0
        or not all_valid_probabilities_finite
        else float(finite_valid_probabilities.max().item())
    )
    mean_entropy = (
        None
        if entropy_values.numel() == 0
        else float(entropy_values.mean().item())
    )
    return {
        "probabilities": tensor_statistics(
            probabilities, valid_mask=valid_pair_mask
        ),
        "scores": (
            None
            if scores is None
            else tensor_statistics(scores, valid_mask=valid_pair_mask)
        ),
        "valid_row_count": int(row_valid.sum().item()),
        "all_masked_row_count": int((~row_valid).sum().item()),
        "normalization_max_abs_error": max_error,
        "normalization_mean_abs_error": mean_error,
        "masked_probability_absolute_mass": masked_mass,
        "minimum_attention_weight": minimum_attention_weight,
        "maximum_attention_weight": maximum_attention_weight,
        "mean_entropy": mean_entropy,
        "attention_min_weight": minimum_attention_weight,
        "attention_max_weight": maximum_attention_weight,
        "attention_entropy": mean_entropy,
        "effective_key_count_per_head": per_head,
        "invalid_query_mass": invalid_query_mass,
        "invalid_key_mass": masked_mass,
        "valid_row_probability_sum_max_absolute_error": max_error,
        "nonfinite_valid_probability_count": int(
            valid_probabilities.numel() - finite_valid_probabilities.numel()
        ),
        "nonfinite_masked_probability_count": int(
            masked_values.numel() - finite_masked_values.numel()
        ),
        "per_head_valid_key_counts": per_head,
    }


def _diagnostic_masked_max(
    values: Tensor,
    valid_mask: Tensor,
    *,
    dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    detached = values.detach()
    valid = valid_mask.detach().bool()
    masked = detached.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    pooled, winners = masked.max(dim=dim)
    group_valid = valid.any(dim=dim)
    pooled = pooled.masked_fill(~group_valid.unsqueeze(-1), 0.0)
    winners = winners.masked_fill(~group_valid.unsqueeze(-1), -1)
    return pooled, group_valid, winners


def pooling_statistics(
    values: Tensor,
    valid_mask: Tensor,
    pooled: Tensor,
    *,
    dim: int,
    winning_indices: Tensor | None = None,
) -> dict[str, Any]:
    """Audit masked max pooling, including winner distribution and empties."""

    expected, group_valid, expected_winners = _diagnostic_masked_max(
        values, valid_mask, dim=dim
    )
    winners = expected_winners if winning_indices is None else winning_indices.detach()
    valid_winners = winners[winners.ge(0)].to(device="cpu")
    histogram = Counter(int(index) for index in valid_winners.reshape(-1).tolist())
    expanded_valid = valid_mask.detach().bool().unsqueeze(-1).expand_as(values)
    gathered_valid = torch.gather(
        expanded_valid,
        dim,
        winners.clamp_min(0).unsqueeze(dim),
    ).squeeze(dim)
    selected_padding_count = int(
        (winners.ge(0) & ~gathered_valid).sum().item()
    )
    difference = (pooled.detach() - expected).abs()
    finite_difference = difference[torch.isfinite(difference)]
    winner_histogram = {str(key): histogram[key] for key in sorted(histogram)}
    return {
        "pool_dimension": int(dim),
        "source": tensor_statistics(values, valid_mask=valid_mask),
        "pooled": tensor_statistics(pooled, valid_mask=group_valid),
        "valid_group_count": int(group_valid.sum().item()),
        "empty_group_count": int((~group_valid).sum().item()),
        "invalid_winner_count": int(winners.lt(0).sum().item()),
        "selected_padding_count": selected_padding_count,
        "masked_pooling_selected_padding": selected_padding_count > 0,
        "winner_histogram": winner_histogram,
        "masked_pool_pad_selection_count": selected_padding_count,
        "invalid_pool_winner_count": int(winners.lt(0).sum().item()),
        "pooling_argmax_position_histogram": winner_histogram,
        "recomputed_max_abs_error": (
            0.0
            if finite_difference.numel() == 0
            else float(finite_difference.max().item())
        ),
    }


def _output_field(output: Any, name: str) -> Tensor | None:
    if isinstance(output, Mapping):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    return value if torch.is_tensor(value) else None


def _first_tensor(output: Any) -> Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, Mapping):
        for item in output.values():
            if torch.is_tensor(item):
                return item
    return None


def _parameter_stage(name: str) -> str:
    prefix = name.split(".", 1)[0]
    return {
        "hpe": "H1",
        "h1_to_d1": "D1",
        "gru_branch": "D1",
        "cnn_branch": "D1",
        "dwaf": "D1",
        "d1_to_transformer": "TRANSFORMER_INPUT",
        "special_token_embedding": "TRANSFORMER_INPUT",
        "transformer": "TRANSFORMER",
        "classifier": "CLASSIFIER",
    }.get(prefix, "OTHER")


def _role_contract(role: str) -> tuple[str, str, str]:
    """Map one runtime role to its frozen observation/hook/module identity."""

    if role.startswith("EMBEDDING"):
        return "OBS-V3-002", "HOOK-V3-EMBEDDING", "EMBEDDING_OR_CBOW"
    if role.startswith("H1"):
        return "OBS-V3-003", "HOOK-V3-H1", "HPE_H1"
    if role.startswith("D1.gru"):
        observation = "OBS-V3-022" if "attention" in role else "OBS-V3-004"
        return observation, "HOOK-V3-GRU", "GRU"
    if role.startswith("D1.cnn"):
        observation = "OBS-V3-023" if "gate" in role else "OBS-V3-005"
        return observation, "HOOK-V3-CNN", "CNN"
    if role.startswith("D1.dwaf"):
        observation = "OBS-V3-024" if "attention" in role else "OBS-V3-006"
        return observation, "HOOK-V3-DWAF", "DWAF_D1"
    if role.startswith("TRANSFORMER"):
        if "attention" in role:
            observation = "OBS-V3-025"
        elif "layer_1" in role:
            observation = "OBS-V3-007"
        elif "layer_3" in role:
            observation = "OBS-V3-008"
        else:
            observation = "OBS-V3-009"
        return observation, "HOOK-V3-TRANSFORMER", "STRUCTURAL_TRANSFORMER"
    if "segment_attention" in role:
        return "OBS-V3-026", "HOOK-V3-C1", "C1_CLASSIFIER"
    if "token_to_path" in role:
        return "OBS-V3-010", "HOOK-V3-C1", "C1_CLASSIFIER"
    if "path_to_segment" in role:
        return "OBS-V3-011", "HOOK-V3-C1", "C1_CLASSIFIER"
    if role.startswith("CLASSIFIER"):
        observation = "OBS-V3-015" if "logit" in role else "OBS-V3-014"
        return observation, "HOOK-V3-C1", "C1_CLASSIFIER"
    return "OBS-V3-001", "HOOK-V3-INPUT", "INPUT_AND_MASK"


@dataclass(frozen=True)
class _HookSpec:
    name: str
    module: nn.Module
    callback: Callable[..., None]
    with_kwargs: bool = False
    pre: bool = False


class Task34ObservabilitySession(AbstractContextManager["Task34ObservabilitySession"]):
    """One removable observation surface for direct and chunked Task34 calls."""

    def __init__(
        self,
        model: nn.Module,
        *,
        batch: Mapping[str, Any] | None = None,
        record_identity: Mapping[str, Any] | None = None,
        enabled: bool = True,
        strict_six_layers: bool = True,
    ) -> None:
        self.model = model
        self.batch = batch
        self.record_identity = dict(record_identity or {})
        self.enabled = bool(enabled)
        self.strict_six_layers = bool(strict_six_layers)
        self._handles: list[Any] = []
        self._active = False
        self._path_cursors: MutableMapping[str, int] = defaultdict(int)
        self._word_embedding_call = 0
        self._pending_attention: dict[int, dict[str, Any]] = defaultdict(dict)
        self._tensor_records: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)
        self._mask_records: dict[str, dict[str, Any]] = {}
        self._mask_health: dict[str, Any] = {}
        self._attention_records: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)
        self._pooling_records: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)
        self._hook_counts: Counter[str] = Counter()
        self._gradient_record: dict[str, Any] | None = None
        self._parameter_records: MutableMapping[str, dict[str, Any]] = {}
        self._update_record: dict[str, Any] | None = None
        self._loss_records: MutableMapping[str, dict[str, Any]] = {}
        self._classifier_summary: dict[str, Any] = {}
        self._baseline: dict[str, Tensor] | None = None
        self._forward_count = 0

    @property
    def labels(self) -> Tensor | None:
        if self.batch is None:
            return None
        labels = self.batch.get("labels")
        return labels if torch.is_tensor(labels) else None

    def _identity_for_role(self, role: str) -> dict[str, Any]:
        observation_id, hook_id, module_id = _role_contract(role)
        return {
            **self.record_identity,
            "observation_id": observation_id,
            "hook_id": hook_id,
            "module_id": module_id,
            "tensor_role": role,
        }

    def _packed_parent_indices(self) -> Tensor | None:
        if self.batch is None:
            return None
        value = self.batch.get("packed_path_to_sample")
        if value is None:
            value = self.batch.get("path_to_sample")
        return value if torch.is_tensor(value) else None

    def _packed_padding_mask(self) -> Tensor | None:
        if self.batch is None:
            return None
        value = self.batch.get("packed_padding_mask")
        if value is None:
            candidate = self.batch.get("padding_mask")
            if torch.is_tensor(candidate) and candidate.ndim == 2:
                value = candidate
        return value.bool() if torch.is_tensor(value) else None

    def _path_context(
        self,
        role: str,
        value: Tensor,
        *,
        mask_kind: str = "token",
    ) -> tuple[Tensor | None, Tensor | None]:
        if value.ndim == 0:
            return None, None
        rows = int(value.shape[0])
        cursor = self._path_cursors[role]
        parents = self._packed_parent_indices()
        padding = self._packed_padding_mask()
        parent_slice: Tensor | None = None
        valid: Tensor | None = None
        if parents is not None and cursor + rows <= parents.shape[0]:
            parent_slice = parents[cursor : cursor + rows]
        if padding is not None and cursor + rows <= padding.shape[0]:
            token_valid = ~padding[cursor : cursor + rows]
            if mask_kind == "token":
                valid = token_valid
            elif mask_kind == "pair":
                valid = token_valid[:, None, :, None] & token_valid[:, None, None, :]
            elif mask_kind == "row":
                valid = token_valid.any(dim=-1)
        self._path_cursors[role] = cursor + rows
        return parent_slice, valid

    def _record_tensor(
        self,
        role: str,
        value: Tensor,
        *,
        valid_mask: Tensor | None = None,
        sample_indices: Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        record: dict[str, Any] = {
            "record_identity": self._identity_for_role(role),
            "statistics": tensor_statistics(value, valid_mask=valid_mask),
            "metadata": dict(metadata or {}),
        }
        labels = self.labels
        if labels is not None and value.ndim > 0:
            record["class_conditioned"] = class_conditioned_statistics(
                value,
                labels,
                sample_indices=sample_indices,
                valid_mask=valid_mask,
            )
        if "gate" in role.lower():
            gate = _to_cpu(value).to(dtype=torch.float64)
            finite_gate = gate[torch.isfinite(gate)]
            if finite_gate.numel() > 0:
                bounded = finite_gate.clamp(0.0, 1.0)
                safe = bounded.clamp(
                    torch.finfo(torch.float64).tiny,
                    1.0 - torch.finfo(torch.float64).eps,
                )
                entropy = -(
                    safe * safe.log() + (1.0 - safe) * (1.0 - safe).log()
                )
                record["gate_statistics"] = {
                    "gate_below_epsilon_ratio": float(
                        bounded.le(1e-6).float().mean().item()
                    ),
                    "gate_above_one_minus_epsilon_ratio": float(
                        bounded.ge(1.0 - 1e-6).float().mean().item()
                    ),
                    "gate_bernoulli_entropy": float(entropy.mean().item()),
                }
        self._tensor_records[role].append(record)

    def _record_path_tensor(
        self,
        role: str,
        value: Tensor,
        *,
        mask_kind: str = "token",
        valid_mask: Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        parents, inferred_mask = self._path_context(
            role, value, mask_kind=mask_kind
        )
        self._record_tensor(
            role,
            value,
            valid_mask=valid_mask if valid_mask is not None else inferred_mask,
            sample_indices=parents,
            metadata=metadata,
        )

    def record_attention(
        self,
        role: str,
        *,
        probabilities: Tensor,
        valid_pair_mask: Tensor,
        scores: Tensor | None = None,
        output: Tensor | None = None,
        dropped_probabilities: Tensor | None = None,
        score_components: Mapping[str, Tensor] | None = None,
        sample_indices: Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record attention diagnostics without retaining live tensors."""

        if not self.enabled:
            return
        record = attention_statistics(
            probabilities, valid_pair_mask=valid_pair_mask, scores=scores
        )
        record["metadata"] = dict(metadata or {})
        record["record_identity"] = self._identity_for_role(role)
        record["profile_family"] = str(
            getattr(self.model, "family_id", "UNKNOWN")
        )
        record["attention_layer_index"] = (metadata or {}).get("layer_index")
        record["attention_head_index"] = None
        structural_bias = (score_components or {}).get("structural_bias")
        if torch.is_tensor(structural_bias):
            structural_statistics = tensor_statistics(
                structural_bias, valid_mask=valid_pair_mask
            )
            record.update(
                {
                    "structural_bias_minimum": structural_statistics["minimum"],
                    "structural_bias_maximum": structural_statistics["maximum"],
                    "structural_bias_mean": structural_statistics["mean"],
                    "structural_bias_standard_deviation": structural_statistics[
                        "standard_deviation"
                    ],
                }
            )
        if self.labels is not None and probabilities.ndim > 0:
            record["class_conditioned"] = class_conditioned_statistics(
                probabilities,
                self.labels,
                sample_indices=sample_indices,
                valid_mask=valid_pair_mask,
            )
        self._attention_records[role].append(record)
        self._record_tensor(
            f"{role}.probabilities",
            probabilities,
            valid_mask=valid_pair_mask,
            sample_indices=sample_indices,
            metadata={"attention_component": "pre_dropout_probabilities"},
        )
        if scores is not None:
            self._record_tensor(
                f"{role}.scores",
                scores,
                valid_mask=valid_pair_mask,
                sample_indices=sample_indices,
                metadata={"attention_component": "pre_softmax_scores"},
            )
        for name, component in (score_components or {}).items():
            self._record_tensor(
                f"{role}.{name}",
                component,
                valid_mask=valid_pair_mask,
                sample_indices=sample_indices,
                metadata={"attention_component": name},
            )
        if dropped_probabilities is not None:
            self._record_tensor(
                f"{role}.post_dropout_probabilities",
                dropped_probabilities,
                valid_mask=valid_pair_mask,
                sample_indices=sample_indices,
                metadata={"normalization_expected": False},
            )
        if output is not None:
            self._record_tensor(
                f"{role}.output",
                output,
                sample_indices=sample_indices,
                metadata={"attention_component": "projected_output"},
            )

    def record_pooling(
        self,
        role: str,
        *,
        values: Tensor,
        valid_mask: Tensor,
        pooled: Tensor,
        dim: int,
        winning_indices: Tensor | None = None,
        sample_indices: Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a detached masked-pooling recomputation and its provenance."""

        if not self.enabled:
            return
        record = pooling_statistics(
            values,
            valid_mask,
            pooled,
            dim=dim,
            winning_indices=winning_indices,
        )
        record["metadata"] = dict(metadata or {})
        record["record_identity"] = self._identity_for_role(role)
        self._pooling_records[role].append(record)
        group_valid = valid_mask.detach().bool().any(dim=dim)
        self._record_tensor(
            f"{role}.pooled",
            pooled,
            valid_mask=group_valid,
            sample_indices=sample_indices,
            metadata=metadata,
        )

    def _capture_batch_masks(self) -> None:
        if not self.enabled or self.batch is None:
            return

        def add(role: str, key: str, semantics: str, *, ndim: int | None = None) -> None:
            value = self.batch.get(key)
            if torch.is_tensor(value) and (ndim is None or value.ndim == ndim):
                self._mask_records[role] = mask_statistics(value, semantics=semantics)

        add("token_valid_mask", "token_valid_mask", "True means a real hierarchy token")
        add("token_padding_mask", "token_padding_mask", "True means hierarchy padding")
        add(
            "sample_path_valid_mask",
            "path_valid_mask",
            "True means a path slot exists for the sample, shape [B,P]",
            ndim=2,
        )
        add(
            "segment_path_valid_mask",
            "path_segment_valid_mask",
            "True means a path is valid inside a segment, shape [B,S,P]",
            ndim=3,
        )
        path_valid = self.batch.get("path_valid_mask")
        if torch.is_tensor(path_valid) and path_valid.ndim == 3:
            self._mask_records["segment_path_valid_mask"] = mask_statistics(
                path_valid,
                semantics="True means a path is valid inside a segment, shape [B,S,P]",
            )
        add("segment_valid_mask", "segment_valid_mask", "True means a segment exists")
        packed = self._packed_padding_mask()
        if packed is not None:
            self._mask_records["packed_token_padding_mask"] = mask_statistics(
                packed, semantics="True means packed-path token padding"
            )
            self._mask_records["packed_token_valid_mask"] = mask_statistics(
                ~packed, semantics="True means a real packed-path token"
            )
        add(
            "packed_attention_mask",
            "packed_attention_mask",
            "True means an attention pair is permitted",
        )
        token_valid = self.batch.get("token_valid_mask")
        segment_paths = self.batch.get("path_segment_valid_mask")
        segment_valid = self.batch.get("segment_valid_mask")
        if torch.is_tensor(token_valid):
            valid = token_valid.detach().bool()
            self._mask_health["token_valid_count"] = int(valid.sum().item())
            self._mask_health["token_valid_ratio"] = float(valid.float().mean().item())
            self._mask_health["all_padding_path_count"] = int(
                (~valid.any(dim=-1)).sum().item()
            )
            if valid.ndim >= 4:
                self._mask_health["empty_segment_count_from_tokens"] = int(
                    (~valid.any(dim=(-1, -2))).sum().item()
                )
                self._mask_health["empty_function_count_from_tokens"] = int(
                    (~valid.any(dim=(-1, -2, -3))).sum().item()
                )
        sample_paths = self.batch.get("path_valid_mask")
        if torch.is_tensor(sample_paths) and sample_paths.ndim == 2:
            self._mask_health["path_valid_count"] = int(
                sample_paths.detach().bool().sum().item()
            )
            self._mask_health["path_valid_ratio"] = float(
                sample_paths.detach().bool().float().mean().item()
            )
            self._mask_health["sample_path_valid_ratio"] = float(
                sample_paths.detach().bool().float().mean().item()
            )
        if torch.is_tensor(segment_paths):
            valid = segment_paths.detach().bool()
            self._mask_health.setdefault(
                "path_valid_count", int(valid.sum().item())
            )
            self._mask_health.setdefault(
                "path_valid_ratio", float(valid.float().mean().item())
            )
            self._mask_health["segment_path_valid_ratio"] = float(
                valid.float().mean().item()
            )
            self._mask_health["empty_segment_count"] = int(
                (~valid.any(dim=-1)).sum().item()
            )
        if torch.is_tensor(segment_valid):
            valid = segment_valid.detach().bool()
            self._mask_health["segment_valid_count"] = int(valid.sum().item())
            self._mask_health["segment_valid_ratio"] = float(
                valid.float().mean().item()
            )
            self._mask_health["empty_function_count"] = int(
                (~valid.any(dim=-1)).sum().item()
            )
        packed = self._packed_padding_mask()
        if packed is not None:
            self._mask_health["packed_token_valid_ratio"] = float(
                (~packed).float().mean().item()
            )
            self._mask_health["all_padding_token_row_count"] = int(
                packed.all(dim=-1).sum().item()
            )

    def _hook_hpe(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is None:
            return
        self._hook_counts["H1.output"] += 1
        self._record_path_tensor("H1.output", tensor)
        observed_method = getattr(_module, "observability", None)
        if callable(observed_method):
            observed = observed_method()
            for name in (
                "semantic_projected",
                "structural_base",
                "delta_effect",
                "ref_vector",
                "structural_with_ref",
                "vector_gate",
            ):
                value = observed.get(name) if isinstance(observed, Mapping) else None
                if torch.is_tensor(value):
                    self._record_path_tensor(f"H1.{name}", value)

    def _hook_word_embedding(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is None:
            return
        role = (
            "EMBEDDING.cbow_token_output"
            if self._word_embedding_call % 2 == 0
            else "EMBEDDING.cbow_ref_definition_output"
        )
        self._word_embedding_call += 1
        self._hook_counts[role] += 1
        self._record_path_tensor(role, tensor)

    def _simple_path_hook(
        self,
        role: str,
        *,
        output_index: int | None = None,
        mask_kind: str = "token",
        metadata: Mapping[str, Any] | None = None,
    ) -> Callable[[nn.Module, Any, Any], None]:
        def callback(_module: nn.Module, _inputs: Any, output: Any) -> None:
            selected = output
            if output_index is not None and isinstance(output, (tuple, list)):
                selected = output[output_index]
            tensor = _first_tensor(selected)
            if tensor is not None:
                self._hook_counts[role] += 1
                self._record_path_tensor(
                    role, tensor, mask_kind=mask_kind, metadata=metadata
                )

        return callback

    def _hook_gru_branch(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 3:
            return
        weighted, attention, pooled = output[:3]
        for role, value, kind in (
            ("D1.gru.weighted_tokens", weighted, "token"),
            ("D1.gru.token_attention", attention, "token"),
            ("D1.gru.pooled", pooled, "row"),
        ):
            if torch.is_tensor(value):
                self._hook_counts[role] += 1
                self._record_path_tensor(role, value, mask_kind=kind)
        if torch.is_tensor(attention):
            parents, valid = self._path_context(
                "D1.gru.attention.parents", attention, mask_kind="token"
            )
            if valid is None:
                valid = torch.ones_like(attention, dtype=torch.bool)
            self.record_attention(
                "D1.gru.attention",
                probabilities=attention,
                valid_pair_mask=valid,
                output=weighted if torch.is_tensor(weighted) else None,
                sample_indices=parents,
                metadata={"attention_axis": "token"},
            )
            self._hook_counts["D1.gru.attention"] += 1

    def _hook_cnn_branch(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return
        for role, value, kind in (
            ("D1.cnn.gated_tokens", output[0], "token"),
            ("D1.cnn.channel_gate", output[1], "row"),
        ):
            if torch.is_tensor(value):
                self._hook_counts[role] += 1
                self._record_path_tensor(role, value, mask_kind=kind)

    def _hook_cnn_convolution(
        self, _module: nn.Module, _inputs: Any, output: Any
    ) -> None:
        tensor = _first_tensor(output)
        if tensor is None:
            return
        # Conv1d is [N,H,T], while every other token observation is [N,T,H].
        # Normalize the diagnostic view so the packed token mask has one
        # unambiguous axis and no legacy tensor is modified.
        diagnostic = tensor.transpose(1, 2) if tensor.ndim == 3 else tensor
        role = "D1.cnn.convolution_output"
        self._hook_counts[role] += 1
        self._record_path_tensor(role, diagnostic)

    def _hook_dwaf(self, module: nn.Module, _inputs: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return
        fused, probabilities = output[:2]
        if not (torch.is_tensor(fused) and torch.is_tensor(probabilities)):
            return
        parents, pair_mask = self._path_context(
            "D1.dwaf.attention", probabilities, mask_kind="pair"
        )
        if pair_mask is None:
            pair_mask = torch.ones_like(probabilities, dtype=torch.bool)
        observed_method = getattr(module, "observability", None)
        observed = observed_method() if callable(observed_method) else {}
        q = observed.get("q") if isinstance(observed, Mapping) else None
        k = observed.get("k") if isinstance(observed, Mapping) else None
        scores: Tensor | None = None
        if torch.is_tensor(q) and torch.is_tensor(k):
            heads = int(getattr(module, "num_heads", probabilities.shape[1]))
            width = int(q.shape[-1])
            head_dim = width // heads
            qh = q.view(q.shape[0], q.shape[1], heads, head_dim).transpose(1, 2)
            kh = k.view(k.shape[0], k.shape[1], heads, head_dim).transpose(1, 2)
            scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(head_dim)
        self.record_attention(
            "D1.dwaf.attention",
            probabilities=probabilities,
            valid_pair_mask=pair_mask,
            scores=scores,
            output=fused,
            sample_indices=parents,
            metadata={"sources": "original_Q, gru_K, cnn_V", "head_count": probabilities.shape[1]},
        )
        for name in ("q", "k", "v", "context", "concatenated", "fused"):
            value = observed.get(name) if isinstance(observed, Mapping) else None
            if torch.is_tensor(value):
                self._record_path_tensor(f"D1.dwaf.{name}", value)
        self._hook_counts["D1.dwaf.attention"] += 1

    def _hook_transformer_pre(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        structural = args[0] if args else kwargs.get("structural_input")
        token_embeddings = getattr(structural, "token_embeddings", None)
        if torch.is_tensor(token_embeddings):
            self._hook_counts["TRANSFORMER.input"] += 1
            padding = getattr(structural, "padding_mask", None)
            self._record_path_tensor(
                "TRANSFORMER.input",
                token_embeddings,
                valid_mask=(~padding.bool()) if torch.is_tensor(padding) else None,
            )

    def _hook_attention_pre(
        self,
        layer_index: int,
    ) -> Callable[[nn.Module, tuple[Any, ...], Mapping[str, Any]], None]:
        def callback(
            _module: nn.Module,
            _args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> None:
            self._pending_attention[layer_index]["structural"] = kwargs.get(
                "structural_input"
            )

        return callback

    def _hook_qkv(
        self, layer_index: int
    ) -> Callable[[nn.Module, Any, Any], None]:
        def callback(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                self._pending_attention[layer_index]["qkv"] = tensor.detach()
                self._record_path_tensor(
                    f"TRANSFORMER.layer_{layer_index + 1}.qkv_projection", tensor
                )

        return callback

    def _hook_relation_bias(
        self, layer_index: int
    ) -> Callable[[nn.Module, Any, Any], None]:
        def callback(_module: nn.Module, inputs: Any, output: Any) -> None:
            if isinstance(output, Mapping):
                self._pending_attention[layer_index]["bias_parts"] = {
                    name: value.detach()
                    for name, value in output.items()
                    if torch.is_tensor(value)
                }
            if inputs:
                self._pending_attention[layer_index]["structural"] = inputs[0]

        return callback

    def _hook_attention_dropout(
        self, layer_index: int, attention_module: nn.Module
    ) -> Callable[[nn.Module, Any, Any], None]:
        def callback(_module: nn.Module, inputs: Any, output: Any) -> None:
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return
            probabilities = inputs[0]
            pending = self._pending_attention[layer_index]
            structural = pending.get("structural")
            valid_pair_method = getattr(structural, "valid_pair_mask", None)
            if callable(valid_pair_method):
                # Includes both padding and any restrictive structural
                # attention_mask from the frozen Transformer input contract.
                pair_mask = valid_pair_method().bool().unsqueeze(1)
            else:
                pair_mask = torch.ones_like(probabilities, dtype=torch.bool)
            qkv = pending.get("qkv")
            raw_scores: Tensor | None = None
            biased_scores: Tensor | None = None
            components: dict[str, Tensor] = {}
            if torch.is_tensor(qkv):
                config = getattr(attention_module, "config", None)
                heads = int(getattr(config, "num_heads", probabilities.shape[1]))
                head_dim = int(getattr(config, "head_dim", qkv.shape[-1] // (3 * heads)))
                query, key, _value = qkv.chunk(3, dim=-1)
                query = query.view(query.shape[0], query.shape[1], heads, head_dim).transpose(1, 2)
                key = key.view(key.shape[0], key.shape[1], heads, head_dim).transpose(1, 2)
                raw_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
                components["raw_qk_scores"] = raw_scores
            bias_parts = pending.get("bias_parts", {})
            if isinstance(bias_parts, Mapping) and bias_parts:
                combined: Tensor | None = None
                for name, value in bias_parts.items():
                    if torch.is_tensor(value):
                        components[name] = value
                        combined = value if combined is None else combined + value
                if combined is not None:
                    components["combined_structural_bias"] = combined
                    if raw_scores is not None:
                        biased_scores = raw_scores + combined
            scores = biased_scores if biased_scores is not None else raw_scores
            parents, _ = self._path_context(
                f"TRANSFORMER.layer_{layer_index + 1}.attention.parents",
                probabilities,
                mask_kind="pair",
            )
            role = f"TRANSFORMER.layer_{layer_index + 1}.attention"
            self.record_attention(
                role,
                probabilities=probabilities,
                valid_pair_mask=pair_mask,
                scores=scores,
                dropped_probabilities=output,
                score_components=components,
                sample_indices=parents,
                metadata={"layer_index_zero_based": layer_index},
            )
            self._hook_counts[role] += 1

        return callback

    def _hook_self_attention_output(
        self, layer_index: int
    ) -> Callable[[nn.Module, Any, Any], None]:
        role = f"TRANSFORMER.layer_{layer_index + 1}.attention_output"
        return self._simple_path_hook(role, output_index=0)

    def _hook_transformer_layer(
        self, layer_index: int
    ) -> Callable[[nn.Module, Any, Any], None]:
        role = f"TRANSFORMER.layer_{layer_index + 1}.output"
        return self._simple_path_hook(
            role,
            output_index=0,
            metadata={"layer_index_zero_based": layer_index},
        )

    def _hook_classifier(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        if len(args) < 2 or not torch.is_tensor(args[0]) or not torch.is_tensor(args[1]):
            return
        hierarchy = args[0]
        token_valid = args[1].bool()
        if hierarchy.ndim != 5 or token_valid.shape != hierarchy.shape[:-1]:
            return
        batch_indices = torch.arange(hierarchy.shape[0], device=hierarchy.device)
        self._record_tensor(
            "CLASSIFIER.token_representations",
            hierarchy,
            valid_mask=token_valid,
            sample_indices=batch_indices,
        )
        supplied_paths = kwargs.get("path_valid_mask")
        effective_token_valid = token_valid
        if torch.is_tensor(supplied_paths):
            effective_token_valid = token_valid & supplied_paths.bool().unsqueeze(-1)
        path_vectors, actual_paths, token_winners = _diagnostic_masked_max(
            hierarchy, effective_token_valid, dim=3
        )
        self.record_pooling(
            "CLASSIFIER.token_to_path_max",
            values=hierarchy,
            valid_mask=effective_token_valid,
            pooled=path_vectors,
            dim=3,
            winning_indices=token_winners,
            sample_indices=batch_indices,
            metadata={"causal_in_frozen_classifier": True},
        )
        segment_vectors, actual_segments, path_winners = _diagnostic_masked_max(
            path_vectors, actual_paths, dim=2
        )
        supplied_segments = kwargs.get("segment_valid_mask")
        if torch.is_tensor(supplied_segments):
            actual_segments = actual_segments & supplied_segments.bool()
            segment_vectors = segment_vectors.masked_fill(
                ~actual_segments.unsqueeze(-1), 0.0
            )
            path_winners = path_winners.masked_fill(
                ~actual_segments.unsqueeze(-1), -1
            )
        self.record_pooling(
            "CLASSIFIER.path_to_segment_max",
            values=path_vectors,
            valid_mask=actual_paths,
            pooled=segment_vectors,
            dim=2,
            winning_indices=path_winners,
            sample_indices=batch_indices,
            metadata={"causal_in_frozen_classifier": True},
        )
        self._mask_records["classifier_segment_path_valid_mask"] = mask_statistics(
            actual_paths,
            semantics="classifier output: valid path inside [B,S,P] hierarchy",
        )
        self._mask_records["classifier_segment_valid_mask"] = mask_statistics(
            actual_segments,
            semantics="classifier output: segment survives hierarchy masks",
        )
        for name in (
            "segment_logits",
            "segment_probabilities",
            "segment_attention_scores",
            "segment_attention_weights",
            "final_logits",
            "final_probability",
        ):
            value = _output_field(output, name)
            if value is not None:
                mask: Tensor | None = None
                if name.startswith("segment_"):
                    mask = actual_segments
                self._record_tensor(
                    f"CLASSIFIER.{name}",
                    value,
                    valid_mask=mask,
                    sample_indices=batch_indices,
                    metadata={"causal_in_frozen_classifier": True},
                )
        weights = _output_field(output, "segment_attention_weights")
        if torch.is_tensor(weights) and weights.shape == actual_segments.shape:
            self.record_attention(
                "CLASSIFIER.segment_attention",
                probabilities=weights,
                valid_pair_mask=actual_segments,
                sample_indices=batch_indices,
                metadata={"attention_axis": "segment"},
            )
            self._hook_counts["CLASSIFIER.segment_attention"] += 1
            derived_function = (
                segment_vectors.detach() * weights.detach().unsqueeze(-1)
            ).sum(dim=1)
            self._record_tensor(
                "CLASSIFIER.derived_weighted_segment_representation",
                derived_function,
                sample_indices=batch_indices,
                metadata={
                    "diagnostic_only": True,
                    "causal_in_frozen_classifier": False,
                    "reason": "frozen C1 aggregates segment probabilities, not this vector",
                },
            )
        final_probability = _output_field(output, "final_probability")
        final_logits = _output_field(output, "final_logits")
        classifier_parameters: dict[str, Any] = {}
        for name, parameter in _module.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                classifier_parameters[name] = tensor_statistics(parameter)
        decision_logits: Tensor | None = final_logits
        if torch.is_tensor(final_logits) and final_logits.ndim == 2:
            if final_logits.shape[-1] == 2:
                decision_logits = final_logits[:, 1] - final_logits[:, 0]
            elif final_logits.shape[-1] == 1:
                decision_logits = final_logits[:, 0]
        classifier_weight_l2 = math.sqrt(
            sum(
                float(row["l2_norm"]) ** 2
                for name, row in classifier_parameters.items()
                if name.endswith("weight") and row["l2_norm"] is not None
            )
        )
        signed_bias_values = [
            float(row["mean"])
            for name, row in classifier_parameters.items()
            if name.endswith("bias") and row["mean"] is not None
        ]
        probability_statistics = (
            None
            if final_probability is None
            else tensor_statistics(final_probability)
        )
        decision_logit_statistics = (
            None
            if decision_logits is None
            else tensor_statistics(decision_logits)
        )
        self._classifier_summary = {
            "family_profile": str(getattr(self.model, "family_id", "UNKNOWN")),
            "parameters": classifier_parameters,
            "classifier_weight_l2": classifier_weight_l2,
            "classifier_weight_l2_norm": classifier_weight_l2,
            "classifier_signed_bias": (
                None
                if not signed_bias_values
                else math.fsum(signed_bias_values)
            ),
            "classifier_bias": {
                name: row
                for name, row in classifier_parameters.items()
                if name.endswith("bias")
            },
            "prediction_positive_rate": (
                None
                if final_probability is None or final_probability.numel() == 0
                # Protocol V3 freezes equality at 0.5 to class zero.
                else float(final_probability.detach().gt(0.5).float().mean().item())
            ),
            "probability_statistics": probability_statistics,
            "decision_logit_statistics": decision_logit_statistics,
            "probability_mean": (
                None if probability_statistics is None else probability_statistics["mean"]
            ),
            "probability_standard_deviation": (
                None
                if probability_statistics is None
                else probability_statistics["standard_deviation"]
            ),
            "logit_mean": (
                None
                if decision_logit_statistics is None
                else decision_logit_statistics["mean"]
            ),
            "logit_standard_deviation": (
                None
                if decision_logit_statistics is None
                else decision_logit_statistics["standard_deviation"]
            ),
        }
        self._hook_counts["CLASSIFIER.output"] += 1

    def _hook_specs(self) -> list[_HookSpec]:
        specs: list[_HookSpec] = []

        def add(
            name: str,
            module: Any,
            callback: Callable[..., None],
            *,
            with_kwargs: bool = False,
            pre: bool = False,
        ) -> None:
            if isinstance(module, nn.Module):
                specs.append(_HookSpec(name, module, callback, with_kwargs, pre))

        hpe = getattr(self.model, "hpe", None)
        add("H1", hpe, self._hook_hpe)
        add("word_embedding", getattr(hpe, "word_embedding", None), self._hook_word_embedding)
        add(
            "h1_to_d1",
            getattr(self.model, "h1_to_d1", None),
            self._simple_path_hook("D1.input_projection"),
        )
        gru = getattr(self.model, "gru_branch", None)
        add("gru_branch", gru, self._hook_gru_branch)
        add(
            "gru_hidden",
            getattr(gru, "dropout", None),
            self._simple_path_hook("D1.gru.hidden_sequence"),
        )
        cnn = getattr(self.model, "cnn_branch", None)
        add("cnn_branch", cnn, self._hook_cnn_branch)
        add(
            "cnn_conv",
            getattr(cnn, "conv", None),
            self._hook_cnn_convolution,
        )
        add("dwaf", getattr(self.model, "dwaf", None), self._hook_dwaf)
        add(
            "d1_to_transformer",
            getattr(self.model, "d1_to_transformer", None),
            self._simple_path_hook("TRANSFORMER.projected_input"),
        )
        add(
            "special_token_embedding",
            getattr(self.model, "special_token_embedding", None),
            self._simple_path_hook("TRANSFORMER.special_token_embedding"),
        )
        transformer = getattr(self.model, "transformer", None)
        add(
            "transformer_input",
            transformer,
            self._hook_transformer_pre,
            with_kwargs=True,
            pre=True,
        )
        layers = getattr(transformer, "layers", None)
        if layers is not None:
            layer_list = list(layers)
            if self.strict_six_layers and len(layer_list) != REQUIRED_TRANSFORMER_LAYER_COUNT:
                raise ObservabilityContractError(
                    f"Task39S requires all six Transformer layers, found {len(layer_list)}"
                )
            for index, layer in enumerate(layer_list):
                add(f"layer_{index + 1}", layer, self._hook_transformer_layer(index))
                attention = getattr(layer, "self_attention", None)
                add(
                    f"attention_{index + 1}_pre",
                    attention,
                    self._hook_attention_pre(index),
                    with_kwargs=True,
                    pre=True,
                )
                add(
                    f"attention_{index + 1}_output",
                    attention,
                    self._hook_self_attention_output(index),
                )
                add(
                    f"qkv_{index + 1}",
                    getattr(attention, "qkv_projection", None),
                    self._hook_qkv(index),
                )
                add(
                    f"bias_{index + 1}",
                    getattr(attention, "relation_bias", None),
                    self._hook_relation_bias(index),
                )
                add(
                    f"attention_dropout_{index + 1}",
                    getattr(attention, "attention_dropout", None),
                    self._hook_attention_dropout(index, attention),
                )
        add(
            "classifier",
            getattr(self.model, "classifier", None),
            self._hook_classifier,
            with_kwargs=True,
        )
        return specs

    def _register(self, spec: _HookSpec) -> Any:
        if spec.pre:
            if spec.with_kwargs:
                return spec.module.register_forward_pre_hook(
                    spec.callback, with_kwargs=True
                )
            return spec.module.register_forward_pre_hook(spec.callback)
        if spec.with_kwargs:
            return spec.module.register_forward_hook(spec.callback, with_kwargs=True)
        return spec.module.register_forward_hook(spec.callback)

    def __enter__(self) -> "Task34ObservabilitySession":
        if self._active:
            raise ObservabilityContractError("observation session is already active")
        self._active = True
        self._path_cursors.clear()
        self._word_embedding_call = 0
        self._pending_attention.clear()
        self._capture_batch_masks()
        if self.enabled:
            try:
                specs = self._hook_specs()
                registered_names = {spec.name for spec in specs}
                required_names = {
                    "H1",
                    "word_embedding",
                    "h1_to_d1",
                    "gru_branch",
                    "cnn_branch",
                    "dwaf",
                    "d1_to_transformer",
                    "transformer_input",
                    "classifier",
                }
                for index in range(1, REQUIRED_TRANSFORMER_LAYER_COUNT + 1):
                    required_names.update(
                        {
                            f"layer_{index}",
                            f"attention_{index}_pre",
                            f"attention_{index}_output",
                            f"qkv_{index}",
                            f"bias_{index}",
                            f"attention_dropout_{index}",
                        }
                    )
                missing = sorted(required_names - registered_names)
                if missing:
                    raise ObservabilityContractError(
                        "required Task34 observability hooks missing: "
                        + ", ".join(missing)
                    )
                for spec in specs:
                    self._handles.append(self._register(spec))
            except BaseException:
                self.remove()
                raise
        self._forward_count += 1
        return self

    def remove(self) -> None:
        """Remove every hook; safe to call repeatedly and after exceptions."""

        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        # Do not retain detached GPU activations or StructuralTransformerInput
        # objects after the forward-only observation scope.
        self._pending_attention.clear()
        self._active = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.remove()
        return False

    def record_loss(
        self,
        loss: Tensor,
        *,
        name: str = "loss",
        unreduced_loss: Tensor | None = None,
        labels: Tensor | None = None,
        logical_loss: float | None = None,
    ) -> dict[str, Any]:
        record = tensor_statistics(loss) if self.enabled else {"enabled": False}
        if self.enabled:
            if unreduced_loss is not None:
                if labels is None or labels.ndim != 1:
                    raise ObservabilityContractError(
                        "unreduced loss requires one label per sample"
                    )
                values = _to_cpu(unreduced_loss).to(dtype=torch.float64).reshape(-1)
                targets = _to_cpu(labels).long().reshape(-1)
                if values.shape != targets.shape:
                    raise ObservabilityContractError(
                        "unreduced loss and labels must have identical shape"
                    )
                for class_id in (0, 1):
                    selected = values[targets.eq(class_id)]
                    record[f"class_{class_id}_loss_mean"] = (
                        None if selected.numel() == 0 else float(selected.mean().item())
                    )
                    record[f"class_{class_id}_loss_variance"] = (
                        None
                        if selected.numel() == 0
                        else float(selected.var(unbiased=False).item())
                    )
                recomputed = float(values.mean().item())
                effective = (
                    float(loss.detach().cpu())
                    if logical_loss is None
                    else float(logical_loss)
                )
                record["effective_total_loss"] = effective
                record["logical_loss_recomputed"] = recomputed
                record["logical_loss_recomputation_error"] = abs(
                    recomputed - effective
                )
            self._loss_records[name] = record
        return record

    def record_parameters(self, *, name: str = "parameters") -> dict[str, Any]:
        """Record detached parameter distributions, grouped by frozen stage."""

        if not self.enabled:
            return {"enabled": False}
        rows: dict[str, Any] = {}
        stages: MutableMapping[str, list[float]] = defaultdict(list)
        for parameter_name, parameter in self.model.named_parameters():
            statistics = tensor_statistics(parameter)
            rows[parameter_name] = {
                "stage": _parameter_stage(parameter_name),
                "parameter_name_or_module": parameter_name,
                "requires_grad": bool(parameter.requires_grad),
                "current_parameter_sha256": _tensor_sha256(parameter),
                "statistics": statistics,
            }
            norm = statistics["l2_norm"]
            if norm is not None:
                stages[_parameter_stage(parameter_name)].append(float(norm) ** 2)
        record = {
            "parameters": rows,
            "stage_l2": {
                stage: math.sqrt(sum(squares)) for stage, squares in stages.items()
            },
        }
        self._parameter_records[name] = record
        return record

    def capture_parameter_baseline(self) -> dict[str, Tensor]:
        """Clone parameters to CPU for a later exact update measurement."""

        if not self.enabled:
            self._baseline = {}
        else:
            self._baseline = {
                name: _to_cpu(parameter).clone()
                for name, parameter in self.model.named_parameters()
            }
        return dict(self._baseline)

    def record_gradients(self) -> dict[str, Any]:
        """Record per-parameter and per-stage gradient norms and anomalies."""

        if not self.enabled:
            return {"enabled": False}
        rows: dict[str, Any] = {}
        stage_squares: MutableMapping[str, float] = defaultdict(float)
        missing: list[str] = []
        nonfinite: list[str] = []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                if parameter.requires_grad:
                    missing.append(name)
                rows[name] = {
                    "stage": _parameter_stage(name),
                    "parameter_name_or_module": name,
                    "requires_grad": bool(parameter.requires_grad),
                    "observed": False,
                    "gradient_none_count": 1,
                    "all_zero_gradient_tensor_count": 0,
                    "zero_gradient_element_count": 0,
                    "zero_gradient_element_ratio": None,
                    "gradient_l2_norm": None,
                }
                continue
            statistics = tensor_statistics(parameter.grad)
            rows[name] = {
                "stage": _parameter_stage(name),
                "parameter_name_or_module": name,
                "requires_grad": bool(parameter.requires_grad),
                "observed": True,
                "gradient_none_count": 0,
                "all_zero_gradient_tensor_count": int(
                    statistics["zero_count"] == statistics["element_count"]
                ),
                "zero_gradient_element_count": statistics["zero_count"],
                "zero_gradient_element_ratio": statistics["zero_ratio"],
                "gradient_l2_norm": statistics["l2_norm"],
                "statistics": statistics,
                "gradient_to_parameter_l2_ratio": (
                    None
                    if statistics["l2_norm"] is None
                    else float(statistics["l2_norm"])
                    / max(
                        float(torch.linalg.vector_norm(parameter.detach().float()).item()),
                        torch.finfo(torch.float32).tiny,
                    )
                ),
            }
            if statistics["finite_ratio"] != 1.0:
                nonfinite.append(name)
            if statistics["l2_norm"] is not None:
                stage_squares[_parameter_stage(name)] += float(
                    statistics["l2_norm"]
                ) ** 2
        self._gradient_record = {
            "parameters": rows,
            "missing_trainable_gradients": missing,
            "nonfinite_gradients": nonfinite,
            "all_observed_gradients_finite": not nonfinite,
            "stage_gradient_l2": {
                stage: math.sqrt(value) for stage, value in stage_squares.items()
            },
            "global_gradient_l2": math.sqrt(sum(stage_squares.values())),
        }
        return self._gradient_record

    def record_parameter_updates(
        self,
        before: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        """Measure exact optimizer updates against a named CPU snapshot."""

        if not self.enabled:
            return {"enabled": False}
        baseline = before if before is not None else self._baseline
        if baseline is None:
            raise ObservabilityContractError(
                "capture_parameter_baseline() or an explicit snapshot is required"
            )
        rows: dict[str, Any] = {}
        stage_squares: MutableMapping[str, float] = defaultdict(float)
        for name, parameter in self.model.named_parameters():
            if name not in baseline:
                raise ObservabilityContractError(f"new parameter after snapshot: {name}")
            initial = baseline[name].detach().to(device="cpu")
            current = _to_cpu(parameter)
            if initial.shape != current.shape:
                raise ObservabilityContractError(f"parameter shape drift: {name}")
            delta = current - initial.to(dtype=current.dtype)
            absolute = float(torch.linalg.vector_norm(delta.float()).item())
            base = float(torch.linalg.vector_norm(initial.float()).item())
            stage = _parameter_stage(name)
            stage_squares[stage] += absolute * absolute
            rows[name] = {
                "stage": stage,
                "parameter_name_or_module": name,
                "element_count": int(delta.numel()),
                "changed_element_count": int(delta.ne(0).sum().item()),
                "absolute_l2": absolute,
                "parameter_update_l2_norm": absolute,
                "relative_l2": absolute
                / max(base, torch.finfo(torch.float32).tiny),
                "parameter_update_relative_l2_norm": absolute
                / max(base, torch.finfo(torch.float32).tiny),
                "initial_parameter_sha256": _tensor_sha256(initial),
                "current_parameter_sha256": _tensor_sha256(current),
                "maximum_absolute_update": (
                    0.0 if delta.numel() == 0 else float(delta.abs().max().item())
                ),
                "finite": bool(torch.isfinite(delta).all().item()),
            }
        self._update_record = {
            "parameters": rows,
            "stage_update_l2": {
                stage: math.sqrt(value) for stage, value in stage_squares.items()
            },
            "global_update_l2": math.sqrt(sum(stage_squares.values())),
        }
        return self._update_record

    def runtime_coverage_gaps(self) -> list[str]:
        """Return missing layer-hook records for this forward-only scope."""

        if not self.enabled:
            return ["DIAGNOSTICS_DISABLED"]
        required = {
            "H1.output",
            "D1.gru.weighted_tokens",
            "D1.gru.attention",
            "D1.cnn.gated_tokens",
            "D1.dwaf.attention",
            "TRANSFORMER.input",
            "CLASSIFIER.segment_attention",
            "CLASSIFIER.output",
        }
        for index in range(1, REQUIRED_TRANSFORMER_LAYER_COUNT + 1):
            required.update(
                {
                    f"TRANSFORMER.layer_{index}.attention",
                    f"TRANSFORMER.layer_{index}.attention_output",
                    f"TRANSFORMER.layer_{index}.output",
                }
            )
        gaps = [name for name in sorted(required) if self._hook_counts[name] <= 0]
        if not any(
            count > 0 and name.startswith("EMBEDDING.")
            for name, count in self._hook_counts.items()
        ):
            gaps.append("EMBEDDING.output")
        mask_alternatives = {
            "path_valid_mask": {
                "sample_path_valid_mask",
                "classifier_segment_path_valid_mask",
            },
            "segment_valid_mask": {
                "segment_path_valid_mask",
                "segment_valid_mask",
                "classifier_segment_valid_mask",
            },
            "token_padding_mask": {
                "packed_token_padding_mask",
                "packed_token_valid_mask",
            },
        }
        for role, alternatives in mask_alternatives.items():
            if not alternatives.intersection(self._mask_records):
                gaps.append(f"MASK.{role}")
        return sorted(gaps)

    def validate_runtime_coverage(self) -> None:
        gaps = self.runtime_coverage_gaps()
        if gaps:
            raise ObservabilityContractError(
                "OBSERVABILITY_GAP_RUNTIME: " + ", ".join(gaps)
            )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe copy of all aggregate observation records."""

        layer_coverage = {
            f"layer_{index}": int(
                self._hook_counts[f"TRANSFORMER.layer_{index}.output"]
            )
            for index in range(1, REQUIRED_TRANSFORMER_LAYER_COUNT + 1)
        }
        gaps = self.runtime_coverage_gaps()
        implemented_observation_ids = {
            row["record_identity"]["observation_id"]
            for records in self._tensor_records.values()
            for row in records
            if row.get("record_identity", {}).get("observation_id")
        }
        if self._mask_records:
            implemented_observation_ids.add("OBS-V3-001")
        if self._pooling_records:
            implemented_observation_ids.add("OBS-V3-012")
        if any(
            role.startswith("CLASSIFIER.segment")
            for role in self._tensor_records
        ):
            implemented_observation_ids.add("OBS-V3-013")
        implemented_observation_ids.update(
            row["record_identity"]["observation_id"]
            for records in self._attention_records.values()
            for row in records
            if row.get("record_identity", {}).get("observation_id")
        )
        if self._classifier_summary:
            implemented_observation_ids.update(
                {"OBS-V3-014", "OBS-V3-015", "OBS-V3-016"}
            )
        if self._loss_records:
            implemented_observation_ids.add("OBS-V3-017")
        if self._gradient_record is not None:
            implemented_observation_ids.update({"OBS-V3-018", "OBS-V3-019"})
        if self._update_record is not None:
            implemented_observation_ids.add("OBS-V3-020")
        selected_padding_count = sum(
            int(row.get("selected_padding_count", 0))
            for records in self._pooling_records.values()
            for row in records
        )
        invalid_winner_count = sum(
            int(row.get("invalid_winner_count", 0))
            for records in self._pooling_records.values()
            for row in records
        )
        argmax_histogram: Counter[str] = Counter()
        for records in self._pooling_records.values():
            for row in records:
                argmax_histogram.update(row.get("winner_histogram") or {})
        mask_summary = {
            **self._mask_health,
            "masked_pool_pad_selection_count": selected_padding_count,
            "invalid_pool_winner_count": invalid_winner_count,
            "pooling_argmax_position_histogram": {
                key: argmax_histogram[key] for key in sorted(argmax_histogram)
            },
        }
        return {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "protocol_version": PROTOCOL_VERSION,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "record_identity": dict(self.record_identity),
            "enabled": self.enabled,
            "forward_scope_count": self._forward_count,
            "active": self._active,
            "mask_roles": dict(self._mask_records),
            "mask_health": dict(self._mask_health),
            "mask_summary": mask_summary,
            "tensors": {name: list(rows) for name, rows in self._tensor_records.items()},
            "attention": {
                name: list(rows) for name, rows in self._attention_records.items()
            },
            "pooling": {
                name: list(rows) for name, rows in self._pooling_records.items()
            },
            "losses": dict(self._loss_records),
            "classifier": dict(self._classifier_summary),
            "parameters": dict(self._parameter_records),
            "gradients": self._gradient_record,
            "updates": self._update_record,
            "hook_counts": dict(self._hook_counts),
            "model_family_profile": str(
                getattr(self.model, "family_id", "UNKNOWN")
            ),
            "transformer_layer_coverage": layer_coverage,
            "all_six_transformer_layers_observed": all(
                count > 0 for count in layer_coverage.values()
            ),
            "collector_scope": "FORWARD_LAYER_HOOK_PLUS_OPTIONAL_POST_FORWARD_STATE",
            "layer_hook_coverage_status": "PASS" if not gaps else "FAIL",
            "layer_hook_coverage_gaps": gaps,
            "runtime_observability_status": (
                "LAYER_HOOK_SCOPE_PASS"
                if not gaps
                else "OBSERVABILITY_GAP_RUNTIME"
            ),
            "runtime_observability_gaps": gaps,
            "full_protocol_runtime_status": "REQUIRES_RUN_LEVEL_AGGREGATOR",
            "implemented_observation_ids_in_scope": sorted(
                implemented_observation_ids
            ),
            "non_interference_contract": {
                "evidence_kind": "PAIRED_ON_OFF_TEST_REQUIRED",
                "observer_implementation_uses_forward_hooks_only": True,
                "observer_writes_parameters_or_buffers": False,
                "observer_calls_rng_operations": False,
                "observer_replaces_returned_tensors": False,
                "live_activation_tensors_retained_after_scope": False,
            },
        }


# A shorter public name for runner integrations while retaining the explicit
# Task34 session name in protocol evidence.
ObservabilityV3 = Task34ObservabilitySession


class WelfordAccumulatorV3:
    """Float64 mergeable population statistics for run-level aggregation."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: Tensor | Sequence[float]) -> None:
        tensor = torch.as_tensor(values).detach().to(device="cpu", dtype=torch.float64)
        flat = tensor.reshape(-1)
        if not bool(torch.isfinite(flat).all().item()):
            raise ObservabilityContractError(
                "Welford aggregation refuses non-finite values"
            )
        for value in flat.tolist():
            self.count += 1
            delta = float(value) - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (float(value) - self.mean)

    def merge(self, other: "WelfordAccumulatorV3") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.m2 = other.count, other.mean, other.m2
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total

    def snapshot(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "variance": None,
                "standard_deviation": None,
                "missing_reason": "NO_OBSERVATIONS",
                "algorithm": "WELFORD_COUNT_MEAN_M2",
                "accumulator_dtype": "float64",
            }
        variance = self.m2 / self.count
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": variance,
            "standard_deviation": math.sqrt(max(0.0, variance)),
            "missing_reason": None,
            "algorithm": "WELFORD_COUNT_MEAN_M2",
            "accumulator_dtype": "float64",
        }


class ProtocolObservationLedgerV3:
    """Fail-closed run-level coverage ledger for the 33 frozen observations."""

    ALL_OBSERVATION_IDS = frozenset(
        f"OBS-V3-{index:03d}" for index in range(1, 34)
    )

    def __init__(
        self,
        *,
        required_observation_ids: Iterable[str] | None = None,
    ) -> None:
        required = set(
            self.ALL_OBSERVATION_IDS
            if required_observation_ids is None
            else required_observation_ids
        )
        unknown = required - self.ALL_OBSERVATION_IDS
        if unknown:
            raise ObservabilityContractError(
                "unknown required observations: " + ", ".join(sorted(unknown))
            )
        self.required = frozenset(required)
        self._records: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)

    def record(self, observation_id: str, payload: Mapping[str, Any]) -> None:
        if observation_id not in self.ALL_OBSERVATION_IDS:
            raise ObservabilityContractError(
                f"unknown Protocol V3 observation: {observation_id}"
            )
        if not payload:
            raise ObservabilityContractError("observation payload must be nonempty")
        record = dict(payload)
        try:
            json.dumps(record, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ObservabilityContractError(
                f"observation payload is not finite JSON: {observation_id}"
            ) from exc
        self._records[observation_id].append(record)

    def snapshot(self) -> dict[str, Any]:
        observed = set(self._records)
        missing = sorted(self.required - observed)
        return {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "required_observation_ids": sorted(self.required),
            "observed_observation_ids": sorted(observed),
            "missing_observation_ids": missing,
            "records": {
                key: list(self._records[key]) for key in sorted(self._records)
            },
            "status": "PASS" if not missing else "OBSERVABILITY_GAP_RUNTIME",
        }


__all__ = [
    "OBSERVABILITY_SCHEMA_VERSION",
    "REQUIRED_TRANSFORMER_LAYER_COUNT",
    "ObservabilityContractError",
    "ObservabilityV3",
    "Task34ObservabilitySession",
    "ProtocolObservationLedgerV3",
    "WelfordAccumulatorV3",
    "attention_statistics",
    "class_conditioned_statistics",
    "mask_statistics",
    "pooling_statistics",
    "tensor_statistics",
]
