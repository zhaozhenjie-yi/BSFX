from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


BCE_SINGLE_SIGMOID_V1 = "BCE_SINGLE_SIGMOID_V1"
CE_TWO_LOGIT_V1 = "CE_TWO_LOGIT_V1"
SUPPORTED_OUTPUT_PROTOCOLS = frozenset(
    {BCE_SINGLE_SIGMOID_V1, CE_TWO_LOGIT_V1}
)
POSITIVE_CLASS_INDEX = 1
POSITIVE_THRESHOLD = 0.5
THRESHOLD_RULE = "probability > 0.5; equality maps to class 0"


def normalize_output_protocol(protocol: str) -> str:
    normalized = str(protocol).strip().upper()
    aliases = {
        "BCE": BCE_SINGLE_SIGMOID_V1,
        "SINGLE_LOGIT": BCE_SINGLE_SIGMOID_V1,
        "SINGLE_SIGMOID": BCE_SINGLE_SIGMOID_V1,
        "CE": CE_TWO_LOGIT_V1,
        "TWO_LOGITS": CE_TWO_LOGIT_V1,
        "TWO_LOGIT": CE_TWO_LOGIT_V1,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_OUTPUT_PROTOCOLS:
        raise ValueError(
            f"unsupported Task34 output protocol {protocol!r}; "
            f"expected one of {sorted(SUPPORTED_OUTPUT_PROTOCOLS)}"
        )
    return normalized


def _single_logits(outputs: Tensor) -> Tensor:
    if outputs.ndim == 1:
        return outputs
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        return outputs[:, 0]
    raise ValueError("BCE_SINGLE_SIGMOID_V1 requires logits shaped [B] or [B,1]")


def _two_logits(outputs: Tensor) -> Tensor:
    if outputs.ndim != 2 or outputs.shape[1] != 2:
        raise ValueError("CE_TWO_LOGIT_V1 requires logits shaped [B,2]")
    return outputs


def positive_probabilities(outputs: Tensor, protocol: str) -> Tensor:
    """Map either frozen family output to one positive-class probability.

    PAPER outputs are logits, not already-sigmoided probabilities. ENGINEERING
    outputs are two logits with positive class index one. Keeping this adapter
    explicit prevents a second sigmoid/softmax during evaluation.
    """

    if not torch.is_tensor(outputs):
        raise TypeError("model outputs must be a torch.Tensor")
    normalized = normalize_output_protocol(protocol)
    if normalized == BCE_SINGLE_SIGMOID_V1:
        probabilities = torch.sigmoid(_single_logits(outputs))
    else:
        probabilities = torch.softmax(_two_logits(outputs), dim=-1)[
            :, POSITIVE_CLASS_INDEX
        ]
    return probabilities


def binary_classification_loss(
    outputs: Tensor,
    labels: Tensor,
    protocol: str,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Apply the loss frozen for a Task33 protocol family."""

    normalized = normalize_output_protocol(protocol)
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B]")
    if labels.numel() != outputs.shape[0]:
        raise ValueError("output and label batch sizes differ")
    if not torch.all((labels == 0) | (labels == 1)).item():
        raise ValueError("Task34 function labels must be binary values 0 or 1")
    if normalized == BCE_SINGLE_SIGMOID_V1:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            _single_logits(outputs),
            labels.to(dtype=outputs.dtype),
            reduction=reduction,
        )
    return torch.nn.functional.cross_entropy(
        _two_logits(outputs),
        labels.to(dtype=torch.long),
        reduction=reduction,
    )


def threshold_probabilities(probabilities: Tensor) -> Tensor:
    if probabilities.ndim != 1:
        raise ValueError("positive probabilities must have shape [B]")
    if not torch.isfinite(probabilities).all().item():
        raise ValueError("positive probabilities contain NaN or Inf")
    if not torch.all((probabilities >= 0.0) & (probabilities <= 1.0)).item():
        raise ValueError("positive probabilities must be in [0,1]")
    # The strict comparison is deliberate: p == 0.5 belongs to class zero.
    return probabilities.gt(POSITIVE_THRESHOLD).to(dtype=torch.long)


def _roc_auc(probabilities: list[float], labels: list[int]) -> float:
    """Compute binary ROC-AUC using average ranks, including score ties."""

    positives = sum(label == 1 for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both label classes")

    ordered = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    offset = 0
    while offset < len(ordered):
        end = offset + 1
        while end < len(ordered) and ordered[end][0] == ordered[offset][0]:
            end += 1
        # Ranks are one-based; all tied observations receive the average rank.
        average_rank = ((offset + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for _, label in ordered[offset:end]
        )
        offset = end

    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def evaluate_probabilities(
    probabilities: Tensor | Sequence[float],
    labels: Tensor | Sequence[int],
    *,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Evaluate one run under the frozen threshold and N/A rules.

    Undefined denominators are represented by ``None`` plus a machine-readable
    reason in ``metric_status``. They are never silently changed to zero.
    """

    probability_tensor = torch.as_tensor(probabilities, dtype=torch.float64)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    if probability_tensor.ndim != 1 or label_tensor.ndim != 1:
        raise ValueError("probabilities and labels must both have shape [B]")
    if probability_tensor.shape != label_tensor.shape:
        raise ValueError("probability and label shapes differ")
    if not torch.isfinite(probability_tensor).all().item():
        raise ValueError("positive probabilities contain NaN or Inf")
    if not torch.all(
        (probability_tensor >= 0.0) & (probability_tensor <= 1.0)
    ).item():
        raise ValueError("positive probabilities must be in [0,1]")
    if not torch.all((label_tensor == 0) | (label_tensor == 1)).item():
        raise ValueError("labels must contain only binary values 0 or 1")

    predicted = threshold_probabilities(probability_tensor)
    tp = int(((predicted == 1) & (label_tensor == 1)).sum().item())
    tn = int(((predicted == 0) & (label_tensor == 0)).sum().item())
    fp = int(((predicted == 1) & (label_tensor == 0)).sum().item())
    fn = int(((predicted == 0) & (label_tensor == 1)).sum().item())
    support = int(label_tensor.numel())
    positive_support = tp + fn
    negative_support = tn + fp
    predicted_positive_count = tp + fp

    metric_status: dict[str, str] = {}

    def ratio(
        numerator: int,
        denominator: int,
        name: str,
        undefined_reason: str,
    ) -> float | None:
        if denominator == 0:
            metric_status[name] = undefined_reason
            return None
        metric_status[name] = "OK"
        return numerator / denominator

    accuracy = ratio(tp + tn, support, "accuracy", "N/A_NO_SUPPORT")
    precision = ratio(
        tp,
        predicted_positive_count,
        "precision",
        "N/A_NO_PREDICTED_POSITIVES",
    )
    recall = ratio(tp, positive_support, "recall", "N/A_NO_POSITIVE_LABELS")
    fpr = ratio(fp, negative_support, "fpr", "N/A_NO_NEGATIVE_LABELS")
    predicted_positive_rate = ratio(
        predicted_positive_count,
        support,
        "predicted_positive_rate",
        "N/A_NO_SUPPORT",
    )

    if precision is None or recall is None:
        f1 = None
        metric_status["f1"] = "N/A_UNDEFINED_PRECISION_OR_RECALL"
    elif precision + recall == 0.0:
        # Both denominators exist, so a zero F1 is mathematically defined.
        f1 = 0.0
        metric_status["f1"] = "OK"
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
        metric_status["f1"] = "OK"

    if positive_support == 0 or negative_support == 0:
        roc_auc = None
        metric_status["roc_auc"] = (
            "N/A_NO_SUPPORT" if support == 0 else "N/A_ONE_CLASS"
        )
    else:
        roc_auc = _roc_auc(
            [float(value) for value in probability_tensor.tolist()],
            [int(value) for value in label_tensor.tolist()],
        )
        metric_status["roc_auc"] = "OK"

    reason = None if failure_reason is None else str(failure_reason).strip() or None
    run_failure_count = int(reason is not None)
    na_metrics = sorted(
        name for name, status in metric_status.items() if status.startswith("N/A_")
    )
    return {
        "metric_schema_version": "task34-evaluator-v1",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": predicted_positive_rate,
        "support": support,
        "positive_support": positive_support,
        "negative_support": negative_support,
        "run_failure": reason is not None,
        "run_failure_count": run_failure_count,
        "run_failure_rate": float(run_failure_count),
        "failure_state": "FAIL" if reason is not None else "PASS",
        "failure_reason": reason,
        "positive_class_index": POSITIVE_CLASS_INDEX,
        "threshold": POSITIVE_THRESHOLD,
        "threshold_rule": THRESHOLD_RULE,
        "metric_status": metric_status,
        "n_a_metrics": na_metrics,
    }


def evaluate_outputs(
    outputs: Tensor,
    labels: Tensor,
    protocol: str,
    *,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    probabilities = positive_probabilities(outputs, protocol)
    report = evaluate_probabilities(
        probabilities.detach().cpu(),
        labels.detach().cpu(),
        failure_reason=failure_reason,
    )
    report["output_protocol"] = normalize_output_protocol(protocol)
    return report


def build_prediction_rows(
    sample_ids: Sequence[str],
    labels: Tensor | Sequence[int],
    probabilities: Tensor | Sequence[float],
) -> list[dict[str, Any]]:
    probability_tensor = torch.as_tensor(probabilities, dtype=torch.float64)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    if probability_tensor.ndim != 1 or label_tensor.ndim != 1:
        raise ValueError("probabilities and labels must both have shape [B]")
    if (
        len(sample_ids) != probability_tensor.numel()
        or label_tensor.shape != probability_tensor.shape
    ):
        raise ValueError("sample IDs, labels and probabilities must have the same length")
    if len({str(sample_id) for sample_id in sample_ids}) != len(sample_ids):
        raise ValueError("sample IDs must be unique in a prediction artifact")
    predicted = threshold_probabilities(probability_tensor)
    if not torch.all((label_tensor == 0) | (label_tensor == 1)).item():
        raise ValueError("prediction labels must contain only 0 or 1")
    return [
        {
            "sample_id": str(sample_id),
            "label": int(label),
            "probability_pos": float(probability),
            "pred": int(prediction),
        }
        for sample_id, label, probability, prediction in zip(
            sample_ids,
            label_tensor.tolist(),
            probability_tensor.tolist(),
            predicted.tolist(),
        )
    ]


def assert_finite_metrics(report: dict[str, Any]) -> None:
    """Reject non-finite numeric metrics while permitting explicit N/A values."""

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "fpr",
        "predicted_positive_rate",
        "run_failure_rate",
    ):
        value = report.get(key)
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"metric {key} is not finite")
