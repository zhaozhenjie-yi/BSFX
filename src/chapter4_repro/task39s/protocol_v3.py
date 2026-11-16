"""Core contracts for Task39S Protocol V3.

This namespace is additive and train/validation bounded.  Importing it does
not construct a model, resolve a dataset, or execute an experiment.  The
formal V3 D0 matrix intentionally has no execution entry point in Task39S.1.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


PROTOCOL_ID = "TASK39S_PROTOCOL_V3"
PROTOCOL_VERSION = "V3"
IMPLEMENTATION_REVISION = "task39s_protocol_v3_freeze_v1"

PAPER_PROFILE = "paper_repro_task39s_protocol_v3"
ENGINEERING_PROFILE = "engineering_aligned_task39s_protocol_v3"
PAPER_BASE_PROFILE = "paper_repro_v1"
ENGINEERING_BASE_PROFILE = "engineering_aligned_v1"
POSITIVE_CLASS_INDEX = 1
PROBABILITY_THRESHOLD = 0.5
LOW_VARIANCE_THRESHOLD = 1e-12

QUALIFICATION_EXTERNAL_PREREQUISITES = frozenset(
    {
        "metric_recomputation_pass",
        "family_contract_pass",
        "sampler_contract_pass",
        "mask_contract_pass",
        "optimizer_closure_pass",
        "diagnostics_non_interference_pass",
    }
)
TINY_GATE_TECHNICAL_PREREQUISITES = frozenset(
    {*QUALIFICATION_EXTERNAL_PREREQUISITES, "observability_finite_pass"}
)

TINY_SELECTION_SEED = 123
TINY_SELECTION_DOMAIN = "TINY_BALANCED_OVERFIT_V1"
TINY_SELECTION_PAYLOAD = (
    "TASK39S_PROTOCOL_V3\\0TINY_BALANCED_OVERFIT_V1\\0"
    "{dataset}\\0{label}\\0{seed}\\0{sample_id}"
)

TERMINAL_STATUSES = frozenset(
    {
        "COLLAPSE_ALL_POSITIVE",
        "COLLAPSE_ALL_NEGATIVE",
        "COLLAPSE_LOW_VARIANCE",
        "COLLAPSE_CLASS_BLIND",
        "NUMERICAL_FAILURE",
        "RUNTIME_FAILURE",
        "NON_COLLAPSED_NOT_QUALIFIED",
        "QUALIFIED",
    }
)

FORBIDDEN_SPLIT_NAMES = frozenset(
    {
        "validation",
        "val",
        "test",
        "test_build",
        "sealed",
        "sealed_test",
        "sealed_test_sources",
        "can_sealed_test",
    }
)


class ProtocolViolation(RuntimeError):
    """Fail-closed Protocol V3 contract violation."""


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_tiny_train_split(split: str) -> None:
    if str(split).strip().lower() != "train":
        raise ProtocolViolation("tiny-overfit accepts the train split only")


def assert_train_only_path(path: str | Path) -> Path:
    candidate = Path(path)
    lowered = {part.lower() for part in candidate.parts}
    if lowered & FORBIDDEN_SPLIT_NAMES:
        raise ProtocolViolation(f"forbidden tiny-overfit path: {candidate}")
    if "train_build" not in lowered or not candidate.name.startswith("train_"):
        raise ProtocolViolation(
            "tiny-overfit paths must be explicit train_build/train_* artifacts"
        )
    return candidate


def _profile_family(profile_id: str) -> str:
    if profile_id in {PAPER_PROFILE, PAPER_BASE_PROFILE, "PAPER_REPRO_V1"}:
        return "PAPER"
    if profile_id in {
        ENGINEERING_PROFILE,
        ENGINEERING_BASE_PROFILE,
        "ENGINEERING_ALIGNED_V1",
    }:
        return "ENGINEERING"
    raise ProtocolViolation(f"unsupported Protocol V3 profile: {profile_id!r}")


def decision_margin(outputs: Tensor, *, profile_id: str) -> Tensor:
    """Return the sole V3 decision scale for both output families.

    Low-variance checks must use this margin, never flattened two-logit values.
    """

    family = _profile_family(profile_id)
    if family == "PAPER":
        if outputs.ndim == 2 and outputs.shape[-1] == 1:
            return outputs[:, 0]
        if outputs.ndim == 1:
            return outputs
        raise ProtocolViolation("paper profile requires [B] or [B,1] raw logits")
    if outputs.ndim != 2 or outputs.shape[-1] != 2:
        raise ProtocolViolation("engineering profile requires [B,2] raw logits")
    return outputs[:, POSITIVE_CLASS_INDEX] - outputs[:, 0]


def positive_probabilities(outputs: Tensor, *, profile_id: str) -> Tensor:
    family = _profile_family(profile_id)
    margin = decision_margin(outputs, profile_id=profile_id)
    if family == "PAPER":
        return torch.sigmoid(margin)
    # Keep the frozen family identity explicit instead of applying sigmoid to
    # the derived margin, even though the two expressions are algebraically
    # equivalent in exact arithmetic.
    return torch.softmax(outputs, dim=-1)[:, POSITIVE_CLASS_INDEX]


def family_loss(outputs: Tensor, labels: Tensor, *, profile_id: str) -> Tensor:
    family = _profile_family(profile_id)
    if labels.ndim != 1 or labels.shape[0] != outputs.shape[0]:
        raise ProtocolViolation("labels and outputs must share batch dimension")
    if family == "PAPER":
        logits = decision_margin(outputs, profile_id=profile_id)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(dtype=logits.dtype)
        )
    decision_margin(outputs, profile_id=profile_id)
    return torch.nn.functional.cross_entropy(outputs, labels.long())


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _roc_auc(probabilities: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(probabilities)
    positive_sum = sum(
        rank for rank, label in zip(ranks, labels) if label == 1
    )
    return (
        positive_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _average_precision(
    probabilities: list[float], labels: list[int]
) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(
        zip(probabilities, labels), key=lambda item: item[0], reverse=True
    )
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        tied_positive = sum(label for _, label in ordered[start:end])
        true_positive += tied_positive
        false_positive += (end - start) - tied_positive
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return area


def _population_stats(values: Tensor) -> dict[str, Any]:
    flat = values.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    finite = torch.isfinite(flat)
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    if flat.numel() == 0 or not bool(finite.all().item()):
        return {
            "count": int(flat.numel()),
            "finite_ratio": (
                1.0
                if flat.numel() == 0
                else float(finite.sum().item()) / int(flat.numel())
            ),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "variance": None,
            "l2_norm": None,
            "zero_ratio": None,
        }
    return {
        "count": int(flat.numel()),
        "finite_ratio": 1.0,
        "nan_count": 0,
        "inf_count": 0,
        "minimum": float(flat.min().item()),
        "maximum": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "standard_deviation": float(flat.std(unbiased=False).item()),
        "variance": float(flat.var(unbiased=False).item()),
        "l2_norm": float(torch.linalg.vector_norm(flat).item()),
        "zero_ratio": float(flat.eq(0).sum().item()) / int(flat.numel()),
    }


def metric_adapter_v3(
    outputs: Tensor | Sequence[float] | Sequence[Sequence[float]],
    labels: Tensor | Sequence[int],
    *,
    profile_id: str,
    qualification: Mapping[str, float] | None = None,
    qualification_prerequisites: Mapping[str, bool] | None = None,
    low_variance_threshold: float = LOW_VARIANCE_THRESHOLD,
) -> dict[str, Any]:
    """Compute the unique V3 metric and collapse-classification record."""

    logits = torch.as_tensor(outputs)
    target = torch.as_tensor(labels, dtype=torch.long)
    if target.ndim != 1 or logits.shape[0] != target.shape[0]:
        raise ProtocolViolation("metric outputs and labels must align on [B]")
    if target.numel() == 0:
        raise ProtocolViolation("metrics require nonempty support")
    if not bool(((target == 0) | (target == 1)).all().item()):
        raise ProtocolViolation("metrics require binary labels")

    margin = decision_margin(logits, profile_id=profile_id)
    finite_logits = bool(torch.isfinite(logits).all().item())
    finite_margin = bool(torch.isfinite(margin).all().item())
    if finite_logits and finite_margin:
        probabilities = positive_probabilities(logits, profile_id=profile_id)
        finite_probabilities = bool(torch.isfinite(probabilities).all().item())
        bounded_probabilities = bool(
            ((probabilities >= 0) & (probabilities <= 1)).all().item()
        )
    else:
        probabilities = torch.full_like(margin, float("nan"), dtype=torch.float32)
        finite_probabilities = False
        bounded_probabilities = False

    qualification_uses_f1_only = (
        qualification is not None and set(qualification) == {"f1"}
    )
    if qualification_uses_f1_only:
        raise ProtocolViolation("Protocol V3 forbids F1-only qualification")

    numerical_failure = not (
        finite_logits
        and finite_margin
        and finite_probabilities
        and bounded_probabilities
    )
    if numerical_failure:
        unavailable = {
            "confusion_matrix": {"tn": None, "fp": None, "fn": None, "tp": None},
            "TN": None,
            "FP": None,
            "FN": None,
            "TP": None,
            "support": int(target.numel()),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "sensitivity": None,
            "specificity": None,
            "f1": None,
            "balanced_accuracy": None,
            "mcc": None,
            "mcc_denominator": None,
            "mcc_denominator_zero": None,
            "roc_auc": None,
            "pr_auc": None,
            "pr_auc_definition": (
                "AVERAGE_PRECISION_NON_INTERPOLATED_DISTINCT_SCORE_THRESHOLDS"
            ),
            "prediction_positive_rate": None,
            "probability_mean": None,
            "probability_std": None,
            "logit_mean": None,
            "logit_std": None,
        }
        return {
            "metric_schema_version": "task39s-protocol-v3-metrics-v1",
            "protocol_id": PROTOCOL_ID,
            "profile_id": profile_id,
            "positive_class_index": POSITIVE_CLASS_INDEX,
            "threshold": PROBABILITY_THRESHOLD,
            "threshold_rule": "POSITIVE_IF_STRICTLY_GREATER",
            "status": "NUMERICAL_FAILURE",
            "collapse_detected": False,
            "collapse_flags": [],
            "numerical_failure": True,
            **unavailable,
            "logit_statistics": _population_stats(logits),
            "decision_logit_statistics": _population_stats(margin),
            "probability_statistics": _population_stats(probabilities),
            "qualification_uses_f1_only": qualification_uses_f1_only,
            "qualification_checks": {},
            "qualification_prerequisite_checks": {},
            "metric_status": {
                name: "N/A_NUMERICAL_FAILURE"
                for name in (
                    "accuracy",
                    "precision",
                    "sensitivity",
                    "specificity",
                    "f1",
                    "balanced_accuracy",
                    "mcc",
                    "roc_auc",
                    "pr_auc",
                )
            },
        }

    predicted = probabilities.gt(PROBABILITY_THRESHOLD).long()
    true_positive = int(((predicted == 1) & (target == 1)).sum().item())
    true_negative = int(((predicted == 0) & (target == 0)).sum().item())
    false_positive = int(((predicted == 1) & (target == 0)).sum().item())
    false_negative = int(((predicted == 0) & (target == 1)).sum().item())
    support = int(target.numel())
    predicted_positive = true_positive + false_positive

    accuracy = (true_positive + true_negative) / support
    precision = _ratio(true_positive, true_positive + false_positive)
    sensitivity = _ratio(true_positive, true_positive + false_negative)
    specificity = _ratio(true_negative, true_negative + false_positive)
    if precision is None or sensitivity is None:
        f1 = None
    elif precision + sensitivity == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity)
    balanced_accuracy = (
        None
        if sensitivity is None or specificity is None
        else (sensitivity + specificity) / 2.0
    )
    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        None
        if mcc_denominator == 0
        else (
            true_positive * true_negative
            - false_positive * false_negative
        )
        / mcc_denominator
    )
    probability_values = [float(value) for value in probabilities.detach().cpu()]
    label_values = [int(value) for value in target.detach().cpu()]
    probability_stats = _population_stats(probabilities)
    margin_stats = _population_stats(margin)
    positive_rate = predicted_positive / support

    flags: list[str] = []
    if predicted_positive == support:
        flags.append("ALL_POSITIVE_PREDICTION")
    if predicted_positive == 0:
        flags.append("ALL_NEGATIVE_PREDICTION")
    if specificity == 0.0:
        flags.append("ZERO_SPECIFICITY")
    if sensitivity == 0.0:
        flags.append("ZERO_SENSITIVITY")
    if mcc_denominator == 0:
        flags.append("MCC_DENOMINATOR_ZERO")
    if (
        margin_stats["variance"] is not None
        and float(margin_stats["variance"]) <= float(low_variance_threshold)
    ):
        flags.append("DECISION_MARGIN_LOW_VARIANCE")

    if "ALL_POSITIVE_PREDICTION" in flags:
        status = "COLLAPSE_ALL_POSITIVE"
    elif "ALL_NEGATIVE_PREDICTION" in flags:
        status = "COLLAPSE_ALL_NEGATIVE"
    elif any(
        flag in flags
        for flag in (
            "ZERO_SPECIFICITY",
            "ZERO_SENSITIVITY",
            "MCC_DENOMINATOR_ZERO",
        )
    ):
        # Zero class recall can occur without an all-one-class prediction.  A
        # separate class-blind status avoids misreporting it as low variance.
        status = "COLLAPSE_CLASS_BLIND"
    elif "DECISION_MARGIN_LOW_VARIANCE" in flags:
        status = "COLLAPSE_LOW_VARIANCE"
    else:
        status = "NON_COLLAPSED_NOT_QUALIFIED"

    qualification_checks: dict[str, bool] = {}
    qualification_prerequisite_checks: dict[str, bool] = {}
    if not flags and qualification is not None:
        values = {
            "accuracy": accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
        }
        for metric_name, minimum in qualification.items():
            if metric_name not in values:
                raise ProtocolViolation(
                    f"unsupported qualification metric: {metric_name}"
                )
            value = values.get(metric_name)
            qualification_checks[metric_name] = (
                value is not None and float(value) >= float(minimum)
            )
        supplied_prerequisites = dict(qualification_prerequisites or {})
        qualification_prerequisite_checks = {
            name: supplied_prerequisites.get(name) is True
            for name in sorted(QUALIFICATION_EXTERNAL_PREREQUISITES)
        }
        qualification_prerequisite_checks.update(
            {
                "finite_contract_pass": not numerical_failure,
                "noncollapsed_prediction_contract_pass": not flags,
                "balanced_accuracy_defined": balanced_accuracy is not None,
                "specificity_defined": specificity is not None,
                "sensitivity_defined": sensitivity is not None,
                "mcc_defined": mcc is not None,
            }
        )
        if (
            qualification_checks
            and all(qualification_checks.values())
            and all(qualification_prerequisite_checks.values())
        ):
            status = "QUALIFIED"

    if status not in TERMINAL_STATUSES:
        raise AssertionError(f"unregistered V3 terminal status: {status}")
    metric_status = {
        name: "OK" if value is not None else "N/A_UNDEFINED_DENOMINATOR"
        for name, value in {
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
        }.items()
    }
    metric_status["accuracy"] = "OK"
    metric_status["roc_auc"] = (
        "OK" if len(set(label_values)) == 2 else "N/A_SINGLE_LABEL_CLASS"
    )
    metric_status["pr_auc"] = (
        "OK" if sum(label_values) > 0 else "N/A_NO_POSITIVE_LABEL"
    )
    return {
        "metric_schema_version": "task39s-protocol-v3-metrics-v1",
        "protocol_id": PROTOCOL_ID,
        "profile_id": profile_id,
        "positive_class_index": POSITIVE_CLASS_INDEX,
        "threshold": PROBABILITY_THRESHOLD,
        "threshold_rule": "POSITIVE_IF_STRICTLY_GREATER",
        "threshold_equality_class": 0,
        "confusion_matrix": {
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "TP": true_positive,
        "support": support,
        "accuracy": accuracy,
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "mcc_denominator": mcc_denominator,
        "mcc_denominator_zero": mcc_denominator == 0,
        "roc_auc": _roc_auc(probability_values, label_values),
        "pr_auc": _average_precision(probability_values, label_values),
        "pr_auc_definition": (
            "AVERAGE_PRECISION_NON_INTERPOLATED_DISTINCT_SCORE_THRESHOLDS"
        ),
        "prediction_positive_rate": positive_rate,
        "probability_mean": probability_stats["mean"],
        "probability_std": probability_stats["standard_deviation"],
        "logit_mean": margin_stats["mean"],
        "logit_std": margin_stats["standard_deviation"],
        "logit_statistics": _population_stats(logits),
        "decision_logit_statistics": margin_stats,
        "probability_statistics": probability_stats,
        "collapse_detected": bool(flags),
        "collapse_flags": flags,
        "status": status,
        "numerical_failure": False,
        "qualification_uses_f1_only": qualification_uses_f1_only,
        "qualification_checks": qualification_checks,
        "qualification_prerequisite_checks": qualification_prerequisite_checks,
        "metric_status": metric_status,
    }


def independent_metric_recomputation_v3(
    outputs: Tensor | Sequence[float] | Sequence[Sequence[float]],
    labels: Tensor | Sequence[int],
    *,
    profile_id: str,
    reported: Mapping[str, Any],
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Independently recompute the reported metric row from logits and labels.

    This deliberately uses a compact pure-Python confusion/AUC/AP path rather
    than calling :func:`metric_adapter_v3` again.  Probability construction
    still follows the single frozen family mapping.
    """

    logits = torch.as_tensor(outputs)
    target = torch.as_tensor(labels, dtype=torch.long)
    margin = decision_margin(logits, profile_id=profile_id)
    probabilities = positive_probabilities(logits, profile_id=profile_id)
    if not (
        bool(torch.isfinite(logits).all().item())
        and bool(torch.isfinite(margin).all().item())
        and bool(torch.isfinite(probabilities).all().item())
    ):
        return {
            "status": "FAIL",
            "mismatches": ["NONFINITE_RECOMPUTATION_INPUT"],
            "atol": float(atol),
            "rtol": 0.0,
        }

    probability_values = [
        float(value) for value in probabilities.detach().to("cpu", torch.float64)
    ]
    margin_values = [
        float(value) for value in margin.detach().to("cpu", torch.float64)
    ]
    label_values = [int(value) for value in target.detach().cpu()]
    predictions = [int(value > PROBABILITY_THRESHOLD) for value in probability_values]
    tp = sum(prediction == 1 and label == 1 for prediction, label in zip(predictions, label_values))
    tn = sum(prediction == 0 and label == 0 for prediction, label in zip(predictions, label_values))
    fp = sum(prediction == 1 and label == 0 for prediction, label in zip(predictions, label_values))
    fn = sum(prediction == 0 and label == 1 for prediction, label in zip(predictions, label_values))
    support = len(label_values)

    precision = _ratio(tp, tp + fp)
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    if precision is None or sensitivity is None:
        f1 = None
    elif precision + sensitivity == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity)
    balanced_accuracy = (
        None
        if sensitivity is None or specificity is None
        else (sensitivity + specificity) / 2.0
    )
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = None if mcc_denominator == 0 else (tp * tn - fp * fn) / mcc_denominator

    positives = [
        probability
        for probability, label in zip(probability_values, label_values)
        if label == 1
    ]
    negatives = [
        probability
        for probability, label in zip(probability_values, label_values)
        if label == 0
    ]
    roc_auc = None
    if positives and negatives:
        wins = sum(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        )
        roc_auc = wins / (len(positives) * len(negatives))

    pr_auc = None
    if positives:
        ordered = sorted(
            zip(probability_values, label_values),
            key=lambda row: row[0],
            reverse=True,
        )
        true_positive = 0
        false_positive = 0
        previous_recall = 0.0
        area = 0.0
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
                end += 1
            tied_positive = sum(label for _, label in ordered[cursor:end])
            true_positive += tied_positive
            false_positive += end - cursor - tied_positive
            recall = true_positive / len(positives)
            area += (recall - previous_recall) * (
                true_positive / (true_positive + false_positive)
            )
            previous_recall = recall
            cursor = end
        pr_auc = area

    def mean_std(values: Sequence[float]) -> tuple[float, float]:
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        return mean, math.sqrt(variance)

    probability_mean, probability_std = mean_std(probability_values)
    logit_mean, logit_std = mean_std(margin_values)
    expected: dict[str, Any] = {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "support": support,
        "accuracy": (tp + tn) / support,
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "prediction_positive_rate": (tp + fp) / support,
        "probability_mean": probability_mean,
        "probability_std": probability_std,
        "logit_mean": logit_mean,
        "logit_std": logit_std,
    }
    exact_fields = {"confusion_matrix", "TN", "FP", "FN", "TP", "support"}
    mismatches: list[str] = []
    for name, expected_value in expected.items():
        observed_value = reported.get(name)
        if name in exact_fields or expected_value is None or observed_value is None:
            matches = observed_value == expected_value
        else:
            try:
                matches = math.isclose(
                    float(observed_value),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=float(atol),
                )
            except (TypeError, ValueError):
                matches = False
        if not matches:
            mismatches.append(name)
    payload = {
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "atol": float(atol),
        "rtol": 0.0,
        "integer_and_categorical_fields_exact": True,
        "recomputed": expected,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def _selection_key(
    *, dataset: str, label: int, seed: int, sample_id: str
) -> tuple[str, bytes]:
    payload = (
        f"{PROTOCOL_ID}\0{TINY_SELECTION_DOMAIN}\0{dataset}\0"
        f"{label}\0{seed}\0{sample_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), sample_id.encode("utf-8")


def select_tiny_balanced_ids(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    seed: int = TINY_SELECTION_SEED,
    per_class: int = 16,
) -> dict[str, Any]:
    """Select a deterministic 16+/16- cohort from train records only."""

    if not dataset or per_class < 1:
        raise ProtocolViolation("dataset and positive per-class count are required")
    by_id: dict[str, int] = {}
    row_count = 0
    for record in records:
        row_count += 1
        assert_tiny_train_split(str(record.get("split", "")))
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            raise ProtocolViolation("tiny selection record lacks sample_id")
        label = int(record.get("label", -1))
        if label not in {0, 1}:
            raise ProtocolViolation("tiny selection labels must be binary")
        previous = by_id.setdefault(sample_id, label)
        if previous != label:
            raise ProtocolViolation(f"label drift for sample {sample_id!r}")
    if row_count == 0:
        raise ProtocolViolation("tiny selection source is empty")
    selected: dict[str, list[str]] = {}
    for label in (0, 1):
        candidates = sorted(
            (sample_id for sample_id, value in by_id.items() if value == label),
            key=lambda sample_id: _selection_key(
                dataset=dataset,
                label=label,
                seed=seed,
                sample_id=sample_id,
            ),
        )
        if len(candidates) < per_class:
            raise ProtocolViolation(
                f"tiny selection needs {per_class} unique class-{label} rows"
            )
        selected[str(label)] = candidates[:per_class]
    ordered = [
        sample_id
        for pair in zip(selected["1"], selected["0"])
        for sample_id in pair
    ]
    payload = {
        "protocol_id": PROTOCOL_ID,
        "selection_domain": TINY_SELECTION_DOMAIN,
        "dataset": dataset,
        "split": "train",
        "seed": int(seed),
        "selection_payload_template": TINY_SELECTION_PAYLOAD,
        "sort": "SHA256_ASC_THEN_SAMPLE_ID_UTF8_ASC",
        "replacement": False,
        "per_class": int(per_class),
        "class_0_sample_ids": selected["0"],
        "class_1_sample_ids": selected["1"],
        "ordered_sample_ids": ordered,
        "class_counts": {"0": per_class, "1": per_class},
        "total_count": per_class * 2,
        "paper_engineering_same_cohort": True,
    }
    payload["sample_fingerprint"] = canonical_sha256(payload)
    return payload


def verify_tiny_subset(
    records: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    expected = select_tiny_balanced_ids(
        records,
        dataset=str(manifest.get("dataset") or ""),
        seed=int(manifest.get("seed", -1)),
        per_class=int(manifest.get("per_class", -1)),
    )
    if dict(manifest) != expected:
        raise ProtocolViolation("frozen tiny-overfit sample manifest drift")


def balanced_tiny_batches(
    manifest: Mapping[str, Any], *, epoch: int, batch_size: int = 8
) -> tuple[tuple[str, ...], ...]:
    """Return a deterministic no-replacement 4+/4- epoch schedule."""

    if batch_size != 8:
        raise ProtocolViolation("V3 tiny gate freezes logical batch size at 8")
    dataset = str(manifest["dataset"])
    seed = int(manifest["seed"])

    def ordered(label: int) -> list[str]:
        values = [str(value) for value in manifest[f"class_{label}_sample_ids"]]
        return sorted(
            values,
            key=lambda sample_id: (
                hashlib.sha256(
                    (
                        f"{PROTOCOL_ID}\0TINY_BATCH_V1\0{dataset}\0{label}\0"
                        f"{seed}\0{epoch}\0{sample_id}"
                    ).encode("utf-8")
                ).hexdigest(),
                sample_id.encode("utf-8"),
            ),
        )

    positive = ordered(1)
    negative = ordered(0)
    if len(positive) != 16 or len(negative) != 16:
        raise ProtocolViolation("frozen tiny cohort must contain 16 rows per class")
    batches = []
    for start in range(0, 16, 4):
        batch = tuple(
            value
            for pair in zip(
                positive[start : start + 4], negative[start : start + 4]
            )
            for value in pair
        )
        batches.append(batch)
    flattened = [value for batch in batches for value in batch]
    expected = set(positive + negative)
    if len(flattened) != 32 or len(set(flattened)) != 32 or set(flattened) != expected:
        raise AssertionError("tiny batch schedule lost no-replacement coverage")
    return tuple(batches)


def configure_tiny_gate_parameters(model: nn.Module, *, gate: str) -> dict[str, Any]:
    """Configure Gate A classifier-only or Gate B end-to-end membership."""

    if gate not in {"classifier_only", "end_to_end"}:
        raise ProtocolViolation("tiny gate must be classifier_only or end_to_end")
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = gate == "end_to_end" or name.startswith("classifier.")
        parameter.requires_grad_(enabled)
        (trainable if enabled else frozen).append(name)
    if not trainable:
        raise ProtocolViolation("tiny gate has no trainable parameters")
    if gate == "classifier_only" and not frozen:
        raise ProtocolViolation("classifier-only gate did not freeze a backbone")
    return {
        "gate": gate,
        "trainable_parameters": sorted(trainable),
        "frozen_parameters": sorted(frozen),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
    }


def named_parameter_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in model.named_parameters()
    }


def named_parameter_updates(
    model: nn.Module, before: Mapping[str, Tensor]
) -> dict[str, Any]:
    rows: dict[str, dict[str, float | int]] = {}
    classifier_squared = 0.0
    backbone_squared = 0.0
    for name, parameter in model.named_parameters():
        if name not in before:
            raise ProtocolViolation(f"parameter appeared after snapshot: {name}")
        current = parameter.detach().to(device="cpu")
        initial = before[name]
        if current.shape != initial.shape or current.dtype != initial.dtype:
            raise ProtocolViolation(f"parameter identity drift: {name}")
        delta = current - initial
        absolute = float(torch.linalg.vector_norm(delta.float()).item())
        base = float(torch.linalg.vector_norm(initial.float()).item())
        rows[name] = {
            "numel": int(delta.numel()),
            "absolute_l2": absolute,
            "relative_l2": absolute / max(base, torch.finfo(torch.float32).tiny),
        }
        if name.startswith("classifier."):
            classifier_squared += absolute * absolute
        else:
            backbone_squared += absolute * absolute
    return {
        "parameters": rows,
        "classifier_update_l2": math.sqrt(classifier_squared),
        "backbone_update_l2": math.sqrt(backbone_squared),
    }


def optimizer_parameter_closure(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    observed: list[str] = []
    unknown_count = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = by_identity.get(id(parameter))
            if name is None:
                unknown_count += 1
            else:
                observed.append(name)
    duplicates = sorted(
        name for name in set(observed) if observed.count(name) != 1
    )
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    passed = not (unknown_count or duplicates or missing or unexpected)
    return {
        "status": "PASS" if passed else "FAIL",
        "expected_names": sorted(expected),
        "observed_names": observed,
        "missing_names": missing,
        "unexpected_names": unexpected,
        "duplicate_names": duplicates,
        "unknown_parameter_count": unknown_count,
    }


@dataclass(frozen=True)
class TinyGateAssessment:
    gate: str
    status: str
    passed: bool
    blocking_reasons: tuple[str, ...]
    checks: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_tiny_overfit_gate(
    *,
    gate: str,
    metrics: Mapping[str, Any],
    loss_history: Sequence[float],
    updates: Mapping[str, Any],
    gradients_finite: bool,
    parameters_finite: bool,
    optimizer_state_finite: bool,
    classifier_gradient_observed: bool,
    backbone_gradient_observed: bool,
    technical_prerequisites: Mapping[str, bool],
    minimum_relative_loss_decline: float = 0.20,
) -> TinyGateAssessment:
    if gate not in {"classifier_only", "end_to_end"}:
        raise ProtocolViolation("unknown tiny gate")
    if len(loss_history) < 2 or not all(math.isfinite(float(v)) for v in loss_history):
        relative_decline = float("-inf")
    else:
        initial = float(loss_history[0])
        final = float(loss_history[-1])
        relative_decline = (initial - final) / max(
            abs(initial), torch.finfo(torch.float32).tiny
        )
    prediction_rate = metrics.get("prediction_positive_rate")
    confusion = metrics.get("confusion_matrix") or {}

    def numeric_values(value: Any) -> Iterable[float]:
        if isinstance(value, Mapping):
            for child in value.values():
                yield from numeric_values(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                yield from numeric_values(child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield float(value)

    updates_finite = all(math.isfinite(value) for value in numeric_values(updates))
    technical_checks = {
        name: technical_prerequisites.get(name) is True
        for name in sorted(TINY_GATE_TECHNICAL_PREREQUISITES)
    }
    checks = {
        "balanced_accuracy_at_least_0_95": (
            metrics.get("balanced_accuracy") is not None
            and float(metrics["balanced_accuracy"]) >= 0.95
        ),
        "accuracy_at_least_0_95": (
            metrics.get("accuracy") is not None
            and float(metrics["accuracy"]) >= 0.95
        ),
        "both_classes_have_correct_prediction": (
            int(confusion.get("tn", 0)) > 0 and int(confusion.get("tp", 0)) > 0
        ),
        "prediction_positive_rate_strictly_between_zero_and_one": (
            prediction_rate is not None and 0.0 < float(prediction_rate) < 1.0
        ),
        "loss_declined_by_frozen_fraction": (
            relative_decline >= float(minimum_relative_loss_decline)
        ),
        "noncollapsed_metric_status": (
            metrics.get("collapse_detected") is False
            and metrics.get("status") == "QUALIFIED"
        ),
        "all_finite_contract": (
            metrics.get("status") != "NUMERICAL_FAILURE"
            and gradients_finite
            and parameters_finite
            and optimizer_state_finite
            and updates_finite
            and technical_checks["observability_finite_pass"]
        ),
        **technical_checks,
        "classifier_gradient_observed": bool(classifier_gradient_observed),
        "classifier_parameter_update_nonzero": (
            float(updates.get("classifier_update_l2", 0.0)) > 0.0
        ),
        "backbone_parameter_update_contract": (
            float(updates.get("backbone_update_l2", 0.0)) == 0.0
            if gate == "classifier_only"
            else float(updates.get("backbone_update_l2", 0.0)) > 0.0
        ),
        "backbone_gradient_contract": (
            not backbone_gradient_observed
            if gate == "classifier_only"
            else bool(backbone_gradient_observed)
        ),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return TinyGateAssessment(
        gate=gate,
        status="PASS" if not reasons else "BLOCKED",
        passed=not reasons,
        blocking_reasons=reasons,
        checks=checks,
    )
