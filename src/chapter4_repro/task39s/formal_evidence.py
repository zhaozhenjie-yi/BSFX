"""Durable, fail-closed evidence persistence for Task39S tiny gates.

This module is deliberately orthogonal to model math.  It serializes the
already-produced final state, makes metric inputs self-contained, and creates
hash-addressed evidence before a formal attempt may be marked TERMINAL.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .protocol_v3 import (
    ENGINEERING_PROFILE,
    PAPER_PROFILE,
    POSITIVE_CLASS_INDEX,
    PROBABILITY_THRESHOLD,
    ProtocolViolation,
    independent_metric_recomputation_v3,
    metric_adapter_v3,
    positive_probabilities,
)


PREDICTION_SCHEMA_VERSION = "task39s_2ar_final_predictions_v1"
CHECKPOINT_SCHEMA_VERSION = "task39s_2ar_final_checkpoint_v1"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "task39s_2ar_checkpoint_manifest_v1"
PARAMETER_EVIDENCE_SCHEMA_VERSION = "task39s_2ar_parameter_evidence_v1"
FORMAL_ARTIFACT_MANIFEST_SCHEMA_VERSION = (
    "task39s_2ar_formal_artifact_manifest_v1"
)
METRIC_RECOMPUTATION_SCHEMA_VERSION = "task39s_2ar_recomputed_metrics_v1"
METRIC_CONSISTENCY_SCHEMA_VERSION = "task39s_2ar_metric_consistency_v1"

LEGAL_MODEL_TERMINAL_STATUSES = frozenset(
    {
        "QUALIFIED",
        "NON_COLLAPSED_NOT_QUALIFIED",
        "COLLAPSE_ALL_POSITIVE",
        "COLLAPSE_ALL_NEGATIVE",
        "COLLAPSE_LOW_VARIANCE",
        "COLLAPSE_CLASS_BLIND",
    }
)

RECOMPUTED_METRIC_FIELDS: tuple[str, ...] = (
    "confusion_matrix",
    "TN",
    "FP",
    "FN",
    "TP",
    "support",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "pr_auc",
    "prediction_positive_rate",
    "probability_mean",
    "probability_std",
    "logit_mean",
    "logit_std",
)

REQUIRED_FORMAL_ARTIFACTS: tuple[str, ...] = (
    "run_manifest.json",
    "activation_identity.json",
    "cohort_identity.json",
    "final_predictions.json",
    "final_metrics.json",
    "recomputed_metrics.json",
    "metric_consistency.json",
    "diagnostics_summary.json",
    "layer_separation.json",
    "gradient_summary.json",
    "parameter_update_summary.json",
    "mask_pooling_summary.json",
    "attention_summary.json",
    "transformer_summary.json",
    "classifier_summary.json",
    "collapse_classification.json",
    "failure_localization.json",
    "checkpoint_manifest.json",
    "run_status.txt",
)

PARAMETER_MODULES: tuple[str, ...] = (
    "embedding_or_cbow",
    "H1",
    "GRU",
    "CNN",
    "D1",
    "Transformer",
    "C1",
    "classifier",
)


class FormalEvidenceError(RuntimeError):
    """Evidence could not be persisted or independently verified."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        raise FormalEvidenceError(f"write-once evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist the legacy pretty-JSON bytes without materializing them whole.

    The explicit encoder arguments are the complete effective contract of the
    former ``json.dumps`` call.  A fixed-size byte buffer amortizes writes but
    never accumulates the fragment sequence or a full output representation.
    """

    if path.exists():
        raise FormalEvidenceError(f"write-once evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoder = json.JSONEncoder(
        skipkeys=False,
        ensure_ascii=False,
        check_circular=True,
        allow_nan=False,
        sort_keys=True,
        indent=2,
        separators=None,
        default=None,
    )
    buffer_limit = 1 << 20
    try:
        with temporary.open("wb") as handle:
            buffer = bytearray()
            for fragment in encoder.iterencode(dict(payload), _one_shot=True):
                encoded = fragment.encode("utf-8")
                if len(buffer) + len(encoded) > buffer_limit:
                    handle.write(buffer)
                    buffer.clear()
                if len(encoded) >= buffer_limit:
                    handle.write(encoded)
                else:
                    buffer.extend(encoded)
            buffer.extend(b"\n")
            handle.write(buffer)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FormalEvidenceError(f"write-once evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prediction_payload_hash(payload: Mapping[str, Any]) -> str:
    bare = dict(payload)
    bare.pop("prediction_sha256", None)
    return canonical_sha256(bare)


def build_prediction_payload(
    *,
    sample_ids: Sequence[str],
    labels: Sequence[int] | Tensor,
    raw_outputs: Sequence[float] | Sequence[Sequence[float]] | Tensor,
    profile: str,
    threshold: float = PROBABILITY_THRESHOLD,
) -> dict[str, Any]:
    """Build a self-contained final prediction record from raw family output."""

    output = torch.as_tensor(raw_outputs, dtype=torch.float32).detach().cpu()
    target = torch.as_tensor(labels, dtype=torch.long).reshape(-1).detach().cpu()
    ids = [str(value) for value in sample_ids]
    if len(ids) != len(set(ids)):
        raise FormalEvidenceError("final prediction sample IDs must be unique")
    if len(ids) != int(target.numel()) or output.shape[0] != len(ids):
        raise FormalEvidenceError("prediction IDs, labels, and outputs must align")
    if profile == PAPER_PROFILE:
        if output.ndim == 2 and output.shape[1] == 1:
            output = output[:, 0]
        if output.ndim != 1:
            raise FormalEvidenceError("Paper predictions require one raw logit per row")
    elif profile == ENGINEERING_PROFILE:
        if output.ndim != 2 or output.shape[1] != 2:
            raise FormalEvidenceError("Engineering predictions require two raw logits")
    else:
        raise FormalEvidenceError(f"unknown profile: {profile}")
    if not bool(((target == 0) | (target == 1)).all().item()):
        raise FormalEvidenceError("prediction labels must be binary")
    if not bool(torch.isfinite(output).all().item()):
        raise FormalEvidenceError("final raw outputs must be finite")
    probabilities = positive_probabilities(output, profile_id=profile).detach().cpu()
    predictions = probabilities.gt(float(threshold)).long()
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(ids):
        if profile == PAPER_PROFILE:
            raw_logit = float(output[index].item())
            raw_output: float | list[float] = raw_logit
            raw_logits: list[float] | None = None
            positive_logit = raw_logit
        else:
            raw_logits = [float(value) for value in output[index].tolist()]
            raw_output = raw_logits
            raw_logit = None
            positive_logit = raw_logits[POSITIVE_CLASS_INDEX]
        rows.append(
            {
                "sample_id": sample_id,
                "gold_label": int(target[index].item()),
                "raw_output": raw_output,
                "raw_logit": raw_logit,
                "raw_logits": raw_logits,
                "positive_logit": positive_logit,
                "positive_probability": float(probabilities[index].item()),
                "predicted_label": int(predictions[index].item()),
            }
        )
    payload: dict[str, Any] = {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "profile": profile,
        "positive_class_index": POSITIVE_CLASS_INDEX,
        "threshold": float(threshold),
        "threshold_rule": "POSITIVE_IF_STRICTLY_GREATER",
        "sample_count": len(rows),
        "samples": rows,
        "prediction_sha256": "PENDING",
    }
    payload["prediction_sha256"] = _prediction_payload_hash(payload)
    return payload


def validate_prediction_payload(payload: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    if payload.get("prediction_schema_version") != PREDICTION_SCHEMA_VERSION:
        raise FormalEvidenceError("prediction schema version mismatch")
    if payload.get("prediction_sha256") != _prediction_payload_hash(payload):
        raise FormalEvidenceError("prediction payload SHA-256 mismatch")
    rows = payload.get("samples")
    if not isinstance(rows, list) or len(rows) != payload.get("sample_count"):
        raise FormalEvidenceError("prediction sample count mismatch")
    if len({str(row.get("sample_id")) for row in rows}) != len(rows):
        raise FormalEvidenceError("prediction sample IDs are not unique")
    labels = torch.tensor([int(row["gold_label"]) for row in rows], dtype=torch.long)
    profile = str(payload.get("profile"))
    if profile == PAPER_PROFILE:
        outputs = torch.tensor(
            [float(row["raw_logit"]) for row in rows], dtype=torch.float32
        )
    elif profile == ENGINEERING_PROFILE:
        outputs = torch.tensor(
            [[float(value) for value in row["raw_logits"]] for row in rows],
            dtype=torch.float32,
        )
        if outputs.ndim != 2 or outputs.shape[1] != 2:
            raise FormalEvidenceError("Engineering raw logits are malformed")
    else:
        raise FormalEvidenceError("prediction profile is invalid")
    probabilities = positive_probabilities(outputs, profile_id=profile).detach().cpu()
    threshold = float(payload.get("threshold"))
    for index, row in enumerate(rows):
        probability = float(row["positive_probability"])
        if not math.isclose(
            probability,
            float(probabilities[index].item()),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise FormalEvidenceError("persisted probability does not match raw output")
        if int(row["predicted_label"]) != int(probability > threshold):
            raise FormalEvidenceError("persisted prediction violates threshold contract")
    return outputs, labels


def recompute_metrics_from_prediction_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute scalar metrics solely from a persisted prediction payload."""

    outputs, labels = validate_prediction_payload(payload)
    profile = str(payload["profile"])
    adapter_record = metric_adapter_v3(outputs, labels, profile_id=profile)
    independent = independent_metric_recomputation_v3(
        outputs,
        labels,
        profile_id=profile,
        reported=adapter_record,
    )
    if independent.get("status") != "PASS":
        raise FormalEvidenceError(
            "independent prediction-only metric recomputation failed"
        )
    metrics = dict(independent["recomputed"])
    result: dict[str, Any] = {
        "schema_version": METRIC_RECOMPUTATION_SCHEMA_VERSION,
        "input_prediction_sha256": payload["prediction_sha256"],
        "profile": profile,
        "metric_scope": "PREDICTION_ONLY_NO_GATE_QUALIFICATION",
        "undefined_metrics_are_null": True,
        "metrics": metrics,
        "recomputed_metrics_sha256": canonical_sha256(metrics),
    }
    result["canonical_sha256"] = canonical_sha256(result)
    return result


def recompute_metrics_from_files(
    prediction_path: Path,
    metric_contract_path: Path,
) -> dict[str, Any]:
    """Recompute after process-style reload from exactly two frozen inputs."""

    try:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        metric_contract = json.loads(metric_contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalEvidenceError(f"cannot load metric recomputation input: {exc}") from exc
    if not isinstance(prediction, Mapping) or not isinstance(metric_contract, Mapping):
        raise FormalEvidenceError("metric recomputation inputs must be objects")
    if metric_contract.get("protocol_id") != "TASK39S_PROTOCOL_V3":
        raise FormalEvidenceError("metric contract protocol identity mismatch")
    if metric_contract.get("contract_status") != "FROZEN":
        raise FormalEvidenceError("metric contract is not frozen")
    result = recompute_metrics_from_prediction_payload(prediction)
    result["metric_contract_sha256"] = sha256_file(metric_contract_path)
    result.pop("canonical_sha256", None)
    result["canonical_sha256"] = canonical_sha256(result)
    return result


def compare_runtime_and_recomputed_metrics(
    runtime_metrics: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    values = recomputed.get("metrics")
    if not isinstance(values, Mapping):
        raise FormalEvidenceError("recomputed metric payload is malformed")
    rows: list[dict[str, Any]] = []
    all_match = True
    exact = {"confusion_matrix", "TN", "FP", "FN", "TP", "support"}
    for name in RECOMPUTED_METRIC_FIELDS:
        runtime_value = runtime_metrics.get(name)
        recomputed_value = values.get(name)
        difference: float | None
        if name in exact or runtime_value is None or recomputed_value is None:
            match = runtime_value == recomputed_value
            difference = None
        else:
            difference = abs(float(runtime_value) - float(recomputed_value))
            match = difference <= float(tolerance)
        all_match = all_match and match
        rows.append(
            {
                "metric": name,
                "runtime_metric": runtime_value,
                "recomputed_metric": recomputed_value,
                "absolute_difference": difference,
                "tolerance": 0.0 if name in exact else float(tolerance),
                "match": match,
            }
        )
    result: dict[str, Any] = {
        "schema_version": METRIC_CONSISTENCY_SCHEMA_VERSION,
        "status": "PASS" if all_match else "FAIL",
        "all_metrics_match": all_match,
        "rows": rows,
    }
    result["canonical_sha256"] = canonical_sha256(result)
    return result


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().to(device="cpu").contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def hash_named_tensors(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().to(device="cpu").contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        encoded_shape = json.dumps(list(tensor.shape), separators=(",", ":")).encode(
            "ascii"
        )
        for part in (encoded_name, encoded_dtype, encoded_shape, _tensor_bytes(tensor)):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _parameter_role(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("classifier.segment_classifier."):
        return "classifier"
    if lowered.startswith("classifier."):
        return "C1"
    if "word_embedding" in lowered or "cbow" in lowered:
        return "embedding_or_cbow"
    if lowered.startswith("hpe.") or lowered.startswith("h1_to_d1."):
        return "H1"
    if lowered.startswith("gru_branch."):
        return "GRU"
    if lowered.startswith("cnn_branch."):
        return "CNN"
    if lowered.startswith("dwaf.") or lowered.startswith("d1_to_transformer."):
        return "D1"
    if lowered.startswith("transformer.") or lowered.startswith(
        "special_token_embedding."
    ):
        return "Transformer"
    if lowered.startswith("classifier."):
        return "C1"
    return "D1"


def _hash_group(values: Mapping[str, Tensor]) -> dict[str, Any]:
    if not values:
        return {
            "status": "ABSENT_IN_PROFILE",
            "parameter_count": 0,
            "tensor_count": 0,
            "sha256": None,
        }
    return {
        "status": "PRESENT",
        "parameter_count": sum(int(value.numel()) for value in values.values()),
        "tensor_count": len(values),
        "sha256": hash_named_tensors(values),
    }


def parameter_state_hash_record(values: Mapping[str, Tensor]) -> dict[str, Any]:
    copied = {
        str(name): value.detach().to(device="cpu").clone()
        for name, value in values.items()
    }
    classifier = {
        name: value for name, value in copied.items() if name.startswith("classifier.")
    }
    backbone = {
        name: value for name, value in copied.items() if not name.startswith("classifier.")
    }
    per_module: dict[str, Any] = {}
    for module in PARAMETER_MODULES:
        selected = {
            name: value
            for name, value in copied.items()
            if _parameter_role(name) == module
        }
        per_module[module] = _hash_group(selected)
    return {
        "whole_model_hash": hash_named_tensors(copied),
        "backbone_hash": hash_named_tensors(backbone),
        "classifier_hash": hash_named_tensors(classifier),
        "whole_parameter_count": sum(int(value.numel()) for value in copied.values()),
        "backbone_parameter_count": sum(
            int(value.numel()) for value in backbone.values()
        ),
        "classifier_parameter_count": sum(
            int(value.numel()) for value in classifier.values()
        ),
        "per_module_hash": per_module,
    }


def build_parameter_evidence(
    *,
    initial_parameters: Mapping[str, Tensor],
    model: nn.Module,
    gate: str,
) -> dict[str, Any]:
    final_parameters = {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in model.named_parameters()
    }
    initial = parameter_state_hash_record(initial_parameters)
    final = parameter_state_hash_record(final_parameters)
    modules: dict[str, Any] = {}
    for module in PARAMETER_MODULES:
        before = initial["per_module_hash"][module]
        after = final["per_module_hash"][module]
        comparable = before["status"] == after["status"] == "PRESENT"
        modules[module] = {
            "before": before,
            "after": after,
            "comparable": comparable,
            "changed": (
                before["sha256"] != after["sha256"] if comparable else None
            ),
        }
    result: dict[str, Any] = {
        "schema_version": PARAMETER_EVIDENCE_SCHEMA_VERSION,
        "hash_contract": {
            "parameter_sort": "FULLY_QUALIFIED_NAME_ASCENDING",
            "includes_name": True,
            "includes_dtype": True,
            "includes_shape": True,
            "includes_raw_tensor_bytes": True,
        },
        "gate": gate,
        "initial": initial,
        "final": final,
        "backbone_hash_before": initial["backbone_hash"],
        "backbone_hash_after": final["backbone_hash"],
        "classifier_hash_before": initial["classifier_hash"],
        "classifier_hash_after": final["classifier_hash"],
        "gate_a_backbone_freeze_contract": (
            "PASS"
            if gate == "classifier_only"
            and initial["backbone_hash"] == final["backbone_hash"]
            else "FAIL"
            if gate == "classifier_only"
            else "NOT_APPLICABLE"
        ),
        "module_comparisons": modules,
    }
    result["canonical_sha256"] = canonical_sha256(result)
    return result


def _classifier_state(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in values.items()
        if name.startswith("classifier.")
    }


def _state_summary(values: Mapping[str, Tensor]) -> dict[str, Any]:
    rows = []
    for name in sorted(values):
        value = values[name].detach().to(device="cpu")
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "l2_norm": float(torch.linalg.vector_norm(value.float()).item()),
                "sha256": hash_named_tensors({name: value}),
                "role": "bias" if name.endswith("bias") else "weight_or_parameter",
            }
        )
    return {
        "tensor_count": len(rows),
        "parameter_count": sum(int(value.numel()) for value in values.values()),
        "state_sha256": hash_named_tensors(values),
        "tensors": rows,
    }


def persist_classifier_states(
    run_directory: Path,
    *,
    initial_parameters: Mapping[str, Tensor],
    model: nn.Module,
) -> dict[str, Any]:
    initial = _classifier_state(initial_parameters)
    final = _classifier_state(dict(model.named_parameters()))
    if not initial or not final:
        raise FormalEvidenceError("classifier parameter namespace is absent")
    initial_path = run_directory / "classifier_state_initial.pt"
    final_path = run_directory / "classifier_state_final.pt"
    atomic_torch_save(initial_path, {"state_dict": initial})
    atomic_torch_save(final_path, {"state_dict": final})
    return {
        "initial": {
            **_state_summary(initial),
            "path": initial_path.name,
            "file_sha256": sha256_file(initial_path),
        },
        "final": {
            **_state_summary(final),
            "path": final_path.name,
            "file_sha256": sha256_file(final_path),
        },
    }


def persist_final_checkpoint(
    run_directory: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    run_id: str,
    profile: str,
    gate: str,
    runtime_source_head: str,
    activation_fingerprint: str,
    cohort_fingerprint: str,
    seed: int,
    epoch: int,
    optimizer_step: int,
    formal_evidence: bool = True,
) -> dict[str, Any]:
    checkpoint_path = run_directory / "final_checkpoint.pt"
    classifier_state = {
        name: value.detach().to(device="cpu")
        for name, value in model.state_dict().items()
        if name.startswith("classifier.")
    }
    metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": "FINAL_FORMAL_STATE_CHECKPOINT",
        "validation_selection_used": False,
        "run_id": run_id,
        "profile": profile,
        "gate": gate,
        "runtime_source_head": runtime_source_head,
        "activation_fingerprint": activation_fingerprint,
        "cohort_fingerprint": cohort_fingerprint,
        "seed": int(seed),
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "formal_evidence": bool(formal_evidence),
    }
    atomic_torch_save(
        checkpoint_path,
        {
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier_state,
            "optimizer_state_dict": (
                None if optimizer is None else optimizer.state_dict()
            ),
        },
    )
    manifest: dict[str, Any] = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "checkpoint_path": checkpoint_path.name,
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
        "run_id": run_id,
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "profile": profile,
        "gate": gate,
        "formal_evidence": bool(formal_evidence),
        "checkpoint_kind": "FINAL_FORMAL_STATE_CHECKPOINT",
        "validation_selection_used": False,
    }
    manifest["canonical_sha256"] = canonical_sha256(manifest)
    atomic_json(run_directory / "checkpoint_manifest.json", manifest)
    return manifest


def verify_checkpoint_manifest(
    run_directory: Path,
    manifest: Mapping[str, Any],
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    bare = dict(manifest)
    observed_canonical = bare.pop("canonical_sha256", None)
    if observed_canonical != canonical_sha256(bare):
        raise FormalEvidenceError("checkpoint manifest canonical hash mismatch")
    if manifest.get("run_id") != expected_run_id:
        raise FormalEvidenceError("checkpoint manifest run identity mismatch")
    path = run_directory / str(manifest.get("checkpoint_path"))
    if not path.is_file() or sha256_file(path) != manifest.get("sha256"):
        raise FormalEvidenceError("checkpoint file SHA-256 mismatch")
    if path.stat().st_size != manifest.get("size_bytes"):
        raise FormalEvidenceError("checkpoint file size mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("metadata", {}).get("run_id") != expected_run_id:
        raise FormalEvidenceError("checkpoint payload run identity mismatch")
    if checkpoint.get("metadata", {}).get("validation_selection_used") is not False:
        raise FormalEvidenceError("tiny checkpoint cannot use validation selection")
    return {"verdict": "PASS", "checkpoint_sha256": manifest["sha256"]}


def _observation_extract(
    snapshots: Sequence[Mapping[str, Any]], key: str
) -> list[dict[str, Any]]:
    return [
        {"snapshot_index": index, "value": snapshot.get(key, {})}
        for index, snapshot in enumerate(snapshots)
        if snapshot.get(key)
    ]


def _write_supporting_summaries(
    run_directory: Path,
    *,
    result: Mapping[str, Any],
    parameter_evidence: Mapping[str, Any],
    classifier_state: Mapping[str, Any],
) -> None:
    snapshots = result.get("observability_snapshots") or result.get(
        "training_observability_history"
    ) or []
    summaries: dict[str, Mapping[str, Any]] = {
        "diagnostics_summary.json": {
            "schema_version": "task39s_2ar_diagnostics_summary_v1",
            "snapshot_count": len(snapshots),
            "diagnostics_non_interference": (
                result.get("final_metrics", {})
                .get("qualification_prerequisite_checks", {})
                .get("diagnostics_non_interference_pass")
            ),
            "final_finite_state_hashes": result.get("final_finite_state_hashes"),
        },
        "layer_separation.json": {
            "schema_version": "task39s_2ar_layer_separation_v1",
            "earliest_low_variance_or_nonseparable_layer": result.get(
                "earliest_low_variance_or_nonseparable_layer"
            ),
            "snapshot_count": len(snapshots),
        },
        "gradient_summary.json": {
            "schema_version": "task39s_2ar_gradient_summary_v1",
            "gradient_history": result.get("gradient_history", []),
        },
        "parameter_update_summary.json": {
            "schema_version": "task39s_2ar_parameter_update_summary_v1",
            "parameter_updates": result.get("parameter_updates", {}),
            "parameter_evidence_sha256": parameter_evidence.get("canonical_sha256"),
            "gate_a_backbone_freeze_contract": parameter_evidence.get(
                "gate_a_backbone_freeze_contract"
            ),
        },
        "mask_pooling_summary.json": {
            "schema_version": "task39s_2ar_mask_pooling_summary_v1",
            "pooling": _observation_extract(snapshots, "pooling"),
            "mask": _observation_extract(snapshots, "mask"),
        },
        "attention_summary.json": {
            "schema_version": "task39s_2ar_attention_summary_v1",
            "attention": _observation_extract(snapshots, "attention"),
        },
        "transformer_summary.json": {
            "schema_version": "task39s_2ar_transformer_summary_v1",
            "tensor_rows": [
                {
                    "snapshot_index": index,
                    "roles": {
                        name: rows
                        for name, rows in (snapshot.get("tensors") or {}).items()
                        if "TRANSFORMER" in name.upper()
                    },
                }
                for index, snapshot in enumerate(snapshots)
            ],
        },
        "classifier_summary.json": {
            "schema_version": "task39s_2ar_classifier_summary_v1",
            "classifier_state": classifier_state,
            "classifier_hash_before": parameter_evidence.get(
                "classifier_hash_before"
            ),
            "classifier_hash_after": parameter_evidence.get("classifier_hash_after"),
        },
        "collapse_classification.json": {
            "schema_version": "task39s_2ar_collapse_classification_v1",
            "formal_taxonomy": result.get("formal_taxonomy", result.get("status")),
            "collapse_detected": result.get("final_metrics", {}).get(
                "collapse_detected"
            ),
            "collapse_flags": result.get("final_metrics", {}).get(
                "collapse_flags", []
            ),
        },
        "failure_localization.json": {
            "schema_version": "task39s_2ar_failure_localization_v1",
            **dict(result.get("failure_localization") or {}),
        },
    }
    for name, payload in summaries.items():
        value = dict(payload)
        value["canonical_sha256"] = canonical_sha256(value)
        atomic_json(run_directory / name, value)


def _artifact_entry(
    run_directory: Path,
    name: str,
    *,
    required: bool,
    producer: str,
    evidence_level: str,
    absence_reason: str | None = None,
) -> dict[str, Any]:
    path = run_directory / name
    present = path.is_file()
    return {
        "path": name,
        "required": required,
        "present": present,
        "size_bytes": path.stat().st_size if present else None,
        "sha256": sha256_file(path) if present else None,
        "producer": producer,
        "evidence_level": evidence_level,
        "absence_reason": None if present else absence_reason,
    }


def _build_artifact_manifest(
    run_directory: Path,
    *,
    run_id: str,
    formal_success: bool,
    formal_project_evidence: bool = True,
    evidence_level: str = "FORMAL_PROJECT_EVIDENCE",
    absence_reason: str | None = None,
) -> dict[str, Any]:
    names = list(REQUIRED_FORMAL_ARTIFACTS) + [
        "result.json",
        "final_checkpoint.pt",
        "classifier_state_initial.pt",
        "classifier_state_final.pt",
        "model_parameter_state_hash.json",
    ]
    entries = [
        _artifact_entry(
            run_directory,
            name,
            required=name in REQUIRED_FORMAL_ARTIFACTS,
            producer="task39s.formal_evidence",
            evidence_level=evidence_level,
            absence_reason=absence_reason,
        )
        for name in names
    ]
    missing_required = [row["path"] for row in entries if row["required"] and not row["present"]]
    if formal_success and missing_required:
        raise FormalEvidenceError(
            "required formal artifacts missing: " + ", ".join(missing_required)
        )
    payload: dict[str, Any] = {
        "schema_version": FORMAL_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "formal_evidence": bool(formal_project_evidence),
        "formal_success_artifact_contract": formal_success,
        "artifact_count": len(entries),
        "artifacts": entries,
        "missing_required_artifacts": missing_required,
        "all_required_present": not missing_required,
    }
    payload["formal_artifact_manifest_sha256"] = canonical_sha256(payload)
    return payload


def persist_formal_evidence(
    *,
    run_directory: Path,
    result: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    initial_parameters: Mapping[str, Tensor],
    sample_ids: Sequence[str],
    labels: Sequence[int] | Tensor,
    raw_outputs: Sequence[float] | Sequence[Sequence[float]] | Tensor,
    activation_identity: Mapping[str, Any],
    cohort_fingerprint: str,
    seed: int,
    epoch: int,
    optimizer_step: int,
    metric_contract_path: Path,
    formal_project_evidence: bool = True,
    evidence_level: str = "FORMAL_PROJECT_EVIDENCE",
) -> dict[str, Any]:
    """Persist and verify every artifact needed by a successful terminal row."""

    run_directory.mkdir(parents=True, exist_ok=True)
    run_id = str(result["run_id"])
    profile = str(result["profile"])
    gate = str(result["gate"])
    runtime_metrics = result.get("final_metrics")
    if not isinstance(runtime_metrics, Mapping):
        raise FormalEvidenceError("runtime final metrics are absent")
    prediction = build_prediction_payload(
        sample_ids=sample_ids,
        labels=labels,
        raw_outputs=raw_outputs,
        profile=profile,
    )
    result_path = run_directory / "result.json"
    if not result_path.exists():
        atomic_json(result_path, dict(result))
    atomic_json(run_directory / "run_manifest.json", dict(result))
    atomic_json(run_directory / "activation_identity.json", dict(activation_identity))
    cohort_identity = {
        "schema_version": "task39s_2ar_cohort_identity_v1",
        "run_id": run_id,
        "dataset": result.get("dataset"),
        "split": "train" if formal_project_evidence else "synthetic",
        "cohort_fingerprint": cohort_fingerprint,
        "sample_count": len(sample_ids),
        "seed": int(seed),
    }
    cohort_identity["canonical_sha256"] = canonical_sha256(cohort_identity)
    atomic_json(run_directory / "cohort_identity.json", cohort_identity)
    atomic_json(run_directory / "final_predictions.json", prediction)
    final_metrics = dict(runtime_metrics)
    final_metrics["canonical_sha256"] = canonical_sha256(final_metrics)
    atomic_json(run_directory / "final_metrics.json", final_metrics)
    recomputed = recompute_metrics_from_files(
        run_directory / "final_predictions.json",
        metric_contract_path,
    )
    atomic_json(run_directory / "recomputed_metrics.json", recomputed)
    consistency = compare_runtime_and_recomputed_metrics(
        runtime_metrics, recomputed
    )
    if consistency["status"] != "PASS":
        raise FormalEvidenceError("runtime/recomputed metric consistency failed")
    atomic_json(run_directory / "metric_consistency.json", consistency)

    checkpoint = persist_final_checkpoint(
        run_directory,
        model=model,
        optimizer=optimizer,
        run_id=run_id,
        profile=profile,
        gate=gate,
        runtime_source_head=str(result["runtime_source_head"]),
        activation_fingerprint=str(result["activation_fingerprint"]),
        cohort_fingerprint=cohort_fingerprint,
        seed=seed,
        epoch=epoch,
        optimizer_step=optimizer_step,
        formal_evidence=formal_project_evidence,
    )
    verify_checkpoint_manifest(
        run_directory, checkpoint, expected_run_id=run_id
    )
    classifier_state = persist_classifier_states(
        run_directory,
        initial_parameters=initial_parameters,
        model=model,
    )
    parameter_evidence = build_parameter_evidence(
        initial_parameters=initial_parameters,
        model=model,
        gate=gate,
    )
    if (
        gate == "classifier_only"
        and parameter_evidence["gate_a_backbone_freeze_contract"] != "PASS"
    ):
        raise FormalEvidenceError("Gate A backbone freeze contract failed")
    atomic_json(
        run_directory / "model_parameter_state_hash.json", parameter_evidence
    )
    _write_supporting_summaries(
        run_directory,
        result=result,
        parameter_evidence=parameter_evidence,
        classifier_state=classifier_state,
    )
    run_status = (
        f"RUN_ID={run_id}\n"
        f"RUNTIME_COMPLETION_STATUS=COMPLETED\n"
        f"FORMAL_TAXONOMY={result.get('formal_taxonomy', result.get('status'))}\n"
        f"GATE_PASS={'YES' if result.get('assessment', {}).get('passed') else 'NO'}\n"
        "FORMAL_EVIDENCE_COMPLETE=YES\n"
        f"FORMAL_PROJECT_EVIDENCE={'YES' if formal_project_evidence else 'NO'}\n"
    )
    atomic_text(run_directory / "run_status.txt", run_status)
    manifest = _build_artifact_manifest(
        run_directory,
        run_id=run_id,
        formal_success=True,
        formal_project_evidence=formal_project_evidence,
        evidence_level=evidence_level,
    )
    atomic_json(run_directory / "formal_artifact_manifest.json", manifest)
    terminal = {
        "process_return_code": formal_process_exit_code(
            str(result.get("formal_taxonomy", result.get("status"))),
            evidence_complete=True,
        ),
        "runtime_completion_status": "COMPLETED",
        "formal_taxonomy": result.get("formal_taxonomy", result.get("status")),
        "gate_pass": bool(result.get("assessment", {}).get("passed")),
        "optimizer_step_count": int(optimizer_step),
        "artifact_manifest_sha256": sha256_file(
            run_directory / "formal_artifact_manifest.json"
        ),
        "predictions_sha256": sha256_file(run_directory / "final_predictions.json"),
        "final_metrics_sha256": sha256_file(run_directory / "final_metrics.json"),
        "recomputed_metrics_sha256": sha256_file(
            run_directory / "recomputed_metrics.json"
        ),
        "metric_consistency_sha256": sha256_file(
            run_directory / "metric_consistency.json"
        ),
        "checkpoint_manifest_sha256": sha256_file(
            run_directory / "checkpoint_manifest.json"
        ),
        "diagnostics_summary_sha256": sha256_file(
            run_directory / "diagnostics_summary.json"
        ),
        "failure_localization_sha256": sha256_file(
            run_directory / "failure_localization.json"
        ),
        "run_status_sha256": sha256_file(run_directory / "run_status.txt"),
        "absence_reasons": {},
    }
    return {
        "artifact_manifest": manifest,
        "prediction": prediction,
        "recomputed_metrics": recomputed,
        "metric_consistency": consistency,
        "checkpoint_manifest": checkpoint,
        "parameter_evidence": parameter_evidence,
        "classifier_state": classifier_state,
        "terminal_evidence": terminal,
    }


def persist_failure_terminal_evidence(
    *,
    run_directory: Path,
    result: Mapping[str, Any],
    absence_reason: str,
    runtime_completion_status: str = "RUNTIME_FAILURE",
    formal_taxonomy: str = "RUNTIME_FAILURE",
) -> dict[str, Any]:
    """Persist an explicit failure manifest without inventing missing hashes."""

    run_directory.mkdir(parents=True, exist_ok=True)
    run_id = str(result["run_id"])
    run_manifest_path = run_directory / "run_manifest.json"
    if not run_manifest_path.exists():
        atomic_json(run_manifest_path, dict(result))
    failure_path = run_directory / "failure_localization.json"
    if not failure_path.exists():
        payload = {
            "schema_version": "task39s_2ar_failure_localization_v1",
            **dict(result.get("failure_localization") or {}),
            "evidence_persistence_failure": absence_reason,
        }
        payload["canonical_sha256"] = canonical_sha256(payload)
        atomic_json(failure_path, payload)
    status_path = run_directory / "run_status.txt"
    if not status_path.exists():
        atomic_text(
            status_path,
            f"RUN_ID={run_id}\n"
            f"RUNTIME_COMPLETION_STATUS={runtime_completion_status}\n"
            f"FORMAL_TAXONOMY={formal_taxonomy}\n"
            "GATE_PASS=NO\n"
            "FORMAL_EVIDENCE_COMPLETE=NO\n"
            f"ABSENCE_REASON={absence_reason}\n",
        )
    manifest_path = run_directory / "formal_artifact_manifest.json"
    if not manifest_path.exists():
        manifest = _build_artifact_manifest(
            run_directory,
            run_id=run_id,
            formal_success=False,
            absence_reason=absence_reason,
        )
        atomic_json(manifest_path, manifest)
    missing_hashes = {
        "predictions_sha256": "final_predictions.json absent: " + absence_reason,
        "final_metrics_sha256": "final_metrics.json absent: " + absence_reason,
        "recomputed_metrics_sha256": "recomputed_metrics.json absent: " + absence_reason,
        "metric_consistency_sha256": "metric_consistency.json absent: " + absence_reason,
        "checkpoint_manifest_sha256": "checkpoint_manifest.json absent: " + absence_reason,
        "diagnostics_summary_sha256": "diagnostics_summary.json absent: " + absence_reason,
    }
    terminal: dict[str, Any] = {
        "process_return_code": 1,
        "runtime_completion_status": runtime_completion_status,
        "formal_taxonomy": formal_taxonomy,
        "gate_pass": False,
        "optimizer_step_count": int(result.get("optimizer_steps", 0)),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": None,
        "final_metrics_sha256": None,
        "recomputed_metrics_sha256": None,
        "metric_consistency_sha256": None,
        "checkpoint_manifest_sha256": None,
        "diagnostics_summary_sha256": None,
        "failure_localization_sha256": sha256_file(failure_path),
        "run_status_sha256": sha256_file(status_path),
        "absence_reasons": missing_hashes,
    }
    return terminal


def formal_process_exit_code(taxonomy: str, *, evidence_complete: bool) -> int:
    if evidence_complete and taxonomy in LEGAL_MODEL_TERMINAL_STATUSES:
        return 0
    return 1


def build_prediction_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": PREDICTION_SCHEMA_VERSION,
        "title": "Task39S.2AR final prediction evidence",
        "type": "object",
        "required": [
            "prediction_schema_version",
            "profile",
            "positive_class_index",
            "threshold",
            "sample_count",
            "samples",
            "prediction_sha256",
        ],
    }


def build_checkpoint_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "title": "Task39S.2AR final formal checkpoint manifest",
        "type": "object",
        "required": [
            "checkpoint_path",
            "size_bytes",
            "sha256",
            "run_id",
            "epoch",
            "optimizer_step",
            "profile",
            "gate",
            "formal_evidence",
        ],
    }


def build_parameter_evidence_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": PARAMETER_EVIDENCE_SCHEMA_VERSION,
        "title": "Task39S.2AR deterministic parameter state evidence",
        "type": "object",
        "required": [
            "hash_contract",
            "gate",
            "initial",
            "final",
            "backbone_hash_before",
            "backbone_hash_after",
            "classifier_hash_before",
            "classifier_hash_after",
        ],
    }


def build_formal_artifact_manifest_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": FORMAL_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "title": "Task39S.2AR formal artifact closure manifest",
        "type": "object",
        "required": [
            "run_id",
            "formal_evidence",
            "artifact_count",
            "artifacts",
            "all_required_present",
            "formal_artifact_manifest_sha256",
        ],
    }
