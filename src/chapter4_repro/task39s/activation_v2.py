"""Derived Task39S.2AR activation bound to the repaired runtime commit."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .attempt_ledger import (
    LEDGER_RECORD_V2,
    AttemptLedger,
    AttemptLedgerError,
    build_attempt_ledger_schema_v2,
)
from .formal_evidence import (
    build_checkpoint_schema,
    build_formal_artifact_manifest_schema,
    build_parameter_evidence_schema,
    build_prediction_schema,
)
from .protocol_v3 import ENGINEERING_PROFILE, PAPER_PROFILE
from .source_integrity_v2 import (
    BASE_PROTOCOL_V3_FINGERPRINT,
    S1_SOURCE_BASELINE_HEAD,
    S2R_RUNTIME_SOURCE_HEAD,
    build_s2ar_evidence_allowlist,
    build_s2ar_runtime_source_manifest,
    canonical_sha256,
    repository_root_from_module,
    sha256_file,
    verify_s2ar_runtime_source_integrity,
)


ACTIVATION_V2_ID = "TASK39S_2AR_TINY_EXECUTION_ACTIVATION_V2"
ACTIVATION_V2_MANIFEST_NAME = "protocol_v3_tiny_execution_activation_v2.json"
ACTIVATION_V2_SCHEMA_NAME = "protocol_v3_tiny_execution_activation_v2_schema.json"
ACTIVATION_V2_FINGERPRINT_NAME = "protocol_v3_tiny_execution_activation_v2_fingerprint.txt"
ACTIVATION_V2_FREEZE_NAME = "protocol_v3_tiny_execution_activation_v2_freeze_manifest.json"
RUNTIME_MANIFEST_NAME = "task39s_2ar_runtime_source_manifest.json"
RUNTIME_HASHES_NAME = "task39s_2ar_runtime_source_hashes.sha256"
EVIDENCE_ALLOWLIST_NAME = "task39s_2ar_evidence_allowlist.json"
LEDGER_SCHEMA_NAME = "task39s_2ar_attempt_ledger_schema_v2.json"
PREDICTION_SCHEMA_NAME = "task39s_2ar_prediction_schema.json"
CHECKPOINT_SCHEMA_NAME = "task39s_2ar_checkpoint_schema.json"
PARAMETER_SCHEMA_NAME = "task39s_2ar_parameter_evidence_schema.json"
ARTIFACT_SCHEMA_NAME = "task39s_2ar_formal_artifact_manifest_schema.json"
OLD_ACTIVATION_NAME = "protocol_v3_tiny_execution_activation.json"
OLD_ACTIVATION_FINGERPRINT_NAME = "protocol_v3_tiny_execution_activation_fingerprint.txt"
TINY_CONTRACT_NAME = "protocol_v3_tiny_overfit_contract.json"
ENTRYPOINT_NAME = "scripts/run_task39s_v3_tiny_overfit.py"
RECOMPUTE_TOOL_NAME = "scripts/recompute_task39s_tiny_metrics.py"
ATTEMPT_LEDGER_PATH = "artifacts/chapter4_task39s_tiny_overfit_v3/activation/attempt_ledger.jsonl"
SEED = 123
ORIGINAL_ACTIVATION_FINGERPRINT = "355cfe8a23d8b466cde03dc804aa096853c135ec41302bd0887c4fb89491c6e3"
HISTORICAL_ACTIVATION_FINGERPRINT = ORIGINAL_ACTIVATION_FINGERPRINT
HISTORICAL_ACTIVATION_STATUS = (
    "HISTORICAL_VALID_BUT_RUNTIME_SUPERSEDED_FOR_FORMAL_EXECUTION"
)
RUNTIME_V2_MANIFEST_NAME = RUNTIME_MANIFEST_NAME
LEDGER_V2_SCHEMA_NAME = LEDGER_SCHEMA_NAME


class ActivationV2Error(RuntimeError):
    """Derived activation v2 failed closed."""


def _pretty(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivationV2Error(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActivationV2Error(f"JSON object required: {path}")
    return value


def _atomic(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _active_counts(root: Path) -> tuple[int, int]:
    candidates = _load(root / "protocol_v3_candidate_register.json")
    extensions = _load(root / "protocol_v3_extension_register.json")
    active_b = sum(bool(row.get("active_in_v3_base") or row.get("active")) for row in candidates.get("candidates", []))
    active_c = sum(bool(row.get("active_in_v3_base")) for row in extensions.get("extensions", []))
    return active_b, active_c


def _authorized_runs() -> list[dict[str, Any]]:
    from .activation import AUTHORIZED_RUNS

    return [dict(row) for row in AUTHORIZED_RUNS]


def _required_fields() -> tuple[str, ...]:
    return (
        "schema_version", "activation_id", "activation_generation",
        "activation_purpose", "authorized_task", "parent_source_head",
        "parent_protocol_v3_fingerprint", "historical_s2r_runtime_source_head",
        "historical_s2r_activation_fingerprint", "historical_s2r_activation_status",
        "runtime_source_head", "runtime_source_tree_digest", "tiny_contract_sha256",
        "authorized_entrypoint", "authorized_entrypoint_sha256",
        "metric_recomputation_tool", "metric_recomputation_tool_sha256",
        "prediction_schema_sha256", "checkpoint_schema_sha256",
        "parameter_evidence_schema_sha256", "formal_artifact_manifest_schema_sha256",
        "attempt_ledger_schema_sha256", "attempt_ledger_record_schema_version",
        "authorized_run_namespace", "authorized_run_count", "authorized_runs",
        "authorized_datasets", "authorized_profiles", "authorized_gate_types",
        "authorized_seeds", "cohort_fingerprints", "run_matrix_fingerprint",
        "canonical_attempt_limit_per_run", "total_attempt_limit", "attempt_ledger_path",
        "data_scope", "sealed_test_access_allowed", "validation_access_allowed",
        "test_access_allowed", "v3_d0_allowed", "v3_b_active_candidate_count",
        "v3_c_active_extension_count", "formula_4_2_included",
        "model_semantics_changed", "activation_created_at", "activation_fingerprint",
    )


REQUIRED_V2_FIELDS = frozenset(_required_fields())


def build_activation_v2_schema() -> dict[str, Any]:
    fields = _required_fields()
    properties = {field: {} for field in fields}
    properties.update(
        {
            "schema_version": {"const": "task39s_2ar_tiny_execution_activation_v2"},
            "activation_id": {"const": ACTIVATION_V2_ID},
            "activation_purpose": {"const": "TRAIN_ONLY_TINY_BALANCED_OVERFIT_GATE"},
            "authorized_task": {"const": "TASK39S_2A_REAUTHORIZED_TINY_GATE_EXECUTION"},
            "parent_source_head": {"const": S1_SOURCE_BASELINE_HEAD},
            "parent_protocol_v3_fingerprint": {"const": BASE_PROTOCOL_V3_FINGERPRINT},
            "historical_s2r_runtime_source_head": {"const": S2R_RUNTIME_SOURCE_HEAD},
            "historical_s2r_activation_fingerprint": {"const": HISTORICAL_ACTIVATION_FINGERPRINT},
            "historical_s2r_activation_status": {"const": HISTORICAL_ACTIVATION_STATUS},
            "authorized_run_count": {"const": 8},
            "canonical_attempt_limit_per_run": {"const": 1},
            "total_attempt_limit": {"const": 8},
            "attempt_ledger_record_schema_version": {"const": LEDGER_RECORD_V2},
            "data_scope": {"const": "TRAIN_ONLY_FROZEN_COHORT"},
            "sealed_test_access_allowed": {"const": False},
            "validation_access_allowed": {"const": False},
            "test_access_allowed": {"const": False},
            "v3_d0_allowed": {"const": False},
            "v3_b_active_candidate_count": {"const": 0},
            "v3_c_active_extension_count": {"const": 0},
            "formula_4_2_included": {"const": False},
            "model_semantics_changed": {"const": False},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "task39s_2ar_tiny_execution_activation_v2_schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(fields),
        "properties": properties,
    }


def build_activation_v2_fingerprint_payload(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    bare = dict(manifest)
    bare.pop("activation_fingerprint", None)
    return {
        "schema_version": "task39s_2ar_activation_v2_manifest_payload_v1",
        "manifest_payload": bare,
    }


def activation_v2_fingerprint_payload(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "task39s_2ar_activation_v2_fingerprint_payload_v1",
        "activation_manifest_payload": build_activation_v2_fingerprint_payload(manifest),
        "activation_validator_sha256": sha256_file(Path(__file__)),
        "source_integrity_sha256": sha256_file(root / "BSFX/chapter4_repro/chapter4_repro/task39s/source_integrity_v2.py"),
        "formal_evidence_source_sha256": sha256_file(root / "BSFX/chapter4_repro/chapter4_repro/task39s/formal_evidence.py"),
    }


def generate_activation_v2_bundle(
    root: Path,
    *,
    runtime_source_head: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    output_names = (
        ACTIVATION_V2_SCHEMA_NAME, ACTIVATION_V2_MANIFEST_NAME,
        ACTIVATION_V2_FINGERPRINT_NAME, ACTIVATION_V2_FREEZE_NAME,
        RUNTIME_MANIFEST_NAME, RUNTIME_HASHES_NAME, EVIDENCE_ALLOWLIST_NAME,
        LEDGER_SCHEMA_NAME, PREDICTION_SCHEMA_NAME, CHECKPOINT_SCHEMA_NAME,
        PARAMETER_SCHEMA_NAME, ARTIFACT_SCHEMA_NAME,
    )
    if any((root / name).exists() for name in output_names):
        raise FileExistsError("refusing to overwrite activation v2 freeze")
    if (root / "protocol_v3_fingerprint.txt").read_text(encoding="ascii").strip() != BASE_PROTOCOL_V3_FINGERPRINT:
        raise ActivationV2Error("Base Protocol V3 fingerprint drift")
    contract = _load(root / TINY_CONTRACT_NAME)
    if contract.get("execution_authorized_by_this_contract") is not False:
        raise ActivationV2Error("base tiny contract must remain non-authorizing")
    if _active_counts(root) != (0, 0):
        raise ActivationV2Error("V3-B/V3-C must remain inactive")
    old_fp = (root / OLD_ACTIVATION_FINGERPRINT_NAME).read_text(encoding="ascii").strip()
    if old_fp != ORIGINAL_ACTIVATION_FINGERPRINT:
        raise ActivationV2Error("historical S2R activation fingerprint drift")
    old_manifest = _load(root / OLD_ACTIVATION_NAME)
    if old_manifest.get("activation_fingerprint") != old_fp:
        raise ActivationV2Error("historical S2R activation manifest drift")

    schema = build_activation_v2_schema()
    ledger_schema = build_attempt_ledger_schema_v2()
    prediction_schema = build_prediction_schema()
    checkpoint_schema = build_checkpoint_schema()
    parameter_schema = build_parameter_evidence_schema()
    artifact_schema = build_formal_artifact_manifest_schema()
    allowlist = build_s2ar_evidence_allowlist()
    runtime_manifest, runtime_hashes = build_s2ar_runtime_source_manifest(
        root, runtime_source_head=runtime_source_head
    )
    runs = _authorized_runs()
    cohorts = {
        dataset: contract["sample_manifests"][dataset]["sample_fingerprint"]
        for dataset in ("bigvul", "sard")
    }
    manifest: dict[str, Any] = {
        "schema_version": "task39s_2ar_tiny_execution_activation_v2",
        "activation_id": ACTIVATION_V2_ID,
        "activation_generation": "S2AR_V2",
        "activation_purpose": "TRAIN_ONLY_TINY_BALANCED_OVERFIT_GATE",
        "authorized_task": "TASK39S_2A_REAUTHORIZED_TINY_GATE_EXECUTION",
        "parent_source_head": S1_SOURCE_BASELINE_HEAD,
        "parent_protocol_v3_fingerprint": BASE_PROTOCOL_V3_FINGERPRINT,
        "historical_s2r_runtime_source_head": S2R_RUNTIME_SOURCE_HEAD,
        "historical_s2r_activation_fingerprint": old_fp,
        "historical_s2r_activation_status": HISTORICAL_ACTIVATION_STATUS,
        "runtime_source_head": runtime_source_head,
        "runtime_source_tree_digest": runtime_manifest["runtime_source_tree_digest"],
        "tiny_contract_sha256": sha256_file(root / TINY_CONTRACT_NAME),
        "authorized_entrypoint": ENTRYPOINT_NAME,
        "authorized_entrypoint_sha256": sha256_file(root / ENTRYPOINT_NAME),
        "metric_recomputation_tool": RECOMPUTE_TOOL_NAME,
        "metric_recomputation_tool_sha256": sha256_file(root / RECOMPUTE_TOOL_NAME),
        "prediction_schema_sha256": hashlib.sha256(_pretty(prediction_schema)).hexdigest(),
        "checkpoint_schema_sha256": hashlib.sha256(_pretty(checkpoint_schema)).hexdigest(),
        "parameter_evidence_schema_sha256": hashlib.sha256(_pretty(parameter_schema)).hexdigest(),
        "formal_artifact_manifest_schema_sha256": hashlib.sha256(_pretty(artifact_schema)).hexdigest(),
        "attempt_ledger_schema_sha256": hashlib.sha256(_pretty(ledger_schema)).hexdigest(),
        "attempt_ledger_record_schema_version": LEDGER_RECORD_V2,
        "authorized_run_namespace": "artifacts/chapter4_task39s_tiny_overfit_v3/runs",
        "authorized_run_count": 8,
        "authorized_runs": runs,
        "authorized_datasets": ["bigvul", "sard"],
        "authorized_profiles": [PAPER_PROFILE, ENGINEERING_PROFILE],
        "authorized_gate_types": ["classifier_only", "end_to_end"],
        "authorized_seeds": [SEED],
        "cohort_fingerprints": cohorts,
        "run_matrix_fingerprint": canonical_sha256(runs),
        "canonical_attempt_limit_per_run": 1,
        "total_attempt_limit": 8,
        "attempt_ledger_path": ATTEMPT_LEDGER_PATH,
        "data_scope": "TRAIN_ONLY_FROZEN_COHORT",
        "sealed_test_access_allowed": False,
        "validation_access_allowed": False,
        "test_access_allowed": False,
        "v3_d0_allowed": False,
        "v3_b_active_candidate_count": 0,
        "v3_c_active_extension_count": 0,
        "formula_4_2_included": False,
        "model_semantics_changed": False,
        "activation_created_at": created_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "activation_fingerprint": "PENDING",
    }
    fingerprint = canonical_sha256(activation_v2_fingerprint_payload(root, manifest))
    manifest["activation_fingerprint"] = fingerprint
    freeze: dict[str, Any] = {
        "schema_version": "task39s_2ar_activation_v2_freeze_manifest_v1",
        "freeze_status": "FROZEN",
        "verdict": "PASS",
        "runtime_source_head": runtime_source_head,
        "base_protocol_v3_fingerprint": BASE_PROTOCOL_V3_FINGERPRINT,
        "historical_s2r_activation_fingerprint": old_fp,
        "activation_v2_fingerprint": fingerprint,
        "activation_v2_fingerprint_payload": activation_v2_fingerprint_payload(root, manifest),
        "activation_manifest_sha256": hashlib.sha256(_pretty(manifest)).hexdigest(),
        "runtime_source_manifest_sha256": hashlib.sha256(_pretty(runtime_manifest)).hexdigest(),
        "formal_attempt_count_at_freeze": 0,
        "formal_optimizer_step_count_at_freeze": 0,
        "v3_d0_run_count_at_freeze": 0,
    }
    freeze["canonical_sha256"] = canonical_sha256(freeze)
    values = {
        ACTIVATION_V2_SCHEMA_NAME: _pretty(schema),
        ACTIVATION_V2_MANIFEST_NAME: _pretty(manifest),
        ACTIVATION_V2_FINGERPRINT_NAME: (fingerprint + "\n").encode("ascii"),
        ACTIVATION_V2_FREEZE_NAME: _pretty(freeze),
        RUNTIME_MANIFEST_NAME: _pretty(runtime_manifest),
        RUNTIME_HASHES_NAME: runtime_hashes.encode("utf-8"),
        EVIDENCE_ALLOWLIST_NAME: _pretty(allowlist),
        LEDGER_SCHEMA_NAME: _pretty(ledger_schema),
        PREDICTION_SCHEMA_NAME: _pretty(prediction_schema),
        CHECKPOINT_SCHEMA_NAME: _pretty(checkpoint_schema),
        PARAMETER_SCHEMA_NAME: _pretty(parameter_schema),
        ARTIFACT_SCHEMA_NAME: _pretty(artifact_schema),
    }
    for name, value in values.items():
        _atomic(root / name, value)
    return {
        "verdict": "PASS",
        "runtime_source_head": runtime_source_head,
        "activation_v2_fingerprint": fingerprint,
        "outputs": list(output_names),
    }


def validate_activation_v2(
    manifest_path: Path,
    *,
    manifest: Mapping[str, Any] | None,
    run_id: str,
    dataset: str,
    profile: str,
    gate: str,
    seed: int = SEED,
    repository_root: Path | None = None,
    require_canonical_path: bool = True,
):
    from .activation import (
        _VALIDATION_GUARD,
        ActivationError,
        ValidatedActivationContext,
    )

    root = (repository_root or repository_root_from_module()).resolve()
    path = manifest_path.resolve()
    if require_canonical_path and path != (root / ACTIVATION_V2_MANIFEST_NAME).resolve():
        raise ActivationError("activation v2 manifest path is not canonical")
    payload = dict(manifest or _load(path))
    if set(payload) != set(_required_fields()):
        raise ActivationError("activation v2 field closure mismatch")
    if _load(root / ACTIVATION_V2_SCHEMA_NAME) != build_activation_v2_schema():
        raise ActivationError("activation v2 schema drift")
    fixed = {
        "schema_version": "task39s_2ar_tiny_execution_activation_v2",
        "activation_id": ACTIVATION_V2_ID,
        "activation_generation": "S2AR_V2",
        "parent_source_head": S1_SOURCE_BASELINE_HEAD,
        "parent_protocol_v3_fingerprint": BASE_PROTOCOL_V3_FINGERPRINT,
        "historical_s2r_runtime_source_head": S2R_RUNTIME_SOURCE_HEAD,
        "historical_s2r_activation_fingerprint": ORIGINAL_ACTIVATION_FINGERPRINT,
        "historical_s2r_activation_status": HISTORICAL_ACTIVATION_STATUS,
        "attempt_ledger_record_schema_version": LEDGER_RECORD_V2,
        "authorized_run_count": 8,
        "canonical_attempt_limit_per_run": 1,
        "total_attempt_limit": 8,
        "attempt_ledger_path": ATTEMPT_LEDGER_PATH,
        "data_scope": "TRAIN_ONLY_FROZEN_COHORT",
        "sealed_test_access_allowed": False,
        "validation_access_allowed": False,
        "test_access_allowed": False,
        "v3_d0_allowed": False,
        "v3_b_active_candidate_count": 0,
        "v3_c_active_extension_count": 0,
        "formula_4_2_included": False,
        "model_semantics_changed": False,
    }
    for key, expected in fixed.items():
        if payload.get(key) != expected:
            raise ActivationError(f"activation v2 invariant mismatch: {key}")
    if (root / "protocol_v3_fingerprint.txt").read_text(encoding="ascii").strip() != BASE_PROTOCOL_V3_FINGERPRINT:
        raise ActivationError("Base Protocol V3 fingerprint mismatch")
    contract = _load(root / TINY_CONTRACT_NAME)
    if contract.get("execution_authorized_by_this_contract") is not False:
        raise ActivationError("base tiny contract authorization changed")
    if sha256_file(root / TINY_CONTRACT_NAME) != payload["tiny_contract_sha256"]:
        raise ActivationError("base tiny contract hash mismatch")
    runs = _authorized_runs()
    if payload.get("authorized_runs") != runs or payload.get("run_matrix_fingerprint") != canonical_sha256(runs):
        raise ActivationError("activation v2 exact matrix mismatch")
    expected_cohorts = {dataset_name: contract["sample_manifests"][dataset_name]["sample_fingerprint"] for dataset_name in ("bigvul", "sard")}
    if payload.get("cohort_fingerprints") != expected_cohorts:
        raise ActivationError("activation v2 cohort fingerprint mismatch")
    if _active_counts(root) != (0, 0):
        raise ActivationError("activation v2 V3-B/V3-C boundary failed")
    file_hashes = {
        "authorized_entrypoint_sha256": ENTRYPOINT_NAME,
        "metric_recomputation_tool_sha256": RECOMPUTE_TOOL_NAME,
        "prediction_schema_sha256": PREDICTION_SCHEMA_NAME,
        "checkpoint_schema_sha256": CHECKPOINT_SCHEMA_NAME,
        "parameter_evidence_schema_sha256": PARAMETER_SCHEMA_NAME,
        "formal_artifact_manifest_schema_sha256": ARTIFACT_SCHEMA_NAME,
        "attempt_ledger_schema_sha256": LEDGER_SCHEMA_NAME,
    }
    for field, relative in file_hashes.items():
        if sha256_file(root / relative) != payload.get(field):
            raise ActivationError(f"activation v2 bound file drift: {relative}")
    integrity = verify_s2ar_runtime_source_integrity(
        root,
        runtime_manifest_path=root / RUNTIME_MANIFEST_NAME,
        evidence_allowlist_path=root / EVIDENCE_ALLOWLIST_NAME,
    )
    if integrity["verdict"] != "PASS" or integrity["runtime_source_head"] != payload["runtime_source_head"]:
        raise ActivationError("activation v2 runtime source integrity failed")
    runtime_manifest = _load(root / RUNTIME_MANIFEST_NAME)
    if runtime_manifest.get("runtime_source_tree_digest") != payload["runtime_source_tree_digest"]:
        raise ActivationError("activation v2 runtime tree digest mismatch")
    freeze = _load(root / ACTIVATION_V2_FREEZE_NAME)
    observed = freeze.pop("canonical_sha256", None)
    if observed != canonical_sha256(freeze):
        raise ActivationError("activation v2 freeze canonical hash mismatch")
    fingerprint = canonical_sha256(activation_v2_fingerprint_payload(root, payload))
    if not (
        fingerprint == payload.get("activation_fingerprint")
        == (root / ACTIVATION_V2_FINGERPRINT_NAME).read_text(encoding="ascii").strip()
        == freeze.get("activation_v2_fingerprint")
    ):
        raise ActivationError("activation v2 fingerprint mismatch")
    selected = [row for row in runs if row["run_id"] == run_id]
    if len(selected) != 1:
        raise ActivationError("requested run ID is not authorized")
    run = selected[0]
    requested = {"dataset": dataset, "profile": profile, "gate": gate, "seed": seed}
    if any(run.get(key) != value for key, value in requested.items()):
        raise ActivationError("requested run identity does not match activation v2")
    ledger = AttemptLedger(
        root / ATTEMPT_LEDGER_PATH,
        activation_fingerprint=fingerprint,
        authorized_run_ids=[row["run_id"] for row in runs],
        total_attempt_limit=8,
        record_schema_version=LEDGER_RECORD_V2,
    )
    try:
        ledger.verify()
        ledger.assert_can_start(run_id)
    except AttemptLedgerError as exc:
        raise ActivationError(f"activation v2 ledger preflight failed: {exc}") from exc
    return ValidatedActivationContext(
        _VALIDATION_GUARD,
        activation_fingerprint=fingerprint,
        runtime_source_head=str(payload["runtime_source_head"]),
        run=run,
        manifest_path=path,
        source_integrity=integrity,
        ledger_path=root / ATTEMPT_LEDGER_PATH,
        authorized_run_ids=[row["run_id"] for row in runs],
        total_attempt_limit=8,
        ledger_schema_version=LEDGER_RECORD_V2,
        activation_generation="S2AR_V2",
    )
