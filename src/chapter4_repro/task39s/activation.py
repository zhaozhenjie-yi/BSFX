"""Fail-closed Task39S.2A tiny-gate execution activation boundary."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attempt_ledger import AttemptLedger, AttemptLedgerError, build_attempt_ledger_schema
from .protocol_v3 import ENGINEERING_PROFILE, PAPER_PROFILE
from .source_integrity_v2 import (
    BASE_PROTOCOL_V3_FINGERPRINT,
    S1_SOURCE_BASELINE_HEAD,
    SourceIntegrityError,
    build_evidence_allowlist,
    build_runtime_source_manifest,
    canonical_sha256,
    repository_root_from_module,
    sha256_file,
    verify_runtime_source_integrity,
)


ACTIVATION_ID = "TASK39S_2R_TINY_EXECUTION_ACTIVATION_V1"
ACTIVATION_PURPOSE = "TRAIN_ONLY_TINY_BALANCED_OVERFIT_GATE"
AUTHORIZED_TASK = "TASK39S_2A_AUTHORIZED_TINY_GATE_EXECUTION"
ACTIVATION_MANIFEST_NAME = "protocol_v3_tiny_execution_activation.json"
ACTIVATION_SCHEMA_NAME = "protocol_v3_tiny_execution_activation_schema.json"
ACTIVATION_FINGERPRINT_NAME = (
    "protocol_v3_tiny_execution_activation_fingerprint.txt"
)
ACTIVATION_FREEZE_NAME = (
    "protocol_v3_tiny_execution_activation_freeze_manifest.json"
)
ACTIVATION_V2_MANIFEST_NAME = "protocol_v3_tiny_execution_activation_v2.json"
RUNTIME_MANIFEST_NAME = "task39s_2r_runtime_source_manifest.json"
RUNTIME_HASHES_NAME = "task39s_2r_runtime_source_hashes.sha256"
EVIDENCE_ALLOWLIST_NAME = "task39s_2r_evidence_allowlist.json"
ATTEMPT_LEDGER_SCHEMA_NAME = "task39s_2r_attempt_ledger_schema.json"
TINY_CONTRACT_NAME = "protocol_v3_tiny_overfit_contract.json"
ENTRYPOINT_NAME = "scripts/run_task39s_v3_tiny_overfit.py"
VALIDATOR_NAME = (
    "BSFX/chapter4_repro/chapter4_repro/task39s/activation.py"
)
SOURCE_INTEGRITY_NAME = (
    "BSFX/chapter4_repro/chapter4_repro/task39s/source_integrity_v2.py"
)
ATTEMPT_LEDGER_SOURCE_NAME = (
    "BSFX/chapter4_repro/chapter4_repro/task39s/attempt_ledger.py"
)
AUTHORIZED_RUN_NAMESPACE = "artifacts/chapter4_task39s_tiny_overfit_v3/runs"
ATTEMPT_LEDGER_PATH = (
    "artifacts/chapter4_task39s_tiny_overfit_v3/activation/attempt_ledger.jsonl"
)
ACTIVATION_IDENTITY_PATH = (
    "artifacts/chapter4_task39s_tiny_overfit_v3/activation/activation_identity.json"
)
SEED = 123


class ActivationError(RuntimeError):
    """Fail-closed activation validation error."""


def _run(
    run_id: str,
    sequence: int,
    dataset: str,
    profile: str,
    gate: str,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "run_id": run_id,
        "dataset": dataset,
        "profile": profile,
        "gate": gate,
        "gate_label": "GATE_A_CLASSIFIER_ONLY"
        if gate == "classifier_only"
        else "GATE_B_END_TO_END",
        "seed": SEED,
        "split": "train",
        "artifact_directory": f"{AUTHORIZED_RUN_NAMESPACE}/{run_id}",
        "result_path": f"{AUTHORIZED_RUN_NAMESPACE}/{run_id}/result.json",
    }


AUTHORIZED_RUNS: tuple[dict[str, Any], ...] = (
    _run("T39S2A-01-BIGVUL-PAPER-GATE-A", 1, "bigvul", PAPER_PROFILE, "classifier_only"),
    _run(
        "T39S2A-02-BIGVUL-ENGINEERING-GATE-A",
        2,
        "bigvul",
        ENGINEERING_PROFILE,
        "classifier_only",
    ),
    _run("T39S2A-03-SARD-PAPER-GATE-A", 3, "sard", PAPER_PROFILE, "classifier_only"),
    _run(
        "T39S2A-04-SARD-ENGINEERING-GATE-A",
        4,
        "sard",
        ENGINEERING_PROFILE,
        "classifier_only",
    ),
    _run("T39S2A-05-BIGVUL-PAPER-GATE-B", 5, "bigvul", PAPER_PROFILE, "end_to_end"),
    _run(
        "T39S2A-06-BIGVUL-ENGINEERING-GATE-B",
        6,
        "bigvul",
        ENGINEERING_PROFILE,
        "end_to_end",
    ),
    _run("T39S2A-07-SARD-PAPER-GATE-B", 7, "sard", PAPER_PROFILE, "end_to_end"),
    _run(
        "T39S2A-08-SARD-ENGINEERING-GATE-B",
        8,
        "sard",
        ENGINEERING_PROFILE,
        "end_to_end",
    ),
)

REQUIRED_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "activation_id",
        "activation_purpose",
        "authorized_task",
        "parent_source_head",
        "parent_protocol_v3_fingerprint",
        "runtime_source_head",
        "runtime_source_tree_digest",
        "tiny_contract_sha256",
        "activation_schema_sha256",
        "authorized_entrypoint",
        "authorized_entrypoint_sha256",
        "authorized_run_namespace",
        "authorized_run_count",
        "authorized_runs",
        "authorized_datasets",
        "authorized_profiles",
        "authorized_gate_types",
        "authorized_seeds",
        "cohort_fingerprints",
        "run_matrix_fingerprint",
        "canonical_attempt_limit_per_run",
        "total_attempt_limit",
        "infrastructure_retry_policy",
        "attempt_ledger_path",
        "attempt_ledger_schema_sha256",
        "data_scope",
        "sealed_test_access_allowed",
        "validation_access_allowed",
        "test_access_allowed",
        "v3_d0_allowed",
        "v3_b_active_candidate_count",
        "v3_c_active_extension_count",
        "formula_4_2_included",
        "model_semantics_changed",
        "activation_created_at",
        "activation_fingerprint",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_activation_schema() -> dict[str, Any]:
    properties = {field: {} for field in sorted(REQUIRED_MANIFEST_FIELDS)}
    properties.update(
        {
            "schema_version": {"const": "task39s_2r_tiny_execution_activation_v1"},
            "activation_id": {"const": ACTIVATION_ID},
            "activation_purpose": {"const": ACTIVATION_PURPOSE},
            "authorized_task": {"const": AUTHORIZED_TASK},
            "parent_source_head": {"const": S1_SOURCE_BASELINE_HEAD},
            "parent_protocol_v3_fingerprint": {
                "const": BASE_PROTOCOL_V3_FINGERPRINT
            },
            "authorized_run_count": {"const": 8},
            "canonical_attempt_limit_per_run": {"const": 1},
            "total_attempt_limit": {"const": 8},
            "data_scope": {"const": "TRAIN_ONLY_FROZEN_COHORT"},
            "sealed_test_access_allowed": {"const": False},
            "validation_access_allowed": {"const": False},
            "test_access_allowed": {"const": False},
            "v3_d0_allowed": {"const": False},
            "v3_b_active_candidate_count": {"const": 0},
            "v3_c_active_extension_count": {"const": 0},
            "formula_4_2_included": {"const": False},
            "model_semantics_changed": {"const": False},
            "authorized_runs": {"type": "array", "minItems": 8, "maxItems": 8},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "task39s_2r_tiny_execution_activation_schema_v1",
        "title": "Task39S.2A train-only tiny gate execution activation",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(REQUIRED_MANIFEST_FIELDS),
        "properties": properties,
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ActivationError(f"{label} does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivationError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActivationError(f"{label} must be a JSON object")
    return value


def _contract(root: Path) -> dict[str, Any]:
    payload = _load_json(root / TINY_CONTRACT_NAME, label="base tiny contract")
    if payload.get("execution_authorized_by_this_contract") is not False:
        raise ActivationError(
            "base tiny contract must remain non-authorizing; activation is separate"
        )
    return payload


def _base_active_counts(root: Path) -> tuple[int, int]:
    candidates = _load_json(
        root / "protocol_v3_candidate_register.json", label="candidate register"
    )
    extensions = _load_json(
        root / "protocol_v3_extension_register.json", label="extension register"
    )
    active_b = sum(
        bool(row.get("active_in_v3_base") or row.get("active"))
        for row in candidates.get("candidates", [])
    )
    active_c = sum(
        bool(row.get("active_in_v3_base"))
        for row in extensions.get("extensions", [])
    )
    return active_b, active_c


def activation_manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    bare = dict(manifest)
    bare.pop("activation_fingerprint", None)
    return _sha256_bytes(_canonical_bytes(bare))


def build_activation_fingerprint_payload(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "task39s_2r_activation_fingerprint_payload_v1",
        "activation_id": manifest["activation_id"],
        "base_source_head": manifest["parent_source_head"],
        "base_protocol_v3_fingerprint": manifest[
            "parent_protocol_v3_fingerprint"
        ],
        "runtime_source_head": manifest["runtime_source_head"],
        "runtime_source_tree_digest": manifest["runtime_source_tree_digest"],
        "tiny_contract_sha256": manifest["tiny_contract_sha256"],
        "activation_manifest_payload_sha256": activation_manifest_payload_sha256(
            manifest
        ),
        "activation_schema_sha256": manifest["activation_schema_sha256"],
        "run_matrix_fingerprint": manifest["run_matrix_fingerprint"],
        "authorized_runs": manifest["authorized_runs"],
        "cohort_fingerprints": manifest["cohort_fingerprints"],
        "authorized_entrypoint_sha256": manifest[
            "authorized_entrypoint_sha256"
        ],
        "activation_validator_sha256": sha256_file(root / VALIDATOR_NAME),
        "source_integrity_v2_sha256": sha256_file(root / SOURCE_INTEGRITY_NAME),
        "attempt_ledger_source_sha256": sha256_file(
            root / ATTEMPT_LEDGER_SOURCE_NAME
        ),
        "attempt_ledger_schema_sha256": manifest[
            "attempt_ledger_schema_sha256"
        ],
        "authorized_run_namespace": manifest["authorized_run_namespace"],
        "attempt_ledger_path": manifest["attempt_ledger_path"],
        "data_scope": manifest["data_scope"],
        "sealed_test_access_allowed": manifest["sealed_test_access_allowed"],
        "validation_access_allowed": manifest["validation_access_allowed"],
        "test_access_allowed": manifest["test_access_allowed"],
        "v3_d0_allowed": manifest["v3_d0_allowed"],
        "v3_b_active_candidate_count": manifest[
            "v3_b_active_candidate_count"
        ],
        "v3_c_active_extension_count": manifest[
            "v3_c_active_extension_count"
        ],
        "formula_4_2_included": manifest["formula_4_2_included"],
        "model_semantics_changed": manifest["model_semantics_changed"],
    }


def _atomic_write_new(path: Path, value: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def generate_activation_bundle(
    root: Path,
    *,
    runtime_source_head: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Generate the derived activation freeze after runtime commit A."""

    root = root.resolve()
    targets = [
        root / ACTIVATION_SCHEMA_NAME,
        root / ACTIVATION_MANIFEST_NAME,
        root / ACTIVATION_FINGERPRINT_NAME,
        root / ACTIVATION_FREEZE_NAME,
        root / RUNTIME_MANIFEST_NAME,
        root / RUNTIME_HASHES_NAME,
        root / EVIDENCE_ALLOWLIST_NAME,
        root / ATTEMPT_LEDGER_SCHEMA_NAME,
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite an activation freeze output")

    schema = build_activation_schema()
    schema_bytes = _pretty_bytes(schema)
    ledger_schema = build_attempt_ledger_schema()
    ledger_schema_bytes = _pretty_bytes(ledger_schema)
    allowlist = build_evidence_allowlist()
    runtime_manifest, runtime_hashes = build_runtime_source_manifest(
        root, runtime_source_head=runtime_source_head
    )
    contract = _contract(root)
    active_b, active_c = _base_active_counts(root)
    if active_b != 0 or active_c != 0:
        raise ActivationError("cannot generate activation with active V3-B/V3-C")
    cohort_fingerprints = {
        dataset: contract["sample_manifests"][dataset]["sample_fingerprint"]
        for dataset in ("bigvul", "sard")
    }
    runs = [dict(row) for row in AUTHORIZED_RUNS]
    manifest: dict[str, Any] = {
        "schema_version": "task39s_2r_tiny_execution_activation_v1",
        "activation_id": ACTIVATION_ID,
        "activation_purpose": ACTIVATION_PURPOSE,
        "authorized_task": AUTHORIZED_TASK,
        "parent_source_head": S1_SOURCE_BASELINE_HEAD,
        "parent_protocol_v3_fingerprint": BASE_PROTOCOL_V3_FINGERPRINT,
        "runtime_source_head": runtime_source_head,
        "runtime_source_tree_digest": runtime_manifest[
            "runtime_source_tree_digest"
        ],
        "tiny_contract_sha256": sha256_file(root / TINY_CONTRACT_NAME),
        "activation_schema_sha256": _sha256_bytes(schema_bytes),
        "authorized_entrypoint": ENTRYPOINT_NAME,
        "authorized_entrypoint_sha256": sha256_file(root / ENTRYPOINT_NAME),
        "authorized_run_namespace": AUTHORIZED_RUN_NAMESPACE,
        "authorized_run_count": 8,
        "authorized_runs": runs,
        "authorized_datasets": ["bigvul", "sard"],
        "authorized_profiles": [PAPER_PROFILE, ENGINEERING_PROFILE],
        "authorized_gate_types": ["classifier_only", "end_to_end"],
        "authorized_seeds": [SEED],
        "cohort_fingerprints": cohort_fingerprints,
        "run_matrix_fingerprint": canonical_sha256(runs),
        "canonical_attempt_limit_per_run": 1,
        "total_attempt_limit": 8,
        "infrastructure_retry_policy": contract.get(
            "infrastructure_retry_policy",
            {
                "allowed": True,
                "only_before_first_optimizer_step": True,
                "same_run_id_seed_config_cohort_required": True,
                "original_failure_evidence_must_be_retained": True,
            },
        ),
        "attempt_ledger_path": ATTEMPT_LEDGER_PATH,
        "attempt_ledger_schema_sha256": _sha256_bytes(ledger_schema_bytes),
        "data_scope": "TRAIN_ONLY_FROZEN_COHORT",
        "sealed_test_access_allowed": False,
        "validation_access_allowed": False,
        "test_access_allowed": False,
        "v3_d0_allowed": False,
        "v3_b_active_candidate_count": 0,
        "v3_c_active_extension_count": 0,
        "formula_4_2_included": False,
        "model_semantics_changed": False,
        "activation_created_at": created_at
        or datetime.now().astimezone().isoformat(timespec="seconds"),
        "activation_fingerprint": "PENDING",
    }
    fingerprint_payload = build_activation_fingerprint_payload(root, manifest)
    fingerprint = canonical_sha256(fingerprint_payload)
    manifest["activation_fingerprint"] = fingerprint
    manifest_bytes = _pretty_bytes(manifest)
    freeze: dict[str, Any] = {
        "schema_version": "task39s_2r_activation_freeze_manifest_v1",
        "activation_id": ACTIVATION_ID,
        "freeze_status": "FROZEN",
        "verdict": "PASS",
        "base_protocol_v3_fingerprint": BASE_PROTOCOL_V3_FINGERPRINT,
        "runtime_source_head": runtime_source_head,
        "activation_fingerprint": fingerprint,
        "activation_fingerprint_name": "TASK39S_2R_ACTIVATION_FINGERPRINT",
        "activation_fingerprint_payload": fingerprint_payload,
        "activation_manifest_sha256": _sha256_bytes(manifest_bytes),
        "activation_manifest_payload_sha256": activation_manifest_payload_sha256(
            manifest
        ),
        "activation_schema_sha256": _sha256_bytes(schema_bytes),
        "runtime_source_manifest_sha256": _sha256_bytes(
            _pretty_bytes(runtime_manifest)
        ),
        "evidence_allowlist_sha256": _sha256_bytes(_pretty_bytes(allowlist)),
        "attempt_ledger_schema_sha256": _sha256_bytes(ledger_schema_bytes),
        "authorized_run_count": 8,
        "formal_tiny_run_attempted_at_freeze": 0,
        "formal_optimizer_step_count_at_freeze": 0,
        "v3_d0_run_count_at_freeze": 0,
    }
    freeze["canonical_sha256"] = canonical_sha256(freeze)
    for path, value in (
        (root / ACTIVATION_SCHEMA_NAME, schema_bytes),
        (root / RUNTIME_MANIFEST_NAME, _pretty_bytes(runtime_manifest)),
        (root / RUNTIME_HASHES_NAME, runtime_hashes.encode("utf-8")),
        (root / EVIDENCE_ALLOWLIST_NAME, _pretty_bytes(allowlist)),
        (root / ATTEMPT_LEDGER_SCHEMA_NAME, ledger_schema_bytes),
        (root / ACTIVATION_MANIFEST_NAME, manifest_bytes),
        (root / ACTIVATION_FINGERPRINT_NAME, (fingerprint + "\n").encode("ascii")),
        (root / ACTIVATION_FREEZE_NAME, _pretty_bytes(freeze)),
    ):
        _atomic_write_new(path, value)
    return {
        "verdict": "PASS",
        "activation_fingerprint": fingerprint,
        "runtime_source_head": runtime_source_head,
        "authorized_run_count": 8,
        "outputs": [str(path.relative_to(root)) for path in targets],
    }


_VALIDATION_GUARD = object()


class ValidatedActivationContext:
    """Opaque, run-bound proof returned only by successful validation."""

    __slots__ = (
        "_guard",
        "activation_fingerprint",
        "runtime_source_head",
        "run",
        "manifest_path",
        "source_integrity",
        "ledger_path",
        "authorized_run_ids",
        "total_attempt_limit",
        "ledger_schema_version",
        "activation_generation",
    )

    def __init__(
        self,
        guard: object,
        *,
        activation_fingerprint: str,
        runtime_source_head: str,
        run: Mapping[str, Any],
        manifest_path: Path,
        source_integrity: Mapping[str, Any],
        ledger_path: Path,
        authorized_run_ids: Sequence[str],
        total_attempt_limit: int,
        ledger_schema_version: str = "task39s_2r_attempt_ledger_record_v1",
        activation_generation: str = "S2R_V1",
    ) -> None:
        if guard is not _VALIDATION_GUARD:
            raise ActivationError("validated context cannot be constructed externally")
        self._guard = guard
        self.activation_fingerprint = activation_fingerprint
        self.runtime_source_head = runtime_source_head
        self.run = dict(run)
        self.manifest_path = manifest_path
        self.source_integrity = dict(source_integrity)
        self.ledger_path = ledger_path
        self.authorized_run_ids = tuple(authorized_run_ids)
        self.total_attempt_limit = int(total_attempt_limit)
        self.ledger_schema_version = str(ledger_schema_version)
        self.activation_generation = str(activation_generation)

    def public_record(self) -> dict[str, Any]:
        return {
            "activation_fingerprint": self.activation_fingerprint,
            "runtime_source_head": self.runtime_source_head,
            "run": dict(self.run),
            "manifest_path": str(self.manifest_path),
            "source_integrity_verdict": self.source_integrity.get("verdict"),
            "ledger_path": str(self.ledger_path),
            "ledger_schema_version": self.ledger_schema_version,
            "activation_generation": self.activation_generation,
        }


def require_validated_context(
    context: ValidatedActivationContext | None,
    *,
    run_id: str,
    dataset: str,
    profile: str,
    gate: str,
    seed: int = SEED,
) -> ValidatedActivationContext:
    if not isinstance(context, ValidatedActivationContext):
        raise ActivationError("real tiny execution requires validated activation context")
    if context._guard is not _VALIDATION_GUARD:
        raise ActivationError("validated activation context guard mismatch")
    expected = {
        "run_id": run_id,
        "dataset": dataset,
        "profile": profile,
        "gate": gate,
        "seed": seed,
    }
    if any(context.run.get(key) != value for key, value in expected.items()):
        raise ActivationError("validated activation context cannot be reused across runs")
    return context


def _validate_schema(schema: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if schema != build_activation_schema():
        raise ActivationError("activation schema drift")
    if set(manifest) != REQUIRED_MANIFEST_FIELDS:
        missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
        extra = sorted(set(manifest) - REQUIRED_MANIFEST_FIELDS)
        raise ActivationError(
            f"activation schema field mismatch; missing={missing}, extra={extra}"
        )


def validate_activation(
    manifest_path: Path | None,
    *,
    run_id: str,
    dataset: str,
    profile: str,
    gate: str,
    seed: int = SEED,
    repository_root: Path | None = None,
    require_canonical_path: bool = True,
) -> ValidatedActivationContext:
    root = (repository_root or repository_root_from_module()).resolve()
    if manifest_path is None:
        raise ActivationError("--activation-manifest is required for real tiny gates")
    manifest_path = manifest_path.resolve()
    canonical_path = (root / ACTIVATION_MANIFEST_NAME).resolve()
    canonical_v2_path = (root / ACTIVATION_V2_MANIFEST_NAME).resolve()
    from .activation_v3 import ACTIVATION_V3_MANIFEST_NAME
    canonical_v3_path = (root / ACTIVATION_V3_MANIFEST_NAME).resolve()
    from .activation_task39s4 import TASK39S4_ACTIVATION_MANIFEST_NAME
    canonical_task39s4_path = (root / TASK39S4_ACTIVATION_MANIFEST_NAME).resolve()
    from .activation_task39s5a1 import TASK39S5A1_ACTIVATION_MANIFEST_NAME
    canonical_task39s5a1_path = (
        root / TASK39S5A1_ACTIVATION_MANIFEST_NAME
    ).resolve()
    from .activation_task39s5a3 import TASK39S5A3_ACTIVATION_MANIFEST_NAME
    canonical_task39s5a3_path = (
        root / TASK39S5A3_ACTIVATION_MANIFEST_NAME
    ).resolve()
    if require_canonical_path and manifest_path not in {
        canonical_path,
        canonical_v2_path,
        canonical_v3_path,
        canonical_task39s4_path,
        canonical_task39s5a1_path,
        canonical_task39s5a3_path,
    }:
        raise ActivationError("activation manifest path is not the frozen canonical path")
    manifest = _load_json(manifest_path, label="activation manifest")
    if manifest.get("schema_version") == "task39s_2ar_tiny_execution_activation_v2":
        from .activation_v2 import validate_activation_v2

        return validate_activation_v2(
            manifest_path,
            manifest=manifest,
            run_id=run_id,
            dataset=dataset,
            profile=profile,
            gate=gate,
            seed=seed,
            repository_root=root,
            require_canonical_path=require_canonical_path,
        )
    if manifest.get("schema_version") == "task39s_2ar_2_tiny_recovery_activation_v3":
        from .activation_v3 import validate_activation_v3

        return validate_activation_v3(
            manifest_path,
            manifest=manifest,
            run_id=run_id,
            dataset=dataset,
            profile=profile,
            gate=gate,
            seed=seed,
            repository_root=root,
            require_canonical_path=require_canonical_path,
        )
    if manifest.get("schema_version") == "task39s_4_classifier_budget_activation_v1":
        from .activation_task39s4 import validate_task39s4_activation

        return validate_task39s4_activation(
            manifest_path,
            manifest=manifest,
            run_id=run_id,
            dataset=dataset,
            profile=profile,
            gate=gate,
            seed=seed,
            repository_root=root,
            require_canonical_path=require_canonical_path,
        )
    if manifest.get("schema_version") == "task39s_5a1_classifier_dropout_activation_v1":
        from .activation_task39s5a1 import validate_task39s5a1_activation

        return validate_task39s5a1_activation(
            manifest_path,
            manifest=manifest,
            run_id=run_id,
            dataset=dataset,
            profile=profile,
            gate=gate,
            seed=seed,
            repository_root=root,
            require_canonical_path=require_canonical_path,
        )
    if manifest.get("schema_version") == "task39s_5a3_factorial_activation_v1":
        from .activation_task39s5a3 import validate_task39s5a3_activation

        return validate_task39s5a3_activation(
            manifest_path,
            manifest=manifest,
            run_id=run_id,
            dataset=dataset,
            profile=profile,
            gate=gate,
            seed=seed,
            repository_root=root,
            require_canonical_path=require_canonical_path,
        )
    if require_canonical_path and manifest_path != canonical_path:
        raise ActivationError("activation manifest is not the S2R v1 canonical path")
    schema_path = root / ACTIVATION_SCHEMA_NAME
    schema = _load_json(schema_path, label="activation schema")
    _validate_schema(schema, manifest)
    if sha256_file(schema_path) != manifest.get("activation_schema_sha256"):
        raise ActivationError("activation schema SHA-256 mismatch")
    fixed = {
        "schema_version": "task39s_2r_tiny_execution_activation_v1",
        "activation_id": ACTIVATION_ID,
        "activation_purpose": ACTIVATION_PURPOSE,
        "authorized_task": AUTHORIZED_TASK,
        "parent_source_head": S1_SOURCE_BASELINE_HEAD,
        "parent_protocol_v3_fingerprint": BASE_PROTOCOL_V3_FINGERPRINT,
        "authorized_entrypoint": ENTRYPOINT_NAME,
        "authorized_run_namespace": AUTHORIZED_RUN_NAMESPACE,
        "authorized_run_count": 8,
        "canonical_attempt_limit_per_run": 1,
        "total_attempt_limit": 8,
        "data_scope": "TRAIN_ONLY_FROZEN_COHORT",
        "sealed_test_access_allowed": False,
        "validation_access_allowed": False,
        "test_access_allowed": False,
        "v3_d0_allowed": False,
        "v3_b_active_candidate_count": 0,
        "v3_c_active_extension_count": 0,
        "formula_4_2_included": False,
        "model_semantics_changed": False,
        "attempt_ledger_path": ATTEMPT_LEDGER_PATH,
    }
    for key, expected in fixed.items():
        if manifest.get(key) != expected:
            raise ActivationError(f"activation invariant mismatch: {key}")
    if (root / "protocol_v3_fingerprint.txt").read_text(encoding="ascii").strip() != BASE_PROTOCOL_V3_FINGERPRINT:
        raise ActivationError("base Protocol V3 fingerprint mismatch")
    contract = _contract(root)
    if sha256_file(root / TINY_CONTRACT_NAME) != manifest["tiny_contract_sha256"]:
        raise ActivationError("base tiny contract SHA-256 mismatch")
    if manifest.get("authorized_runs") != [dict(row) for row in AUTHORIZED_RUNS]:
        raise ActivationError("activation exact 8-run matrix mismatch")
    if manifest.get("run_matrix_fingerprint") != canonical_sha256(
        manifest["authorized_runs"]
    ):
        raise ActivationError("activation run-matrix fingerprint mismatch")
    if manifest.get("authorized_datasets") != ["bigvul", "sard"]:
        raise ActivationError("activation dataset scope mismatch")
    if manifest.get("authorized_profiles") != [PAPER_PROFILE, ENGINEERING_PROFILE]:
        raise ActivationError("activation profile scope mismatch")
    if manifest.get("authorized_gate_types") != ["classifier_only", "end_to_end"]:
        raise ActivationError("activation gate scope mismatch")
    if manifest.get("authorized_seeds") != [SEED]:
        raise ActivationError("activation seed scope mismatch")
    expected_cohorts = {
        dataset_name: contract["sample_manifests"][dataset_name]["sample_fingerprint"]
        for dataset_name in ("bigvul", "sard")
    }
    if manifest.get("cohort_fingerprints") != expected_cohorts:
        raise ActivationError("activation cohort fingerprint mismatch")
    active_b, active_c = _base_active_counts(root)
    if active_b != 0 or active_c != 0:
        raise ActivationError("base V3-B/V3-C activation count is nonzero")
    if sha256_file(root / ENTRYPOINT_NAME) != manifest["authorized_entrypoint_sha256"]:
        raise ActivationError("authorized entrypoint SHA-256 mismatch")
    if sha256_file(root / ATTEMPT_LEDGER_SCHEMA_NAME) != manifest[
        "attempt_ledger_schema_sha256"
    ]:
        raise ActivationError("attempt ledger schema SHA-256 mismatch")
    integrity = verify_runtime_source_integrity(
        root,
        runtime_manifest_path=root / RUNTIME_MANIFEST_NAME,
        evidence_allowlist_path=root / EVIDENCE_ALLOWLIST_NAME,
    )
    if integrity["verdict"] != "PASS":
        raise ActivationError(
            "runtime source integrity failed: " + "; ".join(integrity["issues"])
        )
    if integrity["runtime_source_head"] != manifest["runtime_source_head"]:
        raise ActivationError("activation runtime source head mismatch")
    runtime_manifest = _load_json(
        root / RUNTIME_MANIFEST_NAME, label="runtime source manifest"
    )
    if runtime_manifest.get("runtime_source_tree_digest") != manifest[
        "runtime_source_tree_digest"
    ]:
        raise ActivationError("activation runtime source tree digest mismatch")
    freeze = _load_json(root / ACTIVATION_FREEZE_NAME, label="activation freeze")
    freeze_canonical = freeze.get("canonical_sha256")
    freeze_bare = dict(freeze)
    freeze_bare.pop("canonical_sha256", None)
    if freeze_canonical != canonical_sha256(freeze_bare):
        raise ActivationError("activation freeze canonical hash mismatch")
    if freeze.get("activation_manifest_sha256") != sha256_file(manifest_path):
        raise ActivationError("activation manifest file SHA-256 mismatch")
    if freeze.get("activation_manifest_payload_sha256") != activation_manifest_payload_sha256(manifest):
        raise ActivationError("activation manifest payload SHA-256 mismatch")
    fingerprint_payload = build_activation_fingerprint_payload(root, manifest)
    fingerprint = canonical_sha256(fingerprint_payload)
    fingerprint_file = (root / ACTIVATION_FINGERPRINT_NAME).read_text(
        encoding="ascii"
    ).strip()
    if (
        freeze.get("activation_fingerprint_payload") != fingerprint_payload
        or freeze.get("activation_fingerprint") != fingerprint
        or manifest.get("activation_fingerprint") != fingerprint
        or fingerprint_file != fingerprint
    ):
        raise ActivationError("activation fingerprint mismatch")

    matches = [row for row in manifest["authorized_runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ActivationError("requested run ID is not in the authorized matrix")
    selected = matches[0]
    requested = {
        "dataset": dataset,
        "profile": profile,
        "gate": gate,
        "seed": seed,
    }
    if any(selected.get(key) != value for key, value in requested.items()):
        raise ActivationError("requested run identity does not match authorization")
    ledger_path = root / ATTEMPT_LEDGER_PATH
    ledger = AttemptLedger(
        ledger_path,
        activation_fingerprint=fingerprint,
        authorized_run_ids=[row["run_id"] for row in manifest["authorized_runs"]],
        total_attempt_limit=int(manifest["total_attempt_limit"]),
    )
    try:
        ledger.verify()
        ledger.assert_can_start(run_id)
    except AttemptLedgerError as exc:
        raise ActivationError(f"attempt ledger preflight failed: {exc}") from exc
    return ValidatedActivationContext(
        _VALIDATION_GUARD,
        activation_fingerprint=fingerprint,
        runtime_source_head=str(manifest["runtime_source_head"]),
        run=selected,
        manifest_path=manifest_path,
        source_integrity=integrity,
        ledger_path=ledger_path,
        authorized_run_ids=[row["run_id"] for row in manifest["authorized_runs"]],
        total_attempt_limit=int(manifest["total_attempt_limit"]),
    )


def validate_only_result(context: ValidatedActivationContext) -> dict[str, Any]:
    return {
        "schema_version": "task39s_2r_activation_validate_only_v1",
        "status": "PASS",
        "evidence_class": "ACTIVATION_VALIDATION_ONLY",
        "validated_activation_context": context.public_record(),
        "train_content_read_count": 0,
        "validation_content_read_count": 0,
        "test_content_read_count": 0,
        "sealed_test_content_read_count": 0,
        "model_build_count": 0,
        "optimizer_build_count": 0,
        "optimizer_step_count": 0,
        "formal_attempt_consumed": False,
    }


def dry_run_result(context: ValidatedActivationContext) -> dict[str, Any]:
    result = validate_only_result(context)
    result.update(
        {
            "schema_version": "task39s_2r_activation_dry_run_v1",
            "evidence_class": "DRY_RUN_PREFLIGHT_NOT_FORMAL_EVIDENCE",
            "artifact_namespace": str(
                Path(str(context.run["artifact_directory"])).parent
            ),
            "data_boundary_guard": "TRAIN_ONLY_METADATA_NO_SAMPLE_CONTENT_READ",
            "attempt_ledger_verdict": "PASS",
        }
    )
    return result
