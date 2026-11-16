"""Derived runtime-source integrity for Task39S.2R activation evidence.

Protocol V3's base source freeze intentionally remains bound to the S1 commit.
This module adds a second identity layer: runtime authorities are frozen at a
repair commit, while later commits may contain only explicitly allowlisted
activation and evidence paths.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


S1_SOURCE_BASELINE_HEAD = "fc0b3cbf9d87c90c0610662777ff496a9265872d"
BASE_PROTOCOL_V3_FINGERPRINT = (
    "fc1faf4af8d94cf968ae73fab0aafc0c0ed6068827d11bdc5f754933583d4df9"
)
S2R_RUNTIME_SOURCE_HEAD = "fd42f1415eab6d152c462b7a01342530bbcd37a3"

RUNTIME_AUTHORITY_PATHS: tuple[str, ...] = (
    "BSFX/chapter4_repro/chapter4_repro/task39s/__init__.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/activation.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/activation_v2.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/activation_v2.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/attempt_ledger.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/formal_evidence.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/source_integrity_v2.py",
    "scripts/check_task39s_runtime_source_integrity_v2.py",
    "scripts/generate_task39s_tiny_activation.py",
    "scripts/generate_task39s_tiny_activation_v2.py",
    "scripts/recompute_task39s_tiny_metrics.py",
    "scripts/run_task39s_2ar_synthetic_formal_evidence.py",
    "scripts/run_task39s_v3_tiny_overfit.py",
    "scripts/validate_task39s_tiny_activation.py",
    "scripts/validate_task39s_tiny_activation_v2.py",
)

S2AR_EVIDENCE_ALLOWLIST: tuple[dict[str, str], ...] = (
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_v2_schema.json"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_v2.json"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_v2_fingerprint.txt"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_v2_freeze_manifest.json"},
    {"kind": "file", "path": "task39s_2ar_prediction_schema.json"},
    {"kind": "file", "path": "task39s_2ar_checkpoint_schema.json"},
    {"kind": "file", "path": "task39s_2ar_parameter_evidence_schema.json"},
    {"kind": "file", "path": "task39s_2ar_formal_artifact_manifest_schema.json"},
    {"kind": "file", "path": "task39s_2ar_attempt_ledger_schema_v2.json"},
    {"kind": "file", "path": "task39s_2ar_runtime_source_manifest.json"},
    {"kind": "file", "path": "task39s_2ar_runtime_source_hashes.sha256"},
    {"kind": "file", "path": "task39s_2ar_evidence_allowlist.json"},
    {"kind": "file", "path": "task39s_2ar_preserved_s2a_blocker_manifest.json"},
    {"kind": "file", "path": "task39s_2ar_runtime_capability_audit_v2.json"},
    {"kind": "file", "path": "task39s_2ar_activation_v2_validation.json"},
    {"kind": "file", "path": "task39s_2ar_synthetic_formal_evidence_report.json"},
    {"kind": "file", "path": "task39s_2ar_metric_recomputation_test_report.json"},
    {"kind": "file", "path": "task39s_2ar_checkpoint_test_report.json"},
    {"kind": "file", "path": "task39s_2ar_parameter_hash_test_report.json"},
    {"kind": "file", "path": "task39s_2ar_ledger_v2_test_report.json"},
    {"kind": "file", "path": "task39s_2ar_source_integrity_report.json"},
    {"kind": "file", "path": "task39s_2ar_data_access_audit.json"},
    {"kind": "file", "path": "task39s_2ar_test_summary.json"},
    {"kind": "file", "path": "task39s_2ar_evidence_manifest.json"},
    {"kind": "file", "path": "task39s_2ar_checksums.sha256"},
    {"kind": "file", "path": "TASK39S_2AR_REPORT.md"},
    {"kind": "file", "path": "TASK39S_2AR_STATUS.txt"},
    {"kind": "file", "path": "task39s_2a_reauthorized_execution_instructions.md"},
    {"kind": "file", "path": "TASK39S_2A_REPORT.md"},
    {"kind": "file", "path": "TASK39S_2A_STATUS.txt"},
    {"kind": "file", "path": "task39s_2a_gate_matrix.csv"},
    {"kind": "file", "path": "task39s_2a_dataset_profile_summary.csv"},
    {"kind": "file", "path": "task39s_2a_run_manifest.json"},
    {"kind": "file", "path": "task39s_2a_loss_trajectory_summary.csv"},
    {"kind": "file", "path": "task39s_2a_metric_summary.csv"},
    {"kind": "file", "path": "task39s_2a_metric_recomputation_summary.csv"},
    {"kind": "file", "path": "task39s_2a_layer_separation_summary.csv"},
    {"kind": "file", "path": "task39s_2a_gradient_summary.csv"},
    {"kind": "file", "path": "task39s_2a_parameter_update_summary.csv"},
    {"kind": "file", "path": "task39s_2a_attention_summary.csv"},
    {"kind": "file", "path": "task39s_2a_transformer_summary.csv"},
    {"kind": "file", "path": "task39s_2a_c1_pooling_summary.csv"},
    {"kind": "file", "path": "task39s_2a_classifier_summary.csv"},
    {"kind": "file", "path": "task39s_2a_failure_localization.csv"},
    {"kind": "file", "path": "task39s_2a_ab_combination_summary.csv"},
    {"kind": "file", "path": "task39s_2a_data_access_audit.json"},
    {"kind": "file", "path": "task39s_2a_attempt_ledger_audit.json"},
    {"kind": "file", "path": "task39s_2a_protocol_integrity.json"},
    {"kind": "file", "path": "task39s_2a_evidence_manifest.json"},
    {"kind": "file", "path": "task39s_2a_checksums.sha256"},
    {"kind": "file", "path": "task39s_2a_next_stage_recommendation.md"},
    {"kind": "directory", "path": "artifacts/chapter4_task39s_tiny_overfit_v3/s2a"},
    {"kind": "file", "path": "artifacts/chapter4_task39s_tiny_overfit_v3/cohorts/bigvul_real_cohort_verification.json"},
    {"kind": "file", "path": "artifacts/chapter4_task39s_tiny_overfit_v3/cohorts/sard_real_cohort_verification.json"},
    {"kind": "file", "path": "artifacts/chapter4_task39s_tiny_overfit_v3/cohorts/cohort_content_access_audit.json"},
    {"kind": "directory", "path": "artifacts/chapter4_task39s_runtime_blocker_repair"},
)

PROTECTED_BASE_PATHS: tuple[str, ...] = (
    "protocol_v3_tiny_overfit_contract.json",
    "protocol_v3_metric_contract.json",
    "protocol_v3_candidate_register.json",
    "protocol_v3_extension_register.json",
    "protocol_v3_fingerprint.txt",
    "configs/protocol_v3.yaml",
    "configs/protocol_v3_paper.yaml",
    "configs/protocol_v3_engineering.yaml",
)

DEFAULT_EVIDENCE_ALLOWLIST: tuple[dict[str, str], ...] = (
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_schema.json"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation.json"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_fingerprint.txt"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_freeze_manifest.json"},
    {"kind": "file", "path": "task39s_2r_runtime_source_manifest.json"},
    {"kind": "file", "path": "task39s_2r_runtime_source_hashes.sha256"},
    {"kind": "file", "path": "task39s_2r_evidence_allowlist.json"},
    {"kind": "file", "path": "task39s_2r_attempt_ledger_schema.json"},
    {"kind": "file", "path": "task39s_2r_activation_validation_report.json"},
    {"kind": "file", "path": "task39s_2r_source_integrity_report.json"},
    {"kind": "file", "path": "task39s_2r_data_access_audit.json"},
    {"kind": "file", "path": "task39s_2r_preserved_s2_blocker_manifest.json"},
    {"kind": "file", "path": "task39s_2r_test_summary.json"},
    {"kind": "file", "path": "task39s_2r_checksums.sha256"},
    {"kind": "file", "path": "TASK39S_2R_REPORT.md"},
    {"kind": "file", "path": "TASK39S_2R_STATUS.txt"},
    {"kind": "file", "path": "task39s_2a_execution_instructions.md"},
    {"kind": "file", "path": "TASK39S_2_REPORT.md"},
    {"kind": "file", "path": "TASK39S_2_STATUS.txt"},
    {"kind": "file", "path": "task39s_2_checksums.sha256"},
    {"kind": "file", "path": "task39s_2_data_access_audit.json"},
    {"kind": "file", "path": "task39s_2_dataset_profile_summary.csv"},
    {"kind": "file", "path": "task39s_2_evidence_manifest.json"},
    {"kind": "file", "path": "task39s_2_failure_localization.csv"},
    {"kind": "file", "path": "task39s_2_gate_matrix.csv"},
    {"kind": "file", "path": "task39s_2_gradient_update_summary.csv"},
    {"kind": "file", "path": "task39s_2_layer_separation_summary.csv"},
    {"kind": "file", "path": "task39s_2_metric_recomputation_summary.csv"},
    {"kind": "file", "path": "task39s_2_next_stage_recommendation.md"},
    {"kind": "file", "path": "task39s_2_protocol_integrity.json"},
    {"kind": "file", "path": "task39s_2_run_manifest.json"},
    {"kind": "directory", "path": "artifacts/chapter4_task39s_activation_repair"},
    {"kind": "directory", "path": "artifacts/chapter4_task39s_tiny_overfit_v3"},
)


class SourceIntegrityError(RuntimeError):
    """Fail-closed runtime-source identity error."""


def repository_root_from_module() -> Path:
    return Path(__file__).resolve().parents[4]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or "\\" in value
    ):
        raise SourceIntegrityError(f"unsafe repository-relative path: {value!r}")
    return candidate.as_posix()


def _run_git(root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and completed.returncode != 0:
        raise SourceIntegrityError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def git_blob(root: Path, commit: str, relative: str) -> bytes:
    relative = _safe_relative(relative)
    completed = subprocess.run(
        ("git", "show", f"{commit}:{relative}"),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise SourceIntegrityError(
            f"required runtime authority absent from {commit}: {relative}"
        )
    return completed.stdout


def git_blob_if_present(root: Path, commit: str, relative: str) -> bytes | None:
    """Return a committed blob, or ``None`` for frozen post-commit evidence."""

    relative = _safe_relative(relative)
    completed = subprocess.run(
        ("git", "show", f"{commit}:{relative}"),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode == 0:
        return completed.stdout
    return None


def build_evidence_allowlist() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "task39s_2r_evidence_allowlist_v1",
        "policy": "EXACT_FILE_OR_DIRECTORY_PREFIX_NO_GLOBS",
        "entries": [dict(row) for row in DEFAULT_EVIDENCE_ALLOWLIST],
        "runtime_or_protected_paths_may_be_allowlisted": False,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def build_s2ar_evidence_allowlist() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "task39s_2ar_evidence_allowlist_v1",
        "policy": "EXACT_FILE_OR_DIRECTORY_PREFIX_NO_GLOBS",
        "entries": [dict(row) for row in S2AR_EVIDENCE_ALLOWLIST],
        "runtime_or_protected_paths_may_be_allowlisted": False,
        "historical_s2r_runtime_source_head": S2R_RUNTIME_SOURCE_HEAD,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def _canonical_valid(payload: Mapping[str, Any]) -> bool:
    observed = payload.get("canonical_sha256")
    bare = dict(payload)
    bare.pop("canonical_sha256", None)
    return isinstance(observed, str) and observed == canonical_sha256(bare)


def _allowlist_entries(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    if not _canonical_valid(payload):
        raise SourceIntegrityError("evidence allowlist canonical hash mismatch")
    if payload.get("policy") != "EXACT_FILE_OR_DIRECTORY_PREFIX_NO_GLOBS":
        raise SourceIntegrityError("unsupported evidence allowlist policy")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SourceIntegrityError("evidence allowlist entries missing")
    result: list[dict[str, str]] = []
    for row in entries:
        if not isinstance(row, Mapping) or row.get("kind") not in {
            "file",
            "directory",
        }:
            raise SourceIntegrityError("invalid evidence allowlist entry")
        relative = _safe_relative(str(row.get("path", "")))
        if any(
            relative == protected or relative.startswith(protected + "/")
            for protected in (*RUNTIME_AUTHORITY_PATHS, *PROTECTED_BASE_PATHS)
        ):
            raise SourceIntegrityError(
                f"runtime/protected path cannot be evidence-allowlisted: {relative}"
            )
        result.append({"kind": str(row["kind"]), "path": relative})
    return result


def path_is_allowlisted(relative: str, entries: Sequence[Mapping[str, str]]) -> bool:
    relative = _safe_relative(relative)
    for row in entries:
        path = str(row["path"])
        if row["kind"] == "file" and relative == path:
            return True
        if row["kind"] == "directory" and (
            relative == path or relative.startswith(path + "/")
        ):
            return True
    return False


def build_runtime_source_manifest(
    root: Path,
    *,
    runtime_source_head: str,
    schema_version: str = "task39s_2r_runtime_source_manifest_v1",
    evidence_allowlist_path: str = "task39s_2r_evidence_allowlist.json",
) -> tuple[dict[str, Any], str]:
    root = root.resolve()
    if _run_git(root, "rev-parse", runtime_source_head) != runtime_source_head:
        raise SourceIntegrityError("runtime source head is not a full commit identity")
    if not _run_git(
        root,
        "merge-base",
        "--is-ancestor",
        S1_SOURCE_BASELINE_HEAD,
        runtime_source_head,
        check=False,
    ) == "":
        # merge-base --is-ancestor emits no stdout; return code is checked below.
        raise SourceIntegrityError("unexpected merge-base output")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", S1_SOURCE_BASELINE_HEAD, runtime_source_head),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise SourceIntegrityError("runtime source head is not a descendant of S1")

    files: list[dict[str, Any]] = []
    hash_lines: list[str] = []
    for relative in RUNTIME_AUTHORITY_PATHS:
        blob = git_blob(root, runtime_source_head, relative)
        path = root / relative
        if not path.is_file() or sha256_file(path) != sha256_bytes(blob):
            raise SourceIntegrityError(
                f"runtime authority worktree differs from commit: {relative}"
            )
        record = {
            "path": relative,
            "sha256": sha256_bytes(blob),
            "size_bytes": len(blob),
        }
        files.append(record)
        hash_lines.append(f"{record['sha256']}  {relative}")
    protected: list[dict[str, Any]] = []
    for relative in PROTECTED_BASE_PATHS:
        path = root / relative
        if not path.is_file():
            raise SourceIntegrityError(f"protected base file missing: {relative}")
        worktree_hash = sha256_file(path)
        blob = git_blob_if_present(root, runtime_source_head, relative)
        if blob is not None and worktree_hash != sha256_bytes(blob):
            raise SourceIntegrityError(f"protected base file drift: {relative}")
        protected.append(
            {
                "path": relative,
                "sha256": worktree_hash,
                "size_bytes": path.stat().st_size,
                "present_in_runtime_head": blob is not None,
                "identity_source": (
                    "RUNTIME_HEAD_GIT_BLOB"
                    if blob is not None
                    else "S1_POST_FREEZE_WORKTREE_EVIDENCE"
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "s1_source_baseline_head": S1_SOURCE_BASELINE_HEAD,
        "base_protocol_v3_fingerprint": BASE_PROTOCOL_V3_FINGERPRINT,
        "runtime_source_head": runtime_source_head,
        "runtime_source_files": files,
        "runtime_source_file_count": len(files),
        "runtime_source_tree_digest": canonical_sha256(files),
        "protected_base_files": protected,
        "protected_base_file_count": len(protected),
        "evidence_allowlist_path": evidence_allowlist_path,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload, "\n".join(hash_lines) + "\n"


def build_s2ar_runtime_source_manifest(
    root: Path,
    *,
    runtime_source_head: str,
) -> tuple[dict[str, Any], str]:
    return build_runtime_source_manifest(
        root,
        runtime_source_head=runtime_source_head,
        schema_version="task39s_2ar_runtime_source_manifest_v1",
        evidence_allowlist_path="task39s_2ar_evidence_allowlist.json",
    )


S2AR2_RUNTIME_AUTHORITY_PATHS: tuple[str, ...] = (
    "BSFX/chapter4_repro/chapter4_repro/task39s/__init__.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/activation.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/activation_v2.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/activation_v3.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/attempt_ledger.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/formal_evidence.py",
    "BSFX/chapter4_repro/chapter4_repro/task39s/source_integrity_v2.py",
    "scripts/check_task39s_runtime_source_integrity_v2.py",
    "scripts/generate_task39s_tiny_activation.py",
    "scripts/generate_task39s_tiny_activation_v2.py",
    "scripts/generate_task39s_2ar_2_recovery_activation.py",
    "scripts/recompute_task39s_tiny_metrics.py",
    "scripts/run_task39s_2ar_2_memory_probe.py",
    "scripts/run_task39s_2ar_2_equivalence.py",
    "scripts/run_task39s_2ar_synthetic_formal_evidence.py",
    "scripts/run_task39s_v3_tiny_overfit.py",
    "scripts/validate_task39s_tiny_activation.py",
    "scripts/validate_task39s_tiny_activation_v2.py",
)


S2AR2_EVIDENCE_ALLOWLIST: tuple[dict[str, str], ...] = (
    {"kind": "file", "path": "task39s_bigvul_path_chunking_contract.json"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_v3_schema.json"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_v3.json"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_v3_fingerprint.txt"},
    {"kind": "file", "path": "protocol_v3_tiny_execution_activation_v3_freeze_manifest.json"},
    {"kind": "file", "path": "task39s_2ar_2_runtime_source_manifest.json"},
    {"kind": "file", "path": "task39s_2ar_2_runtime_source_hashes.sha256"},
    {"kind": "file", "path": "task39s_2ar_2_evidence_allowlist.json"},
    {"kind": "file", "path": "task39s_2ar_2_recovery_ledger_schema.json"},
    {"kind": "file", "path": "task39s_2ar_2_preserved_s2a_manifest.json"},
    {"kind": "file", "path": "task39s_2ar_2_oom_root_cause.json"},
    {"kind": "file", "path": "task39s_2ar_2_forward_equivalence.json"},
    {"kind": "file", "path": "task39s_2ar_2_backward_equivalence.json"},
    {"kind": "file", "path": "task39s_2ar_2_optimizer_equivalence.json"},
    {"kind": "file", "path": "task39s_2ar_2_hierarchy_equivalence.json"},
    {"kind": "file", "path": "task39s_2ar_2_memory_selection.json"},
    {"kind": "file", "path": "task39s_2ar_2_activation_validation.json"},
    {"kind": "file", "path": "task39s_2ar_2_source_integrity.json"},
    {"kind": "file", "path": "task39s_2ar_2_data_access_audit.json"},
    {"kind": "file", "path": "task39s_2ar_2_test_summary.json"},
    {"kind": "file", "path": "task39s_2ar_2_evidence_manifest.json"},
    {"kind": "file", "path": "task39s_2ar_2_checksums.sha256"},
    {"kind": "file", "path": "TASK39S_2AR_2_REPORT.md"},
    {"kind": "file", "path": "TASK39S_2AR_2_STATUS.txt"},
    {"kind": "file", "path": "task39s_2ar_2a_recovery_execution_instructions.md"},
    {"kind": "directory", "path": "artifacts/chapter4_task39s_bigvul_memory_repair"},
    {"kind": "directory", "path": "artifacts/chapter4_task39s_tiny_overfit_v3/activation"},
    {"kind": "directory", "path": "artifacts/chapter4_task39s_tiny_overfit_v3/runs_recovered"},
)


def build_s2ar2_runtime_source_manifest(
    root: Path,
    *,
    runtime_source_head: str,
) -> tuple[dict[str, Any], str]:
    """V3 runtime manifest bound to Commit E (BigVul memory-safe path chunking)."""

    root = root.resolve()
    if _run_git(root, "rev-parse", runtime_source_head) != runtime_source_head:
        raise SourceIntegrityError("s2ar2 runtime source head is not a full commit identity")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", S1_SOURCE_BASELINE_HEAD, runtime_source_head),
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise SourceIntegrityError("s2ar2 runtime source head is not a descendant of S1")
    s2ar_ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", S2R_RUNTIME_SOURCE_HEAD, runtime_source_head),
        cwd=root,
        check=False,
    )
    if s2ar_ancestor.returncode != 0:
        raise SourceIntegrityError(
            "s2ar2 runtime source head is not a descendant of S2AR runtime source head"
        )

    files: list[dict[str, Any]] = []
    hash_lines: list[str] = []
    for relative in S2AR2_RUNTIME_AUTHORITY_PATHS:
        blob = git_blob(root, runtime_source_head, relative)
        path = root / relative
        if not path.is_file() or sha256_file(path) != sha256_bytes(blob):
            raise SourceIntegrityError(
                f"s2ar2 runtime authority worktree differs from commit: {relative}"
            )
        record = {
            "path": relative,
            "sha256": sha256_bytes(blob),
            "size_bytes": len(blob),
        }
        files.append(record)
        hash_lines.append(f"{record['sha256']}  {relative}")
    protected: list[dict[str, Any]] = []
    for relative in PROTECTED_BASE_PATHS:
        path = root / relative
        if not path.is_file():
            raise SourceIntegrityError(f"s2ar2 protected base file missing: {relative}")
        worktree_hash = sha256_file(path)
        blob = git_blob_if_present(root, runtime_source_head, relative)
        if blob is not None and worktree_hash != sha256_bytes(blob):
            raise SourceIntegrityError(f"s2ar2 protected base file drift: {relative}")
        protected.append(
            {
                "path": relative,
                "sha256": worktree_hash,
                "size_bytes": path.stat().st_size,
                "present_in_runtime_head": blob is not None,
                "identity_source": (
                    "RUNTIME_HEAD_GIT_BLOB"
                    if blob is not None
                    else "S1_POST_FREEZE_WORKTREE_EVIDENCE"
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "task39s_2ar_2_runtime_source_manifest_v1",
        "s1_source_baseline_head": S1_SOURCE_BASELINE_HEAD,
        "base_protocol_v3_fingerprint": BASE_PROTOCOL_V3_FINGERPRINT,
        "runtime_source_head": runtime_source_head,
        "runtime_source_files": files,
        "runtime_source_file_count": len(files),
        "runtime_source_tree_digest": canonical_sha256(files),
        "protected_base_files": protected,
        "protected_base_file_count": len(protected),
        "evidence_allowlist_path": "task39s_2ar_2_evidence_allowlist.json",
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload, "\n".join(hash_lines) + "\n"


def build_s2ar2_evidence_allowlist() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "task39s_2ar_2_evidence_allowlist_v1",
        "policy": "EXACT_FILE_OR_DIRECTORY_PREFIX_NO_GLOBS",
        "entries": [dict(row) for row in S2AR2_EVIDENCE_ALLOWLIST],
        "runtime_or_protected_paths_may_be_allowlisted": False,
        "historical_s2ar_runtime_source_head": S2R_RUNTIME_SOURCE_HEAD,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def verify_s2ar2_runtime_source_integrity(
    root: Path,
    *,
    runtime_manifest_path: Path,
    evidence_allowlist_path: Path,
    current_head: str | None = None,
) -> dict[str, Any]:
    """Alias for verify_runtime_source_integrity so v3 callers have a clear name.

    The underlying verifier is generation-agnostic: it inspects whatever
    manifest/allowlist paths are supplied.
    """

    return verify_runtime_source_integrity(
        root,
        runtime_manifest_path=runtime_manifest_path,
        evidence_allowlist_path=evidence_allowlist_path,
        current_head=current_head,
    )


# The canonical s2ar verifier below (verify_runtime_source_integrity) is reused
# by s2ar2 callers via verify_s2ar2_runtime_source_integrity. The historical
# alias verify_s2ar_runtime_source_integrity is defined later in this module.


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceIntegrityError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceIntegrityError(f"JSON object required: {path}")
    return value


def verify_runtime_source_integrity(
    root: Path,
    *,
    runtime_manifest_path: Path,
    evidence_allowlist_path: Path,
    current_head: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    manifest = _load_json(runtime_manifest_path)
    allowlist = _load_json(evidence_allowlist_path)
    issues: list[str] = []
    if not _canonical_valid(manifest):
        issues.append("runtime source manifest canonical hash mismatch")
    try:
        entries = _allowlist_entries(allowlist)
    except SourceIntegrityError as exc:
        issues.append(str(exc))
        entries = []
    runtime_head = str(manifest.get("runtime_source_head", ""))
    current = current_head or _run_git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", runtime_head, current),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ancestor.returncode != 0:
        issues.append("current HEAD is not a descendant of runtime source HEAD")

    records = manifest.get("runtime_source_files")
    if not isinstance(records, list) or not records:
        issues.append("runtime source records missing")
        records = []
    if canonical_sha256(records) != manifest.get("runtime_source_tree_digest"):
        issues.append("runtime source tree digest mismatch")
    for row in records:
        try:
            relative = _safe_relative(str(row["path"]))
            expected = str(row["sha256"])
            if sha256_bytes(git_blob(root, runtime_head, relative)) != expected:
                issues.append(f"runtime-head blob drift: {relative}")
            if sha256_bytes(git_blob(root, current, relative)) != expected:
                issues.append(f"current-head runtime authority drift: {relative}")
            if sha256_file(root / relative) != expected:
                issues.append(f"worktree runtime authority drift: {relative}")
        except (KeyError, OSError, SourceIntegrityError) as exc:
            issues.append(f"runtime authority verification failure: {exc}")
    protected = manifest.get("protected_base_files")
    if not isinstance(protected, list) or not protected:
        issues.append("protected base records missing")
        protected = []
    for row in protected:
        try:
            relative = _safe_relative(str(row["path"]))
            expected = str(row["sha256"])
            runtime_blob = git_blob_if_present(root, runtime_head, relative)
            expected_in_runtime = row.get("present_in_runtime_head") is True
            if (runtime_blob is not None) != expected_in_runtime:
                issues.append(f"runtime-head protected base presence drift: {relative}")
            elif runtime_blob is not None and sha256_bytes(runtime_blob) != expected:
                issues.append(f"runtime-head protected base drift: {relative}")
            current_blob = git_blob_if_present(root, current, relative)
            if current_blob is not None and sha256_bytes(current_blob) != expected:
                issues.append(f"current-head protected base drift: {relative}")
            if expected_in_runtime and current_blob is None:
                issues.append(f"current-head protected base missing: {relative}")
            if sha256_file(root / relative) != expected:
                issues.append(f"worktree protected base drift: {relative}")
        except (KeyError, OSError, SourceIntegrityError) as exc:
            issues.append(f"protected base verification failure: {exc}")

    committed_changes = (
        _run_git(root, "diff", "--name-only", f"{runtime_head}..{current}").splitlines()
        if runtime_head and current != runtime_head
        else []
    )
    disallowed_committed = sorted(
        relative
        for relative in committed_changes
        if not path_is_allowlisted(relative, entries)
    ) if entries else committed_changes
    if disallowed_committed:
        issues.extend(
            f"post-runtime commit changed non-evidence path: {relative}"
            for relative in disallowed_committed
        )
    worktree_changes = sorted(
        set(_run_git(root, "diff", "--name-only", "HEAD").splitlines())
        | set(_run_git(root, "diff", "--cached", "--name-only").splitlines())
    )
    disallowed_worktree = sorted(
        relative
        for relative in worktree_changes
        if not path_is_allowlisted(relative, entries)
    ) if entries else worktree_changes
    if disallowed_worktree:
        issues.extend(
            f"worktree changed non-evidence path: {relative}"
            for relative in disallowed_worktree
        )

    return {
        "schema_version": "task39s_2r_source_integrity_report_v1",
        "verdict": "PASS" if not issues else "FAIL",
        "issues": issues,
        "s1_source_baseline_head": manifest.get("s1_source_baseline_head"),
        "runtime_source_head": runtime_head,
        "current_evidence_head": current,
        "runtime_source_head_is_ancestor": ancestor.returncode == 0,
        "runtime_source_file_count": len(records),
        "protected_base_file_count": len(protected),
        "committed_changes_after_runtime_head": committed_changes,
        "disallowed_committed_changes": disallowed_committed,
        "tracked_worktree_changes": worktree_changes,
        "disallowed_tracked_worktree_changes": disallowed_worktree,
        "untracked_files_policy": "OUT_OF_SCOPE_PREEXISTING_UNTRACKED_NOT_RUNTIME_AUTHORITY",
    }


def source_closure_recheck(root: Path) -> dict[str, Any]:
    """Rebuild closure at current HEAD without comparing against the S1 file."""

    from scripts.check_task39s_source_closure_v1 import (
        DEFAULT_ENTRYPOINTS,
        DEFAULT_RUNTIME_AUTHORITIES,
        DEFAULT_SOURCE_ROOTS,
        build_manifests,
    )

    bundle = build_manifests(
        root.resolve(),
        source_roots=DEFAULT_SOURCE_ROOTS,
        entrypoints=DEFAULT_ENTRYPOINTS,
        runtime_authorities=DEFAULT_RUNTIME_AUTHORITIES,
    )
    return {
        "source_closure": bundle.source_manifest.get("verdict"),
        "import_closure": bundle.import_manifest.get("verdict"),
        "source_checks": bundle.source_manifest.get("checks"),
        "source_counts": bundle.source_manifest.get("counts"),
        "import_checks": bundle.import_manifest.get("checks"),
        "import_counts": bundle.import_manifest.get("counts"),
    }


def verify_s2ar_runtime_source_integrity(
    root: Path,
    *,
    runtime_manifest_path: Path,
    evidence_allowlist_path: Path,
    current_head: str | None = None,
) -> dict[str, Any]:
    report = verify_runtime_source_integrity(
        root,
        runtime_manifest_path=runtime_manifest_path,
        evidence_allowlist_path=evidence_allowlist_path,
        current_head=current_head,
    )
    return {
        **report,
        "schema_version": "task39s_2ar_source_integrity_report_v1",
        "historical_s2r_runtime_source_head": S2R_RUNTIME_SOURCE_HEAD,
        "runtime_authority_generation": "S2AR_COMMIT_C",
    }
