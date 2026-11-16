"""Result-blind R6E preflight-to-first-formal transition verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


TRANSITION_SCHEMA_VERSION = "task39s-r6e-preflight-to-first-formal-transition-v1"
TRANSITION_VERIFICATION_SCHEMA = "task39s-r6e-transition-verification-v1"
FIRST_FORMAL_RUN_ID = "T39S-R6E-BP-V1-D00-P1R"
PREFLIGHT_SEQUENCE_FREEZE_CANONICAL_SHA256 = (
    "9a51a1cea5309b71f7dec6c212849075195a2e4f0dc30c99294cf1dd4ad17741"
)
PREFLIGHT_SEQUENCE_FREEZE_FILE_SHA256 = (
    "f4714f015e6c64e829f864d8475d695532ad6d19d47203653261cb2b46f44648"
)
PREFLIGHT_FREEZE_MANIFEST_CANONICAL_SHA256 = (
    "c77840d79da23a316323e0edd8fe9f6836a659ac36f4fc4a28058b8d4b4b53ef"
)
PREFLIGHT_FREEZE_MANIFEST_FILE_SHA256 = (
    "2544a7dd8dd3e24e3e4d71588f0a7d9a807ee5d3a102186e8d3460d08c485104"
)
PREFLIGHT_PROVENANCE_EXECUTABLE_FREEZE_ID = (
    "48220cabd11843b57da57b7f92b62252f2b0f3c2c48df8854e37d5b7a60e6ce5"
)
PREFLIGHT_PROVENANCE_EXECUTABLE_MANIFEST_FILE_SHA256 = (
    "c13bd56a87e6d6d802e997fe3476caef6996da3b1c671fac7d5bf652a1b7d7fd"
)
R6C_REPRESENTATION_FREEZE_ID = (
    "5de73ed5bf2cac3296b3ad559f623137af42740a381bd0e8f4eb32eea704ae4e"
)
R6C_REPRESENTATION_MANIFEST_FILE_SHA256 = (
    "30cfad92b37a3ad326f7c0cadf3aa7066e135ef7f338bb23bd2060445d72485a"
)
PREFLIGHT_STATUS = "PASS"
PREFLIGHT_SCIENTIFIC_ASSESSMENT = "NOT_APPLICABLE"
SEQUENCE_ORDER = (
    "BP-V1",
    "BP-V2",
    "BE-V1",
    "BE-V2",
    "SP-V1",
    "SP-V2",
    "SE-V1",
    "SE-V2",
)

_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "loss",
        "prediction",
        "predictions",
        "accuracy",
        "balanced_accuracy",
        "balanced accuracy",
        "qualification",
        "qualification_outcome",
        "scientific_metric",
        "timing_projection",
    }
)


class TransitionContractFailure(ValueError):
    """Raised when the current preflight-to-formal contract does not verify."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def transition_canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_repository_path(root: Path, value: Any, label: str) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise TransitionContractFailure(f"unsafe {label} path")
    candidate = root / relative
    if not candidate.is_file():
        raise TransitionContractFailure(f"missing {label}")
    return candidate


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransitionContractFailure(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise TransitionContractFailure(f"invalid {label} object")
    return value


def _canonical_identity(payload: Mapping[str, Any], label: str) -> str:
    expected = str(payload.get("canonical_sha256") or "")
    basis = {key: value for key, value in payload.items() if key != "canonical_sha256"}
    actual = transition_canonical_sha256(basis)
    if len(expected) != 64 or actual != expected:
        raise TransitionContractFailure(f"{label} canonical SHA-256 mismatch")
    return actual


def _assert_no_result_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_RESULT_KEYS:
                raise TransitionContractFailure(
                    f"result-bearing transition field is forbidden: {key}"
                )
            _assert_no_result_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_result_keys(child)


def _verify_frozen_file(
    root: Path,
    *,
    relative_path: Any,
    expected_file_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path = _safe_repository_path(root, relative_path, label)
    if _sha256_file(path) != expected_file_sha256:
        raise TransitionContractFailure(f"{label} file SHA-256 mismatch")
    return path, _load_json(path, label)


def _verify_manifest_references(root: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "task39s_r6e0r10r2_preflight_8of8_audit.json",
        "task39s_r6e0r10r2_preflight_sequence_freeze.json",
    }:
        raise TransitionContractFailure("preflight freeze manifest artifact set mismatch")
    for relative, expected in artifacts.items():
        candidate = _safe_repository_path(root, relative, "preflight freeze artifact")
        if _sha256_file(candidate) != str(expected):
            raise TransitionContractFailure(
                f"preflight freeze artifact SHA-256 mismatch: {relative}"
            )


def _verify_executable_freeze_identity(
    payload: Mapping[str, Any], expected_freeze_id: str, label: str
) -> None:
    observed = str(payload.get("NEW_R6E_EXECUTABLE_SOURCE_FREEZE_ID") or "")
    basis = payload.get("freeze_basis")
    if observed != expected_freeze_id or not isinstance(basis, Mapping):
        raise TransitionContractFailure(f"{label} identity mismatch")
    if transition_canonical_sha256(basis) != expected_freeze_id:
        raise TransitionContractFailure(f"{label} canonical basis mismatch")


def _verify_current_formal_freeze(
    root: Path,
    payload: Mapping[str, Any],
    expected_freeze_id: str,
) -> None:
    _verify_executable_freeze_identity(payload, expected_freeze_id, "formal executable freeze")
    for section in ("source_sha256", "contract_sha256", "scientific_artifact_sha256"):
        entries = payload.get(section)
        if not isinstance(entries, Mapping) or not entries:
            raise TransitionContractFailure(f"formal executable freeze {section} is empty")
        for relative, expected in entries.items():
            candidate = _safe_repository_path(root, relative, "formal executable freeze reference")
            if _sha256_file(candidate) != str(expected):
                raise TransitionContractFailure(
                    f"formal executable freeze reference mismatch: {relative}"
                )


def verify_preflight_to_formal_transition(
    transition: Mapping[str, Any] | None,
    *,
    expected_transition_sha256: str,
    expected_formal_executable_freeze_id: str,
    expected_formal_executable_manifest_path: str,
    expected_formal_executable_manifest_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    """Verify only identity, execution closure, sequence and persistence inputs."""

    if transition is None or not isinstance(transition, Mapping):
        raise TransitionContractFailure("current preflight-to-formal transition is missing")
    _assert_no_result_keys(transition)
    transition_hash = _canonical_identity(transition, "transition artifact")
    if transition_hash != str(expected_transition_sha256):
        raise TransitionContractFailure("authority transition identity mismatch")

    exact = {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "transition_scope": "CURRENT_PREFLIGHT_8_OF_8_TO_FIRST_FORMAL_BP_V1",
        "transition_status": "CURRENT",
        "first_formal_run_id": FIRST_FORMAL_RUN_ID,
        "preflight_sequence_freeze_canonical_sha256": PREFLIGHT_SEQUENCE_FREEZE_CANONICAL_SHA256,
        "preflight_provenance_executable_freeze_id": PREFLIGHT_PROVENANCE_EXECUTABLE_FREEZE_ID,
        "r6c_representation_freeze_id": R6C_REPRESENTATION_FREEZE_ID,
        "formal_executable_freeze_id": str(expected_formal_executable_freeze_id),
        "current_freeze_preflight_8_of_8": PREFLIGHT_STATUS,
        "preflight_8_of_8_scientific_assessment": PREFLIGHT_SCIENTIFIC_ASSESSMENT,
    }
    for key, expected in exact.items():
        if transition.get(key) != expected:
            raise TransitionContractFailure(f"transition contract mismatch: {key}")
    if tuple(transition.get("sequence_order") or ()) != SEQUENCE_ORDER:
        raise TransitionContractFailure("transition sequence order mismatch")
    if transition.get("result_fields_read") != []:
        raise TransitionContractFailure("transition must attest empty result_fields_read")

    sequence_path, sequence = _verify_frozen_file(
        repository_root,
        relative_path=transition.get("preflight_sequence_freeze_path"),
        expected_file_sha256=PREFLIGHT_SEQUENCE_FREEZE_FILE_SHA256,
        label="preflight sequence freeze",
    )
    if transition.get("preflight_sequence_freeze_file_sha256") != _sha256_file(sequence_path):
        raise TransitionContractFailure("transition sequence-freeze file binding mismatch")
    if _canonical_identity(sequence, "preflight sequence freeze") != PREFLIGHT_SEQUENCE_FREEZE_CANONICAL_SHA256:
        raise TransitionContractFailure("preflight sequence freeze identity mismatch")
    if tuple(sequence.get("sequence_order") or ()) != SEQUENCE_ORDER:
        raise TransitionContractFailure("preflight sequence freeze order mismatch")
    if sequence.get("CURRENT_FREEZE_PREFLIGHT_8_OF_8") != PREFLIGHT_STATUS:
        raise TransitionContractFailure("preflight sequence freeze is not PASS")
    if sequence.get("PREFLIGHT_8_OF_8_SCIENTIFIC_ASSESSMENT") != PREFLIGHT_SCIENTIFIC_ASSESSMENT:
        raise TransitionContractFailure("preflight scientific assessment boundary mismatch")
    if sequence.get("r6e_executable_source_freeze_id") != PREFLIGHT_PROVENANCE_EXECUTABLE_FREEZE_ID:
        raise TransitionContractFailure("preflight sequence provenance freeze mismatch")
    if sequence.get("r6c_representation_freeze_id") != R6C_REPRESENTATION_FREEZE_ID:
        raise TransitionContractFailure("preflight sequence representation freeze mismatch")
    runs = sequence.get("runs")
    if not isinstance(runs, list) or len(runs) != len(SEQUENCE_ORDER):
        raise TransitionContractFailure("preflight sequence run count mismatch")
    for index, (short_id, row) in enumerate(zip(SEQUENCE_ORDER, runs), 1):
        if not isinstance(row, Mapping) or (
            row.get("sequence_index") != index
            or row.get("short_run_id") != short_id
            or row.get("execution_taxonomy") != "VALID_PREFLIGHT_RESULT"
        ):
            raise TransitionContractFailure("preflight sequence row closure mismatch")

    _, freeze_manifest = _verify_frozen_file(
        repository_root,
        relative_path=transition.get("preflight_freeze_manifest_path"),
        expected_file_sha256=PREFLIGHT_FREEZE_MANIFEST_FILE_SHA256,
        label="preflight freeze manifest",
    )
    if transition.get("preflight_freeze_manifest_file_sha256") != PREFLIGHT_FREEZE_MANIFEST_FILE_SHA256:
        raise TransitionContractFailure("transition preflight manifest file binding mismatch")
    if transition.get("preflight_freeze_manifest_canonical_sha256") != PREFLIGHT_FREEZE_MANIFEST_CANONICAL_SHA256:
        raise TransitionContractFailure("transition preflight manifest canonical binding mismatch")
    if _canonical_identity(freeze_manifest, "preflight freeze manifest") != PREFLIGHT_FREEZE_MANIFEST_CANONICAL_SHA256:
        raise TransitionContractFailure("preflight freeze manifest identity mismatch")
    if freeze_manifest.get("preflight_sequence_freeze_canonical_sha256") != PREFLIGHT_SEQUENCE_FREEZE_CANONICAL_SHA256:
        raise TransitionContractFailure("preflight manifest sequence binding mismatch")
    if freeze_manifest.get("CURRENT_FREEZE_PREFLIGHT_8_OF_8") != PREFLIGHT_STATUS:
        raise TransitionContractFailure("preflight freeze manifest is not PASS")
    if freeze_manifest.get("PREFLIGHT_8_OF_8_SCIENTIFIC_ASSESSMENT") != PREFLIGHT_SCIENTIFIC_ASSESSMENT:
        raise TransitionContractFailure("preflight manifest assessment boundary mismatch")
    _verify_manifest_references(repository_root, freeze_manifest)

    _, provenance = _verify_frozen_file(
        repository_root,
        relative_path=transition.get("preflight_provenance_executable_manifest_path"),
        expected_file_sha256=PREFLIGHT_PROVENANCE_EXECUTABLE_MANIFEST_FILE_SHA256,
        label="preflight provenance executable manifest",
    )
    if transition.get("preflight_provenance_executable_manifest_file_sha256") != PREFLIGHT_PROVENANCE_EXECUTABLE_MANIFEST_FILE_SHA256:
        raise TransitionContractFailure("transition provenance manifest binding mismatch")
    _verify_executable_freeze_identity(
        provenance,
        PREFLIGHT_PROVENANCE_EXECUTABLE_FREEZE_ID,
        "preflight provenance executable freeze",
    )

    _, representation = _verify_frozen_file(
        repository_root,
        relative_path=transition.get("r6c_representation_freeze_manifest_path"),
        expected_file_sha256=R6C_REPRESENTATION_MANIFEST_FILE_SHA256,
        label="R6C representation freeze manifest",
    )
    if transition.get("r6c_representation_freeze_manifest_file_sha256") != R6C_REPRESENTATION_MANIFEST_FILE_SHA256:
        raise TransitionContractFailure("transition representation manifest binding mismatch")
    if representation.get("REPRESENTATION_FREEZE_ID") != R6C_REPRESENTATION_FREEZE_ID:
        raise TransitionContractFailure("R6C representation freeze identity mismatch")

    if transition.get("formal_executable_source_manifest_path") != expected_formal_executable_manifest_path:
        raise TransitionContractFailure("transition formal manifest path mismatch")
    if transition.get("formal_executable_source_manifest_file_sha256") != expected_formal_executable_manifest_sha256:
        raise TransitionContractFailure("transition formal manifest file binding mismatch")
    formal_path, formal_freeze = _verify_frozen_file(
        repository_root,
        relative_path=expected_formal_executable_manifest_path,
        expected_file_sha256=expected_formal_executable_manifest_sha256,
        label="formal executable source manifest",
    )
    if _sha256_file(formal_path) != transition.get("formal_executable_source_manifest_file_sha256"):
        raise TransitionContractFailure("formal executable manifest changed during verification")
    _verify_current_formal_freeze(
        repository_root,
        formal_freeze,
        str(expected_formal_executable_freeze_id),
    )

    return {
        "schema_version": TRANSITION_VERIFICATION_SCHEMA,
        "valid": True,
        "mode": "FIRST_FORMAL_TRANSITION_MODE",
        "first_formal_run_id": FIRST_FORMAL_RUN_ID,
        "transition_canonical_sha256": transition_hash,
        "preflight_sequence_freeze_canonical_sha256": PREFLIGHT_SEQUENCE_FREEZE_CANONICAL_SHA256,
        "preflight_provenance_executable_freeze_id": PREFLIGHT_PROVENANCE_EXECUTABLE_FREEZE_ID,
        "r6c_representation_freeze_id": R6C_REPRESENTATION_FREEZE_ID,
        "formal_executable_freeze_id": str(expected_formal_executable_freeze_id),
        "current_freeze_preflight_8_of_8": PREFLIGHT_STATUS,
        "preflight_8_of_8_scientific_assessment": PREFLIGHT_SCIENTIFIC_ASSESSMENT,
        "result_fields_read": [],
    }


__all__ = [
    "FIRST_FORMAL_RUN_ID",
    "PREFLIGHT_FREEZE_MANIFEST_CANONICAL_SHA256",
    "PREFLIGHT_FREEZE_MANIFEST_FILE_SHA256",
    "PREFLIGHT_PROVENANCE_EXECUTABLE_FREEZE_ID",
    "PREFLIGHT_PROVENANCE_EXECUTABLE_MANIFEST_FILE_SHA256",
    "PREFLIGHT_SCIENTIFIC_ASSESSMENT",
    "PREFLIGHT_SEQUENCE_FREEZE_CANONICAL_SHA256",
    "PREFLIGHT_SEQUENCE_FREEZE_FILE_SHA256",
    "PREFLIGHT_STATUS",
    "R6C_REPRESENTATION_FREEZE_ID",
    "R6C_REPRESENTATION_MANIFEST_FILE_SHA256",
    "SEQUENCE_ORDER",
    "TRANSITION_SCHEMA_VERSION",
    "TRANSITION_VERIFICATION_SCHEMA",
    "TransitionContractFailure",
    "transition_canonical_sha256",
    "verify_preflight_to_formal_transition",
]
