"""R6C materialization helpers for frozen Clang AST V2 representations."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

from .frontend import canonical_json_sha256
from .schema import (
    CANONICAL_SCHEMA_VERSION,
    PATH_SCHEMA_VERSION,
    SEGMENT_SCHEMA_VERSION,
)


REPRESENTATION_FAMILY = "clang_ast_semantic_paths_v2"
REPRESENTATION_RECORD_SCHEMA = "task39s-r6c-v2-representation-record-v1"
COMPILE_CONTEXT_DIGEST_SCHEMA = "task39s-r6c-compile-context-digest-v1"
RELATION_REGISTRY_ID = "task39s-r6b-relation-registry-v1"
PATH_BUDGET_POLICY_ID = "task39s-r6b-deterministic-path-budget-v1"
SEGMENT_CONTRACT_ID = f"{SEGMENT_SCHEMA_VERSION}:core50:context5"


_LOCAL_REF_TOKEN = re.compile(
    r"^\[REF:f\d{4}(?::|%3A)REF(?::|%3A)\d{4}\]$",
    re.IGNORECASE,
)
_EXTERNAL_REF_TOKEN = re.compile(r"^\[REF:EXTERNAL:[^\]]+\]$")


def vocabulary_token(token: str) -> str:
    """Route stable declaration identity out of the global lexical vocabulary.

    The exact ``fNNNN:REF:NNNN`` value remains in SegmentRecord ``ref_links`` and
    relation metadata.  Only the finite identity *role* reaches token IDs.
    """

    if _LOCAL_REF_TOKEN.fullmatch(token):
        return "[REF:LOCAL_DECL]"
    if _EXTERNAL_REF_TOKEN.fullmatch(token):
        return "[REF:EXTERNAL_DECL]"
    if token == "[REF:UNRESOLVED_REF]":
        return token
    if token.startswith("[REF:"):
        return "[REF:SCHEMA_OTHER]"
    return token


def token_namespace(token: str) -> str:
    for namespace in ("NODE", "REL", "ROLE", "TYPE", "REF"):
        if token.startswith(f"[{namespace}:"):
            return namespace
    return "OTHER"


def _stable_ref_payload(canonical: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "declarations": [
            {
                "node_id": node["node_id"],
                "stable_ref_id": node["declaration_metadata"].get("stable_ref_id"),
                "reference_identity": node["declaration_metadata"].get("reference_identity"),
            }
            for node in canonical["nodes"]
            if node["declaration_metadata"].get("stable_ref_id")
            or node["declaration_metadata"].get("reference_identity")
        ],
        "decl_use": [
            relation
            for relation in canonical["relations"]
            if relation["relation_type"] == "DECL_USE"
        ],
    }


def representation_hashes(frontend_result: Mapping[str, Any]) -> dict[str, Any]:
    canonical = frontend_result.get("canonical")
    representation = frontend_result["representation"]
    return {
        "canonical_ast_sha256": frontend_result.get("canonical_ast_sha256"),
        "relations_sha256": canonical_json_sha256(canonical["relations"]) if canonical else None,
        "stable_ref_sha256": canonical_json_sha256(_stable_ref_payload(canonical)) if canonical else None,
        "paths_sha256": representation["path_output_sha256"] if canonical else None,
        "segments_sha256": representation["segment_output_sha256"] if canonical else None,
    }


def build_representation_record(
    frontend_result: Mapping[str, Any],
    *,
    contract_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build one complete eligible representation record and its digest."""

    parse_class = str(frontend_result["acceptance"]["parse_class"])
    if parse_class not in {"CLEAN", "RECOVERED_ACCEPTED"}:
        raise ValueError(f"noneligible parse class cannot be materialized: {parse_class}")
    canonical = frontend_result["canonical"]
    representation = frontend_result["representation"]
    hashes = representation_hashes(frontend_result)
    relation_counts = Counter(
        relation["relation_type"] for relation in canonical["relations"]
    )
    stable_ref_count = sum(
        bool(node["declaration_metadata"].get("stable_ref_id"))
        for node in canonical["nodes"]
    )
    unresolved_ref_count = sum(
        relation["relation_type"] == "DECL_USE" and not relation["resolved"]
        for relation in canonical["relations"]
    )
    raw_path_count = sum(
        int(row["raw_candidate_count"]) for row in representation["function_budget"]
    )
    deduplicated_path_count = sum(
        int(row["dedup_count"]) for row in representation["function_budget"]
    )
    kept_path_count = sum(
        int(row["kept_count"]) for row in representation["function_budget"]
    )
    dropped_path_count = sum(
        int(row["dropped_count"]) for row in representation["function_budget"]
    )
    cap_reasons: Counter[str] = Counter()
    for row in representation["function_budget"]:
        cap_reasons.update(row["drop_reason"])
    compile_context_sha256 = canonical_json_sha256(
        {
            "schema_version": COMPILE_CONTEXT_DIGEST_SCHEMA,
            "compile_context": frontend_result["compile_context"],
        }
    )
    payload = {
        "schema_version": REPRESENTATION_RECORD_SCHEMA,
        "representation_family": REPRESENTATION_FAMILY,
        "sample_id": frontend_result["sample_id"],
        "dataset": frontend_result["dataset"],
        "split": "train",
        "source_sha256": frontend_result["source_sha256"],
        "compile_context_sha256": compile_context_sha256,
        "compile_context": frontend_result["compile_context"],
        "parse_class": parse_class,
        "traceability": {
            "canonical_schema_id": CANONICAL_SCHEMA_VERSION,
            "relation_registry_id": RELATION_REGISTRY_ID,
            "path_schema_id": PATH_SCHEMA_VERSION,
            "path_budget_policy_id": PATH_BUDGET_POLICY_ID,
            "segment_contract_id": SEGMENT_CONTRACT_ID,
            "contract_sha256": dict(sorted(contract_sha256.items())),
        },
        **hashes,
        "node_count": len(canonical["nodes"]),
        "relation_counts": {
            name: int(relation_counts[name])
            for name in (
                "AST_CHILD",
                "DECL_USE",
                "CALL_CALLEE",
                "TYPE_OF",
                "MEMBER_BASE",
                "CONTROL_ROLE",
            )
        },
        "stable_ref_count": stable_ref_count,
        "unresolved_ref_count": unresolved_ref_count,
        "raw_path_count": raw_path_count,
        "deduplicated_path_count": deduplicated_path_count,
        "kept_path_count": kept_path_count,
        "dropped_path_count": dropped_path_count,
        "cap_hit": any(row["cap_hit"] for row in representation["function_budget"]),
        "cap_reason": dict(sorted(cap_reasons.items())),
        "segment_count": len(representation["segments"]),
        "canonical_ast": canonical,
        "semantic_paths": representation["paths"],
        "segments": representation["segments"],
        "function_budget": representation["function_budget"],
    }
    payload["representation_sha256"] = canonical_json_sha256(payload)
    return payload


def compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the inventory/determinism view without large representation arrays."""

    excluded = {"canonical_ast", "semantic_paths", "segments"}
    return {key: value for key, value in record.items() if key not in excluded}
