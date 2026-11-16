"""Deterministic Clang JSON-AST shadow frontend for Task39S.R6A.

This module is deliberately disconnected from the V1 dataset and learner entrypoints.
It accepts source text, invokes the system Clang through stdin, and removes volatile
Clang pointer IDs before hashing a compact canonical AST/relation representation.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SCHEMA_VERSION = "task39s-r6a-clang-shadow-v1"
WRAPPER_VERSION = "task39s-r6a-line-remap-wrapper-v1"
RELATION_ORDER = (
    "AST_CHILD",
    "DECL_USE",
    "CALL_CALLEE",
    "TYPE_OF",
    "MEMBER_BASE",
    "CONTROL_ROLE",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _node_children(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [item for item in node.get("inner", []) if isinstance(item, dict) and item.get("kind")]


def _walk(node: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    if node.get("kind"):
        yield dict(node)
    for child in _node_children(node):
        yield from _walk(child)


def _has_body(node: Mapping[str, Any]) -> bool:
    return any(child.get("kind") == "CompoundStmt" for child in _node_children(node))


def _included_location(value: Any) -> bool:
    if isinstance(value, dict):
        if "includedFrom" in value:
            return True
        return any(_included_location(v) for v in value.values())
    if isinstance(value, list):
        return any(_included_location(v) for v in value)
    return False


def _source_function_candidates(ast: Mapping[str, Any], function_name: str | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node in _walk(ast):
        if node.get("kind") != "FunctionDecl" or not _has_body(node):
            continue
        if function_name and node.get("name") != function_name:
            continue
        location_blob = {"loc": node.get("loc"), "range": node.get("range")}
        if _included_location(location_blob):
            continue
        candidates.append(node)
    if function_name and not candidates:
        for node in _walk(ast):
            if node.get("kind") == "FunctionDecl" and node.get("name") == function_name and _has_body(node):
                candidates.append(node)
    return candidates


def _normalize_point(point: Any) -> dict[str, Any] | None:
    if not isinstance(point, dict):
        return None
    spelling = point.get("spellingLoc") if isinstance(point.get("spellingLoc"), dict) else {}
    expansion = point.get("expansionLoc") if isinstance(point.get("expansionLoc"), dict) else {}
    source = point
    if not any(k in point for k in ("line", "presumedLine", "col")):
        source = spelling or expansion or point
    result: dict[str, Any] = {}
    line = source.get("presumedLine", source.get("line"))
    if line is not None:
        result["line"] = int(line)
    if source.get("col") is not None:
        result["column"] = int(source["col"])
    if source.get("tokLen") is not None:
        result["token_length"] = int(source["tokLen"])
    presumed = source.get("presumedFile") or point.get("presumedFile")
    file_name = presumed or source.get("file") or point.get("file")
    if file_name:
        result["source_domain"] = "shadow_input" if str(file_name) == "<stdin>" or str(file_name).startswith("r6a/") else "external"
    if point.get("isMacroArgExpansion"):
        result["macro_arg_expansion"] = True
    return result or None


def _normalize_range(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    begin = _normalize_point(value.get("begin"))
    end = _normalize_point(value.get("end"))
    if not begin and not end:
        return None
    return {"begin": begin, "end": end}


def _external_decl_key(ref: Mapping[str, Any]) -> str:
    stable = {
        "kind": ref.get("kind", "UnknownDecl"),
        "name": ref.get("name", ""),
        "type": (ref.get("type") or {}).get("qualType", "") if isinstance(ref.get("type"), dict) else "",
    }
    return "external:" + canonical_json_sha256(stable)[:20]


def _classify_diagnostics(stderr: str, returncode: int, ast_available: bool) -> list[str]:
    lowered = stderr.lower()
    categories: list[str] = []
    patterns = (
        ("missing_include", ("file not found", "no such file or directory")),
        ("unknown_type", ("unknown type name", "does not name a type")),
        ("undeclared_identifier", ("undeclared identifier", "use of undeclared identifier")),
        ("syntax_error", ("expected ", "extraneous", "unterminated")),
        ("redefinition", ("redefinition", "conflicting types")),
        ("language_mismatch", ("requires 'typename'", "cannot use 'try' with exceptions disabled")),
    )
    for category, needles in patterns:
        if any(needle in lowered for needle in needles):
            categories.append(category)
    if returncode and ast_available:
        categories.append("clang_recovery_ast")
    if returncode and not categories:
        categories.append("other_clang_error")
    return sorted(set(categories))


def _control_roles(kind: str, children: list[dict[str, Any]]) -> list[str]:
    if kind == "IfStmt":
        return ["condition", "then", "else"][: len(children)]
    if kind == "WhileStmt":
        return ["condition", "body"][: len(children)]
    if kind == "DoStmt":
        return ["body", "condition"][: len(children)]
    if kind == "SwitchStmt":
        return ["condition", "body"][: len(children)]
    if kind == "ForStmt":
        base = ["init", "condition", "increment", "body"]
        if len(children) == 5:
            base = ["init", "condition_variable", "condition", "increment", "body"]
        if len(children) == 4:
            return base
        return [f"slot_{i}" for i in range(len(children))]
    return []


def _find_callee_reference(node: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for descendant in _walk(node):
        ref = descendant.get("referencedDecl")
        if descendant.get("kind") == "DeclRefExpr" and isinstance(ref, dict) and ref.get("kind") in {"FunctionDecl", "CXXMethodDecl", "FunctionTemplateDecl"}:
            return ref
    return None


@dataclass(frozen=True)
class ShadowInvocation:
    dataset: str
    sample_id: str
    source_sha256: str
    language: str
    language_reason: str
    function_name: str | None
    virtual_file: str
    compiler: str
    args: tuple[str, ...]
    wrapper_version: str
    wrapper_sha256: str
    wrapped_source: str

    @property
    def args_sha256(self) -> str:
        return canonical_json_sha256(list(self.args))


class ClangShadowFrontend:
    """Small shadow frontend; it has no imports from V1 representation or training."""

    def __init__(self, clang: str, clangxx: str, sard_include: Path | None = None, timeout_seconds: int = 30) -> None:
        self.clang = str(Path(clang).resolve())
        self.clangxx = str(Path(clangxx).resolve())
        self.sard_include = sard_include.resolve() if sard_include else None
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def resolve_language(record: Mapping[str, Any], source: str) -> tuple[str, str]:
        declared = str(record.get("language", "")).lower()
        original = str(record.get("source_file_name", "")).lower()
        if declared in {"c++", "cpp", "cxx"}:
            return "c++", "manifest_language"
        if Path(original).suffix in {".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}:
            return "c++", "original_source_extension"
        if re.search(r"\b(class|namespace|template)\b|::|\bnullptr\b", source):
            return "c++", "source_syntax_probe"
        return "c", "manifest_or_generated_c"

    def make_invocation(self, dataset: str, record: Mapping[str, Any], source: str, source_sha256: str) -> ShadowInvocation:
        sample_id = str(record["sample_id"])
        language, reason = self.resolve_language(record, source)
        extension = "cpp" if language == "c++" else "c"
        virtual_file = f"r6a/{dataset}/{sample_id}.{extension}"
        wrapper_prefix = f"/* {WRAPPER_VERSION}: diagnostic location remap only */\n#line 1 \"{virtual_file}\"\n"
        wrapped = wrapper_prefix + source
        compiler = self.clangxx if language == "c++" else self.clang
        args = [
            compiler,
            "-x",
            "c++" if language == "c++" else "c",
            "-std=gnu++14" if language == "c++" else "-std=gnu11",
            "-fsyntax-only",
            "-ferror-limit=0",
            "-Wno-everything",
        ]
        if dataset == "sard" and self.sard_include:
            args.extend(["-I", str(self.sard_include)])
        args.extend(["-Xclang", "-ast-dump=json"])
        function_name = str(record.get("function_name") or "") or None
        if dataset == "sard" and function_name:
            args.extend(["-Xclang", "-ast-dump-filter", "-Xclang", function_name])
        args.append("-")
        return ShadowInvocation(
            dataset=dataset,
            sample_id=sample_id,
            source_sha256=source_sha256,
            language=language,
            language_reason=reason,
            function_name=function_name,
            virtual_file=virtual_file,
            compiler=compiler,
            args=tuple(args),
            wrapper_version=WRAPPER_VERSION,
            wrapper_sha256=sha256_bytes(wrapper_prefix.encode("utf-8")),
            wrapped_source=wrapped,
        )

    def parse(self, invocation: ShadowInvocation) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                list(invocation.args),
                input=invocation.wrapped_source,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
            )
            returncode = proc.returncode
            stderr = proc.stderr
            try:
                payload = proc.stdout.strip()
                if payload:
                    decoder = json.JSONDecoder()
                    objects: list[dict[str, Any]] = []
                    cursor = 0
                    while cursor < len(payload):
                        value, cursor = decoder.raw_decode(payload, cursor)
                        if isinstance(value, dict):
                            objects.append(value)
                        while cursor < len(payload) and payload[cursor].isspace():
                            cursor += 1
                    ast = objects[0] if len(objects) == 1 else {"kind": "TranslationUnitDecl", "inner": objects}
                else:
                    ast = None
            except json.JSONDecodeError:
                ast = None
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            stderr = (exc.stderr or "") + f"\nR6A_TIMEOUT_{self.timeout_seconds}s"
            ast = None
        raw_ast_json_available = isinstance(ast, dict)
        functions = _source_function_candidates(ast, invocation.function_name) if raw_ast_json_available else []
        ast_available = bool(functions)
        if returncode == 0 and ast_available:
            parse_status = "PARSE_OK"
        elif ast_available:
            parse_status = "PARSE_RECOVERED"
        else:
            parse_status = "PARSE_BLOCKED"
        canonical = self.canonicalize(functions) if functions else None
        stderr_digest = sha256_bytes(stderr.encode("utf-8"))
        return {
            "sample_id": invocation.sample_id,
            "dataset": invocation.dataset,
            "split": "train",
            "source_sha256": invocation.source_sha256,
            "language": invocation.language,
            "language_reason": invocation.language_reason,
            "compiler": invocation.compiler,
            "args": list(invocation.args),
            "args_sha256": invocation.args_sha256,
            "wrapper_version": invocation.wrapper_version,
            "wrapper_sha256": invocation.wrapper_sha256,
            "parse_status": parse_status,
            "returncode": returncode,
            "stderr_sha256": stderr_digest,
            "stderr_line_count": len(stderr.splitlines()),
            "diagnostic_categories": _classify_diagnostics(stderr, returncode, ast_available),
            "raw_ast_json_available": raw_ast_json_available,
            "ast_available": ast_available,
            "ast_sha256": canonical_json_sha256(canonical) if canonical else None,
            "canonical": canonical,
        }

    @staticmethod
    def canonicalize(functions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        raw_to_canonical: dict[str, str] = {}
        raw_nodes: dict[str, Mapping[str, Any]] = {}

        def visit(raw: Mapping[str, Any], parent: str | None, child_index: int, function_id: str) -> str:
            canonical_id = f"n{len(nodes) + 1:06d}"
            raw_id = str(raw.get("id", ""))
            if raw_id:
                raw_to_canonical[raw_id] = canonical_id
                raw_nodes[canonical_id] = raw
            node: dict[str, Any] = {
                "id": canonical_id,
                "kind": raw.get("kind", "Unknown"),
                "parent": parent,
                "child_index": child_index,
                "function_id": function_id,
                "implicit": bool(raw.get("isImplicit", False)),
            }
            for source_key, target_key in (
                ("name", "name"),
                ("opcode", "operator"),
                ("value", "literal"),
                ("valueCategory", "value_category"),
                ("castKind", "cast_kind"),
                ("storageClass", "storage_class"),
                ("tagUsed", "tag_used"),
            ):
                if raw.get(source_key) not in (None, ""):
                    node[target_key] = raw[source_key]
            type_obj = raw.get("type")
            if isinstance(type_obj, dict) and type_obj.get("qualType"):
                node["qualified_type"] = type_obj["qualType"]
            location = _normalize_point(raw.get("loc"))
            source_range = _normalize_range(raw.get("range"))
            if location:
                node["location"] = location
            if source_range:
                node["source_range"] = source_range
            nodes.append(node)
            if parent:
                relations.append({"type": "AST_CHILD", "source": parent, "target": canonical_id, "role": f"child_{child_index}"})
            for idx, child in enumerate(_node_children(raw)):
                visit(child, canonical_id, idx, function_id)
            return canonical_id

        for function_index, function in enumerate(functions):
            visit(function, None, function_index, f"f{function_index + 1:04d}")

        node_by_id = {node["id"]: node for node in nodes}
        declaration_kinds = {"VarDecl", "ParmVarDecl", "FunctionDecl", "CXXMethodDecl", "FieldDecl", "EnumConstantDecl", "TypedefDecl"}
        function_decl_ids: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            if node["kind"] in declaration_kinds:
                function_decl_ids[node["function_id"]].append(node["id"])
        ref_ids: dict[str, str] = {}
        for function_id, decl_ids in sorted(function_decl_ids.items()):
            for index, decl_id in enumerate(decl_ids, 1):
                ref_ids[decl_id] = f"{function_id}:REF:{index:04d}"
                node_by_id[decl_id]["declaration_identity"] = ref_ids[decl_id]

        for canonical_id, raw in raw_nodes.items():
            kind = str(raw.get("kind"))
            ref = raw.get("referencedDecl")
            if kind == "DeclRefExpr" and isinstance(ref, dict):
                raw_target = str(ref.get("id", ""))
                target = raw_to_canonical.get(raw_target) or _external_decl_key(ref)
                relation = {"type": "DECL_USE", "source": canonical_id, "target": target, "role": "references_declaration"}
                if target in ref_ids:
                    relation["ref_id"] = ref_ids[target]
                    node_by_id[canonical_id]["reference_identity"] = ref_ids[target]
                relations.append(relation)
            if kind == "CallExpr":
                callee = _find_callee_reference(raw)
                if callee:
                    target = raw_to_canonical.get(str(callee.get("id", ""))) or _external_decl_key(callee)
                    relations.append({"type": "CALL_CALLEE", "source": canonical_id, "target": target, "role": "resolved_direct_callee"})
            if kind == "MemberExpr":
                children = _node_children(raw)
                if children:
                    child_raw_id = str(children[0].get("id", ""))
                    target = raw_to_canonical.get(child_raw_id)
                    if target:
                        relations.append({"type": "MEMBER_BASE", "source": canonical_id, "target": target, "role": "base_expression"})
            if node_by_id[canonical_id].get("qualified_type"):
                type_name = str(node_by_id[canonical_id]["qualified_type"])
                relations.append({"type": "TYPE_OF", "source": canonical_id, "target": "type:" + canonical_json_sha256(type_name)[:20], "role": type_name})
            children = _node_children(raw)
            roles = _control_roles(kind, children)
            for index, role in enumerate(roles):
                child_raw_id = str(children[index].get("id", ""))
                target = raw_to_canonical.get(child_raw_id)
                if target:
                    relations.append({"type": "CONTROL_ROLE", "source": canonical_id, "target": target, "role": role})

        relation_rank = {name: index for index, name in enumerate(RELATION_ORDER)}
        relations.sort(key=lambda item: (relation_rank.get(item["type"], 99), item["source"], item["target"], item.get("role", "")))
        return {
            "schema_version": SCHEMA_VERSION,
            "functions": len(list(functions)) if not isinstance(functions, list) else len(functions),
            "nodes": nodes,
            "relations": relations,
        }


def summarize_node_inventory(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    functions_by_row: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for result in results:
        canonical = result.get("canonical")
        if not canonical:
            continue
        for node in canonical["nodes"]:
            implicit = "implicit" if node.get("implicit") else "explicit"
            key = (str(result["dataset"]), str(result["language"]), str(node["kind"]))
            rows[key][implicit] += 1
            functions_by_row[key].add((str(result["sample_id"]), str(node["function_id"])))
    v1_mapping = {
        "FunctionDecl": "FuncDef/Decl",
        "ParmVarDecl": "Decl/TypeDecl",
        "VarDecl": "Decl/TypeDecl",
        "DeclRefExpr": "ID",
        "IntegerLiteral": "Constant:int",
        "FloatingLiteral": "Constant:float/double",
        "StringLiteral": "Constant:string",
        "CharacterLiteral": "Constant:char",
        "BinaryOperator": "BinaryOp",
        "UnaryOperator": "UnaryOp",
        "CompoundAssignOperator": "Assignment/BinaryOp",
        "CallExpr": "FuncCall",
        "MemberExpr": "StructRef",
        "IfStmt": "If",
        "ForStmt": "For",
        "WhileStmt": "While",
        "DoStmt": "DoWhile",
        "ReturnStmt": "Return",
        "CompoundStmt": "Compound",
        "ImplicitCastExpr": "no direct node; pycparser omits implicit cast",
        "CXXRecordDecl": "no faithful pycparser equivalent",
        "CXXMethodDecl": "no faithful pycparser equivalent",
    }
    inventory = []
    for key in sorted(rows):
        dataset, language, kind = key
        counts = rows[key]
        inventory.append(
            {
                "dataset": dataset,
                "language": language,
                "clang_kind": kind,
                "node_count": sum(counts.values()),
                "function_count": len(functions_by_row[key]),
                "explicit_count": counts["explicit"],
                "implicit_count": counts["implicit"],
                "v1_pycparser_equivalent": v1_mapping.get(kind, "generic/none; requires R6B mapping review"),
            }
        )
    return {"schema_version": "task39s-r6a-node-inventory-v1", "inventory": inventory}


def summarize_relations(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    pairs: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ref_proofs: list[dict[str, Any]] = []
    for result in results:
        canonical = result.get("canonical")
        if not canonical:
            continue
        node_kind = {node["id"]: node["kind"] for node in canonical["nodes"]}
        ref_groups: dict[str, list[str]] = defaultdict(list)
        for relation in canonical["relations"]:
            relation_type = relation["type"]
            counts[relation_type] += 1
            source_kind = node_kind.get(relation["source"], "synthetic")
            target_kind = node_kind.get(relation["target"], "external_or_synthetic")
            pairs[relation_type][(source_kind, target_kind)] += 1
            if len(examples[relation_type]) < 5:
                examples[relation_type].append(
                    {
                        "dataset": result["dataset"],
                        "sample_id": result["sample_id"],
                        "source_kind": source_kind,
                        "target_kind": target_kind,
                        "role": relation.get("role"),
                    }
                )
            if relation_type == "DECL_USE" and relation.get("ref_id"):
                ref_groups[str(relation["ref_id"])].append(str(relation["source"]))
        repeated = {key: value for key, value in ref_groups.items() if len(value) >= 2}
        if repeated and len(ref_proofs) < 20:
            ref_proofs.append(
                {
                    "dataset": result["dataset"],
                    "sample_id": result["sample_id"],
                    "repeated_reference_groups": repeated,
                    "proof": "multiple DeclRefExpr nodes map to one stable function-local declaration identity",
                }
            )
    readiness = {
        "AST_CHILD": ("READY", "exact Clang inner-array parent/child order"),
        "DECL_USE": ("READY", "Clang referencedDecl identity; source-local targets carry stable REF IDs"),
        "CALL_CALLEE": ("PARTIAL", "direct DeclRefExpr callees only; indirect/virtual dispatch not claimed"),
        "TYPE_OF": ("READY", "Clang qualified type string represented as synthetic type symbol"),
        "MEMBER_BASE": ("READY", "MemberExpr first expression child"),
        "CONTROL_ROLE": ("PARTIAL", "reliable for common structured statements; malformed/recovery AST and irregular ForStmt slots require R6B fixtures"),
    }
    registry = []
    for relation_type in RELATION_ORDER:
        state, caveat = readiness[relation_type]
        kind_pairs = [
            {"source_kind": pair[0], "target_kind": pair[1], "count": count}
            for pair, count in sorted(pairs[relation_type].items())
        ]
        registry.append(
            {
                "relation_name": relation_type,
                "source_node_kind": sorted({item["source_kind"] for item in kind_pairs}),
                "target_node_kind": sorted({item["target_kind"] for item in kind_pairs}),
                "evidence_source": "Clang 14 JSON AST fields observed in fixed train-only shadow cohort",
                "direction": "source_to_target",
                "cardinality": "zero_or_more per source node; observed aggregate count supplied",
                "observed_count": counts[relation_type],
                "observed_kind_pairs": kind_pairs,
                "readiness": state,
                "recovery_confidence": "high" if state == "READY" else "medium",
                "caveat": caveat,
                "example_count": len(examples[relation_type]),
                "examples": examples[relation_type],
            }
        )
    return {
        "schema_version": "task39s-r6a-relation-registry-v1",
        "scope": "relations observed in the fixed train-only shadow cohort",
        "registry": registry,
        "stable_ref_identity_proofs": ref_proofs,
        "not_claimed": ["CFG", "DFG", "PDG", "CPG", "points-to", "alias", "virtual-dispatch completeness", "interprocedural data flow"],
    }
