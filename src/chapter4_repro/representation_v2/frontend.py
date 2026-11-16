"""Production Clang JSON-AST frontend for Representation V2.

This module is intentionally isolated from the V1 parser and every learner.  It
invokes a pinned system Clang through stdin, preserves recovery provenance, and
turns volatile Clang IDs into deterministic function-local identities.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .schema import (
    CALL_KINDS,
    CANONICAL_SCHEMA_VERSION,
    CONTROL_ROLE_SLOTS,
    DECLARATION_KINDS,
    FRONTEND_SCHEMA_VERSION,
    FUNCTION_KINDS,
    KEEP_NODE_KINDS,
    PARSE_ACCEPTANCE_THRESHOLDS,
    RELATION_ORDER,
    WRAPPER_SCHEMA_ID,
    node_policy,
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


def _raw_children(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [item for item in node.get("inner", []) if isinstance(item, dict) and item.get("kind")]


def _raw_child_slots(node: Mapping[str, Any]) -> list[dict[str, Any] | None]:
    slots: list[dict[str, Any] | None] = []
    for item in node.get("inner", []):
        if isinstance(item, dict) and item.get("kind"):
            slots.append(item)
        else:
            slots.append(None)
    return slots


def _walk(node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    if node.get("kind"):
        yield node
    for child in _raw_children(node):
        yield from _walk(child)


def _has_body(node: Mapping[str, Any]) -> bool:
    return any(child.get("kind") == "CompoundStmt" for child in _raw_children(node))


def _has_included_from(value: Any) -> bool:
    if isinstance(value, dict):
        if "includedFrom" in value:
            return True
        return any(_has_included_from(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_included_from(item) for item in value)
    return False


def _point_line(point: Any) -> int | None:
    if not isinstance(point, dict):
        return None
    for candidate in (point, point.get("spellingLoc"), point.get("expansionLoc")):
        if isinstance(candidate, dict):
            line = candidate.get("presumedLine", candidate.get("line"))
            if line is not None:
                return int(line)
    return None


def _function_sort_key(node: Mapping[str, Any]) -> tuple[Any, ...]:
    location = node.get("loc") if isinstance(node.get("loc"), dict) else {}
    return (
        _point_line(location) or 10**12,
        int(location.get("col", 10**12)) if isinstance(location, dict) else 10**12,
        str(node.get("kind", "")),
        str(node.get("name", "")),
        str((node.get("type") or {}).get("qualType", "")),
    )


def _source_function_candidates(ast: Mapping[str, Any], function_name: str | None) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    for node in _walk(ast):
        if node.get("kind") not in FUNCTION_KINDS or not _has_body(node):
            continue
        if function_name and node.get("name") != function_name:
            continue
        if _has_included_from({"loc": node.get("loc"), "range": node.get("range")}):
            continue
        candidates.append(node)
    if function_name and not candidates:
        for node in _walk(ast):
            if node.get("kind") in FUNCTION_KINDS and node.get("name") == function_name and _has_body(node):
                candidates.append(node)
    unique: dict[str, Mapping[str, Any]] = {}
    anonymous: list[Mapping[str, Any]] = []
    for candidate in candidates:
        raw_id = str(candidate.get("id", ""))
        if raw_id:
            unique.setdefault(raw_id, candidate)
        else:
            anonymous.append(candidate)
    return sorted([*unique.values(), *anonymous], key=_function_sort_key)


def _source_domain(file_name: str | None, virtual_prefix: str) -> str | None:
    if not file_name:
        return None
    if file_name == "<stdin>" or file_name.startswith(virtual_prefix):
        return "TARGET_SOURCE"
    return "EXTERNAL_CONTEXT"


def _normalize_point(
    point: Any,
    virtual_prefix: str,
    fallback_line: int | None = None,
    fallback_domain: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(point, dict):
        return None
    spelling = point.get("spellingLoc") if isinstance(point.get("spellingLoc"), dict) else {}
    expansion = point.get("expansionLoc") if isinstance(point.get("expansionLoc"), dict) else {}
    source = point
    if not any(key in point for key in ("line", "presumedLine", "col", "file", "presumedFile")):
        source = spelling or expansion or point
    result: dict[str, Any] = {}
    line = source.get("presumedLine", source.get("line", fallback_line))
    if line is not None:
        result["line"] = int(line)
    if source.get("col") is not None:
        result["column"] = int(source["col"])
    if source.get("tokLen") is not None:
        result["token_length"] = int(source["tokLen"])
    file_name = (
        source.get("presumedFile")
        or point.get("presumedFile")
        or source.get("file")
        or point.get("file")
    )
    domain = _source_domain(str(file_name), virtual_prefix) if file_name else fallback_domain
    if domain:
        result["source_domain"] = domain
    if point.get("isMacroArgExpansion") or source.get("isMacroArgExpansion"):
        result["macro_arg_expansion"] = True
    if spelling:
        spelling_point = _normalize_point(spelling, virtual_prefix, fallback_line, fallback_domain)
        if spelling_point:
            result["spelling"] = spelling_point
    if expansion:
        expansion_point = _normalize_point(expansion, virtual_prefix, fallback_line, fallback_domain)
        if expansion_point:
            result["expansion"] = expansion_point
    return result or None


def _normalize_span(
    node: Mapping[str, Any],
    virtual_prefix: str,
    fallback_line: int | None = None,
    fallback_domain: str | None = None,
) -> dict[str, Any] | None:
    location = _normalize_point(node.get("loc"), virtual_prefix, fallback_line, fallback_domain)
    location_line = location.get("line") if location else fallback_line
    location_domain = location.get("source_domain") if location else fallback_domain
    raw_range = node.get("range")
    if not isinstance(raw_range, dict):
        return {"location": location} if location else None
    begin = _normalize_point(raw_range.get("begin"), virtual_prefix, location_line, location_domain)
    begin_line = begin.get("line") if begin else location_line
    begin_domain = begin.get("source_domain") if begin else location_domain
    end = _normalize_point(raw_range.get("end"), virtual_prefix, begin_line, begin_domain)
    output: dict[str, Any] = {}
    if location:
        output["location"] = location
    if begin:
        output["begin"] = begin
    if end:
        output["end"] = end
    return output or None


def _qualified_type(node: Mapping[str, Any]) -> str | None:
    type_value = node.get("type")
    if isinstance(type_value, dict) and type_value.get("qualType"):
        return str(type_value["qualType"])
    return None


def _diagnostic_summary(stderr: str, returncode: int, ast_json_available: bool) -> dict[str, Any]:
    lowered = stderr.lower()
    categories: set[str] = set()
    pattern_registry = {
        "TOOLCHAIN_CONTEXT_MISSING": ("file not found", "no such file or directory"),
        "UNKNOWN_TYPE": ("unknown type name", "does not name a type"),
        "UNDECLARED_IDENTIFIER": ("undeclared identifier", "use of undeclared identifier"),
        "SOURCE_SYNTAX_INVALID": (" expected ", "error: expected", "extraneous", "unterminated"),
        "REDEFINITION_OR_CONFLICT": ("redefinition", "conflicting types"),
        "LANGUAGE_MODE_MISMATCH": ("requires 'typename'", "cannot use 'try' with exceptions disabled"),
    }
    padded = " " + lowered
    for category, needles in pattern_registry.items():
        if any(needle in padded for needle in needles):
            categories.add(category)
    error_count = len(re.findall(r"(?m):\s*(?:fatal\s+)?error:", stderr))
    warning_count = len(re.findall(r"(?m):\s*warning:", stderr))
    if returncode and ast_json_available:
        categories.add("CLANG_RECOVERY_OUTPUT")
    if returncode and not categories:
        categories.add("OTHER_CLANG_ERROR")
    return {
        "categories": sorted(categories),
        "error_count": error_count,
        "warning_count": warning_count,
        "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
        "stderr_line_count": len(stderr.splitlines()),
    }


def _parse_json_stream(payload: str) -> Mapping[str, Any] | None:
    payload = payload.strip()
    if not payload:
        return None
    decoder = json.JSONDecoder()
    objects: list[Mapping[str, Any]] = []
    cursor = 0
    while cursor < len(payload):
        value, cursor = decoder.raw_decode(payload, cursor)
        if isinstance(value, dict):
            objects.append(value)
        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
    if not objects:
        return None
    if len(objects) == 1:
        return objects[0]
    return {"kind": "TranslationUnitDecl", "inner": objects}


def _stable_external_symbol(ref: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    type_value = ref.get("type") if isinstance(ref.get("type"), dict) else {}
    stable = {
        "kind": str(ref.get("kind", "UnknownDecl")),
        "name": str(ref.get("name", "")),
        "qualified_type": str(type_value.get("qualType", "")),
        "mangled_name": str(ref.get("mangledName", "")),
    }
    usable = bool(stable["name"] or stable["mangled_name"]) and stable["kind"] not in {
        "UnresolvedLookupExpr",
        "OverloadExpr",
        "FunctionTemplateDecl",
    }
    if not usable:
        return None, stable
    return "external:" + canonical_json_sha256(stable)[:24], stable


def _relation(
    relation_type: str,
    source_id: str,
    target_id: str,
    role: str,
    resolved: bool,
    resolution_kind: str,
    confidence_class: str,
    source_provenance: str,
    **extra: Any,
) -> dict[str, Any]:
    output = {
        "relation_type": relation_type,
        "source_id": source_id,
        "target_id": target_id,
        "direction": "SOURCE_TO_TARGET",
        "role": role,
        "resolved": resolved,
        "resolution_kind": resolution_kind,
        "confidence_class": confidence_class,
        "source_provenance": source_provenance,
    }
    output.update({key: value for key, value in extra.items() if value is not None})
    return output


@dataclass(frozen=True)
class CompileContext:
    sample_id: str
    dataset: str
    split: str
    source_sha256: str
    language: str
    language_source: str
    language_confidence: str
    clang_binary: str
    clang_version: str
    target_triple: str
    language_standard: str
    original_file_extension: str
    generated_file_extension: str
    include_dirs: tuple[str, ...]
    defines: tuple[str, ...]
    undefines: tuple[str, ...]
    wrapper_schema_id: str
    wrapper_sha256: str
    compiler_args: tuple[str, ...]
    context_source: str
    context_layers: tuple[str, ...]
    virtual_file: str

    @property
    def compiler_args_sha256(self) -> str:
        return canonical_json_sha256(list(self.compiler_args))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "source_sha256": self.source_sha256,
            "language": self.language,
            "language_source": self.language_source,
            "language_confidence": self.language_confidence,
            "clang_binary": self.clang_binary,
            "clang_version": self.clang_version,
            "target_triple": self.target_triple,
            "language_standard": self.language_standard,
            "original_file_extension": self.original_file_extension,
            "generated_file_extension": self.generated_file_extension,
            "include_dirs": list(self.include_dirs),
            "defines": list(self.defines),
            "undefines": list(self.undefines),
            "wrapper_schema_id": self.wrapper_schema_id,
            "wrapper_sha256": self.wrapper_sha256,
            "compiler_args": list(self.compiler_args),
            "compiler_args_sha256": self.compiler_args_sha256,
            "context_source": self.context_source,
            "context_layers": list(self.context_layers),
            "virtual_file": self.virtual_file,
        }


@dataclass(frozen=True)
class ProductionInvocation:
    context: CompileContext
    function_name: str | None
    wrapped_source: str


class ProductionClangFrontend:
    """Deterministic, CPU-only Clang V2 representation frontend."""

    def __init__(
        self,
        clang: str,
        clangxx: str,
        clang_version: str,
        target_triple: str,
        sard_include: Path | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.clang = str(Path(clang).resolve())
        self.clangxx = str(Path(clangxx).resolve())
        self.clang_version = clang_version
        self.target_triple = target_triple
        self.sard_include = sard_include.resolve() if sard_include else None
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def resolve_language(record: Mapping[str, Any], source: str) -> tuple[str, str, str]:
        declared = str(record.get("language", "")).lower()
        original = str(record.get("source_file_name", record.get("physical_file_name", ""))).lower()
        if declared in {"c++", "cpp", "cxx"}:
            return "c++", "MANIFEST_LANGUAGE", "HIGH"
        if Path(original).suffix in {".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}:
            return "c++", "ORIGINAL_FILE_EXTENSION", "HIGH"
        if re.search(r"\b(class|namespace|template)\b|::|\bnullptr\b", source):
            return "c++", "SOURCE_SYNTAX_PROBE", "MEDIUM"
        if declared == "c":
            return "c", "MANIFEST_LANGUAGE", "HIGH"
        return "c", "GENERATED_FILE_EXTENSION", "MEDIUM"

    def make_invocation(
        self,
        dataset: str,
        record: Mapping[str, Any],
        source: str,
        source_sha256: str,
        generated_extension: str = ".c",
    ) -> ProductionInvocation:
        if str(record.get("split", "train")) != "train":
            raise RuntimeError(f"R6B train-only guard rejected split={record.get('split')}")
        sample_id = str(record["sample_id"])
        language, language_source, language_confidence = self.resolve_language(record, source)
        standard = "gnu++14" if language == "c++" else "gnu11"
        extension = "cpp" if language == "c++" else "c"
        virtual_file = f"r6b/{dataset}/{sample_id}.{extension}"
        prefix = f"/* {WRAPPER_SCHEMA_ID}: diagnostic remap only */\n#line 1 \"{virtual_file}\"\n"
        compiler = self.clangxx if language == "c++" else self.clang
        include_dirs: list[str] = []
        if dataset == "sard" and self.sard_include:
            include_dirs.append(str(self.sard_include))
        args = [
            compiler,
            "-x",
            "c++" if language == "c++" else "c",
            f"-std={standard}",
            "-fsyntax-only",
            "-ferror-limit=0",
            "-Wno-everything",
        ]
        for include_dir in include_dirs:
            args.extend(["-I", include_dir])
        args.extend(["-Xclang", "-ast-dump=json"])
        function_name = str(record.get("function_name") or "") or None
        if function_name:
            args.extend(["-Xclang", "-ast-dump-filter", "-Xclang", function_name])
        args.append("-")
        original_name = str(record.get("source_file_name", record.get("physical_file_name", "")))
        context_source = "GENERATED_CONTEXT"
        context_layers = ("GENERATED_CONTEXT", "DETERMINISTIC_WRAPPER")
        context = CompileContext(
            sample_id=sample_id,
            dataset=dataset,
            split="train",
            source_sha256=source_sha256,
            language=language,
            language_source=language_source,
            language_confidence=language_confidence,
            clang_binary=compiler,
            clang_version=self.clang_version,
            target_triple=self.target_triple,
            language_standard=standard,
            original_file_extension=Path(original_name).suffix.lower() or "<missing>",
            generated_file_extension=generated_extension.lower() or "<missing>",
            include_dirs=tuple(include_dirs),
            defines=(),
            undefines=(),
            wrapper_schema_id=WRAPPER_SCHEMA_ID,
            wrapper_sha256=sha256_bytes(prefix.encode("utf-8")),
            compiler_args=tuple(args),
            context_source=context_source,
            context_layers=context_layers,
            virtual_file=virtual_file,
        )
        return ProductionInvocation(context=context, function_name=function_name, wrapped_source=prefix + source)

    def parse(self, invocation: ProductionInvocation) -> dict[str, Any]:
        timed_out = False
        json_error = False
        try:
            proc = subprocess.run(
                list(invocation.context.compiler_args),
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
                ast = _parse_json_stream(proc.stdout)
            except (json.JSONDecodeError, ValueError):
                ast = None
                json_error = True
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr_value = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr = stderr_value + f"\nR6B_TIMEOUT_{self.timeout_seconds}s"
            try:
                ast = _parse_json_stream(stdout)
            except (json.JSONDecodeError, ValueError):
                ast = None
                json_error = True
        ast_json_available = isinstance(ast, dict)
        functions = _source_function_candidates(ast, invocation.function_name) if ast_json_available else []
        canonical = self.canonicalize(ast, functions, invocation.context) if functions and ast else None
        diagnostics = _diagnostic_summary(stderr, returncode, ast_json_available)
        if canonical:
            recovery_parse = bool(returncode != 0 or diagnostics["error_count"] > 0)
            canonical["parse_provenance"] = {
                "clang_recovery_parse": recovery_parse,
                "return_code": returncode,
                "diagnostic_categories": list(diagnostics["categories"]),
            }
            for node in canonical["nodes"]:
                node["recovery_provenance"]["clang_recovery_parse"] = recovery_parse
        acceptance = self.classify_acceptance(
            canonical=canonical,
            diagnostics=diagnostics,
            returncode=returncode,
            ast_json_available=ast_json_available,
            timed_out=timed_out,
            json_error=json_error,
        )
        compile_context = invocation.context.to_dict()
        compile_context["representation_context_source"] = (
            "RECOVERY_ONLY" if acceptance["parse_class"] != "CLEAN" else invocation.context.context_source
        )
        output = {
            "schema_version": FRONTEND_SCHEMA_VERSION,
            "sample_id": invocation.context.sample_id,
            "dataset": invocation.context.dataset,
            "split": "train",
            "source_sha256": invocation.context.source_sha256,
            "compile_context": compile_context,
            "return_code": returncode,
            "ast_json_available": ast_json_available,
            "diagnostics": diagnostics,
            "acceptance": acceptance,
            "canonical": canonical,
            "canonical_ast_sha256": canonical_json_sha256(canonical) if canonical else None,
        }
        output["frontend_output_sha256"] = canonical_json_sha256(
            {
                "schema_version": output["schema_version"],
                "sample_id": output["sample_id"],
                "dataset": output["dataset"],
                "source_sha256": output["source_sha256"],
                "compile_context": compile_context,
                "return_code": returncode,
                "diagnostics": diagnostics,
                "acceptance": acceptance,
                "canonical": canonical,
            }
        )
        return output

    @staticmethod
    def canonicalize(
        ast: Mapping[str, Any],
        functions: Sequence[Mapping[str, Any]],
        context: CompileContext,
    ) -> dict[str, Any]:
        global_raw_lookup: dict[str, Mapping[str, Any]] = {}
        for raw in _walk(ast):
            raw_id = str(raw.get("id", ""))
            if raw_id:
                global_raw_lookup.setdefault(raw_id, raw)

        nodes: list[dict[str, Any]] = []
        raw_to_canonical: dict[str, str] = {}
        canonical_to_raw: dict[str, Mapping[str, Any]] = {}
        function_roots: dict[str, str] = {}

        def visit(
            raw: Mapping[str, Any],
            parent_id: str | None,
            child_index: int,
            depth: int,
            function_id: str,
            preorder_counter: list[int],
            inherited_line: int | None = None,
            inherited_domain: str | None = None,
        ) -> str:
            preorder_counter[0] += 1
            canonical_id = f"{function_id}:n{preorder_counter[0]:06d}"
            raw_id = str(raw.get("id", ""))
            if raw_id:
                raw_to_canonical[raw_id] = canonical_id
            canonical_to_raw[canonical_id] = raw
            kind = str(raw.get("kind", "Unknown"))
            policy = node_policy(kind)
            qualified_type = _qualified_type(raw)
            type_value = raw.get("type") if isinstance(raw.get("type"), dict) else {}
            type_metadata: dict[str, Any] = {}
            if qualified_type:
                type_metadata["qualified_type"] = qualified_type
            if type_value.get("desugaredQualType"):
                type_metadata["desugared_qualified_type"] = str(type_value["desugaredQualType"])
            literal_metadata = {
                key: raw[key]
                for key in ("value", "valueCategory")
                if raw.get(key) not in (None, "")
            }
            operator_metadata = {
                key: raw[key]
                for key in ("opcode", "isPostfix")
                if raw.get(key) not in (None, "")
            }
            declaration_metadata = {
                key: raw[key]
                for key in ("name", "mangledName", "storageClass", "tagUsed")
                if raw.get(key) not in (None, "")
            }
            source_span = _normalize_span(raw, "r6b/", inherited_line, inherited_domain)
            recovery_specific = policy == "QUARANTINE_SPECIAL" or kind.startswith("Recovery")
            node = {
                "node_id": canonical_id,
                "function_id": function_id,
                "preorder": preorder_counter[0],
                "node_kind": kind,
                "canonical_node_kind": kind,
                "emission_policy": policy,
                "emits_token": policy in {"KEEP", "QUARANTINE_SPECIAL"},
                "parent_id": parent_id,
                "depth": depth,
                "sibling_index": child_index,
                "child_role": f"CLANG_INNER_{child_index}",
                "implicit": bool(raw.get("isImplicit", False) or kind == "ImplicitCastExpr"),
                "source_span": source_span,
                "type_metadata": type_metadata,
                "literal_metadata": literal_metadata,
                "operator_metadata": operator_metadata,
                "declaration_metadata": declaration_metadata,
                "recovery_provenance": {
                    "recovery_specific_node": recovery_specific,
                    "clang_recovery_parse": False,
                },
                "attributes": {
                    "unregistered_kind": policy == "KEEP" and kind not in KEEP_NODE_KINDS,
                },
            }
            if raw.get("castKind"):
                node["attributes"]["cast_kind"] = str(raw["castKind"])
            nodes.append(node)
            if parent_id is None:
                function_roots[function_id] = canonical_id
            next_line = None
            next_domain = None
            if source_span:
                for point_name in ("location", "begin", "end"):
                    point = source_span.get(point_name) or {}
                    if next_line is None and point.get("line") is not None:
                        next_line = int(point["line"])
                    if next_domain is None and point.get("source_domain"):
                        next_domain = str(point["source_domain"])
            for slot, child in enumerate(_raw_child_slots(raw)):
                if child is not None:
                    visit(
                        child,
                        canonical_id,
                        slot,
                        depth + 1,
                        function_id,
                        preorder_counter,
                        next_line,
                        next_domain,
                    )
            return canonical_id

        for function_index, function in enumerate(functions, 1):
            visit(function, None, 0, 0, f"f{function_index:04d}", [0])

        node_by_id = {node["node_id"]: node for node in nodes}

        declaration_ids_by_function: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            if node["node_kind"] in DECLARATION_KINDS:
                declaration_ids_by_function[node["function_id"]].append(node["node_id"])
        stable_refs: dict[str, str] = {}
        for function_id in sorted(declaration_ids_by_function):
            for index, node_id in enumerate(declaration_ids_by_function[function_id], 1):
                ref_id = f"{function_id}:REF:{index:04d}"
                stable_refs[node_id] = ref_id
                node_by_id[node_id]["declaration_metadata"]["stable_ref_id"] = ref_id

        def first_emitted_descendant(raw: Mapping[str, Any]) -> str | None:
            raw_id = str(raw.get("id", ""))
            canonical_id = raw_to_canonical.get(raw_id)
            if canonical_id and node_by_id[canonical_id]["emits_token"]:
                return canonical_id
            for child in _raw_children(raw):
                found = first_emitted_descendant(child)
                if found:
                    return found
            return None

        # Preserve collapsed/attribute-only provenance on the first emitted
        # descendant while retaining the provenance node itself in ``nodes``.
        for node_id, raw in canonical_to_raw.items():
            node = node_by_id[node_id]
            if node["emission_policy"] == "ATTRIBUTE_ONLY":
                target = first_emitted_descendant(raw)
                children = _raw_children(raw)
                source_type = _qualified_type(children[0]) if children else None
                attribute = {
                    "provenance_node_id": node_id,
                    "cast_kind": raw.get("castKind"),
                    "source_type": source_type,
                    "target_type": _qualified_type(raw),
                    "source_span": node.get("source_span"),
                    "implicit": True,
                }
                if target and target != node_id:
                    node_by_id[target]["attributes"].setdefault("implicit_casts", []).append(attribute)
            elif node["emission_policy"] == "COLLAPSE":
                target = first_emitted_descendant(raw)
                if target and target != node_id:
                    node_by_id[target]["attributes"].setdefault("collapsed_wrappers", []).append(
                        {
                            "provenance_node_id": node_id,
                            "node_kind": node["node_kind"],
                            "source_span": node.get("source_span"),
                        }
                    )

        relations: list[dict[str, Any]] = []
        reference_sites = 0
        unresolved_reference_sites = 0

        def resolve_declaration(
            raw_ref: Mapping[str, Any] | None,
            source_node_id: str,
            unresolved_kind: str,
        ) -> tuple[str, bool, str, str | None, dict[str, Any] | None]:
            nonlocal unresolved_reference_sites
            if isinstance(raw_ref, Mapping):
                raw_target = str(raw_ref.get("id", ""))
                canonical_target = raw_to_canonical.get(raw_target)
                if canonical_target:
                    return (
                        canonical_target,
                        True,
                        "FUNCTION_LOCAL_STABLE_REF",
                        stable_refs.get(canonical_target),
                        None,
                    )
                external_target, external_metadata = _stable_external_symbol(raw_ref)
                if external_target:
                    return external_target, True, "EXTERNAL_SYMBOL_FINGERPRINT", None, external_metadata
            unresolved_reference_sites += 1
            return (
                f"unresolved:ref:{source_node_id}:{unresolved_kind}",
                False,
                unresolved_kind,
                "UNRESOLVED_REF",
                None,
            )

        for node_id, raw in canonical_to_raw.items():
            node = node_by_id[node_id]
            raw_parent = node.get("parent_id")
            if raw_parent:
                relations.append(
                    _relation(
                        "AST_CHILD",
                        raw_parent,
                        node_id,
                        node["child_role"],
                        True,
                        "CLANG_INNER_ORDER",
                        "HIGH",
                        "CLANG_JSON_INNER",
                    )
                )

            kind = node["node_kind"]
            ref: Mapping[str, Any] | None = None
            unresolved_kind = "MISSING_REFERENCED_DECL"
            if kind == "DeclRefExpr":
                candidate = raw.get("referencedDecl")
                ref = candidate if isinstance(candidate, Mapping) else None
                unresolved_kind = "UNRESOLVED_REF"
            elif kind == "MemberExpr":
                member_id = raw.get("referencedMemberDecl")
                if member_id is not None:
                    ref = global_raw_lookup.get(str(member_id))
                    if ref is None:
                        ref = {
                            "kind": "FieldDecl",
                            "name": raw.get("name", ""),
                            "type": raw.get("type", {}),
                        }
                unresolved_kind = "MEMBER_UNRESOLVED_REF"
            if kind in {"DeclRefExpr", "MemberExpr"}:
                reference_sites += 1
                target, resolved, resolution_kind, stable_ref_id, external_metadata = resolve_declaration(
                    ref, node_id, unresolved_kind
                )
                relations.append(
                    _relation(
                        "DECL_USE",
                        node_id,
                        target,
                        "REFERENCES_DECLARATION",
                        resolved,
                        resolution_kind,
                        "HIGH" if resolved else "EXPLICIT_UNRESOLVED",
                        "CLANG_REFERENCED_DECL" if kind == "DeclRefExpr" else "CLANG_REFERENCED_MEMBER_DECL",
                        stable_ref_id=stable_ref_id,
                        external_symbol=external_metadata,
                    )
                )
                node["declaration_metadata"]["reference_identity"] = stable_ref_id or (
                    target if resolved else "UNRESOLVED_REF"
                )

            qualified_type = node["type_metadata"].get("qualified_type")
            if qualified_type:
                type_target = "type:" + canonical_json_sha256(qualified_type)[:24]
                relations.append(
                    _relation(
                        "TYPE_OF",
                        node_id,
                        type_target,
                        qualified_type,
                        True,
                        "QUALIFIED_TYPE_STRING",
                        "HIGH",
                        "CLANG_TYPE_QUALTYPE",
                        qualified_type=qualified_type,
                    )
                )

            if kind == "MemberExpr":
                children = _raw_children(raw)
                target = first_emitted_descendant(children[0]) if children else None
                relations.append(
                    _relation(
                        "MEMBER_BASE",
                        node_id,
                        target or f"unresolved:member-base:{node_id}",
                        "MEMBER_BASE_EXPRESSION" if target else "MEMBER_BASE_UNRESOLVED",
                        bool(target),
                        "CLANG_FIRST_EXPRESSION_CHILD" if target else "MEMBER_BASE_UNRESOLVED",
                        "HIGH" if target else "EXPLICIT_UNRESOLVED",
                        "CLANG_JSON_INNER",
                    )
                )

            if kind in CALL_KINDS:
                call_relation = ProductionClangFrontend._call_relation(
                    node_id,
                    raw,
                    raw_to_canonical,
                    stable_refs,
                    global_raw_lookup,
                )
                relations.append(call_relation)

            if kind in CONTROL_ROLE_SLOTS:
                relations.extend(
                    ProductionClangFrontend._control_relations(
                        node_id,
                        raw,
                        raw_to_canonical,
                        node_by_id,
                        first_emitted_descendant,
                    )
                )

        relation_rank = {name: index for index, name in enumerate(RELATION_ORDER)}
        relations.sort(
            key=lambda item: (
                relation_rank.get(item["relation_type"], 99),
                item["source_id"],
                item["target_id"],
                item["role"],
                item["resolution_kind"],
            )
        )
        recovery_node_count = sum(node["recovery_provenance"]["recovery_specific_node"] for node in nodes)
        emitted_node_count = sum(bool(node["emits_token"]) for node in nodes)
        function_summaries = []
        for function_id in sorted(function_roots):
            root_id = function_roots[function_id]
            root = node_by_id[root_id]
            function_summaries.append(
                {
                    "function_id": function_id,
                    "root_node_id": root_id,
                    "function_name": root["declaration_metadata"].get("name"),
                    "node_count": sum(node["function_id"] == function_id for node in nodes),
                    "emitted_node_count": sum(
                        node["function_id"] == function_id and node["emits_token"] for node in nodes
                    ),
                }
            )
        canonical = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "sample_identity": {
                "sample_id": context.sample_id,
                "dataset": context.dataset,
                "split": context.split,
                "source_sha256": context.source_sha256,
            },
            "function_order": "normalized source location, kind, name, qualified type",
            "node_order": "function-local deterministic preorder over Clang inner slots",
            "relation_order": "registry rank, source ID, target stable ID, role, resolution kind",
            "functions": function_summaries,
            "nodes": nodes,
            "relations": relations,
            "quality": {
                "function_count": len(function_summaries),
                "raw_node_count": len(nodes),
                "emitted_node_count": emitted_node_count,
                "recovery_node_count": recovery_node_count,
                "reference_site_count": reference_sites,
                "unresolved_reference_count": unresolved_reference_sites,
            },
        }
        return canonical

    @staticmethod
    def _call_relation(
        node_id: str,
        raw: Mapping[str, Any],
        raw_to_canonical: Mapping[str, str],
        stable_refs: Mapping[str, str],
        global_raw_lookup: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        children = _raw_children(raw)
        callee_tree = children[0] if children else None
        descendants = list(_walk(callee_tree)) if callee_tree else []
        kinds = {str(item.get("kind", "")) for item in descendants}
        member = next((item for item in descendants if item.get("kind") == "MemberExpr"), None)
        declref = next((item for item in descendants if item.get("kind") == "DeclRefExpr"), None)
        unresolved_lookup = next(
            (item for item in descendants if item.get("kind") in {"UnresolvedLookupExpr", "OverloadExpr"}),
            None,
        )
        if "RecoveryExpr" in kinds:
            resolution_kind = "RECOVERY_UNRESOLVED"
        elif unresolved_lookup is not None:
            resolution_kind = (
                "TEMPLATE_UNRESOLVED" if unresolved_lookup.get("lookups") else "OVERLOAD_UNRESOLVED"
            )
        elif member is not None or raw.get("kind") == "CXXMemberCallExpr":
            resolution_kind = "MEMBER_UNRESOLVED"
        elif declref is not None and isinstance(declref.get("referencedDecl"), Mapping):
            ref = declref["referencedDecl"]
            ref_kind = str(ref.get("kind", ""))
            if ref_kind == "FunctionTemplateDecl":
                resolution_kind = "TEMPLATE_UNRESOLVED"
            elif ref_kind in {"FunctionDecl", "CXXMethodDecl"}:
                raw_target = str(ref.get("id", ""))
                canonical_target = raw_to_canonical.get(raw_target)
                if canonical_target:
                    return _relation(
                        "CALL_CALLEE",
                        node_id,
                        canonical_target,
                        "DIRECT_CALLEE",
                        True,
                        "DIRECT_RESOLVED",
                        "HIGH",
                        "CLANG_CALLEE_DECLREF",
                        stable_ref_id=stable_refs.get(canonical_target),
                    )
                external_target, external_metadata = _stable_external_symbol(ref)
                if external_target:
                    return _relation(
                        "CALL_CALLEE",
                        node_id,
                        external_target,
                        "DIRECT_CALLEE",
                        True,
                        "DIRECT_EXTERNAL",
                        "HIGH",
                        "CLANG_CALLEE_DECLREF",
                        external_symbol=external_metadata,
                    )
                resolution_kind = "DIRECT_EXTERNAL"
            else:
                resolution_kind = "INDIRECT_UNRESOLVED"
        else:
            resolution_kind = "INDIRECT_UNRESOLVED"
        return _relation(
            "CALL_CALLEE",
            node_id,
            f"unresolved:callee:{node_id}:{resolution_kind}",
            "CALLEE_UNRESOLVED",
            False,
            resolution_kind,
            "EXPLICIT_UNRESOLVED",
            "CLANG_CALLEE_EXPRESSION",
        )

    @staticmethod
    def _control_relations(
        node_id: str,
        raw: Mapping[str, Any],
        raw_to_canonical: Mapping[str, str],
        node_by_id: Mapping[str, Mapping[str, Any]],
        first_emitted_descendant: Any,
    ) -> list[dict[str, Any]]:
        kind = str(raw.get("kind"))
        expected_roles = CONTROL_ROLE_SLOTS[kind]
        slots = _raw_child_slots(raw)
        valid_lengths = {len(expected_roles)}
        if kind == "IfStmt":
            valid_lengths = {2, 3}
        if len(slots) not in valid_lengths:
            return [
                _relation(
                    "CONTROL_ROLE",
                    node_id,
                    f"unresolved:control:{node_id}",
                    "CONTROL_ROLE_UNRESOLVED",
                    False,
                    "IRREGULAR_CHILD_SLOTS",
                    "EXPLICIT_UNRESOLVED",
                    "CLANG_JSON_INNER_SLOTS",
                    observed_slot_count=len(slots),
                    expected_slot_count=sorted(valid_lengths),
                )
            ]
        output: list[dict[str, Any]] = []
        for slot_index, raw_child in enumerate(slots):
            if raw_child is None:
                continue
            raw_id = str(raw_child.get("id", ""))
            target = raw_to_canonical.get(raw_id) or first_emitted_descendant(raw_child)
            if target and not node_by_id[target].get("emits_token", True):
                target = first_emitted_descendant(raw_child)
            role = expected_roles[slot_index]
            output.append(
                _relation(
                    "CONTROL_ROLE",
                    node_id,
                    target or f"unresolved:control:{node_id}:{slot_index}",
                    role if target else "CONTROL_ROLE_UNRESOLVED",
                    bool(target),
                    "STRUCTURED_AST_SLOT" if target else "SLOT_TARGET_UNRESOLVED",
                    "HIGH" if target else "EXPLICIT_UNRESOLVED",
                    "CLANG_JSON_INNER_SLOTS",
                    slot_index=slot_index,
                )
            )
        return output

    @staticmethod
    def classify_acceptance(
        canonical: Mapping[str, Any] | None,
        diagnostics: Mapping[str, Any],
        returncode: int,
        ast_json_available: bool,
        timed_out: bool,
        json_error: bool,
    ) -> dict[str, Any]:
        if not canonical:
            reasons = []
            if timed_out:
                reasons.append("CLANG_TIMEOUT")
            if json_error:
                reasons.append("INVALID_AST_JSON")
            if not ast_json_available:
                reasons.append("AST_JSON_UNAVAILABLE")
            reasons.append("TARGET_FUNCTION_NOT_LOCATED")
            return {
                "parse_class": "BLOCKED",
                "classification_reasons": sorted(set(reasons)),
                "target_function_located": False,
                "source_span_integrity": False,
                "ast_structural_completeness": False,
                "declaration_binding_availability": False,
                "recovery_node_presence": False,
                "recovery_node_ratio": None,
                "unresolved_ref_ratio": None,
                "reference_accounting_ratio": None,
                "old_shadow_equivalent": "PARSE_BLOCKED",
            }

        nodes = list(canonical["nodes"])
        relations = list(canonical["relations"])
        function_roots = {item["root_node_id"] for item in canonical["functions"]}
        root_nodes = [node for node in nodes if node["node_id"] in function_roots]
        root_spans_ok = all(
            node.get("source_span", {}).get("begin", {}).get("line") is not None
            and node.get("source_span", {}).get("end", {}).get("line") is not None
            for node in root_nodes
        )
        body_by_function = {
            node["function_id"]
            for node in nodes
            if node["node_kind"] == "CompoundStmt"
            and node.get("source_span", {}).get("begin", {}).get("line") is not None
            and node.get("source_span", {}).get("end", {}).get("line") is not None
        }
        function_ids = {item["function_id"] for item in canonical["functions"]}
        source_span_integrity = bool(root_nodes) and root_spans_ok and body_by_function == function_ids
        ast_structural_completeness = bool(nodes) and body_by_function == function_ids
        recovery_nodes = int(canonical["quality"]["recovery_node_count"])
        emitted_nodes = max(1, int(canonical["quality"]["emitted_node_count"]))
        recovery_ratio = recovery_nodes / emitted_nodes
        reference_sites = int(canonical["quality"]["reference_site_count"])
        unresolved_refs = int(canonical["quality"]["unresolved_reference_count"])
        decl_use_count = sum(item["relation_type"] == "DECL_USE" for item in relations)
        accounting_ratio = decl_use_count / reference_sites if reference_sites else 1.0
        unresolved_ratio = unresolved_refs / reference_sites if reference_sites else 0.0
        declaration_binding_availability = (
            accounting_ratio >= PARSE_ACCEPTANCE_THRESHOLDS["required_reference_accounting_ratio"]
        )
        quarantine_reasons = []
        if not source_span_integrity:
            quarantine_reasons.append("SOURCE_SPAN_INTEGRITY_FAILED")
        if not ast_structural_completeness:
            quarantine_reasons.append("AST_STRUCTURAL_COMPLETENESS_FAILED")
        if not declaration_binding_availability:
            quarantine_reasons.append("REFERENCE_ACCOUNTING_INCOMPLETE")
        if recovery_ratio > PARSE_ACCEPTANCE_THRESHOLDS["max_recovery_node_ratio"]:
            quarantine_reasons.append("RECOVERY_NODE_RATIO_EXCEEDED")
        if unresolved_ratio > PARSE_ACCEPTANCE_THRESHOLDS["max_unresolved_ref_ratio"]:
            quarantine_reasons.append("UNRESOLVED_REF_RATIO_EXCEEDED")

        if quarantine_reasons:
            parse_class = "QUARANTINED"
            reasons = quarantine_reasons
        elif returncode == 0 and diagnostics["error_count"] == 0 and recovery_nodes == 0:
            parse_class = "CLEAN"
            reasons = ["ERROR_FREE_TARGET_FUNCTION_AST"]
        else:
            parse_class = "RECOVERED_ACCEPTED"
            reasons = ["RECOVERY_PROVENANCE_RETAINED", "MINIMUM_COMPLETENESS_CONTRACT_SATISFIED"]
        old_shadow = "PARSE_OK" if returncode == 0 else "PARSE_RECOVERED"
        return {
            "parse_class": parse_class,
            "classification_reasons": reasons,
            "target_function_located": True,
            "source_span_integrity": source_span_integrity,
            "ast_structural_completeness": ast_structural_completeness,
            "declaration_binding_availability": declaration_binding_availability,
            "diagnostic_categories": list(diagnostics["categories"]),
            "recovery_node_presence": recovery_nodes > 0,
            "recovery_node_ratio": recovery_ratio,
            "unresolved_ref_ratio": unresolved_ratio,
            "reference_accounting_ratio": accounting_ratio,
            "old_shadow_equivalent": old_shadow,
        }
