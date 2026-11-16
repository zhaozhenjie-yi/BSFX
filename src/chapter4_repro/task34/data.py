"""Task 34 train-only X1 multi-path tensorization.

This module closes the Task 31/32 adapter gap without changing any historical
artifact.  Its public batch contract has two synchronized views:

* canonical ``[B,S,P,T]`` tensors for the Task 32 hierarchy; and
* packed valid-path ``[N,L]`` tensors for H1/DWAF/structural Transformer work.

``N`` contains real paths only.  ``S`` is a segment ordinal within one path,
and a stable ``P`` slot identifies that same path at every ordinal.  Textual
path IDs are never treated as globally unique; all grouping uses composite
``(sample_id, path_group_id)`` keys.

Exact sibling-parent identity is supplied by a train-only ``ast_main``
sidecar.  Semantic-path depth is deliberately never used to guess parents.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from chapter4_repro.models.structure_sensitive_transformer import (
    REF_RELATION_BIDIRECTIONAL,
    REF_RELATION_DEF_TO_USE,
    REF_RELATION_NONE,
    REF_RELATION_USE_TO_DEF,
    SPECIAL_TOKEN_NONE,
    SPECIAL_TOKEN_PAD,
    SPECIAL_TOKEN_SEG_END,
    SPECIAL_TOKEN_SEG_START,
)
from chapter4_repro.training.dataset import (
    Vocabulary,
    h1_ref_identity_id,
    h1_ref_identity_key,
)


TASK34_LAYOUT_ID = "task34-x1-b-s-p-t-v1"
TASK34_PARENT_CONTRACT = "shared-protocol-train-ast-main-exact-preorder-v1"
TASK34_REF_CONTRACT = "segment-global-endpoint-directional-v1"
TASK34_MULTISEGMENT_THRESHOLD = 50


def _composite_path_key(sample_id: str, path_group_id: str) -> tuple[str, str]:
    return sample_id, path_group_id


def _composite_segment_key(
    sample_id: str,
    path_group_id: str,
    segment_index: int,
) -> tuple[str, str, int]:
    return sample_id, path_group_id, int(segment_index)


def _assert_train_only_jsonl(path: str | Path) -> Path:
    candidate = Path(path)
    lowered = {part.lower() for part in candidate.parts}
    forbidden = {
        "val_build",
        "validation",
        "test_build",
        "sealed_test",
        "sealed_test_sources",
    }
    if lowered & forbidden:
        raise ValueError(f"Task 34 refuses non-train input: {candidate}")
    if "train_build" not in lowered or not candidate.name.startswith("train_"):
        raise ValueError(f"Task 34 requires an explicit train_build/train_* input: {candidate}")
    return candidate


def iter_train_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield an explicitly train-only JSONL and reject split drift."""

    candidate = _assert_train_only_jsonl(path)
    with candidate.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            # ``shared_protocol/*/train_build/train_paths.jsonl`` is a
            # path-level artifact and predates the redundant per-record split
            # column.  The guarded path itself is its train attestation.  If a
            # split is present, however, drift is always rejected.
            if record.get("split") is not None and str(record.get("split")) != "train":
                raise ValueError(
                    f"non-train record in {candidate} at line {line_number}: "
                    f"split={record.get('split')!r}"
                )
            yield record


def read_train_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_train_jsonl(path))


@dataclass(frozen=True)
class ExactParentSidecar:
    """Exact ``(sample,node) -> parent node`` mapping from complete AST paths."""

    parent_by_node: Mapping[tuple[str, int], int | None]
    source_path_keys: tuple[tuple[str, str], ...]
    contract: str = TASK34_PARENT_CONTRACT

    def parent_for(self, sample_id: str, node_id: int) -> int | None:
        key = (str(sample_id), int(node_id))
        if key not in self.parent_by_node:
            raise ValueError(
                "semantic node is absent from the exact train ast_main sidecar: "
                f"sample_id={sample_id!r}, node_id={node_id}"
            )
        return self.parent_by_node[key]


def build_exact_parent_sidecar(
    ast_main_records: Iterable[Mapping[str, Any]],
) -> ExactParentSidecar:
    """Reconstruct exact parent node IDs from complete ``ast_main`` preorder.

    Context-window duplicates are merged by global position.  The algorithm
    fails closed unless every source path covers a contiguous preorder from
    position zero.  It is intentionally unavailable for semantic-path input.
    """

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in ast_main_records:
        if record.get("split") is not None and str(record.get("split")) != "train":
            raise ValueError("exact parent sidecar accepts train records only")
        if str(record.get("path_kind")) != "ast_main":
            raise ValueError("exact parent sidecar accepts path_kind=ast_main only")
        sample_id = str(record.get("sample_id") or "")
        path_id = str(record.get("path_id") or "")
        if not sample_id or not path_id:
            raise ValueError("ast_main sidecar records require sample_id and path_id")
        grouped[(sample_id, path_id)].append(record)
    if not grouped:
        raise ValueError("exact parent sidecar requires at least one ast_main path")

    parent_by_node: dict[tuple[str, int], int | None] = {}
    for (sample_id, path_id), records in sorted(grouped.items()):
        rows: dict[int, tuple[int, int, int]] = {}
        for record in records:
            node_ids = list(record.get("node_ids") or [])
            columns = {
                "node_ids": node_ids,
                "depths": list(record.get("depths") or []),
                "siblings": list(record.get("siblings") or []),
                # Full train_paths ast_main records are already one complete
                # preorder and therefore need no persisted window positions.
                "global_positions": list(record.get("global_positions") or range(len(node_ids))),
            }
            if len({len(values) for values in columns.values()}) != 1:
                raise ValueError(
                    f"unaligned ast_main sidecar columns for {(sample_id, path_id)!r}"
                )
            for position, node_id, depth, sibling in zip(
                columns["global_positions"],
                columns["node_ids"],
                columns["depths"],
                columns["siblings"],
            ):
                position = int(position)
                candidate = (int(node_id), int(depth), int(sibling))
                previous = rows.get(position)
                if previous is not None and previous != candidate:
                    raise ValueError(
                        f"conflicting ast_main node at {(sample_id, path_id, position)!r}"
                    )
                rows[position] = candidate
        positions = sorted(rows)
        if not positions or positions != list(range(len(positions))):
            raise ValueError(
                f"ast_main preorder is not complete/contiguous for {(sample_id, path_id)!r}"
            )

        stack: list[int] = []
        children_by_parent: dict[int, set[int]] = defaultdict(set)
        for position in positions:
            node_id, depth, sibling = rows[position]
            if depth < 0:
                raise ValueError("AST depth must be non-negative")
            while stack and rows[stack[-1]][1] >= depth:
                stack.pop()
            parent_position = stack[-1] if stack else None
            parent_node_id = None if parent_position is None else rows[parent_position][0]
            if parent_position is not None:
                ranks = children_by_parent[parent_node_id]
                if sibling in ranks:
                    raise ValueError(
                        f"duplicate sibling rank {sibling} under node {parent_node_id}"
                    )
                ranks.add(sibling)
            key = (sample_id, node_id)
            previous_parent = parent_by_node.get(key)
            if key in parent_by_node and previous_parent != parent_node_id:
                raise ValueError(f"conflicting exact parent for node {key!r}")
            parent_by_node[key] = parent_node_id
            stack.append(position)

    return ExactParentSidecar(
        parent_by_node=parent_by_node,
        source_path_keys=tuple(sorted(grouped)),
    )


def load_exact_parent_sidecar(path: str | Path) -> ExactParentSidecar:
    return build_exact_parent_sidecar(
        record
        for record in iter_train_jsonl(path)
        if str(record.get("path_kind")) == "ast_main"
    )


@dataclass(frozen=True)
class TokenProvenance:
    sample_id: str
    path_group_id: str
    path_id: str
    segment_id: str
    segment_index: int
    token_index: int
    source_token_index: int | None
    source_global_position: int | None
    node_id: int | None
    role: str
    is_core: bool


@dataclass
class PreparedSegment:
    segment_id: str
    segment_index: int
    token_ids: Tensor
    depth_ids: Tensor
    sibling_order_ids: Tensor
    sibling_parent_ids: Tensor
    ast_node_identity_ids: Tensor
    global_position_ids: Tensor
    local_position_ids: Tensor
    segment_ids: Tensor
    special_token_ids: Tensor
    provenance: tuple[TokenProvenance, ...]
    truncated: bool


@dataclass
class PreparedPath:
    sample_id: str
    path_group_id: str
    path_id: str
    path_kind: str
    path_family: str
    composite_key: tuple[str, str]
    duplicate_weight: float
    source_path_length: int
    segments: tuple[PreparedSegment, ...]
    flat_token_ids: Tensor
    flat_depth_ids: Tensor
    flat_sibling_order_ids: Tensor
    flat_sibling_parent_ids: Tensor
    flat_ast_node_identity_ids: Tensor
    flat_global_position_ids: Tensor
    flat_local_position_ids: Tensor
    flat_delta_seg: Tensor
    flat_ref_identity_ids: Tensor
    flat_ref_definition_token_ids: Tensor
    flat_segment_ids: Tensor
    flat_special_token_ids: Tensor
    ref_link_matrix: Tensor
    segment_offsets: tuple[tuple[int, int], ...]
    resolved_ref_links: tuple[dict[str, Any], ...]
    unresolved_ref_links: tuple[dict[str, Any], ...]


@dataclass
class PreparedSample:
    sample_id: str
    label: int
    paths: tuple[PreparedPath, ...]


@dataclass(frozen=True)
class MultisegmentEligibilityAudit:
    dataset_id: str
    threshold: int
    total_paths: int
    triggered_paths: int
    triggered_multisegment_paths: int
    status: str
    violations: tuple[tuple[str, str, int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_ref_relation(existing: int, candidate: int) -> int:
    if existing in {REF_RELATION_NONE, candidate}:
        return candidate
    return REF_RELATION_BIDIRECTIONAL


def _validate_segment_record(record: Mapping[str, Any]) -> int:
    if str(record.get("split")) != "train":
        raise ValueError("Task 34 X1 adapter accepts train records only")
    if str(record.get("semantic_profile")) != "semantic_x1_v1":
        raise ValueError("Task 34 X1 adapter requires semantic_profile=semantic_x1_v1")
    if str(record.get("path_kind") or "").startswith("semantic_") is False:
        raise ValueError("Task 34 X1 adapter rejects non-semantic paths")
    columns = (
        "node_ids",
        "nodes",
        "node_types",
        "depths",
        "siblings",
        "lines",
        "global_positions",
        "local_positions",
    )
    lengths = {name: len(record.get(name) or []) for name in columns}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"unaligned X1 segment columns: {lengths}")
    length = next(iter(lengths.values()))
    if length <= 0:
        raise ValueError("Task 34 rejects empty X1 segments")
    if list(map(int, record.get("local_positions") or [])) != list(range(length)):
        raise ValueError("X1 segment local_positions must be contiguous from zero")
    return length


def _prepare_segment(
    record: Mapping[str, Any],
    *,
    vocab: Vocabulary,
    parent_sidecar: ExactParentSidecar,
    path_group_id: str,
    max_tokens_per_segment: int | None,
) -> PreparedSegment:
    source_length = _validate_segment_record(record)
    if max_tokens_per_segment is not None:
        if max_tokens_per_segment < 3:
            raise ValueError(
                "max_tokens_per_segment must preserve SEG_START, one content token and SEG_END"
            )
        keep = min(source_length, int(max_tokens_per_segment) - 2)
    else:
        keep = source_length
    truncated = keep < source_length

    sample_id = str(record["sample_id"])
    path_id = str(record["path_id"])
    segment_id = str(record["segment_id"])
    segment_index = int(record["segment_index"])
    nodes = [str(value) for value in record["nodes"][:keep]]
    node_ids = [int(value) for value in record["node_ids"][:keep]]
    depths = [int(value) for value in record["depths"][:keep]]
    siblings = [int(value) for value in record["siblings"][:keep]]
    source_globals = [int(value) for value in record["global_positions"][:keep]]
    source_locals = [int(value) for value in record["local_positions"][:keep]]
    parents = [parent_sidecar.parent_for(sample_id, value) for value in node_ids]
    parent_ids = [-1 if value is None else int(value) for value in parents]

    start_id = int(vocab.encode("[SEG_START]"))
    end_id = int(vocab.encode("[SEG_END]"))
    if start_id in {vocab.pad_id, vocab.unk_id} or end_id in {vocab.pad_id, vocab.unk_id}:
        raise ValueError("SEG_START/SEG_END require reserved vocabulary IDs")
    if start_id == end_id:
        raise ValueError("SEG_START and SEG_END must have distinct vocabulary IDs")

    core_span = [int(value) for value in record.get("core_span") or []]
    if len(core_span) != 2 or core_span[0] < 0 or core_span[1] <= core_span[0]:
        raise ValueError("X1 segment requires a valid core_span")
    roles: list[str] = []
    core_flags: list[bool] = []
    for position in source_globals:
        is_core = core_span[0] <= position < core_span[1]
        core_flags.append(is_core)
        roles.append("core" if is_core else ("left_context" if position < core_span[0] else "right_context"))

    provenance = [
        TokenProvenance(
            sample_id=sample_id,
            path_group_id=path_group_id,
            path_id=path_id,
            segment_id=segment_id,
            segment_index=segment_index,
            token_index=0,
            source_token_index=None,
            source_global_position=None,
            node_id=None,
            role="SEG_START",
            is_core=False,
        )
    ]
    provenance.extend(
        TokenProvenance(
            sample_id=sample_id,
            path_group_id=path_group_id,
            path_id=path_id,
            segment_id=segment_id,
            segment_index=segment_index,
            token_index=index + 1,
            source_token_index=int(source_locals[index]),
            source_global_position=int(source_globals[index]),
            node_id=int(node_ids[index]),
            role=roles[index],
            is_core=core_flags[index],
        )
        for index in range(keep)
    )
    provenance.append(
        TokenProvenance(
            sample_id=sample_id,
            path_group_id=path_group_id,
            path_id=path_id,
            segment_id=segment_id,
            segment_index=segment_index,
            token_index=keep + 1,
            source_token_index=None,
            source_global_position=None,
            node_id=None,
            role="SEG_END",
            is_core=False,
        )
    )

    shifted_global = [0] + [value + 1 for value in source_globals] + [source_globals[-1] + 2]
    shifted_local = [0] + [value + 1 for value in source_locals] + [source_locals[-1] + 2]
    return PreparedSegment(
        segment_id=segment_id,
        segment_index=segment_index,
        token_ids=torch.tensor(
            [start_id] + [int(vocab.encode(value)) for value in nodes] + [end_id],
            dtype=torch.long,
        ),
        depth_ids=torch.tensor([0] + depths + [0], dtype=torch.long),
        sibling_order_ids=torch.tensor([0] + siblings + [0], dtype=torch.long),
        sibling_parent_ids=torch.tensor([-1] + parent_ids + [-1], dtype=torch.long),
        ast_node_identity_ids=torch.tensor([-1] + node_ids + [-1], dtype=torch.long),
        global_position_ids=torch.tensor(shifted_global, dtype=torch.long),
        local_position_ids=torch.tensor(shifted_local, dtype=torch.long),
        segment_ids=torch.full((keep + 2,), segment_index, dtype=torch.long),
        special_token_ids=torch.tensor(
            [SPECIAL_TOKEN_SEG_START] + [SPECIAL_TOKEN_NONE] * keep + [SPECIAL_TOKEN_SEG_END],
            dtype=torch.long,
        ),
        provenance=tuple(provenance),
        truncated=truncated,
    )


def _build_path(
    records: Sequence[Mapping[str, Any]],
    *,
    vocab: Vocabulary,
    parent_sidecar: ExactParentSidecar,
    max_tokens_per_segment: int | None,
    require_all_ref_endpoints: bool,
) -> PreparedPath:
    if not records:
        raise ValueError("path records must not be empty")
    sample_ids = {str(record.get("sample_id")) for record in records}
    path_ids = {str(record.get("path_id")) for record in records}
    group_ids = {str(record.get("path_group_id") or record.get("path_id")) for record in records}
    if len(sample_ids) != 1 or len(path_ids) != 1 or len(group_ids) != 1:
        raise ValueError("one Task 34 path requires one sample/path/group identity")
    sample_id = next(iter(sample_ids))
    path_id = next(iter(path_ids))
    path_group_id = next(iter(group_ids))
    ordered = sorted(records, key=lambda value: int(value.get("segment_index", -1)))
    segment_indices = [int(value.get("segment_index", -1)) for value in ordered]
    if segment_indices != list(range(len(ordered))):
        raise ValueError(
            f"segment indices must be unique/contiguous for {(sample_id, path_group_id)!r}"
        )

    core_spans = [[int(part) for part in record.get("core_span") or []] for record in ordered]
    if any(len(span) != 2 for span in core_spans):
        raise ValueError("every X1 segment requires core_span=[start,end]")
    if core_spans[0][0] != 0 or any(
        previous[1] != current[0] for previous, current in zip(core_spans, core_spans[1:])
    ):
        raise ValueError("X1 core spans must partition the path contiguously from zero")
    source_path_length = core_spans[-1][1]

    segments = tuple(
        _prepare_segment(
            record,
            vocab=vocab,
            parent_sidecar=parent_sidecar,
            path_group_id=path_group_id,
            max_tokens_per_segment=max_tokens_per_segment,
        )
        for record in ordered
    )
    segment_offsets: list[tuple[int, int]] = []
    offset = 0
    endpoint_to_flat: dict[tuple[int, int], int] = {}
    endpoint_to_token: dict[tuple[int, int], str] = {}
    for segment in segments:
        start = offset
        end = start + int(segment.token_ids.numel())
        segment_offsets.append((start, end))
        for local_index, provenance in enumerate(segment.provenance):
            if provenance.source_global_position is None:
                continue
            endpoint = (segment.segment_index, provenance.source_global_position)
            if endpoint in endpoint_to_flat:
                raise ValueError(f"duplicate endpoint in segment copy: {endpoint!r}")
            endpoint_to_flat[endpoint] = start + local_index
            endpoint_to_token[endpoint] = str(ordered[segment.segment_index]["nodes"][provenance.source_token_index])
        offset = end

    flat_token_ids = torch.cat([value.token_ids for value in segments])
    flat_depth_ids = torch.cat([value.depth_ids for value in segments])
    flat_sibling_order_ids = torch.cat([value.sibling_order_ids for value in segments])
    flat_sibling_parent_ids = torch.cat([value.sibling_parent_ids for value in segments])
    flat_ast_node_identity_ids = torch.cat([value.ast_node_identity_ids for value in segments])
    flat_global_position_ids = torch.cat([value.global_position_ids for value in segments])
    flat_local_position_ids = torch.cat([value.local_position_ids for value in segments])
    flat_segment_ids = torch.cat([value.segment_ids for value in segments])
    flat_special_token_ids = torch.cat([value.special_token_ids for value in segments])
    total_tokens = int(flat_token_ids.numel())
    flat_delta_seg = torch.zeros(total_tokens, dtype=torch.float32)
    flat_ref_identity_ids = torch.zeros(total_tokens, dtype=torch.long)
    flat_ref_definition_token_ids = torch.zeros(total_tokens, dtype=torch.long)
    ref_link_matrix = torch.zeros((total_tokens, total_tokens), dtype=torch.long)

    seen_links: set[tuple[Any, ...]] = set()
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    identity_at_token: dict[int, str] = {}
    definition_priority: dict[int, tuple[int, int]] = {}
    bucket_keys: dict[int, str] = {}
    for record in ordered:
        for link in record.get("ref_links") or []:
            required = {
                "identifier",
                "def_global_pos",
                "use_global_pos",
                "def_segment",
                "use_segment",
                "def_kind",
                "use_kind",
                "pair_kind",
            }
            missing = sorted(required - set(link))
            if missing:
                raise ValueError(f"REF link is missing provenance fields: {missing}")
            key = (
                str(link["identifier"]),
                int(link["def_segment"]),
                int(link["def_global_pos"]),
                int(link["use_segment"]),
                int(link["use_global_pos"]),
                str(link["def_kind"]),
                str(link["use_kind"]),
                str(link["pair_kind"]),
            )
            if key in seen_links:
                continue
            seen_links.add(key)
            def_endpoint = (key[1], key[2])
            use_endpoint = (key[3], key[4])
            def_index = endpoint_to_flat.get(def_endpoint)
            use_index = endpoint_to_flat.get(use_endpoint)
            evidence = {
                "identifier": key[0],
                "def_segment": key[1],
                "def_global_position": key[2],
                "use_segment": key[3],
                "use_global_position": key[4],
                "def_kind": key[5],
                "use_kind": key[6],
                "pair_kind": key[7],
                "def_token_index": def_index,
                "use_token_index": use_index,
            }
            if def_index is None or use_index is None:
                unresolved.append(evidence)
                continue

            ref_link_matrix[def_index, use_index] = _merge_ref_relation(
                int(ref_link_matrix[def_index, use_index]), REF_RELATION_DEF_TO_USE
            )
            ref_link_matrix[use_index, def_index] = _merge_ref_relation(
                int(ref_link_matrix[use_index, def_index]), REF_RELATION_USE_TO_DEF
            )
            distance = abs(key[3] - key[1])
            delta_value = float(distance)
            flat_delta_seg[def_index] = max(float(flat_delta_seg[def_index]), delta_value)
            flat_delta_seg[use_index] = max(float(flat_delta_seg[use_index]), delta_value)

            identity_key = h1_ref_identity_key(dict(record), dict(link))
            identity_id = h1_ref_identity_id(identity_key)
            prior_key = bucket_keys.get(identity_id)
            if prior_key is not None and prior_key != identity_key:
                raise ValueError("sample/path-local H1 REF hash collision")
            bucket_keys[identity_id] = identity_key
            definition_text = endpoint_to_token[def_endpoint]
            definition_token_id = int(vocab.encode(definition_text))
            for token_index, is_definition in ((def_index, True), (use_index, False)):
                prior_identity = identity_at_token.get(token_index)
                if prior_identity is not None and prior_identity != identity_key:
                    raise ValueError("multiple H1 REF identities target one designated endpoint")
                identity_at_token[token_index] = identity_key
                flat_ref_identity_ids[token_index] = identity_id
                priority = (1 if is_definition else 0, key[1])
                if priority >= definition_priority.get(token_index, (-1, -1)):
                    flat_ref_definition_token_ids[token_index] = definition_token_id
                    definition_priority[token_index] = priority
            evidence["cross_segment"] = key[1] != key[3]
            resolved.append(evidence)

    if require_all_ref_endpoints and unresolved:
        raise ValueError(
            f"{len(unresolved)} REF link(s) lost a designated endpoint, usually due to truncation"
        )

    first = ordered[0]
    weights = {float(value.get("duplicate_weight", 1.0)) for value in ordered}
    if len(weights) != 1 or next(iter(weights)) <= 0:
        raise ValueError("all segments in one path require one positive duplicate_weight")
    return PreparedPath(
        sample_id=sample_id,
        path_group_id=path_group_id,
        path_id=path_id,
        path_kind=str(first.get("path_kind") or ""),
        path_family=str(first.get("path_family") or ""),
        composite_key=_composite_path_key(sample_id, path_group_id),
        duplicate_weight=next(iter(weights)),
        source_path_length=source_path_length,
        segments=segments,
        flat_token_ids=flat_token_ids,
        flat_depth_ids=flat_depth_ids,
        flat_sibling_order_ids=flat_sibling_order_ids,
        flat_sibling_parent_ids=flat_sibling_parent_ids,
        flat_ast_node_identity_ids=flat_ast_node_identity_ids,
        flat_global_position_ids=flat_global_position_ids,
        flat_local_position_ids=flat_local_position_ids,
        flat_delta_seg=flat_delta_seg,
        flat_ref_identity_ids=flat_ref_identity_ids,
        flat_ref_definition_token_ids=flat_ref_definition_token_ids,
        flat_segment_ids=flat_segment_ids,
        flat_special_token_ids=flat_special_token_ids,
        ref_link_matrix=ref_link_matrix,
        segment_offsets=tuple(segment_offsets),
        resolved_ref_links=tuple(resolved),
        unresolved_ref_links=tuple(unresolved),
    )


def prepare_task34_samples(
    x1_records: Iterable[Mapping[str, Any]],
    *,
    vocab: Vocabulary,
    parent_sidecar: ExactParentSidecar,
    dataset_id: str,
    max_core_path_length: int = TASK34_MULTISEGMENT_THRESHOLD,
    max_tokens_per_segment: int | None = None,
    require_all_ref_endpoints: bool = True,
) -> tuple[list[PreparedSample], MultisegmentEligibilityAudit]:
    """Group X1 records into stable samples/paths and audit segmentation eligibility."""

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in x1_records:
        _validate_segment_record(record)
        sample_id = str(record.get("sample_id") or "")
        path_group_id = str(record.get("path_group_id") or record.get("path_id") or "")
        if not sample_id or not path_group_id:
            raise ValueError("X1 records require sample_id and path_group_id/path_id")
        grouped[sample_id][path_group_id].append(record)
    if not grouped:
        raise ValueError("Task 34 X1 adapter requires at least one record")

    samples: list[PreparedSample] = []
    violations: list[tuple[str, str, int, int]] = []
    triggered = 0
    triggered_multisegment = 0
    total_paths = 0
    for sample_id in sorted(grouped):
        paths: list[PreparedPath] = []
        labels: set[int] = set()
        for path_group_id in sorted(grouped[sample_id]):
            records = grouped[sample_id][path_group_id]
            labels.update(int(record.get("label", -1)) for record in records)
            path = _build_path(
                records,
                vocab=vocab,
                parent_sidecar=parent_sidecar,
                max_tokens_per_segment=max_tokens_per_segment,
                require_all_ref_endpoints=require_all_ref_endpoints,
            )
            paths.append(path)
            total_paths += 1
            if path.source_path_length > int(max_core_path_length):
                triggered += 1
                if len(path.segments) > 1:
                    triggered_multisegment += 1
                else:
                    violations.append(
                        (
                            sample_id,
                            path_group_id,
                            path.source_path_length,
                            len(path.segments),
                        )
                    )
        if len(labels) != 1 or next(iter(labels)) not in {0, 1}:
            raise ValueError(f"sample {sample_id!r} requires one binary label")
        samples.append(
            PreparedSample(
                sample_id=sample_id,
                label=next(iter(labels)),
                paths=tuple(paths),
            )
        )
    if violations:
        status = "TRIGGERED_FAIL"
    elif triggered:
        status = "TRIGGERED_PASS"
    else:
        status = "NOT_TRIGGERED"
    audit = MultisegmentEligibilityAudit(
        dataset_id=str(dataset_id),
        threshold=int(max_core_path_length),
        total_paths=total_paths,
        triggered_paths=triggered,
        triggered_multisegment_paths=triggered_multisegment,
        status=status,
        violations=tuple(violations),
    )
    return samples, audit


class Task34MultipathDataset(Dataset):
    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        vocab: Vocabulary,
        parent_sidecar: ExactParentSidecar,
        dataset_id: str,
        max_core_path_length: int = TASK34_MULTISEGMENT_THRESHOLD,
        max_tokens_per_segment: int | None = None,
        require_all_ref_endpoints: bool = True,
    ) -> None:
        self.samples, self.eligibility_audit = prepare_task34_samples(
            records,
            vocab=vocab,
            parent_sidecar=parent_sidecar,
            dataset_id=dataset_id,
            max_core_path_length=max_core_path_length,
            max_tokens_per_segment=max_tokens_per_segment,
            require_all_ref_endpoints=require_all_ref_endpoints,
        )
        self.vocab = vocab
        self.dataset_id = str(dataset_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreparedSample:
        return self.samples[index]

    def sample_labels(self) -> list[int]:
        return [sample.label for sample in self.samples]


def _pad_1d(value: Tensor, length: int, padding_value: int | float) -> Tensor:
    return torch.nn.functional.pad(value, (0, length - int(value.shape[0])), value=padding_value)


def collate_task34_multipath(
    samples: Sequence[PreparedSample],
    *,
    pad_id: int,
) -> dict[str, Any]:
    """Create synchronized canonical and valid-path-packed dynamic batches."""

    if not samples:
        raise ValueError("Task 34 collate requires at least one sample")
    batch_size = len(samples)
    max_paths = max(len(sample.paths) for sample in samples)
    max_segments = max(len(path.segments) for sample in samples for path in sample.paths)
    max_tokens = max(
        int(segment.token_ids.numel())
        for sample in samples
        for path in sample.paths
        for segment in path.segments
    )
    packed_paths = [path for sample in samples for path in sample.paths]
    max_packed_tokens = max(int(path.flat_token_ids.numel()) for path in packed_paths)

    shape = (batch_size, max_segments, max_paths, max_tokens)
    tensors: dict[str, Tensor] = {
        "token_ids": torch.full(shape, int(pad_id), dtype=torch.long),
        "depth_ids": torch.zeros(shape, dtype=torch.long),
        "sibling_order_ids": torch.zeros(shape, dtype=torch.long),
        "sibling_parent_ids": torch.full(shape, -1, dtype=torch.long),
        "ast_node_identity_ids": torch.full(shape, -1, dtype=torch.long),
        "global_position_ids": torch.zeros(shape, dtype=torch.long),
        "local_position_ids": torch.zeros(shape, dtype=torch.long),
        "delta_seg": torch.zeros(shape, dtype=torch.float32),
        "ref_identity_ids": torch.zeros(shape, dtype=torch.long),
        "ref_definition_token_ids": torch.zeros(shape, dtype=torch.long),
        "segment_ids": torch.full(shape, -1, dtype=torch.long),
        "special_token_ids": torch.full(shape, SPECIAL_TOKEN_PAD, dtype=torch.long),
    }
    token_valid_mask = torch.zeros(shape, dtype=torch.bool)
    path_segment_valid_mask = torch.zeros(
        (batch_size, max_segments, max_paths), dtype=torch.bool
    )
    path_valid_mask = torch.zeros((batch_size, max_paths), dtype=torch.bool)
    path_duplicate_weights = torch.zeros((batch_size, max_paths), dtype=torch.float32)
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)

    provenance: list[dict[str, Any]] = []
    packed_path_to_sample: list[int] = []
    packed_path_slot: list[int] = []
    packed_segment_indices: list[Tensor] = []
    packed_token_indices: list[Tensor] = []
    packed_rows: dict[str, list[Tensor]] = {
        "token_ids": [],
        "depth_ids": [],
        "sibling_order_ids": [],
        "sibling_parent_ids": [],
        "ast_node_identity_ids": [],
        "global_position_ids": [],
        "local_position_ids": [],
        "delta_seg": [],
        "ref_identity_ids": [],
        "ref_definition_token_ids": [],
        "segment_ids": [],
        "special_token_ids": [],
    }
    packed_padding_masks: list[Tensor] = []
    packed_ref_matrices: list[Tensor] = []

    for batch_index, sample in enumerate(samples):
        sample_audit: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "sample_key": (sample.sample_id,),
            "label": sample.label,
            "paths": [],
        }
        for path_slot, path in enumerate(sample.paths):
            path_valid_mask[batch_index, path_slot] = True
            path_duplicate_weights[batch_index, path_slot] = path.duplicate_weight
            path_audit = {
                "path_slot": path_slot,
                "path_key": path.composite_key,
                "path_id": path.path_id,
                "path_group_id": path.path_group_id,
                "path_kind": path.path_kind,
                "path_family": path.path_family,
                "source_path_length": path.source_path_length,
                "resolved_ref_link_count": len(path.resolved_ref_links),
                "segments": [],
            }
            for segment in path.segments:
                segment_index = segment.segment_index
                length = int(segment.token_ids.numel())
                path_segment_valid_mask[batch_index, segment_index, path_slot] = True
                token_valid_mask[batch_index, segment_index, path_slot, :length] = True
                start, end = path.segment_offsets[segment_index]
                values = {
                    "token_ids": segment.token_ids,
                    "depth_ids": segment.depth_ids,
                    "sibling_order_ids": segment.sibling_order_ids,
                    "sibling_parent_ids": segment.sibling_parent_ids,
                    "ast_node_identity_ids": segment.ast_node_identity_ids,
                    "global_position_ids": segment.global_position_ids,
                    "local_position_ids": segment.local_position_ids,
                    "delta_seg": path.flat_delta_seg[start:end],
                    "ref_identity_ids": path.flat_ref_identity_ids[start:end],
                    "ref_definition_token_ids": path.flat_ref_definition_token_ids[start:end],
                    "segment_ids": segment.segment_ids,
                    "special_token_ids": segment.special_token_ids,
                }
                for name, value in values.items():
                    tensors[name][batch_index, segment_index, path_slot, :length] = value
                path_audit["segments"].append(
                    {
                        "segment_key": _composite_segment_key(
                            sample.sample_id, path.path_group_id, segment_index
                        ),
                        "segment_id": segment.segment_id,
                        "segment_index": segment_index,
                        "truncated": segment.truncated,
                        "tokens": [asdict(value) for value in segment.provenance],
                    }
                )
            sample_audit["paths"].append(path_audit)

            packed_path_to_sample.append(batch_index)
            packed_path_slot.append(path_slot)
            valid_length = int(path.flat_token_ids.numel())
            padding_length = max_packed_tokens - valid_length
            packed_values = {
                "token_ids": (path.flat_token_ids, int(pad_id)),
                "depth_ids": (path.flat_depth_ids, 0),
                "sibling_order_ids": (path.flat_sibling_order_ids, 0),
                "sibling_parent_ids": (path.flat_sibling_parent_ids, -1),
                "ast_node_identity_ids": (path.flat_ast_node_identity_ids, -1),
                "global_position_ids": (path.flat_global_position_ids, 0),
                "local_position_ids": (path.flat_local_position_ids, 0),
                "delta_seg": (path.flat_delta_seg, 0.0),
                "ref_identity_ids": (path.flat_ref_identity_ids, 0),
                "ref_definition_token_ids": (path.flat_ref_definition_token_ids, 0),
                "segment_ids": (path.flat_segment_ids, -1),
                "special_token_ids": (path.flat_special_token_ids, SPECIAL_TOKEN_PAD),
            }
            for name, (value, padding_value) in packed_values.items():
                packed_rows[name].append(_pad_1d(value, max_packed_tokens, padding_value))
            packed_padding_masks.append(
                torch.tensor(
                    [False] * valid_length + [True] * padding_length,
                    dtype=torch.bool,
                )
            )
            packed_ref_matrices.append(
                torch.nn.functional.pad(
                    path.ref_link_matrix,
                    (0, padding_length, 0, padding_length),
                    value=REF_RELATION_NONE,
                )
            )
            flat_segments = torch.full((max_packed_tokens,), -1, dtype=torch.long)
            flat_tokens = torch.full((max_packed_tokens,), -1, dtype=torch.long)
            for segment_index, (start, end) in enumerate(path.segment_offsets):
                flat_segments[start:end] = segment_index
                flat_tokens[start:end] = torch.arange(end - start, dtype=torch.long)
            packed_segment_indices.append(flat_segments)
            packed_token_indices.append(flat_tokens)
        provenance.append(sample_audit)

    segment_valid_mask = path_segment_valid_mask.any(dim=2)
    token_padding_mask = ~token_valid_mask
    packed_padding_mask = torch.stack(packed_padding_masks)
    packed_valid = ~packed_padding_mask
    output: dict[str, Any] = {
        **tensors,
        "token_valid_mask": token_valid_mask,
        "token_padding_mask": token_padding_mask,
        "path_segment_valid_mask": path_segment_valid_mask,
        "segment_valid_mask": segment_valid_mask,
        "segment_padding_mask": ~segment_valid_mask,
        "path_valid_mask": path_valid_mask,
        "path_padding_mask": ~path_valid_mask,
        "path_duplicate_weights": path_duplicate_weights,
        "labels": labels,
        "sample_ids": [sample.sample_id for sample in samples],
        "provenance": provenance,
        "layout_id": TASK34_LAYOUT_ID,
        "parent_contract": TASK34_PARENT_CONTRACT,
        "ref_contract": TASK34_REF_CONTRACT,
        "batch_size": batch_size,
        "max_segments": max_segments,
        "max_paths": max_paths,
        "max_tokens": max_tokens,
        "max_packed_tokens": max_packed_tokens,
        "packed_path_count": len(packed_paths),
        "packed_path_to_sample": torch.tensor(packed_path_to_sample, dtype=torch.long),
        "packed_path_slot": torch.tensor(packed_path_slot, dtype=torch.long),
        "packed_segment_indices": torch.stack(packed_segment_indices),
        "packed_token_indices": torch.stack(packed_token_indices),
        "packed_padding_mask": packed_padding_mask,
        "packed_attention_mask": packed_valid.unsqueeze(-1) & packed_valid.unsqueeze(-2),
        "packed_ref_link_matrix": torch.stack(packed_ref_matrices),
    }
    for name, rows in packed_rows.items():
        output[f"packed_{name}"] = torch.stack(rows)
    if int(path_valid_mask.sum()) != len(packed_paths):
        raise AssertionError("packed path count drifted from canonical path_valid_mask")
    return output


def scatter_packed_paths_to_hierarchy(
    packed_values: Tensor,
    batch: Mapping[str, Any],
) -> Tensor:
    """Scatter ``[N,L,D]`` valid-path outputs back to ``[B,S,P,T,D]``."""

    if packed_values.ndim != 3:
        raise ValueError("packed_values must have shape [N,L,D]")
    expected = (
        int(batch["packed_path_count"]),
        int(batch["max_packed_tokens"]),
    )
    if tuple(packed_values.shape[:2]) != expected:
        raise ValueError(f"packed_values leading shape must be {expected}")
    hidden = int(packed_values.shape[-1])
    hierarchy = packed_values.new_zeros(
        (
            int(batch["batch_size"]),
            int(batch["max_segments"]),
            int(batch["max_paths"]),
            int(batch["max_tokens"]),
            hidden,
        )
    )
    for packed_index in range(expected[0]):
        batch_index = int(batch["packed_path_to_sample"][packed_index])
        path_slot = int(batch["packed_path_slot"][packed_index])
        valid = ~batch["packed_padding_mask"][packed_index]
        segment_indices = batch["packed_segment_indices"][packed_index, valid]
        token_indices = batch["packed_token_indices"][packed_index, valid]
        hierarchy[
            batch_index,
            segment_indices,
            path_slot,
            token_indices,
        ] = packed_values[packed_index, valid]
    return hierarchy


def summarize_real_multipath_eligibility(
    samples: Sequence[PreparedSample],
    audit: MultisegmentEligibilityAudit,
) -> dict[str, Any]:
    """Return evidence without turning an untriggered condition into failure."""

    path_counts = [len(sample.paths) for sample in samples]
    all_paths = [path for sample in samples for path in sample.paths]
    segment_counts = [len(path.segments) for path in all_paths]
    path_lengths = [path.source_path_length for path in all_paths]
    sample_has_multisegment = [
        any(len(path.segments) > 1 for path in sample.paths) for sample in samples
    ]
    return {
        "dataset_id": audit.dataset_id,
        "sample_count": len(samples),
        "semantic_path_count": len(all_paths),
        "segment_count": sum(segment_counts),
        "path_length_min": min(path_lengths, default=0),
        "path_length_max": max(path_lengths, default=0),
        "path_length_mean": (
            float(sum(path_lengths)) / float(len(path_lengths)) if path_lengths else 0.0
        ),
        "path_count_gt50": sum(value > TASK34_MULTISEGMENT_THRESHOLD for value in path_lengths),
        "sample_count_with_p_gt1": sum(value > 1 for value in path_counts),
        "sample_count_with_s_gt1": sum(sample_has_multisegment),
        "max_p": max(path_counts, default=0),
        "max_s": max(segment_counts, default=0),
        "multipath_real_data": "PASS" if any(value > 1 for value in path_counts) else "FAIL",
        "multisegment_eligibility": audit.to_dict(),
    }
