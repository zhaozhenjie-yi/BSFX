from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from chapter4_repro.embedding.aligned_hpe import H1_REF_IDENTITY_BUCKETS
from chapter4_repro.ref_kind_filters import matches_kind_patterns, normalize_kind_patterns


BASE_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SEG_START]", "[SEG_END]"]
REF_MARKER_TOKEN = "[REF]"
REF_TOKEN_MODES = {
    "none",
    "generic",
    "identifier",
    "hash",
    "boundary_generic",
    "boundary_identifier",
    "boundary_hash",
}
REF_CUE_MODES = {"none", "binary", "role"}
REF_CUE_NONE = 0
REF_CUE_DEF = 1
REF_CUE_USE = 2
REF_CUE_BOTH = 3
PATH_KIND_TO_ID = {
    "unknown": 0,
    "ast_main": 1,
    "key_node_path": 2,
    "branch_path": 3,
    "call_path": 4,
    "loop_path": 5,
    "semantic_call_path": 6,
    "semantic_cfg_path": 7,
    "semantic_cfg_loop_path": 8,
    "semantic_data_flow_path": 9,
}
SHELL_CUE_TOKENS = {
    "ID:NULL",
    "ID:free",
    "FuncCall:free",
    "FuncCall:kfree",
    "ID:err",
    "ID:ENOMEM",
    "ID:EINVAL",
    "ID:FALSE",
    "FuncCall:PTR_ERR",
}


def _delta_seg_value(segment_distance: int, mode: str) -> float:
    if mode == "distance":
        return float(segment_distance)
    if mode == "inverse_distance":
        return 1.0 / float(segment_distance)
    if mode == "binary":
        return 1.0
    raise ValueError(f"Unsupported delta_seg mode: {mode}")


def _merge_delta_seg(current: float, candidate: float, mode: str) -> float:
    if current <= 0:
        return candidate
    if mode == "distance":
        return min(current, candidate)
    return max(current, candidate)


def _normalize_ref_token_mode(ref_token_mode: str) -> str:
    normalized = ref_token_mode.strip().lower().replace("-", "_")
    if normalized not in REF_TOKEN_MODES:
        raise ValueError(f"Unsupported ref_token_mode: {ref_token_mode}")
    return normalized


def _normalize_ref_cue_mode(ref_cue_mode: str) -> str:
    normalized = ref_cue_mode.strip().lower().replace("-", "_")
    if normalized not in REF_CUE_MODES:
        raise ValueError(f"Unsupported ref_cue_mode: {ref_cue_mode}")
    return normalized


def _is_boundary_ref_token_mode(ref_token_mode: str) -> bool:
    return ref_token_mode.startswith("boundary_")


def _ref_token_payload_mode(ref_token_mode: str) -> str:
    if _is_boundary_ref_token_mode(ref_token_mode):
        return ref_token_mode[len("boundary_") :]
    return ref_token_mode


def _make_ref_hash_token(record: dict, link: dict) -> str:
    hash_source = "|".join(
        [
            str(record.get("sample_id") or ""),
            str(record.get("path_id") or ""),
            str(link.get("identifier") or ""),
            str(int(link.get("def_global_pos", -1))),
            str(int(link.get("def_segment", -1))),
        ]
    )
    digest = hashlib.sha1(hash_source.encode("utf-8")).hexdigest()[:10]
    return f"[REFID:{digest}]"


def _make_ref_token(record: dict, link: dict, ref_token_mode: str) -> str:
    payload_mode = _ref_token_payload_mode(ref_token_mode)
    if payload_mode == "generic":
        return REF_MARKER_TOKEN
    if payload_mode == "hash":
        return _make_ref_hash_token(record, link)
    identifier = link.get("identifier")
    identifier_text = str(identifier or "").strip()
    if not identifier_text:
        return REF_MARKER_TOKEN
    return f"[REF:{identifier_text}]"


def _merge_ref_cue_value(current: int, *, as_def: bool, as_use: bool, ref_cue_mode: str) -> int:
    if ref_cue_mode == "binary":
        return 1 if (current > 0 or as_def or as_use) else 0

    current_def = current in {REF_CUE_DEF, REF_CUE_BOTH}
    current_use = current in {REF_CUE_USE, REF_CUE_BOTH}
    next_def = current_def or as_def
    next_use = current_use or as_use
    if next_def and next_use:
        return REF_CUE_BOTH
    if next_def:
        return REF_CUE_DEF
    if next_use:
        return REF_CUE_USE
    return REF_CUE_NONE


def path_kind_to_id(path_kind: object) -> int:
    normalized = str(path_kind or "unknown").strip() or "unknown"
    return int(PATH_KIND_TO_ID.get(normalized, PATH_KIND_TO_ID["unknown"]))


def segment_has_shell_cue(tokens: Sequence[str]) -> bool:
    return any(str(token) in SHELL_CUE_TOKENS for token in tokens)


def special_tokens_for_vocab(
    ref_token_mode: str = "none",
    *,
    reserve_ref_token: bool | None = None,
) -> list[str]:
    normalized_mode = _normalize_ref_token_mode(ref_token_mode)
    if reserve_ref_token is None:
        reserve_ref_token = normalized_mode != "none"

    tokens = list(BASE_SPECIAL_TOKENS)
    if reserve_ref_token:
        tokens.append(REF_MARKER_TOKEN)
    return tokens


def normalize_ref_link_kind_filters(
    *,
    keep_ref_def_kinds: Sequence[str] | None = None,
    keep_ref_use_kinds: Sequence[str] | None = None,
    keep_ref_pair_kinds: Sequence[str] | None = None,
    drop_ref_def_kinds: Sequence[str] | None = None,
    drop_ref_use_kinds: Sequence[str] | None = None,
    drop_ref_pair_kinds: Sequence[str] | None = None,
) -> dict[str, frozenset[str]]:
    return {
        "keep_ref_def_kinds": normalize_kind_patterns(keep_ref_def_kinds),
        "keep_ref_use_kinds": normalize_kind_patterns(keep_ref_use_kinds),
        "keep_ref_pair_kinds": normalize_kind_patterns(keep_ref_pair_kinds),
        "drop_ref_def_kinds": normalize_kind_patterns(drop_ref_def_kinds),
        "drop_ref_use_kinds": normalize_kind_patterns(drop_ref_use_kinds),
        "drop_ref_pair_kinds": normalize_kind_patterns(drop_ref_pair_kinds),
    }


def _ref_link_kind_filters_active(kind_filters: dict[str, frozenset[str]]) -> bool:
    return any(kind_filters.values())


def _matches_ref_link_kind_filters(
    link: dict,
    *,
    keep_ref_def_kinds: frozenset[str],
    keep_ref_use_kinds: frozenset[str],
    keep_ref_pair_kinds: frozenset[str],
    drop_ref_def_kinds: frozenset[str],
    drop_ref_use_kinds: frozenset[str],
    drop_ref_pair_kinds: frozenset[str],
) -> bool:
    if not any(
        (
            keep_ref_def_kinds,
            keep_ref_use_kinds,
            keep_ref_pair_kinds,
            drop_ref_def_kinds,
            drop_ref_use_kinds,
            drop_ref_pair_kinds,
        )
    ):
        return True

    def_kind = link.get("def_kind")
    use_kind = link.get("use_kind")
    pair_kind = link.get("pair_kind")
    if def_kind is None or use_kind is None or pair_kind is None:
        raise ValueError(
            "ref link kind filters require annotated ref_links with def_kind/use_kind/pair_kind"
        )

    if keep_ref_def_kinds and not matches_kind_patterns(def_kind, keep_ref_def_kinds):
        return False
    if keep_ref_use_kinds and not matches_kind_patterns(use_kind, keep_ref_use_kinds):
        return False
    if keep_ref_pair_kinds and not matches_kind_patterns(pair_kind, keep_ref_pair_kinds):
        return False
    if drop_ref_def_kinds and matches_kind_patterns(def_kind, drop_ref_def_kinds):
        return False
    if drop_ref_use_kinds and matches_kind_patterns(use_kind, drop_ref_use_kinds):
        return False
    if drop_ref_pair_kinds and matches_kind_patterns(pair_kind, drop_ref_pair_kinds):
        return False
    return True


def filter_ref_links(
    record: dict,
    *,
    keep_ref_def_kinds: frozenset[str],
    keep_ref_use_kinds: frozenset[str],
    keep_ref_pair_kinds: frozenset[str],
    drop_ref_def_kinds: frozenset[str],
    drop_ref_use_kinds: frozenset[str],
    drop_ref_pair_kinds: frozenset[str],
) -> list[dict]:
    ref_links = record.get("ref_links") or []
    if not ref_links:
        return []
    return [
        link
        for link in ref_links
        if _matches_ref_link_kind_filters(
            link,
            keep_ref_def_kinds=keep_ref_def_kinds,
            keep_ref_use_kinds=keep_ref_use_kinds,
            keep_ref_pair_kinds=keep_ref_pair_kinds,
            drop_ref_def_kinds=drop_ref_def_kinds,
            drop_ref_use_kinds=drop_ref_use_kinds,
            drop_ref_pair_kinds=drop_ref_pair_kinds,
        )
    ]


def prepare_record_sequence(
    record: dict,
    *,
    add_boundaries: bool = True,
    ref_token_mode: str = "none",
    keep_ref_def_kinds: Sequence[str] | None = None,
    keep_ref_use_kinds: Sequence[str] | None = None,
    keep_ref_pair_kinds: Sequence[str] | None = None,
    drop_ref_def_kinds: Sequence[str] | None = None,
    drop_ref_use_kinds: Sequence[str] | None = None,
    drop_ref_pair_kinds: Sequence[str] | None = None,
) -> dict[str, list[int] | list[str]]:
    ref_token_mode = _normalize_ref_token_mode(ref_token_mode)
    kind_filters = normalize_ref_link_kind_filters(
        keep_ref_def_kinds=keep_ref_def_kinds,
        keep_ref_use_kinds=keep_ref_use_kinds,
        keep_ref_pair_kinds=keep_ref_pair_kinds,
        drop_ref_def_kinds=drop_ref_def_kinds,
        drop_ref_use_kinds=drop_ref_use_kinds,
        drop_ref_pair_kinds=drop_ref_pair_kinds,
    )

    tokens = list(record["nodes"])
    depths = list(record["depths"])
    siblings = list(record["siblings"])
    global_positions = list(record["global_positions"])
    local_positions = list(record["local_positions"])
    segment_ids = [int(record["segment_index"])] * len(tokens)
    source_global_positions = list(global_positions)

    if ref_token_mode != "none":
        (
            tokens,
            depths,
            siblings,
            global_positions,
            local_positions,
            segment_ids,
            source_global_positions,
        ) = _insert_ref_tokens(
            record,
            tokens=tokens,
            depths=depths,
            siblings=siblings,
            global_positions=global_positions,
            local_positions=local_positions,
            segment_ids=segment_ids,
            source_global_positions=source_global_positions,
            ref_token_mode=ref_token_mode,
            ref_links=filter_ref_links(record, **kind_filters),
        )

    if add_boundaries:
        if not global_positions or not local_positions:
            raise ValueError("segment record must contain at least one node position")
        tokens = ["[SEG_START]"] + tokens + ["[SEG_END]"]
        depths = [0] + depths + [0]
        siblings = [0] + siblings + [0]
        global_positions = [0] + [pos + 1 for pos in global_positions] + [global_positions[-1] + 2]
        local_positions = [0] + [pos + 1 for pos in local_positions] + [local_positions[-1] + 2]
        segment_ids = [segment_ids[0]] + segment_ids + [segment_ids[-1]]
        source_global_positions = [-1] + source_global_positions + [-1]

    return {
        "tokens": tokens,
        "depths": depths,
        "siblings": siblings,
        "global_positions": global_positions,
        "local_positions": local_positions,
        "segment_ids": segment_ids,
        "source_global_positions": source_global_positions,
    }


def _insert_ref_tokens(
    record: dict,
    *,
    tokens: list[str],
    depths: list[int],
    siblings: list[int],
    global_positions: list[int],
    local_positions: list[int],
    segment_ids: list[int],
    source_global_positions: list[int],
    ref_token_mode: str,
    ref_links: Sequence[dict],
) -> tuple[list[str], list[int], list[int], list[int], list[int], list[int], list[int]]:
    if not ref_links:
        return (
            tokens,
            depths,
            siblings,
            global_positions,
            local_positions,
            segment_ids,
            source_global_positions,
        )

    target_segment_index = int(record.get("segment_index", 0))
    position_to_index = {int(position): index for index, position in enumerate(source_global_positions)}
    if _is_boundary_ref_token_mode(ref_token_mode):
        segment_markers = sorted(
            {
                _make_ref_token(record, link, ref_token_mode)
                for link in ref_links
                if int(link.get("use_segment", -1)) == target_segment_index
            }
        )
        if not segment_markers:
            return (
                tokens,
                depths,
                siblings,
                global_positions,
                local_positions,
                segment_ids,
                source_global_positions,
            )

        anchor_index = 0
        new_tokens = list(segment_markers) + list(tokens)
        new_depths = [depths[anchor_index]] * len(segment_markers) + list(depths)
        new_siblings = [siblings[anchor_index]] * len(segment_markers) + list(siblings)
        new_global_positions = [global_positions[anchor_index]] * len(segment_markers) + list(global_positions)
        new_local_positions = [local_positions[anchor_index]] * len(segment_markers) + list(local_positions)
        new_segment_ids = [segment_ids[anchor_index]] * len(segment_markers) + list(segment_ids)
        new_source_global_positions = [-1] * len(segment_markers) + list(source_global_positions)
        return (
            new_tokens,
            new_depths,
            new_siblings,
            new_global_positions,
            new_local_positions,
            new_segment_ids,
            new_source_global_positions,
        )

    marker_buckets: dict[int, set[str]] = defaultdict(set)

    for link in ref_links:
        if int(link.get("use_segment", -1)) != target_segment_index:
            continue
        use_position = int(link.get("use_global_pos", -1))
        anchor_index = position_to_index.get(use_position)
        if anchor_index is None:
            continue
        marker_buckets[anchor_index].add(_make_ref_token(record, link, ref_token_mode))

    if not marker_buckets:
        return (
            tokens,
            depths,
            siblings,
            global_positions,
            local_positions,
            segment_ids,
            source_global_positions,
        )

    new_tokens: list[str] = []
    new_depths: list[int] = []
    new_siblings: list[int] = []
    new_global_positions: list[int] = []
    new_local_positions: list[int] = []
    new_segment_ids: list[int] = []
    new_source_global_positions: list[int] = []

    for index, token in enumerate(tokens):
        for marker in sorted(marker_buckets.get(index, ())):
            new_tokens.append(marker)
            new_depths.append(depths[index])
            new_siblings.append(siblings[index])
            new_global_positions.append(global_positions[index])
            new_local_positions.append(local_positions[index])
            new_segment_ids.append(segment_ids[index])
            new_source_global_positions.append(-1)

        new_tokens.append(token)
        new_depths.append(depths[index])
        new_siblings.append(siblings[index])
        new_global_positions.append(global_positions[index])
        new_local_positions.append(local_positions[index])
        new_segment_ids.append(segment_ids[index])
        new_source_global_positions.append(source_global_positions[index])

    return (
        new_tokens,
        new_depths,
        new_siblings,
        new_global_positions,
        new_local_positions,
        new_segment_ids,
        new_source_global_positions,
    )


def build_ref_features(
    record: dict,
    *,
    sequence_len: int,
    add_boundaries: bool,
    delta_seg_mode: str = "distance",
    delta_seg_targets: str = "both",
    ref_matrix_mode: str = "symmetric",
    source_global_positions: list[int] | None = None,
    keep_ref_def_kinds: Sequence[str] | None = None,
    keep_ref_use_kinds: Sequence[str] | None = None,
    keep_ref_pair_kinds: Sequence[str] | None = None,
    drop_ref_def_kinds: Sequence[str] | None = None,
    drop_ref_use_kinds: Sequence[str] | None = None,
    drop_ref_pair_kinds: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_seg = torch.zeros(sequence_len, dtype=torch.float32)
    ref_matrix = torch.zeros((sequence_len, sequence_len), dtype=torch.float32)
    kind_filters = normalize_ref_link_kind_filters(
        keep_ref_def_kinds=keep_ref_def_kinds,
        keep_ref_use_kinds=keep_ref_use_kinds,
        keep_ref_pair_kinds=keep_ref_pair_kinds,
        drop_ref_def_kinds=drop_ref_def_kinds,
        drop_ref_use_kinds=drop_ref_use_kinds,
        drop_ref_pair_kinds=drop_ref_pair_kinds,
    )
    ref_links = filter_ref_links(record, **kind_filters)
    if not ref_links:
        return delta_seg, ref_matrix

    if source_global_positions is not None:
        position_to_index: dict[int, int] = {}
        for index, position in enumerate(source_global_positions):
            if int(position) < 0:
                continue
            position_to_index.setdefault(int(position), index)
    else:
        global_positions = list(record.get("global_positions") or [])
        offset = 1 if add_boundaries else 0
        position_to_index = {
            int(position): index + offset
            for index, position in enumerate(global_positions)
        }

    for link in ref_links:
        def_segment = int(link.get("def_segment", 0))
        use_segment = int(link.get("use_segment", 0))
        segment_distance = abs(use_segment - def_segment)
        if segment_distance <= 0:
            continue

        def_pos = int(link.get("def_global_pos", -1))
        use_pos = int(link.get("use_global_pos", -1))
        def_index = position_to_index.get(def_pos)
        use_index = position_to_index.get(use_pos)
        delta_value = _delta_seg_value(segment_distance, delta_seg_mode)

        if def_index is not None and delta_seg_targets in {"both", "def"}:
            token_index = def_index
            delta_seg[token_index] = _merge_delta_seg(
                float(delta_seg[token_index].item()),
                delta_value,
                delta_seg_mode,
            )
        if use_index is not None and delta_seg_targets in {"both", "use"}:
            token_index = use_index
            delta_seg[token_index] = _merge_delta_seg(
                float(delta_seg[token_index].item()),
                delta_value,
                delta_seg_mode,
            )

        if def_index is not None and use_index is not None:
            def_token_index = def_index
            use_token_index = use_index
            weight = 1.0 / float(segment_distance)
            if ref_matrix_mode == "symmetric":
                ref_matrix[def_token_index, use_token_index] = max(
                    float(ref_matrix[def_token_index, use_token_index].item()),
                    weight,
                )
                ref_matrix[use_token_index, def_token_index] = max(
                    float(ref_matrix[use_token_index, def_token_index].item()),
                    weight,
                )
            elif ref_matrix_mode == "use_to_def":
                ref_matrix[use_token_index, def_token_index] = max(
                    float(ref_matrix[use_token_index, def_token_index].item()),
                    weight,
                )
            else:
                raise ValueError(f"Unsupported ref_matrix mode: {ref_matrix_mode}")

    return delta_seg, ref_matrix


def build_ref_cue_ids(
    record: dict,
    *,
    sequence_len: int,
    add_boundaries: bool,
    ref_cue_mode: str = "none",
    source_global_positions: list[int] | None = None,
    keep_ref_def_kinds: Sequence[str] | None = None,
    keep_ref_use_kinds: Sequence[str] | None = None,
    keep_ref_pair_kinds: Sequence[str] | None = None,
    drop_ref_def_kinds: Sequence[str] | None = None,
    drop_ref_use_kinds: Sequence[str] | None = None,
    drop_ref_pair_kinds: Sequence[str] | None = None,
) -> torch.Tensor:
    ref_cue_mode = _normalize_ref_cue_mode(ref_cue_mode)
    ref_cue_ids = torch.zeros(sequence_len, dtype=torch.long)
    if ref_cue_mode == "none":
        return ref_cue_ids

    kind_filters = normalize_ref_link_kind_filters(
        keep_ref_def_kinds=keep_ref_def_kinds,
        keep_ref_use_kinds=keep_ref_use_kinds,
        keep_ref_pair_kinds=keep_ref_pair_kinds,
        drop_ref_def_kinds=drop_ref_def_kinds,
        drop_ref_use_kinds=drop_ref_use_kinds,
        drop_ref_pair_kinds=drop_ref_pair_kinds,
    )
    ref_links = filter_ref_links(record, **kind_filters)
    if not ref_links:
        return ref_cue_ids

    if source_global_positions is not None:
        position_to_index: dict[int, int] = {}
        for index, position in enumerate(source_global_positions):
            if int(position) < 0:
                continue
            position_to_index.setdefault(int(position), index)
    else:
        global_positions = list(record.get("global_positions") or [])
        offset = 1 if add_boundaries else 0
        position_to_index = {int(position): index + offset for index, position in enumerate(global_positions)}

    for link in ref_links:
        def_index = position_to_index.get(int(link.get("def_global_pos", -1)))
        use_index = position_to_index.get(int(link.get("use_global_pos", -1)))
        if def_index is not None:
            ref_cue_ids[def_index] = _merge_ref_cue_value(
                int(ref_cue_ids[def_index].item()),
                as_def=True,
                as_use=False,
                ref_cue_mode=ref_cue_mode,
            )
        if use_index is not None:
            ref_cue_ids[use_index] = _merge_ref_cue_value(
                int(ref_cue_ids[use_index].item()),
                as_def=False,
                as_use=True,
                ref_cue_mode=ref_cue_mode,
            )
    return ref_cue_ids


def h1_ref_identity_key(record: dict, link: dict) -> str:
    """Return the frozen sample/path-local identity behind formula (4-9)."""

    identifier = str(link.get("identifier") or "").strip()
    if not identifier:
        raise ValueError("H1 REF requires a non-empty ref-link identifier")
    return "|".join(
        (
            str(record.get("sample_id") or ""),
            str(record.get("path_id") or ""),
            identifier,
        )
    )


def h1_ref_identity_id(identity_key: str, *, buckets: int = H1_REF_IDENTITY_BUCKETS) -> int:
    """Map an identity deterministically to a non-padding embedding index."""

    if buckets < 2:
        raise ValueError("H1 REF identity table must contain at least two entries")
    digest = hashlib.sha256(identity_key.encode("utf-8")).digest()
    return 1 + (int.from_bytes(digest[:8], byteorder="big", signed=False) % (buckets - 1))


def build_h1_definition_node_index(records: Sequence[dict]) -> dict[tuple[str, str, int], str]:
    """Index the real definition-end AST node across all segments of each path."""

    result: dict[tuple[str, str, int], str] = {}
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        path_id = str(record.get("path_id") or "")
        nodes = list(record.get("nodes") or [])
        positions = [int(value) for value in (record.get("global_positions") or [])]
        if len(nodes) != len(positions):
            raise ValueError("H1 REF definition index requires aligned nodes/global_positions")
        for position, node in zip(positions, nodes):
            key = (sample_id, path_id, position)
            node_text = str(node)
            existing = result.get(key)
            if existing is not None and existing != node_text:
                raise ValueError(f"conflicting AST node at H1 definition endpoint {key!r}")
            result[key] = node_text
    return result


def build_h1_ref_identity_features(
    record: dict,
    *,
    sequence_len: int,
    source_global_positions: Sequence[int],
    vocab: "Vocabulary",
    definition_node_index: dict[tuple[str, str, int], str],
    ref_identity_buckets: int = H1_REF_IDENTITY_BUCKETS,
    keep_ref_def_kinds: Sequence[str] | None = None,
    keep_ref_use_kinds: Sequence[str] | None = None,
    keep_ref_pair_kinds: Sequence[str] | None = None,
    drop_ref_def_kinds: Sequence[str] | None = None,
    drop_ref_use_kinds: Sequence[str] | None = None,
    drop_ref_pair_kinds: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build token-aligned REF identity and definition-end-node ids for H1.

    The same identity/definition pair is attached to every visible definition
    or use endpoint.  Distinct identities on one physical AST position are a
    contract conflict and are rejected rather than merged silently.
    """

    identity_ids = torch.zeros(sequence_len, dtype=torch.long)
    definition_token_ids = torch.zeros(sequence_len, dtype=torch.long)
    kind_filters = normalize_ref_link_kind_filters(
        keep_ref_def_kinds=keep_ref_def_kinds,
        keep_ref_use_kinds=keep_ref_use_kinds,
        keep_ref_pair_kinds=keep_ref_pair_kinds,
        drop_ref_def_kinds=drop_ref_def_kinds,
        drop_ref_use_kinds=drop_ref_use_kinds,
        drop_ref_pair_kinds=drop_ref_pair_kinds,
    )
    links = filter_ref_links(record, **kind_filters)
    if not links:
        return identity_ids, definition_token_ids

    position_to_index: dict[int, int] = {}
    for index, position in enumerate(source_global_positions):
        if int(position) >= 0:
            position_to_index.setdefault(int(position), index)

    sample_id = str(record.get("sample_id") or "")
    path_id = str(record.get("path_id") or "")
    bucket_keys: dict[int, str] = {}
    position_identity_keys: dict[int, str] = {}
    position_definition_priority: dict[int, tuple[int, int]] = {}
    for link in links:
        identity_key = h1_ref_identity_key(record, link)
        identity_id = h1_ref_identity_id(identity_key, buckets=ref_identity_buckets)
        previous_key = bucket_keys.get(identity_id)
        if previous_key is not None and previous_key != identity_key:
            raise ValueError(
                "sample/path-local H1 REF hash collision; increase ref_identity_buckets"
            )
        bucket_keys[identity_id] = identity_key

        def_position = int(link.get("def_global_pos", -1))
        definition_node = definition_node_index.get((sample_id, path_id, def_position))
        if definition_node is None:
            raise ValueError(
                f"H1 REF definition endpoint is absent from path nodes: "
                f"sample={sample_id!r}, path={path_id!r}, position={def_position}"
            )
        definition_token_id = int(vocab.encode(definition_node))

        for position in (def_position, int(link.get("use_global_pos", -1))):
            token_index = position_to_index.get(position)
            if token_index is None:
                continue
            existing_key = position_identity_keys.get(token_index)
            if existing_key is not None and existing_key != identity_key:
                raise ValueError(
                    f"multiple H1 REF identities target one AST position: {position}"
                )
            position_identity_keys[token_index] = identity_key
            identity_ids[token_index] = identity_id

            # A compound assignment can be both the use of an older value and
            # the definition of a new value.  The local/new definition wins;
            # otherwise the nearest preceding definition is deterministic.
            candidate_priority = (1 if position == def_position else 0, def_position)
            if candidate_priority >= position_definition_priority.get(token_index, (-1, -1)):
                definition_token_ids[token_index] = definition_token_id
                position_definition_priority[token_index] = candidate_priority

    return identity_ids, definition_token_ids


class Vocabulary:
    def __init__(
        self,
        token_to_id: dict[str, int],
        *,
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.special_tokens = list(special_tokens or [])

    @classmethod
    def from_ordered_tokens(
        cls,
        ordered_tokens: Sequence[str],
        *,
        special_tokens: Sequence[str] | None = None,
    ) -> "Vocabulary":
        token_to_id = {token: index for index, token in enumerate(ordered_tokens)}
        if len(token_to_id) != len(ordered_tokens):
            raise ValueError("ordered_tokens must not contain duplicates")
        return cls(token_to_id, special_tokens=special_tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["[PAD]"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["[UNK]"]

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def ordered_tokens(self) -> list[str]:
        return [self.id_to_token[index] for index in range(len(self.id_to_token))]


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_vocab(
    records: Iterable[dict],
    min_freq: int = 1,
    *,
    ref_token_mode: str = "none",
    special_tokens: Sequence[str] | None = None,
    keep_ref_def_kinds: Sequence[str] | None = None,
    keep_ref_use_kinds: Sequence[str] | None = None,
    keep_ref_pair_kinds: Sequence[str] | None = None,
    drop_ref_def_kinds: Sequence[str] | None = None,
    drop_ref_use_kinds: Sequence[str] | None = None,
    drop_ref_pair_kinds: Sequence[str] | None = None,
) -> Vocabulary:
    ref_token_mode = _normalize_ref_token_mode(ref_token_mode)
    counter: Counter[str] = Counter()
    for record in records:
        prepared = prepare_record_sequence(
            record,
            add_boundaries=False,
            ref_token_mode=ref_token_mode,
            keep_ref_def_kinds=keep_ref_def_kinds,
            keep_ref_use_kinds=keep_ref_use_kinds,
            keep_ref_pair_kinds=keep_ref_pair_kinds,
            drop_ref_def_kinds=drop_ref_def_kinds,
            drop_ref_use_kinds=drop_ref_use_kinds,
            drop_ref_pair_kinds=drop_ref_pair_kinds,
        )
        counter.update(prepared["tokens"])

    reserved_tokens = list(special_tokens_for_vocab(ref_token_mode) if special_tokens is None else special_tokens)
    token_to_id = {token: index for index, token in enumerate(reserved_tokens)}
    for token, freq in counter.items():
        if freq < min_freq or token in token_to_id:
            continue
        token_to_id[token] = len(token_to_id)
    return Vocabulary(token_to_id, special_tokens=reserved_tokens)


class PathSegmentDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        vocab: Vocabulary,
        add_boundaries: bool = True,
        delta_seg_mode: str = "distance",
        delta_seg_targets: str = "both",
        ref_matrix_mode: str = "symmetric",
        ref_token_mode: str = "none",
        ref_cue_mode: str = "none",
        keep_ref_def_kinds: Sequence[str] | None = None,
        keep_ref_use_kinds: Sequence[str] | None = None,
        keep_ref_pair_kinds: Sequence[str] | None = None,
        drop_ref_def_kinds: Sequence[str] | None = None,
        drop_ref_use_kinds: Sequence[str] | None = None,
        drop_ref_pair_kinds: Sequence[str] | None = None,
        cue_keep_ref_def_kinds: Sequence[str] | None = None,
        cue_keep_ref_use_kinds: Sequence[str] | None = None,
        cue_keep_ref_pair_kinds: Sequence[str] | None = None,
        cue_drop_ref_def_kinds: Sequence[str] | None = None,
        cue_drop_ref_use_kinds: Sequence[str] | None = None,
        cue_drop_ref_pair_kinds: Sequence[str] | None = None,
        h1_ref_identity_buckets: int = H1_REF_IDENTITY_BUCKETS,
        enable_h1_ref: bool = False,
    ) -> None:
        self.records = records
        self.vocab = vocab
        self.add_boundaries = add_boundaries
        self.delta_seg_mode = delta_seg_mode
        self.delta_seg_targets = delta_seg_targets
        self.ref_matrix_mode = ref_matrix_mode
        self.ref_token_mode = _normalize_ref_token_mode(ref_token_mode)
        self.ref_cue_mode = _normalize_ref_cue_mode(ref_cue_mode)
        self.h1_ref_identity_buckets = int(h1_ref_identity_buckets)
        self.enable_h1_ref = bool(enable_h1_ref)
        self.h1_definition_node_index = (
            build_h1_definition_node_index(records) if self.enable_h1_ref else {}
        )
        self.ref_link_kind_filters = normalize_ref_link_kind_filters(
            keep_ref_def_kinds=keep_ref_def_kinds,
            keep_ref_use_kinds=keep_ref_use_kinds,
            keep_ref_pair_kinds=keep_ref_pair_kinds,
            drop_ref_def_kinds=drop_ref_def_kinds,
            drop_ref_use_kinds=drop_ref_use_kinds,
            drop_ref_pair_kinds=drop_ref_pair_kinds,
        )
        has_cue_filters = any(
            value is not None
            for value in (
                cue_keep_ref_def_kinds,
                cue_keep_ref_use_kinds,
                cue_keep_ref_pair_kinds,
                cue_drop_ref_def_kinds,
                cue_drop_ref_use_kinds,
                cue_drop_ref_pair_kinds,
            )
        )
        self.ref_cue_link_kind_filters = (
            normalize_ref_link_kind_filters(
                keep_ref_def_kinds=cue_keep_ref_def_kinds,
                keep_ref_use_kinds=cue_keep_ref_use_kinds,
                keep_ref_pair_kinds=cue_keep_ref_pair_kinds,
                drop_ref_def_kinds=cue_drop_ref_def_kinds,
                drop_ref_use_kinds=cue_drop_ref_use_kinds,
                drop_ref_pair_kinds=cue_drop_ref_pair_kinds,
            )
            if has_cue_filters
            else dict(self.ref_link_kind_filters)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        prepared = prepare_record_sequence(
            record,
            add_boundaries=self.add_boundaries,
            ref_token_mode=self.ref_token_mode,
            **self.ref_link_kind_filters,
        )
        tokens = list(prepared["tokens"])
        depths = list(prepared["depths"])
        siblings = list(prepared["siblings"])
        global_positions = list(prepared["global_positions"])
        local_positions = list(prepared["local_positions"])
        segment_ids = list(prepared["segment_ids"])
        source_global_positions = list(prepared["source_global_positions"])

        label = record.get("label")
        if label is None:
            label = -1

        delta_seg, ref_matrix = build_ref_features(
            record,
            sequence_len=len(tokens),
            add_boundaries=self.add_boundaries,
            delta_seg_mode=self.delta_seg_mode,
            delta_seg_targets=self.delta_seg_targets,
            ref_matrix_mode=self.ref_matrix_mode,
            source_global_positions=source_global_positions,
            **self.ref_link_kind_filters,
        )
        ref_cue_ids = build_ref_cue_ids(
            record,
            sequence_len=len(tokens),
            add_boundaries=self.add_boundaries,
            ref_cue_mode=self.ref_cue_mode,
            source_global_positions=source_global_positions,
            **self.ref_cue_link_kind_filters,
        )
        if self.enable_h1_ref:
            ref_identity_ids, ref_definition_token_ids = build_h1_ref_identity_features(
                record,
                sequence_len=len(tokens),
                source_global_positions=source_global_positions,
                vocab=self.vocab,
                definition_node_index=self.h1_definition_node_index,
                ref_identity_buckets=self.h1_ref_identity_buckets,
                **self.ref_link_kind_filters,
            )
        else:
            ref_identity_ids = torch.zeros(len(tokens), dtype=torch.long)
            ref_definition_token_ids = torch.zeros(len(tokens), dtype=torch.long)

        return {
            "token_ids": torch.tensor([self.vocab.encode(token) for token in tokens], dtype=torch.long),
            "depth_ids": torch.tensor(depths, dtype=torch.long),
            "sibling_ids": torch.tensor(siblings, dtype=torch.long),
            "global_pos_ids": torch.tensor(global_positions, dtype=torch.long),
            "local_pos_ids": torch.tensor(local_positions, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "ref_cue_ids": ref_cue_ids,
            "ref_identity_ids": ref_identity_ids,
            "ref_definition_token_ids": ref_definition_token_ids,
            "delta_seg": delta_seg,
            "ref_matrix": ref_matrix,
            "path_kind_id": torch.tensor(path_kind_to_id(record.get("path_kind")), dtype=torch.long),
            "segment_shell_flag": torch.tensor(segment_has_shell_cue(tokens), dtype=torch.bool),
            "segment_union_target_flag": torch.tensor(
                str(record.get("path_kind", "unknown"))
                in {"branch_path", "call_path", "semantic_call_path", "semantic_cfg_path", "semantic_cfg_loop_path"}
                or segment_has_shell_cue(tokens),
                dtype=torch.bool,
            ),
            "path_group_id": str(record.get("path_group_id") or record["path_id"]),
            "path_duplicate_weight": torch.tensor(
                float(record.get("duplicate_weight", 1.0)),
                dtype=torch.float32,
            ),
            "semantic_profile": record.get("semantic_profile"),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": record.get("sample_id"),
            "path_id": record["path_id"],
            "segment_id": record["segment_id"],
            "split": record.get("split"),
        }


def collate_segments(batch: list[dict], pad_id: int) -> dict:
    max_len = max(item["token_ids"].shape[0] for item in batch)
    token_ids = []
    depth_ids = []
    sibling_ids = []
    global_pos_ids = []
    local_pos_ids = []
    segment_ids = []
    ref_cue_ids = []
    ref_identity_ids = []
    ref_definition_token_ids = []
    delta_seg = []
    ref_matrix = []
    padding_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - item["token_ids"].shape[0]
        token_ids.append(torch.nn.functional.pad(item["token_ids"], (0, pad_len), value=pad_id))
        depth_ids.append(torch.nn.functional.pad(item["depth_ids"], (0, pad_len), value=0))
        sibling_ids.append(torch.nn.functional.pad(item["sibling_ids"], (0, pad_len), value=0))
        global_pos_ids.append(torch.nn.functional.pad(item["global_pos_ids"], (0, pad_len), value=0))
        local_pos_ids.append(torch.nn.functional.pad(item["local_pos_ids"], (0, pad_len), value=0))
        segment_ids.append(torch.nn.functional.pad(item["segment_ids"], (0, pad_len), value=0))
        ref_cue_ids.append(torch.nn.functional.pad(item["ref_cue_ids"], (0, pad_len), value=REF_CUE_NONE))
        ref_identity_ids.append(
            torch.nn.functional.pad(item["ref_identity_ids"], (0, pad_len), value=0)
        )
        ref_definition_token_ids.append(
            torch.nn.functional.pad(item["ref_definition_token_ids"], (0, pad_len), value=0)
        )
        delta_seg.append(torch.nn.functional.pad(item["delta_seg"], (0, pad_len), value=0))
        ref_matrix.append(torch.nn.functional.pad(item["ref_matrix"], (0, pad_len, 0, pad_len), value=0.0))
        padding_mask.append(
            torch.tensor([False] * item["token_ids"].shape[0] + [True] * pad_len, dtype=torch.bool)
        )
        labels.append(item["label"])

    return {
        "token_ids": torch.stack(token_ids),
        "depth_ids": torch.stack(depth_ids),
        "sibling_ids": torch.stack(sibling_ids),
        "global_pos_ids": torch.stack(global_pos_ids),
        "local_pos_ids": torch.stack(local_pos_ids),
        "segment_ids": torch.stack(segment_ids),
        "ref_cue_ids": torch.stack(ref_cue_ids),
        "ref_identity_ids": torch.stack(ref_identity_ids),
        "ref_definition_token_ids": torch.stack(ref_definition_token_ids),
        "delta_seg": torch.stack(delta_seg),
        "ref_matrix": torch.stack(ref_matrix),
        "padding_mask": torch.stack(padding_mask),
        "labels": torch.stack(labels),
        "sample_ids": [item["sample_id"] for item in batch],
        "path_ids": [item["path_id"] for item in batch],
        "segment_ids_text": [item["segment_id"] for item in batch],
    }


class SampleGroupedDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        vocab: Vocabulary,
        add_boundaries: bool = True,
        delta_seg_mode: str = "distance",
        delta_seg_targets: str = "both",
        ref_matrix_mode: str = "symmetric",
        ref_token_mode: str = "none",
        ref_cue_mode: str = "none",
        keep_ref_def_kinds: Sequence[str] | None = None,
        keep_ref_use_kinds: Sequence[str] | None = None,
        keep_ref_pair_kinds: Sequence[str] | None = None,
        drop_ref_def_kinds: Sequence[str] | None = None,
        drop_ref_use_kinds: Sequence[str] | None = None,
        drop_ref_pair_kinds: Sequence[str] | None = None,
        cue_keep_ref_def_kinds: Sequence[str] | None = None,
        cue_keep_ref_use_kinds: Sequence[str] | None = None,
        cue_keep_ref_pair_kinds: Sequence[str] | None = None,
        cue_drop_ref_def_kinds: Sequence[str] | None = None,
        cue_drop_ref_use_kinds: Sequence[str] | None = None,
        cue_drop_ref_pair_kinds: Sequence[str] | None = None,
        h1_ref_identity_buckets: int = H1_REF_IDENTITY_BUCKETS,
        enable_h1_ref: bool = False,
    ) -> None:
        self.segment_dataset = PathSegmentDataset(
            records,
            vocab=vocab,
            add_boundaries=add_boundaries,
            delta_seg_mode=delta_seg_mode,
            delta_seg_targets=delta_seg_targets,
            ref_matrix_mode=ref_matrix_mode,
            ref_token_mode=ref_token_mode,
            ref_cue_mode=ref_cue_mode,
            keep_ref_def_kinds=keep_ref_def_kinds,
            keep_ref_use_kinds=keep_ref_use_kinds,
            keep_ref_pair_kinds=keep_ref_pair_kinds,
            drop_ref_def_kinds=drop_ref_def_kinds,
            drop_ref_use_kinds=drop_ref_use_kinds,
            drop_ref_pair_kinds=drop_ref_pair_kinds,
            cue_keep_ref_def_kinds=cue_keep_ref_def_kinds,
            cue_keep_ref_use_kinds=cue_keep_ref_use_kinds,
            cue_keep_ref_pair_kinds=cue_keep_ref_pair_kinds,
            cue_drop_ref_def_kinds=cue_drop_ref_def_kinds,
            cue_drop_ref_use_kinds=cue_drop_ref_use_kinds,
            cue_drop_ref_pair_kinds=cue_drop_ref_pair_kinds,
            h1_ref_identity_buckets=h1_ref_identity_buckets,
            enable_h1_ref=enable_h1_ref,
        )
        grouped: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            sample_id = record.get("sample_id")
            if sample_id is None:
                sample_id = f"sample_{index}"
            grouped.setdefault(sample_id, []).append(index)
        self.sample_ids = list(grouped.keys())
        self.grouped_indices = grouped
        self.records = records

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        sample_id = self.sample_ids[index]
        segment_items = [self.segment_dataset[item_index] for item_index in self.grouped_indices[sample_id]]
        label = segment_items[0]["label"]
        if any(int(item["label"].item()) != int(label.item()) for item in segment_items[1:]):
            raise ValueError(f"sample {sample_id!r} contains inconsistent segment labels")
        return {
            "sample_id": sample_id,
            "label": label,
            "segments": segment_items,
        }

    def sample_labels(self) -> list[int]:
        labels: list[int] = []
        for sample_id in self.sample_ids:
            first_record = self.records[self.grouped_indices[sample_id][0]]
            label = first_record.get("label")
            labels.append(-1 if label is None else int(label))
        return labels


def collate_sample_groups(batch: list[dict], pad_id: int) -> dict:
    max_segments = max(len(item["segments"]) for item in batch)
    max_tokens = max(segment["token_ids"].shape[0] for item in batch for segment in item["segments"])

    def pad_tensor(tensor: torch.Tensor, value: int) -> torch.Tensor:
        pad_len = max_tokens - tensor.shape[0]
        return torch.nn.functional.pad(tensor, (0, pad_len), value=value)

    token_ids = []
    depth_ids = []
    sibling_ids = []
    global_pos_ids = []
    local_pos_ids = []
    segment_ids = []
    ref_cue_ids = []
    ref_identity_ids = []
    ref_definition_token_ids = []
    delta_seg = []
    ref_matrix = []
    path_kind_ids = []
    segment_shell_flags = []
    segment_union_target_flags = []
    path_group_ids = []
    path_duplicate_weights = []
    token_padding_mask = []
    segment_padding_mask = []
    labels = []
    sample_ids = []

    for item in batch:
        sample_ids.append(item["sample_id"])
        labels.append(item["label"])
        sample_token_ids = []
        sample_depth_ids = []
        sample_sibling_ids = []
        sample_global_pos_ids = []
        sample_local_pos_ids = []
        sample_segment_ids = []
        sample_ref_cue_ids = []
        sample_ref_identity_ids = []
        sample_ref_definition_token_ids = []
        sample_delta_seg = []
        sample_ref_matrix = []
        sample_path_kind_ids = []
        sample_segment_shell_flags = []
        sample_segment_union_target_flags = []
        sample_path_group_ids = []
        sample_path_duplicate_weights = []
        sample_token_mask = []

        path_group_map: dict[str, int] = {}

        for segment in item["segments"]:
            sample_token_ids.append(pad_tensor(segment["token_ids"], pad_id))
            sample_depth_ids.append(pad_tensor(segment["depth_ids"], 0))
            sample_sibling_ids.append(pad_tensor(segment["sibling_ids"], 0))
            sample_global_pos_ids.append(pad_tensor(segment["global_pos_ids"], 0))
            sample_local_pos_ids.append(pad_tensor(segment["local_pos_ids"], 0))
            sample_segment_ids.append(pad_tensor(segment["segment_ids"], 0))
            sample_ref_cue_ids.append(pad_tensor(segment["ref_cue_ids"], REF_CUE_NONE))
            sample_ref_identity_ids.append(pad_tensor(segment["ref_identity_ids"], 0))
            sample_ref_definition_token_ids.append(
                pad_tensor(segment["ref_definition_token_ids"], 0)
            )
            sample_delta_seg.append(pad_tensor(segment["delta_seg"], 0))
            sample_ref_matrix.append(
                torch.nn.functional.pad(
                    segment["ref_matrix"],
                    (0, max_tokens - segment["ref_matrix"].shape[1], 0, max_tokens - segment["ref_matrix"].shape[0]),
                    value=0.0,
                )
            )
            sample_path_kind_ids.append(segment["path_kind_id"])
            sample_segment_shell_flags.append(segment["segment_shell_flag"])
            sample_segment_union_target_flags.append(segment["segment_union_target_flag"])
            path_group_text = str(segment["path_group_id"])
            if path_group_text not in path_group_map:
                path_group_map[path_group_text] = len(path_group_map)
            sample_path_group_ids.append(torch.tensor(path_group_map[path_group_text], dtype=torch.long))
            sample_path_duplicate_weights.append(segment["path_duplicate_weight"])
            pad_len = max_tokens - segment["token_ids"].shape[0]
            sample_token_mask.append(
                torch.tensor([False] * segment["token_ids"].shape[0] + [True] * pad_len, dtype=torch.bool)
            )

        while len(sample_token_ids) < max_segments:
            sample_token_ids.append(torch.full((max_tokens,), pad_id, dtype=torch.long))
            sample_depth_ids.append(torch.zeros(max_tokens, dtype=torch.long))
            sample_sibling_ids.append(torch.zeros(max_tokens, dtype=torch.long))
            sample_global_pos_ids.append(torch.zeros(max_tokens, dtype=torch.long))
            sample_local_pos_ids.append(torch.zeros(max_tokens, dtype=torch.long))
            sample_segment_ids.append(torch.zeros(max_tokens, dtype=torch.long))
            sample_ref_cue_ids.append(torch.zeros(max_tokens, dtype=torch.long))
            sample_ref_identity_ids.append(torch.zeros(max_tokens, dtype=torch.long))
            sample_ref_definition_token_ids.append(torch.zeros(max_tokens, dtype=torch.long))
            sample_delta_seg.append(torch.zeros(max_tokens, dtype=torch.long))
            sample_ref_matrix.append(torch.zeros((max_tokens, max_tokens), dtype=torch.float32))
            sample_path_kind_ids.append(torch.tensor(PATH_KIND_TO_ID["unknown"], dtype=torch.long))
            sample_segment_shell_flags.append(torch.tensor(False, dtype=torch.bool))
            sample_segment_union_target_flags.append(torch.tensor(False, dtype=torch.bool))
            sample_path_group_ids.append(torch.tensor(-1, dtype=torch.long))
            sample_path_duplicate_weights.append(torch.tensor(0.0, dtype=torch.float32))
            sample_token_mask.append(torch.ones(max_tokens, dtype=torch.bool))

        valid_segments = len(item["segments"])
        segment_padding = [False] * valid_segments + [True] * (max_segments - valid_segments)

        token_ids.append(torch.stack(sample_token_ids))
        depth_ids.append(torch.stack(sample_depth_ids))
        sibling_ids.append(torch.stack(sample_sibling_ids))
        global_pos_ids.append(torch.stack(sample_global_pos_ids))
        local_pos_ids.append(torch.stack(sample_local_pos_ids))
        segment_ids.append(torch.stack(sample_segment_ids))
        ref_cue_ids.append(torch.stack(sample_ref_cue_ids))
        ref_identity_ids.append(torch.stack(sample_ref_identity_ids))
        ref_definition_token_ids.append(torch.stack(sample_ref_definition_token_ids))
        delta_seg.append(torch.stack(sample_delta_seg))
        ref_matrix.append(torch.stack(sample_ref_matrix))
        path_kind_ids.append(torch.stack(sample_path_kind_ids))
        segment_shell_flags.append(torch.stack(sample_segment_shell_flags))
        segment_union_target_flags.append(torch.stack(sample_segment_union_target_flags))
        path_group_ids.append(torch.stack(sample_path_group_ids))
        path_duplicate_weights.append(torch.stack(sample_path_duplicate_weights))
        token_padding_mask.append(torch.stack(sample_token_mask))
        segment_padding_mask.append(torch.tensor(segment_padding, dtype=torch.bool))

    return {
        "token_ids": torch.stack(token_ids),
        "depth_ids": torch.stack(depth_ids),
        "sibling_ids": torch.stack(sibling_ids),
        "global_pos_ids": torch.stack(global_pos_ids),
        "local_pos_ids": torch.stack(local_pos_ids),
        "segment_ids": torch.stack(segment_ids),
        "ref_cue_ids": torch.stack(ref_cue_ids),
        "ref_identity_ids": torch.stack(ref_identity_ids),
        "ref_definition_token_ids": torch.stack(ref_definition_token_ids),
        "delta_seg": torch.stack(delta_seg),
        "ref_matrix": torch.stack(ref_matrix),
        "path_kind_ids": torch.stack(path_kind_ids),
        "segment_shell_flags": torch.stack(segment_shell_flags),
        "segment_union_target_flags": torch.stack(segment_union_target_flags),
        "path_group_ids": torch.stack(path_group_ids),
        "path_duplicate_weights": torch.stack(path_duplicate_weights),
        "token_padding_mask": torch.stack(token_padding_mask),
        "segment_padding_mask": torch.stack(segment_padding_mask),
        "labels": torch.stack(labels),
        "sample_ids": sample_ids,
    }
