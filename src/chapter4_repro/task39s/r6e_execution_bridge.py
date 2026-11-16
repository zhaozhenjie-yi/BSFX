"""Task39S.R6E matched full-retrain execution bridge.

This module is intentionally execution glue.  It does not generate or rewrite
representations and it does not alter Task34 learner math.  The public
contracts bind the R6D matched supports and embeddings to the existing Task34
consumer while keeping sampler, embedding, and shared-learner RNG streams
independent.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from chapter4_repro.models.structure_sensitive_transformer import (
    REF_RELATION_BIDIRECTIONAL,
    REF_RELATION_DEF_TO_USE,
    REF_RELATION_NONE,
    REF_RELATION_USE_TO_DEF,
    SPECIAL_TOKEN_NONE,
    SPECIAL_TOKEN_SEG_END,
    SPECIAL_TOKEN_SEG_START,
)
from chapter4_repro.representation_v2.materialization import vocabulary_token
from chapter4_repro.representation_v2.matched_embedding import (
    INITIALIZATION_SCHEMA as ENGINEERING_INITIALIZATION_SCHEMA,
    build_matched_random_embedding,
)
from chapter4_repro.task34.data import (
    PreparedPath,
    PreparedSample,
    PreparedSegment,
    TokenProvenance,
    build_exact_parent_sidecar,
    collate_task34_multipath,
    prepare_task34_samples,
)
from chapter4_repro.task34.model import (
    ENGINEERING_ALIGNED_V1,
    PAPER_REPRO_V1,
    Task34IntegratedModel,
)
from chapter4_repro.task34.profile import (
    ENGINEERING_PROFILE_ID,
    PAPER_PROFILE_ID,
    resolve_task34_profile,
)
from chapter4_repro.task39s.formal_evidence import atomic_json as streaming_atomic_json
from chapter4_repro.task39s.r6e_transition_contract import (
    TransitionContractFailure,
    verify_preflight_to_formal_transition,
)
from chapter4_repro.training.dataset import Vocabulary


SAMPLER_SCHEMA = "R6E_MATCHED_LARGE_SUPPORT_SAMPLER_V1"
BRIDGE_SCHEMA = "task39s-r6e-matched-full-retrain-execution-bridge-v1"
RUN_NAMESPACE = "chapter4_task39s_r6e_matched_full_retrain_v1"
SHARED_INITIALIZATION_SCHEMA = "task39s-r6e-shared-learner-rng-isolation-v1"
LIFECYCLE_SCHEMA = "task39s-r6e-lifecycle-v1"
LEDGER_SCHEMA = "task39s-r6e-immutable-ledger-v1"
SEQUENCE_GATE_SCHEMA = "task39s-r6e-result-blind-sequence-gate-v1"
SCIENTIFIC_SEED = 123
LOGICAL_BATCH = 32
OPTIMIZER_STEPS = 10_000
PRESENTATIONS = LOGICAL_BATCH * OPTIMIZER_STEPS
R6C_FREEZE_ID = "5de73ed5bf2cac3296b3ad559f623137af42740a381bd0e8f4eb32eea704ae4e"

RUN_ORDER = (
    "T39S-R6E-BP-V1-D00-P1R",
    "T39S-R6E-BP-V2-D00-P1R",
    "T39S-R6E-BE-V1-D00-P1R",
    "T39S-R6E-BE-V2-D00-P1R",
    "T39S-R6E-SP-V1-D00-P1R",
    "T39S-R6E-SP-V2-D00-P1R",
    "T39S-R6E-SE-V1-D00-P1R",
    "T39S-R6E-SE-V2-D00-P1R",
)

LIFECYCLE_EVENTS = (
    "RUN_DISPATCHED",
    "CHILD_SPAWNED",
    "RUNNER_ENTERED",
    "STARTED",
    "TRAINING_STARTED",
    "HEARTBEAT",
    "TRAINING_ENDPOINT_REACHED",
    "RESULT_ASSEMBLY",
    "RESULT_SERIALIZATION",
    "RESULT_JSON_PERSISTED",
    "EVIDENCE_PERSISTENCE",
    "TERMINAL_CANDIDATE_PERSISTED",
    "RUN_PROCESS_EXITED",
    "TERMINAL",
)

_RESULT_FIELDS = frozenset(
    {
        "ba",
        "balanced_accuracy",
        "accuracy",
        "qualification",
        "qualification_status",
        "collapse",
        "collapse_status",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "loss",
    }
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(canonical_bytes(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def domain_seed(scientific_seed: int, *domain: object) -> int:
    payload = "\0".join([str(int(scientific_seed)), *(str(value) for value in domain)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (
        2**63 - 1
    )


class MatchedLargeSupportSampler:
    """Representation-blind deterministic full-support stream."""

    def __init__(
        self,
        *,
        dataset: str,
        ordered_support_ids: Sequence[str],
        scientific_seed: int = SCIENTIFIC_SEED,
        schema: str = SAMPLER_SCHEMA,
    ) -> None:
        self.dataset = str(dataset).lower()
        self.support_ids = tuple(str(value) for value in ordered_support_ids)
        self.scientific_seed = int(scientific_seed)
        self.schema = str(schema)
        if self.dataset not in {"bigvul", "sard"}:
            raise ValueError(f"unsupported R6E dataset: {dataset!r}")
        if not self.support_ids or len(self.support_ids) != len(set(self.support_ids)):
            raise ValueError("matched support must be nonempty and unique")
        if self.schema != SAMPLER_SCHEMA:
            raise ValueError("unknown R6E matched sampler schema")

    def epoch(self, epoch_index: int) -> tuple[str, ...]:
        if int(epoch_index) < 0:
            raise ValueError("epoch_index must be non-negative")
        values = list(self.support_ids)
        seed = domain_seed(
            self.scientific_seed,
            "SAMPLER",
            self.dataset,
            int(epoch_index),
            self.schema,
        )
        random.Random(seed).shuffle(values)
        return tuple(values)

    def stream(self, presentations: int = PRESENTATIONS) -> Iterator[str]:
        remaining = int(presentations)
        if remaining < 0:
            raise ValueError("presentations must be non-negative")
        epoch_index = 0
        while remaining:
            epoch = self.epoch(epoch_index)
            take = min(remaining, len(epoch))
            yield from epoch[:take]
            remaining -= take
            epoch_index += 1

    def schedule_sha256(self, presentations: int = PRESENTATIONS) -> str:
        digest = hashlib.sha256()
        digest.update(
            canonical_bytes(
                {
                    "dataset": self.dataset,
                    "presentations": int(presentations),
                    "sampler_schema": self.schema,
                    "scientific_seed": self.scientific_seed,
                    "support_sha256": canonical_sha256(list(self.support_ids)),
                }
            )
        )
        for sample_id in self.stream(presentations):
            encoded = sample_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def batches(self, presentations: int = PRESENTATIONS) -> Iterator[tuple[str, ...]]:
        batch: list[str] = []
        for sample_id in self.stream(presentations):
            batch.append(sample_id)
            if len(batch) == LOGICAL_BATCH:
                yield tuple(batch)
                batch.clear()
        if batch:
            raise ValueError("presentation count must be divisible by logical batch")


class JsonlOffsetIndex:
    """Read-only JSONL index; records are fetched without rewriting the source."""

    def __init__(self, path: str | Path, *, key: str = "sample_id") -> None:
        self.path = Path(path)
        self.key = str(key)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.offsets: dict[str, list[int]] = defaultdict(list)
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("split")) != "train":
                    raise ValueError(f"non-train record in {self.path}")
                identity = str(row.get(self.key) or "")
                if not identity:
                    raise ValueError(f"missing {self.key} in {self.path}")
                self.offsets[identity].append(offset)

    def get(self, identity: str) -> list[dict[str, Any]]:
        offsets = self.offsets.get(str(identity))
        if not offsets:
            raise KeyError(f"sample not indexed: {identity}")
        rows: list[dict[str, Any]] = []
        with self.path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                rows.append(json.loads(handle.readline()))
        return rows


class R6CShardIndex:
    """Verified direct reader for frozen R6C representation shard rows."""

    def __init__(self, root: str | Path, dataset: str) -> None:
        self.root = Path(root)
        self.dataset = str(dataset)
        manifest_path = self.root / "validation_materialization_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest_sha256 = sha256_file(manifest_path)
        self.locations: dict[str, tuple[Path, int]] = {}
        for entry in manifest.get("shards") or []:
            if str(entry.get("dataset")) != self.dataset:
                continue
            path = repository_root() / str(entry["path"])
            if sha256_file(path) != str(entry["sha256"]):
                raise ValueError(f"R6C shard hash mismatch: {path}")
            count = 0
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("representation_family") != "clang_ast_semantic_paths_v2":
                        raise ValueError("R6C shard representation family drift")
                    if row.get("split") != "val" or row.get("dataset") != self.dataset:
                        raise ValueError("R6C shard dataset/split drift")
                    sample_id = str(row.get("sample_id") or "")
                    if not sample_id or sample_id in self.locations:
                        raise ValueError("R6C shard sample identity is empty or duplicated")
                    self.locations[sample_id] = (path, offset)
                    count += 1
            if count != int(entry["record_count"]):
                raise ValueError(f"R6C shard record count mismatch: {path}")

    def get(self, sample_id: str) -> dict[str, Any]:
        if str(sample_id) not in self.locations:
            raise KeyError(f"R6C sample not indexed: {sample_id}")
        path, offset = self.locations[str(sample_id)]
        with path.open("rb") as handle:
            handle.seek(offset)
            row = json.loads(handle.readline())
        if str(row["sample_id"]) != str(sample_id):
            raise ValueError("R6C shard offset identity mismatch")
        return row


def _read_train_labels(dataset: str) -> dict[str, int]:
    relative = {
        "bigvul": "BSFX/chapter4_repro/data/processed/bigvul_final_v1/train_manifest_v1.jsonl",
        "sard": "BSFX/chapter4_repro/data/processed/sard_c_only_final_v1/train_manifest_v1.jsonl",
    }[dataset]
    path = repository_root() / relative
    if path.name != "train_manifest_v1.jsonl":
        raise ValueError("R6E label authority must be the train manifest")
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("R6E label authority contains non-train row")
            sample_id = str(row["sample_id"])
            label = int(row["label"])
            if label not in {0, 1} or sample_id in labels:
                raise ValueError("R6E label authority is non-binary or duplicated")
            labels[sample_id] = label
    return labels


def _stable_positive_identity(value: str) -> int:
    result = int.from_bytes(
        hashlib.sha256(("R6E_V2_AST_ID\0" + str(value)).encode("utf-8")).digest()[:8],
        "big",
    ) % (2**63 - 1)
    return result or 1


def _merge_ref(existing: int, candidate: int) -> int:
    if existing in {REF_RELATION_NONE, candidate}:
        return candidate
    return REF_RELATION_BIDIRECTIONAL


def _v2_segment_canonical_ids(record: Mapping[str, Any]) -> list[str]:
    """Project frozen parent-path metadata onto the segment's global window."""
    full_ids = (record.get("metadata") or {}).get("canonical_node_ids")
    positions = record.get("global_positions")
    window = record.get("window_span")
    if not isinstance(full_ids, list) or not full_ids or any(
        not isinstance(value, str) or not value for value in full_ids
    ):
        raise ValueError("invalid V2 full-path canonical identities")
    if not isinstance(positions, list) or not positions or any(
        type(value) is not int or not 0 <= value < len(full_ids) for value in positions
    ):
        raise ValueError("V2 global position outside full-path identity range")
    if (
        not isinstance(window, list) or len(window) != 2
        or any(type(value) is not int for value in window)
        or not 0 <= window[0] < window[1] <= len(full_ids)
        or positions != list(range(window[0], window[1]))
    ):
        raise ValueError("V2 global positions do not match the ordered frozen window")
    columns = [record.get(name) or [] for name in (
        "nodes", "depths", "siblings", "local_positions"
    )]
    if any(len(values) != len(positions) for values in columns):
        raise ValueError("unaligned V2 segment columns")
    if columns[-1] != list(range(len(positions))):
        raise ValueError("V2 local positions are not contiguous")
    return [full_ids[position] for position in positions]


def _v2_segment(
    record: Mapping[str, Any],
    *,
    vocab: Vocabulary,
    parent_by_identity: Mapping[str, int],
    path_group_id: str,
) -> PreparedSegment:
    if record.get("semantic_profile") != "clang_ast_semantic_v2":
        raise ValueError("V2 adapter requires clang_ast_semantic_v2")
    if record.get("split") != "val":
        raise ValueError("V2 adapter accepts train shards only")
    columns = [
        list(record.get(name) or [])
        for name in ("nodes", "depths", "siblings", "global_positions", "local_positions")
    ]
    canonical_ids = _v2_segment_canonical_ids(record)
    columns.append(canonical_ids)
    if not columns[0] or len({len(value) for value in columns}) != 1:
        raise ValueError("unaligned V2 segment columns")
    length = len(columns[0])
    if list(map(int, columns[4])) != list(range(length)):
        raise ValueError("V2 local positions are not contiguous")
    sample_id = str(record["sample_id"])
    path_id = str(record["path_id"])
    segment_id = str(record["segment_id"])
    segment_index = int(record["segment_index"])
    nodes = [vocabulary_token(str(value)) for value in columns[0]]
    depths = [max(0, int(value)) for value in columns[1]]
    siblings = [max(0, int(value)) for value in columns[2]]
    globals_ = [int(value) for value in columns[3]]
    canonical = [str(value) for value in canonical_ids]
    identities = [_stable_positive_identity(value) for value in canonical]
    parents = [int(parent_by_identity.get(value, -1)) for value in canonical]
    core_span = [int(value) for value in record.get("core_span") or []]
    if len(core_span) != 2:
        raise ValueError("V2 segment lacks frozen core span")
    start_id = int(vocab.encode("[SEG_START]"))
    end_id = int(vocab.encode("[SEG_END]"))
    provenance: list[TokenProvenance] = []
    provenance.append(
        TokenProvenance(
            sample_id, path_group_id, path_id, segment_id, segment_index,
            0, None, None, None, "SEG_START", False,
        )
    )
    for index, (position, identity) in enumerate(zip(globals_, identities), 1):
        is_core = core_span[0] <= position < core_span[1]
        role = "core" if is_core else ("left_context" if position < core_span[0] else "right_context")
        provenance.append(
            TokenProvenance(
                sample_id, path_group_id, path_id, segment_id, segment_index,
                index, index - 1, position, identity, role, is_core,
            )
        )
    provenance.append(
        TokenProvenance(
            sample_id, path_group_id, path_id, segment_id, segment_index,
            length + 1, None, None, None, "SEG_END", False,
        )
    )
    return PreparedSegment(
        segment_id=segment_id,
        segment_index=segment_index,
        token_ids=torch.tensor([start_id, *[vocab.encode(value) for value in nodes], end_id]),
        depth_ids=torch.tensor([0, *depths, 0], dtype=torch.long),
        sibling_order_ids=torch.tensor([0, *siblings, 0], dtype=torch.long),
        sibling_parent_ids=torch.tensor([-1, *parents, -1], dtype=torch.long),
        ast_node_identity_ids=torch.tensor([-1, *identities, -1], dtype=torch.long),
        global_position_ids=torch.tensor([0, *[v + 1 for v in globals_], globals_[-1] + 2]),
        local_position_ids=torch.tensor([0, *[int(v) + 1 for v in columns[4]], length + 1]),
        segment_ids=torch.full((length + 2,), segment_index, dtype=torch.long),
        special_token_ids=torch.tensor(
            [SPECIAL_TOKEN_SEG_START, *([SPECIAL_TOKEN_NONE] * length), SPECIAL_TOKEN_SEG_END],
            dtype=torch.long,
        ),
        provenance=tuple(provenance),
        truncated=False,
    )


def _v2_path(records: Sequence[Mapping[str, Any]], *, vocab: Vocabulary) -> PreparedPath:
    ordered = sorted(records, key=lambda value: int(value["segment_index"]))
    if [int(value["segment_index"]) for value in ordered] != list(range(len(ordered))):
        raise ValueError("V2 segment indices must be contiguous")
    first = ordered[0]
    sample_id = str(first["sample_id"])
    path_id = str(first["path_id"])
    path_group_id = str(first.get("path_group_id") or path_id)
    if any(str(value["sample_id"]) != sample_id for value in ordered):
        raise ValueError("V2 path crosses sample identity")
    path_family = str(first["path_family"])
    canonical_by_segment = [_v2_segment_canonical_ids(row) for row in ordered]
    full_ids = first["metadata"]["canonical_node_ids"]
    if any(row["metadata"]["canonical_node_ids"] != full_ids for row in ordered):
        raise ValueError("conflicting V2 full-path identity metadata")
    if {position for row in ordered for position in row["global_positions"]} != set(range(len(full_ids))):
        raise ValueError("V2 windows do not cover the full-path identity array")
    parent_by_identity: dict[str, int] = {}
    if path_family == "AST_COMPATIBLE_STRUCTURAL":
        merged: dict[int, tuple[str, int]] = {}
        for row, canonical in zip(ordered, canonical_by_segment):
            depths = [int(value) for value in row.get("depths") or []]
            positions = [int(value) for value in row.get("global_positions") or []]
            if len({len(canonical), len(depths), len(positions)}) != 1:
                raise ValueError("unaligned V2 structural identity columns")
            for position, identity, depth in zip(positions, canonical, depths):
                previous = merged.get(position)
                candidate = (identity, depth)
                if previous is not None and previous != candidate:
                    raise ValueError("conflicting V2 structural context copy")
                merged[position] = candidate
        stack: list[int] = []
        positions = sorted(merged)
        for index, position in enumerate(positions):
            identity, depth = merged[position]
            while stack and merged[positions[stack[-1]]][1] >= depth:
                stack.pop()
            parent_by_identity[identity] = (
                -1
                if not stack
                else _stable_positive_identity(merged[positions[stack[-1]]][0])
            )
            stack.append(index)
    segments = tuple(
        _v2_segment(
            value,
            vocab=vocab,
            parent_by_identity=parent_by_identity,
            path_group_id=path_group_id,
        )
        for value in ordered
    )
    offsets: list[tuple[int, int]] = []
    offset = 0
    endpoints: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for segment in segments:
        start = offset
        end = start + int(segment.token_ids.numel())
        offsets.append((start, end))
        for local, provenance in enumerate(segment.provenance):
            if provenance.source_global_position is not None:
                endpoints[provenance.source_global_position].append(
                    (segment.segment_index, start + local)
                )
        offset = end
    flat = lambda name: torch.cat([getattr(value, name) for value in segments])
    total = offset
    delta = torch.zeros(total, dtype=torch.float32)
    ref_identity = torch.zeros(total, dtype=torch.long)
    ref_definition = torch.zeros(total, dtype=torch.long)
    ref_matrix = torch.zeros((total, total), dtype=torch.long)
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in ordered:
        for link in record.get("ref_links") or []:
            ref_id = str(link.get("ref_id") or link.get("identifier") or "")
            use_position = int(link.get("use_position", link.get("use_global_pos", -1)))
            definition = link.get("definition_position", link.get("def_global_pos"))
            def_position = None if definition is None else int(definition)
            use_segment = int(link.get("use_segment", record["segment_index"]))
            def_segment = int(link.get("def_segment", record["segment_index"]))
            key = (ref_id, def_segment, def_position, use_segment, use_position)
            if key in seen:
                continue
            seen.add(key)
            defs = [] if def_position is None else endpoints.get(def_position, [])
            uses = endpoints.get(use_position, [])
            evidence = {
                "ref_id": ref_id,
                "definition_position": def_position,
                "use_position": use_position,
                "identity_scope": link.get("identity_scope"),
                "resolved": bool(link.get("resolved")),
            }
            if not defs or not uses:
                unresolved.append(evidence)
                continue
            bucket = _stable_positive_identity("REF\0" + ref_id) % 65535 + 1
            definition_token_id = int(vocab.encode("[REF:LOCAL_DECL]"))
            for actual_def_segment, def_index in defs:
                for actual_use_segment, use_index in uses:
                    ref_matrix[def_index, use_index] = _merge_ref(
                        int(ref_matrix[def_index, use_index]), REF_RELATION_DEF_TO_USE
                    )
                    ref_matrix[use_index, def_index] = _merge_ref(
                        int(ref_matrix[use_index, def_index]), REF_RELATION_USE_TO_DEF
                    )
                    distance = float(abs(actual_use_segment - actual_def_segment))
                    delta[def_index] = max(float(delta[def_index]), distance)
                    delta[use_index] = max(float(delta[use_index]), distance)
                ref_identity[def_index] = bucket
                ref_definition[def_index] = definition_token_id
            for _, use_index in uses:
                ref_identity[use_index] = bucket
                ref_definition[use_index] = definition_token_id
            resolved.append(evidence)
    weights = {float(value.get("duplicate_weight", 1.0)) for value in ordered}
    if len(weights) != 1:
        raise ValueError("V2 path duplicate weights drift across segments")
    core_spans = [[int(part) for part in value["core_span"]] for value in ordered]
    return PreparedPath(
        sample_id=sample_id,
        path_group_id=path_group_id,
        path_id=path_id,
        path_kind=str(first["path_kind"]),
        path_family=path_family,
        composite_key=(sample_id, path_group_id),
        duplicate_weight=next(iter(weights)),
        source_path_length=max(span[1] for span in core_spans),
        segments=segments,
        flat_token_ids=flat("token_ids"),
        flat_depth_ids=flat("depth_ids"),
        flat_sibling_order_ids=flat("sibling_order_ids"),
        flat_sibling_parent_ids=flat("sibling_parent_ids"),
        flat_ast_node_identity_ids=flat("ast_node_identity_ids"),
        flat_global_position_ids=flat("global_position_ids"),
        flat_local_position_ids=flat("local_position_ids"),
        flat_delta_seg=delta,
        flat_ref_identity_ids=ref_identity,
        flat_ref_definition_token_ids=ref_definition,
        flat_segment_ids=flat("segment_ids"),
        flat_special_token_ids=flat("special_token_ids"),
        ref_link_matrix=ref_matrix,
        segment_offsets=tuple(offsets),
        resolved_ref_links=tuple(resolved),
        unresolved_ref_links=tuple(unresolved),
    )


class R6ERepresentationAdapter:
    """Explicit V1/V2 dispatch into the existing Task34 common contract."""

    def __init__(self, dataset: str, representation: str) -> None:
        self.dataset = str(dataset).lower()
        self.representation = str(representation).lower()
        if self.dataset not in {"bigvul", "sard"} or self.representation not in {"v1", "v2"}:
            raise ValueError("unsupported R6E representation dispatch key")
        root = repository_root()
        vocab_manifest = json.loads(
            (root / f"task39s_r6d_{self.representation}_matched_vocab_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        vocab_entry = vocab_manifest["datasets"][self.dataset]
        vocab_path = root / str(vocab_entry["path"])
        if sha256_file(vocab_path) != str(vocab_entry["sha256"]):
            raise ValueError("matched vocabulary hash mismatch")
        self.vocab_entry = vocab_entry
        self.ordered_vocab_tokens = tuple(json.loads(vocab_path.read_text(encoding="utf-8")))
        self.vocab = Vocabulary.from_ordered_tokens(
            self.ordered_vocab_tokens,
            special_tokens=["[PAD]", "[UNK]", "[SEG_START]", "[SEG_END]"],
        )
        self.labels = _read_train_labels(self.dataset)
        if self.representation == "v1":
            base = root / "artifacts/task39s_r6d_matched_representation_v1/inputs" / self.dataset / "v1"
            self.segment_index = JsonlOffsetIndex(base / "train_segments.jsonl")
            path_rows = []
            with (base / "train_paths.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("split") != "train" or row.get("path_kind") != "ast_main":
                            raise ValueError("V1 parent sidecar source drift")
                        path_rows.append(row)
            self.parent_sidecar = build_exact_parent_sidecar(path_rows)
            self.shard_index = None
        else:
            self.segment_index = None
            self.parent_sidecar = None
            self.shard_index = R6CShardIndex(
                root / "artifacts/task39s_r6c_v2_representation_v1", self.dataset
            )

    @property
    def representation_family(self) -> str:
        return "pycparser_semantic_x1_v1" if self.representation == "v1" else "clang_ast_semantic_paths_v2"

    def sample(self, sample_id: str) -> PreparedSample:
        sample_id = str(sample_id)
        if sample_id not in self.labels:
            raise KeyError(f"sample absent from train label authority: {sample_id}")
        label = self.labels[sample_id]
        if self.representation == "v1":
            assert self.segment_index is not None and self.parent_sidecar is not None
            records = [dict(row) for row in self.segment_index.get(sample_id)]
            for row in records:
                if row.get("label") is not None and int(row["label"]) != label:
                    raise ValueError("V1 segment label conflicts with train label authority")
                row["label"] = label
            prepared, _ = prepare_task34_samples(
                records,
                vocab=self.vocab,
                parent_sidecar=self.parent_sidecar,
                dataset_id=self.dataset,
            )
            if len(prepared) != 1 or prepared[0].sample_id != sample_id:
                raise ValueError("V1 adapter did not produce exactly one requested sample")
            return prepared[0]
        assert self.shard_index is not None
        record = self.shard_index.get(sample_id)
        if record.get("representation_family") != "clang_ast_semantic_paths_v2":
            raise ValueError("V2 representation was relabeled")
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for segment in record.get("segments") or []:
            if segment.get("semantic_profile") != "clang_ast_semantic_v2":
                raise ValueError("V2 segment semantic profile drift")
            grouped[str(segment.get("path_group_id") or segment["path_id"])].append(segment)
        paths = tuple(_v2_path(grouped[key], vocab=self.vocab) for key in sorted(grouped))
        if not paths or len(paths) > 256:
            raise ValueError("V2 adapter path count violates frozen budget")
        return PreparedSample(sample_id=sample_id, label=label, paths=paths)

    def collate(self, sample_ids: Sequence[str]) -> dict[str, Any]:
        batch = collate_task34_multipath(
            [self.sample(value) for value in sample_ids], pad_id=self.vocab.pad_id
        )
        batch["representation_family"] = self.representation_family
        batch["representation_dispatch_schema"] = BRIDGE_SCHEMA
        return batch


@dataclass(frozen=True)
class EmbeddingBinding:
    dataset: str
    profile: str
    representation: str
    kind: str
    authority_sha256: str
    vocabulary_sha256: str
    support_sha256: str
    artifact_dir: str | None
    trainable: bool


def resolve_embedding_binding(dataset: str, profile: str, representation: str) -> EmbeddingBinding:
    root = repository_root()
    dataset = str(dataset).lower()
    profile = str(profile).lower()
    representation = str(representation).lower()
    vocab_manifest = json.loads(
        (root / f"task39s_r6d_{representation}_matched_vocab_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    vocab = vocab_manifest["datasets"][dataset]
    support_sha = str(vocab["support_sample_id_sha256"])
    if profile == "paper":
        manifest = json.loads(
            (root / "task39s_r6d_paper_cbow_artifact_manifest.json").read_text(encoding="utf-8")
        )
        entry = manifest["artifacts"][f"{dataset}_{representation}"]
        if str(entry["vocabulary_sha256"]) != str(vocab["sha256"]):
            raise ValueError("Paper CBOW/vocabulary authority mismatch")
        if str(entry["common_support_sample_id_sha256"]) != support_sha:
            raise ValueError("Paper CBOW/support authority mismatch")
        selected = entry[str(entry["selected_artifact"])]
        manifest_path = root / str(selected["manifest_path"])
        if sha256_file(manifest_path) != str(selected["manifest_sha256"]):
            raise ValueError("Paper CBOW manifest hash mismatch")
        artifact_dir = str(manifest_path.parent)
        authority = str(entry["artifact_sha256"])
        return EmbeddingBinding(
            dataset, profile, representation, "MATCHED_PAPER_CBOW", authority,
            str(vocab["sha256"]), support_sha, artifact_dir, True,
        )
    if profile != "engineering":
        raise ValueError("unknown R6E profile")
    contract_path = root / "task39s_r6d_engineering_embedding_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["initialization_schema"] != ENGINEERING_INITIALIZATION_SCHEMA:
        raise ValueError("Engineering embedding schema drift")
    implementation = root / str(contract["matched_initialization_implementation"])
    if sha256_file(implementation) != str(contract["matched_initialization_implementation_sha256"]):
        raise ValueError("Engineering embedding implementation hash mismatch")
    return EmbeddingBinding(
        dataset, profile, representation, "TOKEN_KEYED_NORMAL_0_1",
        sha256_file(contract_path), str(vocab["sha256"]), support_sha, None, True,
    )


def _construct_existing_model(profile: str, vocab_size: int, *, residual_branch_scale: float = 1.0) -> Task34IntegratedModel:
    profile_id = PAPER_PROFILE_ID if profile == "paper" else ENGINEERING_PROFILE_ID
    resolved = resolve_task34_profile(profile_id, repository_root=repository_root())
    modules = resolved.modules
    return Task34IntegratedModel(
        family_id=PAPER_REPRO_V1 if profile == "paper" else ENGINEERING_ALIGNED_V1,
        vocab_size=int(vocab_size),
        padding_idx=0,
        transformer_layers=int(modules["transformer_layers"]),
        transformer_heads=int(modules["transformer_heads"]),
        d_model=512,
        dim_feedforward=int(modules["transformer_ffn_dim"]),
        dropout=0.1,
        classifier_dropout=0.0,
        residual_branch_scale=residual_branch_scale,
    )


def build_r6e_model(dataset: str, profile: str, representation: str, *, residual_branch_scale: float = 1.0) -> tuple[Task34IntegratedModel, dict[str, Any]]:
    """Construct via the existing initializer, then isolate embedding RNG.

    A representation-independent V1-vocabulary template is constructed with
    the existing Task34 constructor.  Every same-name/same-shape non-word-
    embedding parameter is copied from that template.  This preserves the
    existing initializer algorithm byte-for-byte while preventing a larger or
    smaller target vocabulary from advancing the shared learner RNG stream.
    """

    adapter = R6ERepresentationAdapter(dataset, representation)
    root = repository_root()
    v1_tokens = json.loads(
        (
            root
            / "artifacts/task39s_r6d_matched_representation_v1/corpora"
            / dataset
            / "v1/vocab_tokens.json"
        ).read_text(encoding="utf-8")
    )
    shared_seed = domain_seed(SCIENTIFIC_SEED, "SHARED_LEARNER", dataset, profile, SHARED_INITIALIZATION_SCHEMA)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(shared_seed)
        template = _construct_existing_model(profile, len(v1_tokens), residual_branch_scale=residual_branch_scale)
        torch.manual_seed(domain_seed(SCIENTIFIC_SEED, "TARGET_SHELL", dataset, profile, representation))
        model = _construct_existing_model(profile, len(adapter.ordered_vocab_tokens), residual_branch_scale=residual_branch_scale)
    target_parameters = dict(model.named_parameters())
    copied: list[str] = []
    with torch.no_grad():
        for name, value in template.named_parameters():
            if name == "hpe.word_embedding.weight":
                continue
            target = target_parameters.get(name)
            if target is None or target.shape != value.shape:
                raise ValueError(f"shared learner parameter contract drift: {name}")
            target.copy_(value)
            copied.append(name)
    del template
    binding = resolve_embedding_binding(dataset, profile, representation)
    if profile == "paper":
        assert binding.artifact_dir is not None
        metadata = model.initialize_paper_cbow(
            artifact_dir=binding.artifact_dir,
            ordered_vocab_tokens=adapter.ordered_vocab_tokens,
            special_tokens=["[PAD]", "[UNK]", "[SEG_START]", "[SEG_END]"],
            freeze=False,
        )
    else:
        matrix = build_matched_random_embedding(
            adapter.ordered_vocab_tokens,
            embedding_dim=128,
            scientific_seed=SCIENTIFIC_SEED,
        )
        with torch.no_grad():
            model.hpe.word_embedding.weight.copy_(torch.from_numpy(matrix))
        model.hpe.word_embedding.weight.requires_grad_(True)
        metadata = {
            "type": "token_keyed_normal",
            "initialization_schema": ENGINEERING_INITIALIZATION_SCHEMA,
            "matrix_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
            "scientific_seed": SCIENTIFIC_SEED,
        }
    model.embedding_metadata.update(metadata)
    audit = {
        "binding": binding.__dict__,
        "shared_seed": shared_seed,
        "shared_initialization_schema": SHARED_INITIALIZATION_SCHEMA,
        "existing_initializer_preserved": True,
        "copied_shared_parameter_count": len(copied),
        "embedding_parameter": "hpe.word_embedding.weight",
        "embedding_trainable": bool(model.hpe.word_embedding.weight.requires_grad),
    }
    return model, audit


def parameter_module(name: str) -> str:
    if name == "hpe.word_embedding.weight":
        return "embedding/H1"
    if name.startswith("hpe."):
        return "embedding/H1"
    if name.startswith("gru_branch."):
        return "GRU"
    if name.startswith("cnn_branch."):
        return "CNN"
    if name.startswith("dwaf."):
        return "DWAF"
    if name.startswith(("h1_to_d1.", "d1_to_transformer.", "special_token_embedding.")):
        return "projection"
    if name.startswith("transformer."):
        return "Structural Transformer"
    if name.startswith("classifier."):
        return "pooling/classifier"
    return "other"


def build_full_retrain_optimizer(model: nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    registry = json.loads(
        (repository_root() / "task39s_r6d_r6e_experiment_registry.json").read_text(encoding="utf-8")
    )
    protocol = registry["optimization"]
    expected = {
        "optimizer": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "gradient_clip_global_l2": 1,
        "optimizer_steps": 10000,
        "logical_batch": 32,
        "physical_batch": 2,
        "gradient_accumulation": 16,
        "presentations": 320000,
        "scheduler": "NONE",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R6D optimization protocol drift: {key}")
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    frozen = [(name, value) for name, value in model.named_parameters() if not value.requires_grad]
    optimizer = torch.optim.AdamW(
        [value for _, value in trainable],
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    optimizer_ids = {id(value) for group in optimizer.param_groups for value in group["params"]}
    rows = [
        {
            "parameter_name": name,
            "module": parameter_module(name),
            "numel": int(value.numel()),
            "requires_grad": bool(value.requires_grad),
            "optimizer_group": 0 if id(value) in optimizer_ids else None,
            "included": id(value) in optimizer_ids,
        }
        for name, value in model.named_parameters()
    ]
    audit = {
        "training_scope": "FULL_MATCHED_DOWNSTREAM_RETRAIN",
        "protocol": protocol,
        "parameters": rows,
        "trainable_not_in_optimizer": [name for name, value in trainable if id(value) not in optimizer_ids],
        "frozen_in_optimizer": [name for name, value in frozen if id(value) in optimizer_ids],
        "classifier_only_freeze_present": False,
        "gradient_clip_global_l2": float(protocol["gradient_clip_global_l2"]),
    }
    return optimizer, audit


class ImmutableHashChain:
    """Append-only, fsync'd, hash-chained R6E evidence journal/ledger."""

    def __init__(self, path: str | Path, *, schema: str, attempt_id: str, activation_fingerprint: str) -> None:
        self.path = Path(path)
        self.schema = str(schema)
        self.attempt_id = str(attempt_id)
        self.activation_fingerprint = str(activation_fingerprint)
        if len(self.activation_fingerprint) != 64:
            raise ValueError("activation fingerprint must be SHA-256")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        previous = None
        with self.path.open("r", encoding="utf-8") as handle:
            for sequence, line in enumerate(handle, 1):
                row = json.loads(line)
                received = row.pop("record_hash")
                if row.get("sequence") != sequence or row.get("previous_hash") != previous:
                    raise ValueError("R6E hash-chain ordering failure")
                actual = canonical_sha256(row)
                if received != actual:
                    raise ValueError("R6E hash-chain record failure")
                row["record_hash"] = received
                previous = received
                rows.append(row)
        return rows

    def append(self, record_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        with lock.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            rows = self.records()
            row = {
                "schema_version": self.schema,
                "sequence": len(rows) + 1,
                "attempt_id": self.attempt_id,
                "activation_fingerprint": self.activation_fingerprint,
                "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="microseconds"),
                "previous_hash": rows[-1]["record_hash"] if rows else None,
                "record_type": str(record_type),
                "payload": dict(payload),
            }
            row["record_hash"] = canonical_sha256(row)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.records()
            return row


class R6ELifecycle:
    def __init__(self, chain: ImmutableHashChain) -> None:
        if chain.schema != LIFECYCLE_SCHEMA:
            raise ValueError("R6E lifecycle requires its dedicated schema")
        self.chain = chain

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        if event not in LIFECYCLE_EVENTS:
            raise ValueError(f"unknown R6E lifecycle event: {event}")
        prior = [row["record_type"] for row in self.chain.records()]
        non_heartbeat = [value for value in prior if value != "HEARTBEAT"]
        required = [value for value in LIFECYCLE_EVENTS if value != "HEARTBEAT"]
        expected = required[len(non_heartbeat)] if len(non_heartbeat) < len(required) else None
        if event != "HEARTBEAT" and event != expected:
            raise ValueError(f"R6E lifecycle event out of order: expected={expected}, got={event}")
        if event == "HEARTBEAT" and ("TRAINING_STARTED" not in prior or "TRAINING_ENDPOINT_REACHED" in prior):
            raise ValueError("R6E heartbeat outside training interval")
        return self.chain.append(event, payload)


def result_blind_sequence_decision(
    run_id: str,
    predecessor_closure: Mapping[str, Any] | None,
    *,
    first_formal_transition: Mapping[str, Any] | None = None,
    expected_transition_sha256: str = "",
    expected_formal_executable_freeze_id: str = "",
    expected_formal_executable_manifest_path: str = "",
    expected_formal_executable_manifest_sha256: str = "",
) -> dict[str, Any]:
    if run_id not in RUN_ORDER:
        raise ValueError("unknown R6E run ID")
    if predecessor_closure is not None:
        lowered = {str(key).lower() for key in predecessor_closure}
        forbidden = sorted(lowered & _RESULT_FIELDS)
        if forbidden:
            raise ValueError(f"result-bearing fields are forbidden in sequence gate: {forbidden}")
    index = RUN_ORDER.index(run_id)
    transition_verification: dict[str, Any] | None = None
    if index == 0:
        try:
            transition_verification = verify_preflight_to_formal_transition(
                first_formal_transition,
                expected_transition_sha256=expected_transition_sha256,
                expected_formal_executable_freeze_id=(
                    expected_formal_executable_freeze_id
                ),
                expected_formal_executable_manifest_path=(
                    expected_formal_executable_manifest_path
                ),
                expected_formal_executable_manifest_sha256=(
                    expected_formal_executable_manifest_sha256
                ),
                repository_root=repository_root(),
            )
        except TransitionContractFailure:
            open_ = False
            reason = "CURRENT_PREFLIGHT_TO_FIRST_FORMAL_TRANSITION_INVALID"
        else:
            open_ = True
            reason = "CURRENT_PREFLIGHT_TO_FIRST_FORMAL_TRANSITION_VALID"
    else:
        required = {
            "run_process_exited": True,
            "runtime_complete": True,
            "result_json_persisted": True,
            "artifact_manifest_persisted": True,
            "terminal_candidate_persisted": True,
            "terminal_persisted": True,
        }
        open_ = predecessor_closure is not None and all(
            predecessor_closure.get(key) is value for key, value in required.items()
        )
        reason = "PREDECESSOR_EVIDENCE_CLOSED" if open_ else "PREDECESSOR_EVIDENCE_INCOMPLETE"
    return {
        "schema_version": SEQUENCE_GATE_SCHEMA,
        "run_id": run_id,
        "predecessor": None if index == 0 else RUN_ORDER[index - 1],
        "mode": (
            "FIRST_FORMAL_TRANSITION_MODE"
            if index == 0
            else "SUBSEQUENT_FORMAL_PREDECESSOR_MODE"
        ),
        "open": open_,
        "reason": reason,
        "transition_verification": transition_verification,
        "result_fields_read": [],
    }


def validate_preflight_authority(authority: Mapping[str, Any]) -> None:
    required = {
        "namespace": RUN_NAMESPACE,
        "training_scope": "FULL_MATCHED_DOWNSTREAM_RETRAIN",
        "sampler_schema": SAMPLER_SCHEMA,
        "r6c_representation_freeze_id": R6C_FREEZE_ID,
        "sequence_locked": True,
        "formal_training_authorized": False,
        "authority_stage": "R6E0R2_PREFLIGHT_ONLY",
    }
    for key, value in required.items():
        if authority.get(key) != value:
            raise ValueError(f"fresh authority contract mismatch: {key}")
    if authority.get("run_id") not in RUN_ORDER:
        raise ValueError("fresh authority run is not in the frozen sequence")


def bind_streaming_result(path: str | Path, payload: Mapping[str, Any]) -> None:
    """The sole R6E result writer: existing bounded streaming serializer."""

    streaming_atomic_json(Path(path), payload)


__all__ = [
    "BRIDGE_SCHEMA",
    "ENGINEERING_INITIALIZATION_SCHEMA",
    "ImmutableHashChain",
    "LEDGER_SCHEMA",
    "LIFECYCLE_EVENTS",
    "LIFECYCLE_SCHEMA",
    "LOGICAL_BATCH",
    "MatchedLargeSupportSampler",
    "OPTIMIZER_STEPS",
    "PRESENTATIONS",
    "R6C_FREEZE_ID",
    "R6ERepresentationAdapter",
    "R6ELifecycle",
    "RUN_NAMESPACE",
    "RUN_ORDER",
    "SAMPLER_SCHEMA",
    "SCIENTIFIC_SEED",
    "SEQUENCE_GATE_SCHEMA",
    "SHARED_INITIALIZATION_SCHEMA",
    "bind_streaming_result",
    "build_full_retrain_optimizer",
    "build_r6e_model",
    "canonical_sha256",
    "domain_seed",
    "parameter_module",
    "repository_root",
    "resolve_embedding_binding",
    "result_blind_sequence_decision",
    "sha256_file",
    "tensor_sha256",
    "validate_preflight_authority",
]
