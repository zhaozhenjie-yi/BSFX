from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from chapter4_repro.training.dataset import (
    BASE_SPECIAL_TOKENS,
    build_vocab,
    prepare_record_sequence,
    read_jsonl,
)


CORPUS_FORMAT = "chapter4_structural_cbow_corpus_v1"
ARTIFACT_FORMAT = "chapter4_cbow_embeddings_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_cbow_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for section in ("corpus", "pretraining", "integration"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"CBOW config is missing mapping section: {section}")
    return payload


def _ensure_new_output_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing CBOW output directory: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _validate_train_role(path: Path, expected_basename: str) -> None:
    if path.name != expected_basename:
        raise ValueError(
            f"Train-only guard rejected {path}: expected basename {expected_basename!r}"
        )
    forbidden_parts = {"val", "valid", "validation", "test"}
    normalized_parts = {
        token
        for part in path.parts
        for token in part.lower().replace("-", "_").split("_")
    }
    if forbidden_parts & normalized_parts:
        raise ValueError(f"Train-only guard rejected val/test-like input path: {path}")


def _verify_expected_sha(path: Path, expected_sha256: str | None) -> str:
    actual = sha256_file(path)
    if expected_sha256 and actual != str(expected_sha256):
        raise ValueError(
            f"Input SHA-256 mismatch for {path}: expected={expected_sha256}, actual={actual}"
        )
    return actual


def _parent_and_children(depths: list[int]) -> tuple[list[int | None], dict[int, list[int]]]:
    parents: list[int | None] = []
    children: dict[int, list[int]] = defaultdict(list)
    stack: list[int] = []
    for index, depth in enumerate(depths):
        while stack and depths[stack[-1]] >= depth:
            stack.pop()
        parent = stack[-1] if stack else None
        parents.append(parent)
        if parent is not None:
            children[parent].append(index)
        stack.append(index)
    return parents, children


def _structural_context_indices(
    index: int,
    parents: list[int | None],
    children: dict[int, list[int]],
    max_context: int,
) -> list[int]:
    parent = parents[index]
    candidates: list[int] = []
    if parent is not None:
        candidates.append(parent)
    candidates.extend(children.get(index, ()))
    if parent is not None:
        candidates.extend(child for child in children.get(parent, ()) if child != index)

    deduplicated: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate == index or candidate in seen:
            continue
        seen.add(candidate)
        deduplicated.append(candidate)
        if len(deduplicated) >= max_context:
            break
    return deduplicated


def _counter_summary(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"token": token, "count": int(count)}
        for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def build_structural_cbow_corpus(
    *,
    train_segments_path: str | Path,
    train_paths_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    corpus_config = config["corpus"]
    allowed_split = str(corpus_config.get("allowed_split", "train"))
    if allowed_split != "train":
        raise ValueError("CBOW corpus is hard-locked to allowed_split=train")

    segments_path = Path(train_segments_path).expanduser().resolve()
    paths_path = Path(train_paths_path).expanduser().resolve()
    if bool(corpus_config.get("strict_train_basenames", True)):
        _validate_train_role(segments_path, "train_segments.jsonl")
        _validate_train_role(paths_path, "train_paths.jsonl")
    if not segments_path.is_file() or not paths_path.is_file():
        raise FileNotFoundError("Both train_segments.jsonl and train_paths.jsonl must exist")

    expected_inputs = corpus_config.get("expected_inputs") or {}
    segments_sha = _verify_expected_sha(
        segments_path, expected_inputs.get("train_segments_sha256")
    )
    paths_sha = _verify_expected_sha(paths_path, expected_inputs.get("train_paths_sha256"))

    ref_token_mode = str(corpus_config.get("ref_token_mode", "none"))
    if ref_token_mode != "none":
        raise ValueError("CBOW v1 is locked to ref_token_mode=none")
    path_kind = str(corpus_config.get("path_kind", "ast_main"))
    if path_kind != "ast_main":
        raise ValueError("Structural CBOW v1 requires path_kind=ast_main")
    max_context = int(corpus_config.get("max_context", 5))
    if max_context <= 0:
        raise ValueError("max_context must be positive")
    expected_views = ["parent", "direct_children", "siblings"]
    if list(corpus_config.get("context_views") or expected_views) != expected_views:
        raise ValueError(f"CBOW v1 context_views must be {expected_views}")
    if corpus_config.get("overflow_policy", "deterministic_first_k") != "deterministic_first_k":
        raise ValueError("CBOW v1 requires overflow_policy=deterministic_first_k")
    if corpus_config.get("empty_slot_policy", "masked_no_vocab_token") != "masked_no_vocab_token":
        raise ValueError("CBOW v1 requires empty_slot_policy=masked_no_vocab_token")
    if not bool(corpus_config.get("deduplicate_source_nodes", True)):
        raise ValueError("CBOW v1 requires deduplicate_source_nodes=true")

    segment_records = read_jsonl(segments_path)
    if not segment_records:
        raise ValueError("train_segments input is empty")
    train_sample_ids: set[str] = set()
    for record_index, record in enumerate(segment_records):
        split = str(record.get("split") or "").strip().lower()
        if split != "train":
            raise ValueError(
                f"Leakage guard rejected segment record {record_index}: split={split!r}"
            )
        sample_id = str(record.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"Segment record {record_index} has no sample_id")
        train_sample_ids.add(sample_id)

    vocab = build_vocab(segment_records, ref_token_mode=ref_token_mode)
    vocab_tokens = vocab.ordered_tokens()
    vocab_set = set(vocab_tokens)
    prepared_token_counter: Counter[str] = Counter()
    for record in segment_records:
        prepared = prepare_record_sequence(
            record,
            add_boundaries=True,
            ref_token_mode=ref_token_mode,
        )
        prepared_token_counter.update(str(token) for token in prepared["tokens"])

    examples: list[dict[str, Any]] = []
    corpus_token_counter: Counter[str] = Counter()
    context_length_counter: Counter[int] = Counter()
    path_sample_ids: set[str] = set()
    seen_source_nodes: dict[tuple[str, int], str] = {}
    path_records = 0
    ast_main_records = 0
    skipped_empty_context = 0
    duplicate_source_nodes = 0
    corpus_vocab_oov = 0

    with paths_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            path_records += 1
            if "label" in record:
                raise ValueError(
                    f"Corpus path input must not expose labels (line {line_number})"
                )
            record_split = record.get("split")
            if record_split is not None and str(record_split).strip().lower() != "train":
                raise ValueError(
                    f"Leakage guard rejected path record {line_number}: split={record_split!r}"
                )
            if str(record.get("path_kind")) != path_kind:
                continue
            ast_main_records += 1
            sample_id = str(record.get("sample_id") or "").strip()
            if sample_id not in train_sample_ids:
                raise ValueError(
                    f"Leakage guard rejected path sample {sample_id!r}: absent from train segments"
                )
            path_sample_ids.add(sample_id)

            node_ids = [int(value) for value in record.get("node_ids") or []]
            tokens = [str(value) for value in record.get("nodes") or []]
            depths = [int(value) for value in record.get("depths") or []]
            siblings = [int(value) for value in record.get("siblings") or []]
            lengths = {len(node_ids), len(tokens), len(depths), len(siblings)}
            if len(lengths) != 1 or not tokens:
                raise ValueError(f"Malformed ast_main path at line {line_number}")

            unknown_tokens = [token for token in tokens if token not in vocab_set]
            if unknown_tokens:
                corpus_vocab_oov += len(unknown_tokens)
                raise ValueError(
                    f"Path/model vocab mismatch at line {line_number}: {unknown_tokens[:3]}"
                )

            parents, children = _parent_and_children(depths)
            for index, (node_id, target) in enumerate(zip(node_ids, tokens)):
                source_key = (sample_id, node_id)
                previous = seen_source_nodes.get(source_key)
                if previous is not None:
                    duplicate_source_nodes += 1
                    if previous != target:
                        raise ValueError(
                            f"Conflicting token for source node {source_key}: {previous!r} vs {target!r}"
                        )
                    continue
                seen_source_nodes[source_key] = target

                context_indices = _structural_context_indices(
                    index, parents, children, max_context
                )
                if not context_indices:
                    skipped_empty_context += 1
                    continue
                context_tokens = [tokens[context_index] for context_index in context_indices]
                example = {
                    "sample_id": sample_id,
                    "target_node_id": node_id,
                    "target": target,
                    "context": context_tokens,
                }
                examples.append(example)
                corpus_token_counter[target] += 1
                corpus_token_counter.update(context_tokens)
                context_length_counter[len(context_tokens)] += 1

    if path_sample_ids != train_sample_ids:
        missing = sorted(train_sample_ids - path_sample_ids)
        extra = sorted(path_sample_ids - train_sample_ids)
        raise ValueError(
            "Train split attestation failed: train path and segment sample sets differ; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    minimum_examples = int(corpus_config.get("minimum_examples", 1))
    if len(examples) < minimum_examples:
        raise ValueError(
            f"Corpus too small for v1 stop rule: examples={len(examples)}, minimum={minimum_examples}"
        )

    corpus_special_counts = {
        token: int(corpus_token_counter.get(token, 0)) for token in vocab.special_tokens
    }
    stats: dict[str, Any] = {
        "allowed_split": "train",
        "train_sample_count": len(train_sample_ids),
        "train_segment_record_count": len(segment_records),
        "train_path_record_count": path_records,
        "ast_main_path_count": ast_main_records,
        "source_node_count": len(seen_source_nodes),
        "example_count": len(examples),
        "context_token_count": int(sum(len(item["context"]) for item in examples)),
        "mean_context_length": (
            float(sum(len(item["context"]) for item in examples) / len(examples))
            if examples
            else 0.0
        ),
        "context_length_counts": {
            str(key): int(value) for key, value in sorted(context_length_counter.items())
        },
        "skipped_empty_context_count": skipped_empty_context,
        "duplicate_source_node_count": duplicate_source_nodes,
        "model_prepared_token_count": int(sum(prepared_token_counter.values())),
        "model_prepared_unique_token_count": len(prepared_token_counter),
        "model_vocab_size": len(vocab_tokens),
        "model_vocab_special_tokens": list(vocab.special_tokens),
        "corpus_unique_token_count": len(corpus_token_counter),
        "corpus_vocab_oov_count": corpus_vocab_oov,
        "corpus_special_token_counts": corpus_special_counts,
        "model_vocab_tokens_absent_from_corpus_count": len(vocab_set - set(corpus_token_counter)),
        "top_corpus_tokens": _counter_summary(corpus_token_counter),
        "labels_encoded_in_corpus": False,
    }

    if dry_run:
        return {
            "dry_run": True,
            "input_sha256": {
                "train_segments": segments_sha,
                "train_paths": paths_sha,
            },
            "stats": stats,
        }

    target_dir = Path(output_dir).expanduser().resolve()
    _ensure_new_output_dir(target_dir)
    corpus_path = target_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    vocab_path = target_dir / "vocab_tokens.json"
    sample_ids_path = target_dir / "train_sample_ids.json"
    stats_path = target_dir / "corpus_stats.json"
    report_path = target_dir / "corpus_report.md"
    write_json(vocab_path, vocab_tokens)
    write_json(sample_ids_path, sorted(train_sample_ids))
    write_json(stats_path, stats)

    report_lines = [
        "# CBOW train-only corpus report v1",
        "",
        f"- split: `train`",
        f"- samples: `{stats['train_sample_count']}`",
        f"- structural CBOW examples: `{stats['example_count']}`",
        f"- context tokens: `{stats['context_token_count']}`",
        f"- mean context length: `{stats['mean_context_length']:.4f}`",
        f"- model vocab: `{stats['model_vocab_size']}`",
        f"- corpus unique tokens: `{stats['corpus_unique_token_count']}`",
        f"- corpus OOV against model vocab: `{stats['corpus_vocab_oov_count']}`",
        f"- special token counts in structural corpus: `{stats['corpus_special_token_counts']}`",
        f"- labels encoded: `{stats['labels_encoded_in_corpus']}`",
        "",
        "Context order is parent, direct children, then siblings; duplicate nodes are removed and the first five are retained. Empty slots are masked and do not create a vocab row.",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    output_files = {
        "corpus": {"path": corpus_path.name, "sha256": sha256_file(corpus_path)},
        "vocab_tokens": {"path": vocab_path.name, "sha256": sha256_file(vocab_path)},
        "train_sample_ids": {
            "path": sample_ids_path.name,
            "sha256": sha256_file(sample_ids_path),
        },
        "stats": {"path": stats_path.name, "sha256": sha256_file(stats_path)},
        "report": {"path": report_path.name, "sha256": sha256_file(report_path)},
    }
    manifest = {
        "format": CORPUS_FORMAT,
        "allowed_split": "train",
        "config": {
            "path": str(Path(config_path).expanduser().resolve()) if config_path else None,
            "sha256": sha256_file(config_path) if config_path else None,
        },
        "inputs": {
            "train_segments": {"path": str(segments_path), "sha256": segments_sha},
            "train_paths": {"path": str(paths_path), "sha256": paths_sha},
        },
        "context_spec": {
            "path_kind": path_kind,
            "views": ["parent", "direct_children", "siblings"],
            "max_context": max_context,
            "overflow_policy": "deterministic_first_k",
            "empty_slot_policy": "masked_no_vocab_token",
            "deduplicate_source_nodes": True,
        },
        "token_preparation": {
            "vocab_builder": "chapter4_repro.training.dataset.build_vocab",
            "sequence_preparation": "chapter4_repro.training.dataset.prepare_record_sequence",
            "ref_token_mode": ref_token_mode,
            "special_tokens": list(vocab.special_tokens),
        },
        "stats": stats,
        "outputs": output_files,
    }
    manifest_path = target_dir / "corpus_manifest.json"
    write_json(manifest_path, manifest)
    return {"dry_run": False, "manifest_path": str(manifest_path), "manifest": manifest}


class _CBOWNegativeSampling(nn.Module):
    def __init__(self, vocab_size: int, vector_dim: int, generator: torch.Generator) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, vector_dim, sparse=True)
        self.output_embeddings = nn.Embedding(vocab_size, vector_dim, sparse=True)
        with torch.no_grad():
            bound = 0.5 / float(vector_dim)
            self.input_embeddings.weight.uniform_(-bound, bound, generator=generator)
            self.output_embeddings.weight.zero_()

    def forward(
        self,
        contexts: Tensor,
        context_mask: Tensor,
        targets: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        context_vectors = self.input_embeddings(contexts)
        mask = context_mask.unsqueeze(-1).to(context_vectors.dtype)
        hidden = (context_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        positive_vectors = self.output_embeddings(targets)
        negative_vectors = self.output_embeddings(negatives)
        positive_scores = torch.sum(hidden * positive_vectors, dim=-1)
        negative_scores = torch.einsum("bd,bnd->bn", hidden, negative_vectors)
        return -(
            torch.nn.functional.logsigmoid(positive_scores)
            + torch.nn.functional.logsigmoid(-negative_scores).sum(dim=-1)
        ).mean()


def _sample_negatives(
    probabilities: Tensor,
    targets: Tensor,
    negative_samples: int,
    generator: torch.Generator,
) -> Tensor:
    negatives = torch.multinomial(
        probabilities,
        num_samples=targets.numel() * negative_samples,
        replacement=True,
        generator=generator,
    ).view(targets.numel(), negative_samples)
    collisions = negatives.eq(targets.unsqueeze(1))
    while bool(collisions.any()):
        replacements = torch.multinomial(
            probabilities,
            num_samples=int(collisions.sum().item()),
            replacement=True,
            generator=generator,
        )
        negatives[collisions] = replacements
        collisions = negatives.eq(targets.unsqueeze(1))
    return negatives


def _resolve_manifest_file(directory: Path, entry: dict[str, Any], label: str) -> Path:
    path = (directory / str(entry.get("path"))).resolve()
    if path.parent != directory.resolve():
        raise ValueError(f"Manifest {label} path escapes artifact directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest {label} file is missing: {path}")
    actual = sha256_file(path)
    expected = str(entry.get("sha256") or "")
    if actual != expected:
        raise ValueError(f"Manifest {label} SHA-256 mismatch: expected={expected}, actual={actual}")
    return path


def train_cbow_embeddings(
    *,
    corpus_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    corpus_root = Path(corpus_dir).expanduser().resolve()
    corpus_manifest_path = corpus_root / "corpus_manifest.json"
    if not corpus_manifest_path.is_file():
        raise FileNotFoundError(f"Missing corpus manifest: {corpus_manifest_path}")
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if corpus_manifest.get("format") != CORPUS_FORMAT:
        raise ValueError(f"Unsupported corpus format: {corpus_manifest.get('format')!r}")
    if corpus_manifest.get("allowed_split") != "train":
        raise ValueError("CBOW trainer refuses a corpus not attested as train-only")

    corpus_path = _resolve_manifest_file(
        corpus_root, corpus_manifest["outputs"]["corpus"], "corpus"
    )
    vocab_path = _resolve_manifest_file(
        corpus_root, corpus_manifest["outputs"]["vocab_tokens"], "vocab_tokens"
    )
    vocab_tokens = json.loads(vocab_path.read_text(encoding="utf-8"))
    if len(vocab_tokens) != len(set(vocab_tokens)):
        raise ValueError("Corpus vocab_tokens contains duplicates")
    full_token_to_id = {str(token): index for index, token in enumerate(vocab_tokens)}

    pretrain = config["pretraining"]
    if pretrain.get("algorithm") != "cbow_negative_sampling":
        raise ValueError("CBOW v1 requires algorithm=cbow_negative_sampling")
    if int(pretrain.get("min_count", 1)) != 1:
        raise ValueError("CBOW v1 requires min_count=1 to preserve the frozen model vocab")
    vector_dim = int(pretrain["vector_dim"])
    negative_samples = int(pretrain["negative_samples"])
    epochs = int(pretrain["epochs"])
    batch_size = int(pretrain["batch_size"])
    learning_rate = float(pretrain["learning_rate"])
    seed = int(pretrain["seed"])
    power = float(pretrain.get("negative_sampling_power", 0.75))
    device_name = str(pretrain.get("device", "cpu"))
    if vector_dim <= 0 or negative_samples <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("CBOW dimensions, negative samples, epochs and batch size must be positive")
    if device_name != "cpu":
        raise ValueError("CBOW v1 pretraining is locked to CPU for deterministic replay")

    special_tokens = [str(token) for token in corpus_manifest["token_preparation"]["special_tokens"]]
    if special_tokens != list(BASE_SPECIAL_TOKENS):
        raise ValueError(
            f"CBOW v1 special-token contract changed: expected={BASE_SPECIAL_TOKENS}, actual={special_tokens}"
        )
    special_set = set(special_tokens)

    raw_examples: list[tuple[int, list[int]]] = []
    target_counts: Counter[int] = Counter()
    active_full_ids: set[int] = set()
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "label" in record:
                raise ValueError(f"Label leakage detected in corpus line {line_number}")
            target_token = str(record["target"])
            context_tokens = [str(token) for token in record["context"]]
            if target_token in special_set or special_set.intersection(context_tokens):
                raise ValueError(f"Special token leaked into structural corpus line {line_number}")
            try:
                target_id = full_token_to_id[target_token]
                context_ids = [full_token_to_id[token] for token in context_tokens]
            except KeyError as exc:
                raise ValueError(f"Corpus token absent from frozen vocab at line {line_number}") from exc
            if not context_ids:
                raise ValueError(f"Empty context at corpus line {line_number}")
            raw_examples.append((target_id, context_ids))
            target_counts[target_id] += 1
            active_full_ids.add(target_id)
            active_full_ids.update(context_ids)

    minimum_examples = int(pretrain.get("minimum_examples", 1))
    if len(raw_examples) < minimum_examples:
        raise ValueError(
            f"Corpus too small for pretraining stop rule: examples={len(raw_examples)}, minimum={minimum_examples}"
        )
    active_ids = sorted(active_full_ids)
    if len(active_ids) < 2:
        raise ValueError("CBOW corpus must contain at least two active tokens")
    full_to_active = {full_id: row for row, full_id in enumerate(active_ids)}
    active_tokens = [str(vocab_tokens[full_id]) for full_id in active_ids]
    max_context = int(corpus_manifest["context_spec"]["max_context"])
    targets_np = np.empty(len(raw_examples), dtype=np.int64)
    contexts_np = np.zeros((len(raw_examples), max_context), dtype=np.int64)
    masks_np = np.zeros((len(raw_examples), max_context), dtype=np.bool_)
    for row, (target_id, context_ids) in enumerate(raw_examples):
        targets_np[row] = full_to_active[target_id]
        for column, context_id in enumerate(context_ids[:max_context]):
            contexts_np[row, column] = full_to_active[context_id]
            masks_np[row, column] = True

    frequency = torch.tensor(
        [float(target_counts.get(full_id, 0)) for full_id in active_ids], dtype=torch.float64
    )
    # Context-only tokens remain sampleable without inventing an extra vocab row.
    frequency = frequency.clamp_min(1.0).pow(power)
    probabilities = frequency / frequency.sum()
    dry_payload = {
        "dry_run": True,
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "example_count": len(raw_examples),
        "full_vocab_size": len(vocab_tokens),
        "trained_token_count": len(active_tokens),
        "vector_dim": vector_dim,
        "negative_samples": negative_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": device_name,
    }
    if dry_run:
        return dry_payload

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model = _CBOWNegativeSampling(len(active_tokens), vector_dim, generator)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    contexts_all = torch.from_numpy(contexts_np)
    masks_all = torch.from_numpy(masks_np)
    targets_all = torch.from_numpy(targets_np)
    log_lines: list[str] = [json.dumps(dry_payload, sort_keys=True)]

    for epoch in range(epochs):
        order = torch.randperm(len(raw_examples), generator=generator)
        epoch_loss_sum = 0.0
        epoch_examples = 0
        for start in range(0, len(raw_examples), batch_size):
            indices = order[start : start + batch_size]
            targets = targets_all[indices]
            contexts = contexts_all[indices]
            masks = masks_all[indices]
            negatives = _sample_negatives(
                probabilities, targets, negative_samples, generator
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(contexts, masks, targets, negatives)
            loss.backward()
            optimizer.step()
            batch_count = int(indices.numel())
            epoch_loss_sum += float(loss.item()) * batch_count
            epoch_examples += batch_count
        mean_loss = epoch_loss_sum / max(epoch_examples, 1)
        line = f"epoch={epoch + 1} mean_negative_sampling_loss={mean_loss:.8f} examples={epoch_examples}"
        log_lines.append(line)

    target_dir = Path(output_dir).expanduser().resolve()
    _ensure_new_output_dir(target_dir)
    vectors_path = target_dir / "vectors.npy"
    token_to_row_path = target_dir / "token_to_row.json"
    vocab_copy_path = target_dir / "vocab_tokens.json"
    config_output_path = target_dir / "train_config.json"
    log_path = target_dir / "train.log"
    vectors = model.input_embeddings.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    np.save(vectors_path, vectors, allow_pickle=False)
    write_json(token_to_row_path, {token: row for row, token in enumerate(active_tokens)})
    write_json(vocab_copy_path, vocab_tokens)
    write_json(config_output_path, config)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    integration = config["integration"]
    output_entries = {
        "vectors": {"path": vectors_path.name, "sha256": sha256_file(vectors_path)},
        "token_to_row": {
            "path": token_to_row_path.name,
            "sha256": sha256_file(token_to_row_path),
        },
        "vocab_tokens": {
            "path": vocab_copy_path.name,
            "sha256": sha256_file(vocab_copy_path),
        },
        "train_config": {
            "path": config_output_path.name,
            "sha256": sha256_file(config_output_path),
        },
        "train_log": {"path": log_path.name, "sha256": sha256_file(log_path)},
    }
    manifest = {
        "format": ARTIFACT_FORMAT,
        "algorithm": "cbow_negative_sampling",
        "context_mode": "ast_parent_children_siblings",
        "corpus": {
            "manifest_path": str(corpus_manifest_path),
            "manifest_sha256": sha256_file(corpus_manifest_path),
            "corpus_sha256": sha256_file(corpus_path),
            "allowed_split": "train",
        },
        "vector_dim": vector_dim,
        "full_vocab_size": len(vocab_tokens),
        "trained_token_count": len(active_tokens),
        "seed": seed,
        "training": {
            "negative_samples": negative_samples,
            "negative_sampling_power": power,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device_name,
            "input_initialization": "uniform(-0.5/dim, 0.5/dim)",
            "output_initialization": "zeros",
        },
        "integration": {
            "missing_token_policy": "deterministic_normal",
            "missing_init_seed": int(integration["missing_init_seed"]),
            "missing_init_std": float(integration["missing_init_std"]),
            "special_token_policy": {
                "[PAD]": "zeros",
                "[UNK]": "deterministic_normal",
                "[SEG_START]": "deterministic_normal",
                "[SEG_END]": "deterministic_normal",
            },
        },
        "outputs": output_entries,
    }
    manifest_path = target_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "dry_run": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "final_loss": float(log_lines[-1].split("mean_negative_sampling_loss=")[1].split()[0]),
    }
