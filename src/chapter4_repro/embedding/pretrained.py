from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .cbow import ARTIFACT_FORMAT, sha256_file


def _resolve_verified_file(
    artifact_dir: Path,
    outputs: dict[str, dict[str, str]],
    key: str,
) -> Path:
    if key not in outputs:
        raise ValueError(f"Embedding artifact manifest is missing output entry: {key}")
    entry = outputs[key]
    path = (artifact_dir / str(entry.get("path"))).resolve()
    if path.parent != artifact_dir.resolve():
        raise ValueError(f"Embedding artifact path escapes source directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Embedding artifact file is missing: {path}")
    expected = str(entry.get("sha256") or "")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Embedding artifact SHA-256 mismatch for {key}: expected={expected}, actual={actual}"
        )
    return path


def _deterministic_normal(token: str, dim: int, seed: int, std: float) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}\0{token}".encode("utf-8")).digest()
    token_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    generator = np.random.default_rng(token_seed)
    return generator.normal(loc=0.0, scale=std, size=dim).astype(np.float32)


def _first_vocab_mismatch(expected: Sequence[str], actual: Sequence[str]) -> str:
    common = min(len(expected), len(actual))
    for index in range(common):
        if expected[index] != actual[index]:
            return f"row={index}, artifact={expected[index]!r}, model={actual[index]!r}"
    return f"length artifact={len(expected)}, model={len(actual)}"


def load_pretrained_embedding_matrix(
    *,
    artifact_dir: str | Path,
    ordered_vocab_tokens: Sequence[str],
    expected_dim: int,
    special_tokens: Sequence[str],
) -> tuple[Tensor, dict[str, Any]]:
    source_dir = Path(artifact_dir).expanduser().resolve()
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing pretrained embedding manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported embedding artifact format: {manifest.get('format')!r}")
    artifact_dim = int(manifest.get("vector_dim", -1))
    if artifact_dim != int(expected_dim):
        raise ValueError(
            f"Pretrained embedding dimension mismatch: artifact={artifact_dim}, model={expected_dim}"
        )

    outputs = manifest.get("outputs") or {}
    vectors_path = _resolve_verified_file(source_dir, outputs, "vectors")
    token_to_row_path = _resolve_verified_file(source_dir, outputs, "token_to_row")
    vocab_path = _resolve_verified_file(source_dir, outputs, "vocab_tokens")
    artifact_vocab = [
        str(token) for token in json.loads(vocab_path.read_text(encoding="utf-8"))
    ]
    model_vocab = [str(token) for token in ordered_vocab_tokens]
    if artifact_vocab != model_vocab:
        raise ValueError(
            "Pretrained embedding vocab/order mismatch: "
            + _first_vocab_mismatch(artifact_vocab, model_vocab)
        )
    if len(model_vocab) != len(set(model_vocab)):
        raise ValueError("Model vocabulary contains duplicate tokens")
    if int(manifest.get("full_vocab_size", -1)) != len(model_vocab):
        raise ValueError("Embedding artifact full_vocab_size does not match frozen vocabulary")

    token_to_row_payload = json.loads(token_to_row_path.read_text(encoding="utf-8"))
    token_to_row = {str(token): int(row) for token, row in token_to_row_payload.items()}
    if len(token_to_row) != len(set(token_to_row.values())):
        raise ValueError("Embedding artifact token_to_row contains duplicate rows")
    vectors = np.load(vectors_path, allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[1] != int(expected_dim):
        raise ValueError(
            f"Pretrained vectors have invalid shape: {vectors.shape}; expected [N,{expected_dim}]"
        )
    if vectors.shape[0] != len(token_to_row):
        raise ValueError("Pretrained vector row count does not match token_to_row")
    expected_rows = set(range(vectors.shape[0]))
    if set(token_to_row.values()) != expected_rows:
        raise ValueError("Pretrained token_to_row rows must be contiguous from zero")

    model_vocab_set = set(model_vocab)
    unknown_artifact_tokens = sorted(set(token_to_row) - model_vocab_set)
    if unknown_artifact_tokens:
        raise ValueError(
            f"Pretrained artifact contains tokens outside frozen vocab: {unknown_artifact_tokens[:3]}"
        )
    special_list = [str(token) for token in special_tokens]
    special_set = set(special_list)
    trained_specials = sorted(special_set.intersection(token_to_row))
    if trained_specials:
        raise ValueError(
            f"Structural CBOW artifact must not train special-token rows: {trained_specials}"
        )
    for token in special_list:
        if token not in model_vocab_set:
            raise ValueError(f"Required special token is absent from model vocab: {token}")

    integration = manifest.get("integration") or {}
    if integration.get("missing_token_policy") != "deterministic_normal":
        raise ValueError("Unsupported missing-token initialization policy")
    missing_seed = int(integration["missing_init_seed"])
    missing_std = float(integration["missing_init_std"])
    if missing_std <= 0:
        raise ValueError("missing_init_std must be positive")
    special_policy = integration.get("special_token_policy") or {}
    expected_special_policy = {
        "[PAD]": "zeros",
        "[UNK]": "deterministic_normal",
        "[SEG_START]": "deterministic_normal",
        "[SEG_END]": "deterministic_normal",
    }
    if special_policy != expected_special_policy:
        raise ValueError(
            f"Special-token policy mismatch: expected={expected_special_policy}, actual={special_policy}"
        )

    matrix = np.empty((len(model_vocab), int(expected_dim)), dtype=np.float32)
    loaded_count = 0
    missing_tokens: list[str] = []
    for row, token in enumerate(model_vocab):
        if token == "[PAD]":
            matrix[row] = 0.0
        elif token in special_set:
            matrix[row] = _deterministic_normal(token, expected_dim, missing_seed, missing_std)
        elif token in token_to_row:
            matrix[row] = vectors[token_to_row[token]]
            loaded_count += 1
        else:
            matrix[row] = _deterministic_normal(token, expected_dim, missing_seed, missing_std)
            missing_tokens.append(token)
    if not np.isfinite(matrix).all():
        raise ValueError("Pretrained embedding matrix contains non-finite values")

    metadata = {
        "type": "cbow_pretrained",
        "artifact_dir": str(source_dir),
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "vectors_sha256": sha256_file(vectors_path),
        "corpus_manifest_sha256": str(manifest["corpus"]["manifest_sha256"]),
        "algorithm": str(manifest["algorithm"]),
        "vector_dim": int(expected_dim),
        "full_vocab_size": len(model_vocab),
        "loaded_token_count": loaded_count,
        "missing_token_count": len(missing_tokens),
        "missing_token_examples": missing_tokens[:20],
        "missing_token_policy": "deterministic_normal",
        "missing_init_seed": missing_seed,
        "missing_init_std": missing_std,
        "special_token_policy": special_policy,
    }
    return torch.from_numpy(matrix.copy()), metadata


def initialize_word_embedding_from_artifact(
    embedding: nn.Embedding,
    *,
    artifact_dir: str | Path,
    ordered_vocab_tokens: Sequence[str],
    special_tokens: Sequence[str],
    freeze: bool,
) -> dict[str, Any]:
    matrix, metadata = load_pretrained_embedding_matrix(
        artifact_dir=artifact_dir,
        ordered_vocab_tokens=ordered_vocab_tokens,
        expected_dim=int(embedding.embedding_dim),
        special_tokens=special_tokens,
    )
    if tuple(embedding.weight.shape) != tuple(matrix.shape):
        raise ValueError(
            f"Embedding table shape mismatch: model={tuple(embedding.weight.shape)}, artifact={tuple(matrix.shape)}"
        )
    with torch.no_grad():
        embedding.weight.copy_(matrix.to(device=embedding.weight.device, dtype=embedding.weight.dtype))
        if embedding.padding_idx is not None:
            embedding.weight[int(embedding.padding_idx)].zero_()
    embedding.weight.requires_grad_(not freeze)
    metadata["freeze"] = bool(freeze)
    return metadata
