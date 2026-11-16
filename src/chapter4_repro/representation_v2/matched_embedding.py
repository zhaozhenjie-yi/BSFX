"""Deterministic matched random-embedding initialization for Task39S.R6D.

The historical Engineering profile uses a trainable ``nn.Embedding`` with a
standard-normal initialization.  V1 and V2 have different vocabularies, so a
single shape-dependent RNG stream would give a shared token different rows and
would also advance the global model RNG by different amounts.  This adapter
keeps the historical distribution while deriving a private seed from the
scientific seed and the UTF-8 token identity.  The result is independent of
vocabulary size/order and never consumes the downstream model RNG stream.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch


INITIALIZATION_SCHEMA = "task39s-r6d-token-keyed-normal-v1"
SEED_DOMAIN = "TASK39S_R6D_MATCHED_ENGINEERING_EMBEDDING_V1"


def token_seed(scientific_seed: int, token: str) -> int:
    """Return the stable private CPU generator seed for one vocabulary token."""

    payload = (
        f"{SEED_DOMAIN}\0{int(scientific_seed)}\0{str(token)}".encode("utf-8")
    )
    # torch.Generator.manual_seed accepts signed 64-bit values portably.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def build_matched_random_embedding(
    ordered_vocab_tokens: Sequence[str],
    *,
    embedding_dim: int,
    scientific_seed: int,
    padding_token: str = "[PAD]",
) -> np.ndarray:
    """Build a CPU float32 matrix under the frozen token-keyed policy.

    Non-padding rows follow ``Normal(mean=0, std=1)``, matching PyTorch
    ``nn.Embedding.reset_parameters``.  The padding row is exactly zero.
    Duplicate vocabulary entries fail closed.
    """

    tokens = [str(token) for token in ordered_vocab_tokens]
    if len(tokens) != len(set(tokens)):
        raise ValueError("ordered_vocab_tokens contains duplicates")
    if int(embedding_dim) <= 0:
        raise ValueError("embedding_dim must be positive")

    matrix = np.empty((len(tokens), int(embedding_dim)), dtype=np.float32)
    for row, token in enumerate(tokens):
        if token == padding_token:
            matrix[row].fill(0.0)
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(token_seed(scientific_seed, token))
        vector = torch.empty(int(embedding_dim), dtype=torch.float32)
        vector.normal_(mean=0.0, std=1.0, generator=generator)
        matrix[row] = vector.numpy()
    return matrix

