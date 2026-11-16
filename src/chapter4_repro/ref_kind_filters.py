from __future__ import annotations

from collections.abc import Sequence
from fnmatch import fnmatchcase


def normalize_kind_patterns(values: Sequence[str] | None) -> frozenset[str]:
    if not values:
        return frozenset()

    normalized = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return frozenset(normalized)


def matches_kind_patterns(value: object, patterns: frozenset[str]) -> bool:
    if not patterns:
        return True
    text = str(value)
    return any(fnmatchcase(text, pattern) for pattern in patterns)

