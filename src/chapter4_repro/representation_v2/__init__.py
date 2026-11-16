"""Train-only, non-production representation V2 shadow prototypes."""

from .clang_shadow import ClangShadowFrontend, canonical_json_sha256, sha256_file

__all__ = ["ClangShadowFrontend", "canonical_json_sha256", "sha256_file"]
