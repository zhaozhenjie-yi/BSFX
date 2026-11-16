from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK34_PROFILE_SCHEMA = "task34-integrated-profile-v1"
PAPER_PROFILE_ID = "paper_repro_v1"
ENGINEERING_PROFILE_ID = "engineering_aligned_v1"
PAPER_FAMILY_ID = "PAPER_REPRO_V1"
ENGINEERING_FAMILY_ID = "ENGINEERING_ALIGNED_V1"
TASK34_PROFILE_IDS = frozenset({PAPER_PROFILE_ID, ENGINEERING_PROFILE_ID})

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = _PROJECT_ROOT.parents[1]
TASK34_CONFIG_DIR = _PROJECT_ROOT / "configs" / "task34_v1"


class Task34ProfileError(ValueError):
    """Base class for fail-closed Task 34 profile errors."""


class Task34ProfileContractError(Task34ProfileError):
    """Raised when a profile differs from the Task 33 frozen contract."""


class Task34ProfileHashError(Task34ProfileError):
    """Raised when a profile or parent evidence file fails SHA-256 verification."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


_PARENT_EVIDENCE = {
    "task33_scope": {
        "path": "docs/task33_scope_freeze_and_task34_spec.md",
        "sha256": "b42077de0c3c73cba64b7464b4479c37f6fac83215a59ecce286d7488e3a58f1",
    },
    "task33_acceptance": {
        "path": "docs/task33_acceptance_v1.json",
        "sha256": "f4def62941a99a0975b0af7e65201c027ca2d24b3c22be7bdcacc2cc5613c56c",
    },
    "h1_config": {
        "path": "BSFX/chapter4_repro/configs/b3_h1_v1.yaml",
        "sha256": "7ee9d0e51bc434a052e9f7e1d83f0d066eac5b203539409f19a256b915916cfd",
    },
    "d1_config": {
        "path": "BSFX/chapter4_repro/configs/b3_d1_v1.yaml",
        "sha256": "91167c531eae7da7eab2c40f06dc0024ad501d99b098f1662b20795c22062fee",
    },
    "st1_config": {
        "path": "BSFX/chapter4_repro/configs/task31_st1_structure_sensitive.yaml",
        "sha256": "0e0a59867091e41a0fbd00bca32e2dc4d9f454018d052e1eee663e664e85e5b5",
    },
    "c1_config": {
        "path": "BSFX/chapter4_repro/configs/task32_c1_multilevel_classifier.yaml",
        "sha256": "67428b3eaaa2d8ae82a4af9e87dec0b562cf866d4f985a0c27ea565d4f30885e",
    },
    "x1_coverage": {
        "path": "BSFX/chapter4_repro/data/processed/b2_x1_pf1_v1/path_coverage_distribution_v1.json",
        "sha256": "fc9476173c86a3fe255ef3182c0397fe8b74f943871baef60abc4c7d7b6a669d",
    },
}

_DATA_CONTRACT = {
    "datasets": {
        "bigvul": {
            "dataset_id": "bigvul-final-data-v1",
            "split_fingerprint_sha256": (
                "25b5659d5365fdc89e83f25c767aac3a76d5ced29660e7bfcbfaa760b189fae5"
            ),
        },
        "sard": {
            "dataset_id": "sard-c-only-final-v1",
            "split_fingerprint_sha256": (
                "1ab84d3dc989791b9b932fc46feed318f27073ce48f7f0de45da5399ae77c1b0"
            ),
            "cwe_scope": ["CWE-127", "CWE-338"],
            "role": "synthetic",
            "multisegment_condition": "CONDITIONAL_NOT_TRIGGERED",
        },
    },
    "allowed_split": "train",
    "path_profile": "semantic_x1_v1",
    "semantic_contract": "x1-call-cfg-defuse-v1.0.0",
    "max_core_path": 50,
    "context_window": 5,
    "duplicate_policy": "inverse_exact_equivalence_mass_one",
}

_TENSOR_CONTRACT = {
    "canonical_layout": "[B,S,P,T,D]",
    "provenance_levels": ["sample", "path", "segment", "token"],
    "masks": ["token_valid_mask", "path_valid_mask", "segment_valid_mask"],
    "dynamic_padding": True,
    "aggregate_invalid_entries": False,
    "boundaries": ["[SEG_START]", "[SEG_END]"],
    "ref_relation": "directional_def_use_matrix",
    "real_multipath_required": True,
}

_COMMON_MODULES = {
    "hpe": "formula_aligned_hpe_v1",
    "hpe_output_dim": 64,
    "hpe_to_dwaf_projection": "64_to_128",
    "cnn": "masked_gap_sigmoid_channel_gate",
    "dwaf": "token_three_source_mha8_dwaf_v1",
    "dwaf_heads": 8,
    "dwaf_output_dim": 128,
    "transformer_projection": "128_to_512",
    "transformer_layers": 6,
    "transformer_heads": 8,
    "transformer_ffn_dim": 3072,
    "transformer_residual": "standard_post_norm",
}

_PAPER_MODULES = {
    "semantic_embedding": "dataset_specific_train_only_cbow_128",
    "cbow_enabled": True,
    "cbow_seed": 20260715,
    "cbow_context": "parent_direct_children_siblings_k5",
    "cbow_negative_samples": 5,
    "cbow_fine_tune": True,
    **_COMMON_MODULES,
    "gru_attention": "tanh_additive_token_scorer",
    "structural_bias": "absolute_scalar_formula_4_37",
    "classifier": "paper_single_sigmoid_probability_attention",
    "output_protocol": "single_logit",
}

_ENGINEERING_MODULES = {
    "semantic_embedding": "trainable_random_128",
    "cbow_enabled": False,
    "cbow_seed": None,
    "cbow_context": None,
    "cbow_negative_samples": None,
    "cbow_fine_tune": False,
    **_COMMON_MODULES,
    "gru_attention": "linear_token_scorer",
    "structural_bias": "signed_clipped_per_layer_per_head_directional_ref",
    "classifier": "c1_two_logit_probability_attention",
    "output_protocol": "two_logits",
}

_PAPER_LOSS = {
    "protocol_id": "BCE_SINGLE_SIGMOID_V1",
    "implementation": "binary_cross_entropy_with_logits",
    "positive_class_index": 1,
    "threshold": "probability_strictly_greater_than_0_5",
}

_ENGINEERING_LOSS = {
    "protocol_id": "CE_TWO_LOGIT_V1",
    "implementation": "cross_entropy",
    "positive_class_index": 1,
    "threshold": "probability_strictly_greater_than_0_5",
}

_COMMON_QUALIFICATION = {
    "task33_verdict": "GO",
    "scope": "IMPLEMENTATION_INTEGRATION_AND_TRAIN_ONLY_QUALIFICATION",
    "model_forward": True,
    "model_loss_backward": True,
    "model_optimizer_steps": 0,
    "model_parameter_updates": 0,
    "validation_access": False,
    "test_access": False,
    "sealed_test_access": False,
    "epoch_loop": False,
    "s2": False,
    "s4": False,
    "formal_metrics": False,
}

_ARTIFACT_CONTRACT = {
    "namespace": "BSFX/chapter4_repro/artifacts/chapter4_task34_dual_profile_integration_v1",
    "overwrite_existing": False,
    "required_evidence": [
        "raw_log",
        "resolved_manifest",
        "input_fingerprints",
        "shape_provenance_audit",
        "prediction_snapshot",
        "gradient_audit",
        "failure_catalogue",
    ],
}


def _frozen_payload(
    *,
    profile_id: str,
    family_id: str,
    architecture_id: str,
    modules: Mapping[str, Any],
    loss_protocol: Mapping[str, Any],
    cbow_updates_allowed: bool,
) -> dict[str, Any]:
    qualification = {
        **_COMMON_QUALIFICATION,
        "cbow_train_only_unsupervised_optimizer_updates": cbow_updates_allowed,
    }
    return {
        "schema_version": TASK34_PROFILE_SCHEMA,
        "task_id": 34,
        "profile_id": profile_id,
        "family_id": family_id,
        "architecture_id": architecture_id,
        "parent_evidence": _PARENT_EVIDENCE,
        "data_contract": _DATA_CONTRACT,
        "tensor_contract": _TENSOR_CONTRACT,
        "modules": dict(modules),
        "loss_protocol": dict(loss_protocol),
        "qualification": qualification,
        "artifact_contract": _ARTIFACT_CONTRACT,
    }


FROZEN_PROFILE_PAYLOADS = {
    PAPER_PROFILE_ID: _frozen_payload(
        profile_id=PAPER_PROFILE_ID,
        family_id=PAPER_FAMILY_ID,
        architecture_id="task34-paper-repro-v1",
        modules=_PAPER_MODULES,
        loss_protocol=_PAPER_LOSS,
        cbow_updates_allowed=True,
    ),
    ENGINEERING_PROFILE_ID: _frozen_payload(
        profile_id=ENGINEERING_PROFILE_ID,
        family_id=ENGINEERING_FAMILY_ID,
        architecture_id="task34-engineering-aligned-v1",
        modules=_ENGINEERING_MODULES,
        loss_protocol=_ENGINEERING_LOSS,
        cbow_updates_allowed=False,
    ),
}

# These hashes lock the two v1 YAML byte streams.  Changing a switch requires
# a new versioned profile instead of an in-place reinterpretation.
FROZEN_PROFILE_FILE_SHA256 = {
    PAPER_PROFILE_ID: "34197adf8596e0f3c31c86d30eb3dba73146235184deece5434a8db7767650f7",
    ENGINEERING_PROFILE_ID: "a85973d271d2a627c9f465bcd326acfa62ba82b4486d820ded50597ebf4b6a74",
}


def _difference(expected: Any, actual: Any, path: str = "profile") -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: expected type {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f"{path}: key mismatch missing={missing}, extra={extra}"
        for key in expected:
            result = _difference(expected[key], actual[key], f"{path}.{key}")
            if result is not None:
                return result
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: expected list length {len(expected)}, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            result = _difference(expected_item, actual_item, f"{path}[{index}]")
            if result is not None:
                return result
        return None
    if expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def validate_task34_profile_payload(payload: Mapping[str, Any]) -> str:
    """Validate every field against one of the two Task 33 frozen families.

    Validation is deliberately exact: unknown keys, implicit defaults and
    alternate spellings are rejected.  The returned value is the profile ID.
    """

    if not isinstance(payload, Mapping):
        raise Task34ProfileContractError("Task 34 profile must be a mapping")
    profile_id = payload.get("profile_id")
    if profile_id not in TASK34_PROFILE_IDS:
        raise Task34ProfileContractError(
            f"unknown Task 34 profile_id={profile_id!r}; expected {sorted(TASK34_PROFILE_IDS)}"
        )
    expected = FROZEN_PROFILE_PAYLOADS[str(profile_id)]
    actual = dict(payload)
    mismatch = _difference(expected, actual)
    if mismatch is not None:
        raise Task34ProfileContractError(mismatch)
    return str(profile_id)


def _resolve_evidence_path(repository_root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise Task34ProfileContractError("parent evidence paths must be repository-relative")
    resolved = (repository_root / relative_path).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise Task34ProfileContractError(
            f"parent evidence path escapes repository: {relative_path!r}"
        ) from exc
    return resolved


@dataclass(frozen=True)
class ResolvedTask34Profile:
    source_path: Path
    source_sha256: str
    resolved_sha256: str
    profile_id: str
    family_id: str
    architecture_id: str
    loss_protocol_id: str
    raw: dict[str, Any]
    verified_parent_sha256: dict[str, str]

    @property
    def modules(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw["modules"])

    @property
    def qualification(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw["qualification"])

    @property
    def cbow_enabled(self) -> bool:
        return bool(self.raw["modules"]["cbow_enabled"])

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "task34-resolved-profile-manifest-v1",
            "task_id": 34,
            "profile_id": self.profile_id,
            "family_id": self.family_id,
            "architecture_id": self.architecture_id,
            "loss_protocol_id": self.loss_protocol_id,
            "profile_path": str(self.source_path),
            "profile_sha256": self.source_sha256,
            "resolved_profile_sha256": self.resolved_sha256,
            "verified_parent_sha256": dict(self.verified_parent_sha256),
            "module_switches": self.modules,
            "qualification": self.qualification,
        }


def load_task34_profile(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> ResolvedTask34Profile:
    """Load a frozen profile and verify its own and every parent file hash."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Task 34 profile is missing: {source_path}")
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    profile_id = validate_task34_profile_payload(raw)

    source_sha256 = sha256_file(source_path)
    expected_source_sha256 = FROZEN_PROFILE_FILE_SHA256[profile_id]
    if source_sha256 != expected_source_sha256:
        raise Task34ProfileHashError(
            f"profile SHA-256 mismatch for {profile_id}: "
            f"expected={expected_source_sha256}, actual={source_sha256}"
        )

    repo = Path(repository_root).expanduser().resolve() if repository_root else _REPOSITORY_ROOT
    if not repo.is_dir():
        raise FileNotFoundError(f"repository root is missing: {repo}")
    verified_parents: dict[str, str] = {}
    for name, evidence in raw["parent_evidence"].items():
        evidence_path = _resolve_evidence_path(repo, str(evidence["path"]))
        if not evidence_path.is_file():
            raise FileNotFoundError(f"Task 34 parent evidence is missing: {evidence_path}")
        actual_sha256 = sha256_file(evidence_path)
        expected_sha256 = str(evidence["sha256"])
        if actual_sha256 != expected_sha256:
            raise Task34ProfileHashError(
                f"parent evidence SHA-256 mismatch for {name}: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        verified_parents[name] = actual_sha256

    resolved_payload = {
        "profile": raw,
        "profile_sha256": source_sha256,
        "verified_parent_sha256": verified_parents,
    }
    return ResolvedTask34Profile(
        source_path=source_path,
        source_sha256=source_sha256,
        resolved_sha256=canonical_payload_sha256(resolved_payload),
        profile_id=profile_id,
        family_id=str(raw["family_id"]),
        architecture_id=str(raw["architecture_id"]),
        loss_protocol_id=str(raw["loss_protocol"]["protocol_id"]),
        raw=copy.deepcopy(raw),
        verified_parent_sha256=verified_parents,
    )


def resolve_task34_profile(
    profile_id: str,
    *,
    config_dir: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> ResolvedTask34Profile:
    if profile_id not in TASK34_PROFILE_IDS:
        raise Task34ProfileContractError(
            f"unknown Task 34 profile_id={profile_id!r}; expected {sorted(TASK34_PROFILE_IDS)}"
        )
    directory = Path(config_dir).expanduser().resolve() if config_dir else TASK34_CONFIG_DIR
    return load_task34_profile(
        directory / f"{profile_id}.yaml",
        repository_root=repository_root,
    )


def resolve_all_task34_profiles(
    *,
    config_dir: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, ResolvedTask34Profile]:
    return {
        profile_id: resolve_task34_profile(
            profile_id,
            config_dir=config_dir,
            repository_root=repository_root,
        )
        for profile_id in (PAPER_PROFILE_ID, ENGINEERING_PROFILE_ID)
    }
