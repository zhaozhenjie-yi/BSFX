from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .profile import PAPER_FAMILY_ID, ResolvedTask34Profile


class Task34ScopeViolation(PermissionError):
    """Raised when an operation exceeds the Task 33-frozen Task 34 scope."""


class Task34Operation(str, Enum):
    PROFILE_RESOLUTION = "profile_resolution"
    TRAIN_ONLY_ADAPTER_BUILD = "train_only_adapter_build"
    TRAIN_ONLY_MODEL_FORWARD = "train_only_model_forward"
    TRAIN_ONLY_MODEL_LOSS_BACKWARD = "train_only_model_loss_backward"
    TRAIN_ONLY_INTEGRATED_SMOKE = "train_only_integrated_smoke"
    TRAIN_ONLY_PREDICTION_SNAPSHOT = "train_only_prediction_snapshot"
    TRAIN_ONLY_AUDIT = "train_only_audit"
    CBOW_CORPUS_BUILD = "cbow_corpus_build"
    CBOW_UNSUPERVISED_PRETRAIN = "cbow_unsupervised_pretrain"

    MODEL_OPTIMIZER_STEP = "model_optimizer_step"
    SUPERVISED_TRAINING = "supervised_training"
    EPOCH_LOOP = "epoch_loop"
    VALIDATION = "validation"
    TEST = "test"
    SEALED_TEST = "sealed_test"
    S2 = "s2"
    S4 = "s4"
    S25 = "s25"
    FORMAL_RESULT = "formal_result"


_ALLOWED_OPERATIONS = frozenset(
    {
        Task34Operation.PROFILE_RESOLUTION,
        Task34Operation.TRAIN_ONLY_ADAPTER_BUILD,
        Task34Operation.TRAIN_ONLY_MODEL_FORWARD,
        Task34Operation.TRAIN_ONLY_MODEL_LOSS_BACKWARD,
        Task34Operation.TRAIN_ONLY_INTEGRATED_SMOKE,
        Task34Operation.TRAIN_ONLY_PREDICTION_SNAPSHOT,
        Task34Operation.TRAIN_ONLY_AUDIT,
        Task34Operation.CBOW_CORPUS_BUILD,
        Task34Operation.CBOW_UNSUPERVISED_PRETRAIN,
    }
)

_DATA_OPERATIONS = _ALLOWED_OPERATIONS - {Task34Operation.PROFILE_RESOLUTION}
_CBOW_OPERATIONS = frozenset(
    {Task34Operation.CBOW_CORPUS_BUILD, Task34Operation.CBOW_UNSUPERVISED_PRETRAIN}
)


@dataclass(frozen=True)
class Task34ScopeRequest:
    operation: Task34Operation | str
    split: str | None = None
    model_optimizer_steps: int = 0
    model_parameter_updates: int = 0
    cbow_optimizer_steps: int = 0
    labels_consumed_by_cbow: bool = False
    supervised_training: bool = False
    epoch_loop: bool = False
    validation_access: bool = False
    test_access: bool = False
    sealed_test_access: bool = False
    s2: bool = False
    s4: bool = False
    formal_metrics: bool = False
    overwrite_existing_artifact: bool = False


@dataclass(frozen=True)
class Task34ScopeAuthorization:
    operation: str
    profile_id: str
    family_id: str
    split: str | None
    model_optimizer_steps: int
    model_parameter_updates: int
    cbow_optimizer_steps: int
    decision: str = "ALLOW"
    parent_scope: str = "TASK33_FROZEN_TASK34_V1"

    def audit_record(self) -> dict[str, Any]:
        return {
            "schema_version": "task34-scope-authorization-v1",
            "decision": self.decision,
            "parent_scope": self.parent_scope,
            "operation": self.operation,
            "profile_id": self.profile_id,
            "family_id": self.family_id,
            "split": self.split,
            "model_optimizer_steps": self.model_optimizer_steps,
            "model_parameter_updates": self.model_parameter_updates,
            "cbow_optimizer_steps": self.cbow_optimizer_steps,
            "validation_access": False,
            "test_access": False,
            "sealed_test_access": False,
            "s2": False,
            "s4": False,
            "formal_metrics": False,
        }


def _normalize_operation(value: Task34Operation | str) -> Task34Operation:
    if isinstance(value, Task34Operation):
        return value
    try:
        return Task34Operation(str(value))
    except ValueError as exc:
        raise Task34ScopeViolation(f"unknown Task 34 operation: {value!r}") from exc


def _require_nonnegative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Task34ScopeViolation(f"{name} must be a non-negative integer")


class Task34ScopeGuard:
    """Fail-closed execution guard for the Task 33-frozen Task 34 boundary.

    Function-label loss and backward on a train-only qualification batch are
    permitted.  A supervised optimizer update is not.  The sole optimizer
    exception is PAPER's label-free, train-only structural CBOW pretraining.
    """

    def __init__(self, profile: ResolvedTask34Profile) -> None:
        self.profile = profile

    def authorize(self, request: Task34ScopeRequest) -> Task34ScopeAuthorization:
        operation = _normalize_operation(request.operation)
        for name in (
            "model_optimizer_steps",
            "model_parameter_updates",
            "cbow_optimizer_steps",
        ):
            _require_nonnegative(name, getattr(request, name))

        if operation not in _ALLOWED_OPERATIONS:
            raise Task34ScopeViolation(
                f"operation {operation.value!r} is prohibited by the Task 33 Task 34 scope"
            )
        if request.model_optimizer_steps != 0 or request.model_parameter_updates != 0:
            raise Task34ScopeViolation("Task 34 requires model optimizer steps and updates to equal 0")
        if request.supervised_training:
            raise Task34ScopeViolation("Task 34 prohibits supervised training and optimizer updates")
        if request.epoch_loop:
            raise Task34ScopeViolation("Task 34 prohibits epoch loops")
        if request.validation_access or request.split == "val":
            raise Task34ScopeViolation("Task 34 prohibits validation access")
        if request.test_access or request.split == "test":
            raise Task34ScopeViolation("Task 34 prohibits test access")
        if request.sealed_test_access or request.split == "sealed_test":
            raise Task34ScopeViolation("Task 34 prohibits sealed-test access")
        if request.s2 or request.s4:
            raise Task34ScopeViolation("Task 34 prohibits S2 and S4")
        if request.formal_metrics:
            raise Task34ScopeViolation("Task 34 prohibits formal metrics/results")
        if request.overwrite_existing_artifact:
            raise Task34ScopeViolation("Task 34 artifacts must be created without overwrite")

        if operation in _DATA_OPERATIONS and request.split != "train":
            raise Task34ScopeViolation(
                f"operation {operation.value!r} requires the explicit train split"
            )
        if operation == Task34Operation.PROFILE_RESOLUTION and request.split is not None:
            raise Task34ScopeViolation("profile resolution must not open a data split")

        if operation in _CBOW_OPERATIONS:
            if self.profile.family_id != PAPER_FAMILY_ID or not self.profile.cbow_enabled:
                raise Task34ScopeViolation(
                    "CBOW corpus/pretraining is allowed only for PAPER_REPRO_V1"
                )
            if request.labels_consumed_by_cbow:
                raise Task34ScopeViolation("CBOW pretraining must be label-free")
            if operation == Task34Operation.CBOW_CORPUS_BUILD and request.cbow_optimizer_steps != 0:
                raise Task34ScopeViolation("CBOW corpus construction cannot perform optimizer steps")
        elif request.cbow_optimizer_steps != 0:
            raise Task34ScopeViolation(
                "CBOW optimizer steps are allowed only in cbow_unsupervised_pretrain"
            )

        return Task34ScopeAuthorization(
            operation=operation.value,
            profile_id=self.profile.profile_id,
            family_id=self.profile.family_id,
            split=request.split,
            model_optimizer_steps=request.model_optimizer_steps,
            model_parameter_updates=request.model_parameter_updates,
            cbow_optimizer_steps=request.cbow_optimizer_steps,
        )


def authorize_task34_operation(
    profile: ResolvedTask34Profile,
    request: Task34ScopeRequest,
) -> Task34ScopeAuthorization:
    return Task34ScopeGuard(profile).authorize(request)
