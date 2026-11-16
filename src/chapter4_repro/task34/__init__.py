"""Task 34 frozen profile resolution and execution-scope guards."""

from .profile import (
    ENGINEERING_FAMILY_ID,
    ENGINEERING_PROFILE_ID,
    PAPER_FAMILY_ID,
    PAPER_PROFILE_ID,
    ResolvedTask34Profile,
    Task34ProfileContractError,
    Task34ProfileError,
    Task34ProfileHashError,
    load_task34_profile,
    resolve_all_task34_profiles,
    resolve_task34_profile,
    validate_task34_profile_payload,
)
from .scope import (
    Task34Operation,
    Task34ScopeAuthorization,
    Task34ScopeGuard,
    Task34ScopeRequest,
    Task34ScopeViolation,
    authorize_task34_operation,
)

__all__ = [
    "ENGINEERING_FAMILY_ID",
    "ENGINEERING_PROFILE_ID",
    "PAPER_FAMILY_ID",
    "PAPER_PROFILE_ID",
    "ResolvedTask34Profile",
    "Task34Operation",
    "Task34ProfileContractError",
    "Task34ProfileError",
    "Task34ProfileHashError",
    "Task34ScopeAuthorization",
    "Task34ScopeGuard",
    "Task34ScopeRequest",
    "Task34ScopeViolation",
    "authorize_task34_operation",
    "load_task34_profile",
    "resolve_all_task34_profiles",
    "resolve_task34_profile",
    "validate_task34_profile_payload",
]
