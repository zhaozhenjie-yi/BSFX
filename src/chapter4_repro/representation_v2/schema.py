"""Frozen production contracts for the Clang AST V2 frontend.

The constants in this module are representation contracts, not model knobs.  They
must not depend on labels, validation metrics, or learner behavior.
"""

from __future__ import annotations

from typing import Final


CANONICAL_SCHEMA_VERSION: Final = "clang-ast-semantic-v2-canonical-v1"
FRONTEND_SCHEMA_VERSION: Final = "task39s-r6b-production-frontend-v1"
WRAPPER_SCHEMA_ID: Final = "task39s-r6b-line-remap-wrapper-v1"
PATH_SCHEMA_VERSION: Final = "clang-ast-semantic-v2-paths-v1"
SEGMENT_SCHEMA_VERSION: Final = "clang-ast-semantic-v2-segments-v1"
VOCABULARY_SCHEMA_VERSION: Final = "clang-ast-semantic-v2-vocabulary-v1"

PARSE_CLASSES: Final = (
    "CLEAN",
    "RECOVERED_ACCEPTED",
    "QUARANTINED",
    "BLOCKED",
)

RELATION_ORDER: Final = (
    "AST_CHILD",
    "DECL_USE",
    "CALL_CALLEE",
    "TYPE_OF",
    "MEMBER_BASE",
    "CONTROL_ROLE",
)

RELATION_READINESS: Final = {
    "AST_CHILD": "READY",
    "DECL_USE": "READY",
    "CALL_CALLEE": "PARTIAL_DIRECT_ONLY",
    "TYPE_OF": "READY",
    "MEMBER_BASE": "READY",
    "CONTROL_ROLE": "PARTIAL_STRUCTURED_ONLY",
}

# ParenExpr is the only production-collapsed kind in v1.  The fixture suite
# checks that eliminating it from token emission leaves declaration identity and
# semantic relation identity unchanged.
COLLAPSE_NODE_KINDS: Final = frozenset({"ParenExpr"})
ATTRIBUTE_ONLY_NODE_KINDS: Final = frozenset({"ImplicitCastExpr"})
QUARANTINE_SPECIAL_NODE_KINDS: Final = frozenset(
    {
        "RecoveryExpr",
        "UnresolvedLookupExpr",
        "CXXDependentScopeMemberExpr",
    }
)

# All other observed nodes are kept.  Listing the evidence-backed core kinds
# here makes the freeze explicit; an unseen kind is conservatively KEEP with an
# ``unregistered_kind`` provenance flag instead of being silently deleted.
KEEP_NODE_KINDS: Final = frozenset(
    {
        "ArraySubscriptExpr",
        "BinaryOperator",
        "BreakStmt",
        "CStyleCastExpr",
        "CXXBoolLiteralExpr",
        "CXXConstCastExpr",
        "CXXConstructorDecl",
        "CXXDestructorDecl",
        "CXXDynamicCastExpr",
        "CXXFunctionalCastExpr",
        "CXXMemberCallExpr",
        "CXXMethodDecl",
        "CXXRecordDecl",
        "CXXReinterpretCastExpr",
        "CXXStaticCastExpr",
        "CallExpr",
        "CaseStmt",
        "CharacterLiteral",
        "CompoundAssignOperator",
        "CompoundStmt",
        "ConditionalOperator",
        "ConstantExpr",
        "ContinueStmt",
        "DeclRefExpr",
        "DeclStmt",
        "DefaultStmt",
        "DoStmt",
        "FieldDecl",
        "FloatingLiteral",
        "ForStmt",
        "FunctionDecl",
        "FunctionTemplateDecl",
        "GotoStmt",
        "IfStmt",
        "InitListExpr",
        "IntegerLiteral",
        "LabelStmt",
        "LambdaExpr",
        "MaterializeTemporaryExpr",
        "MemberExpr",
        "NullStmt",
        "ParmVarDecl",
        "PredefinedExpr",
        "RecordDecl",
        "ReturnStmt",
        "StringLiteral",
        "SwitchStmt",
        "UnaryExprOrTypeTraitExpr",
        "UnaryOperator",
        "VarDecl",
        "WhileStmt",
    }
)

EXPLICIT_CAST_KINDS: Final = frozenset(
    {
        "CStyleCastExpr",
        "CXXConstCastExpr",
        "CXXDynamicCastExpr",
        "CXXFunctionalCastExpr",
        "CXXReinterpretCastExpr",
        "CXXStaticCastExpr",
    }
)

DECLARATION_KINDS: Final = frozenset(
    {
        "BindingDecl",
        "CXXConstructorDecl",
        "CXXDestructorDecl",
        "CXXMethodDecl",
        "EnumConstantDecl",
        "FieldDecl",
        "FunctionDecl",
        "FunctionTemplateDecl",
        "ParmVarDecl",
        "StructuredBindingDecl",
        "TypedefDecl",
        "TypeAliasDecl",
        "VarDecl",
    }
)

FUNCTION_KINDS: Final = frozenset(
    {
        "CXXConstructorDecl",
        "CXXDestructorDecl",
        "CXXMethodDecl",
        "FunctionDecl",
    }
)

CALL_KINDS: Final = frozenset(
    {
        "CallExpr",
        "CXXMemberCallExpr",
        "CXXOperatorCallExpr",
    }
)

CONTROL_ROLE_SLOTS: Final = {
    "IfStmt": ("IF_CONDITION", "IF_THEN", "IF_ELSE"),
    "WhileStmt": ("WHILE_CONDITION", "WHILE_BODY"),
    "DoStmt": ("DO_BODY", "DO_CONDITION"),
    "SwitchStmt": ("SWITCH_CONDITION", "SWITCH_BODY"),
    # Clang serializes five positional slots and uses empty objects for absent
    # init/condition-variable/condition/increment values.
    "ForStmt": (
        "FOR_INIT",
        "FOR_CONDITION_VARIABLE",
        "FOR_CONDITION",
        "FOR_INCREMENT",
        "FOR_BODY",
    ),
}

PARSE_ACCEPTANCE_THRESHOLDS: Final = {
    "max_recovery_node_ratio": 0.25,
    "max_unresolved_ref_ratio": 0.50,
    "required_reference_accounting_ratio": 1.0,
}

# A global count of 256 stays above the observed V1 context P99 (168) and max
# (201), while bounding the R6A BigVul direct-semantic max of 1182.  Structural
# paths get a reserved cap before semantic priorities are applied.
PATH_BUDGET: Final = {
    "per_function_global_cap": 256,
    "structural_cap": 64,
    "per_relation_direct_cap": {
        "DECL_USE": 64,
        "CALL_CALLEE": 32,
        "MEMBER_BASE": 32,
        "CONTROL_ROLE": 48,
        "TYPE_OF": 64,
    },
    "per_relation_hybrid_cap": {
        "DECL_USE": 32,
        "CALL_CALLEE": 16,
        "MEMBER_BASE": 16,
        "CONTROL_ROLE": 24,
        "TYPE_OF": 0,
    },
    "semantic_priority": (
        "DECL_USE",
        "CALL_CALLEE",
        "MEMBER_BASE",
        "CONTROL_ROLE",
        "TYPE_OF",
    ),
    "hybrid_max_ast_bridge_nodes": 4,
    "hybrid_max_tokens": 16,
    "cycle_policy": "reject a candidate whose ordered canonical node IDs repeat",
    "unresolved_policy": "retain explicit unresolved direct paths; never guess a target",
    "overflow_order": "family reserve, semantic priority, direct before hybrid, function/source preorder, target stable ID, role, canonical path hash",
}

SEGMENT_CORE_LENGTH: Final = 50
SEGMENT_CONTEXT: Final = 5

TOKEN_NAMESPACES: Final = (
    "[NODE:...]",
    "[REL:...]",
    "[ROLE:...]",
    "[TYPE:...]",
    "[REF:...]",
)

NOT_CLAIMED: Final = (
    "CFG",
    "DFG",
    "PDG",
    "CPG",
    "LLVM IR",
    "SSA",
    "MemorySSA",
    "alias analysis",
    "points-to analysis",
    "complete virtual dispatch",
    "complete interprocedural data flow",
)


def node_policy(kind: str) -> str:
    """Return the frozen token-emission policy for a Clang node kind."""

    if kind in COLLAPSE_NODE_KINDS:
        return "COLLAPSE"
    if kind in ATTRIBUTE_ONLY_NODE_KINDS:
        return "ATTRIBUTE_ONLY"
    if kind in QUARANTINE_SPECIAL_NODE_KINDS:
        return "QUARANTINE_SPECIAL"
    return "KEEP"

