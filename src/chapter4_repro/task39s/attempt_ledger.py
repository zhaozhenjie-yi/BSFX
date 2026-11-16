"""Write-once hash-chained canonical-attempt ledger for Task39S.2A."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


class AttemptLedgerError(RuntimeError):
    """Fail-closed attempt-ledger error."""


LEDGER_RECORD_V1 = "task39s_2r_attempt_ledger_record_v1"
LEDGER_RECORD_V2 = "task39s_2ar_attempt_ledger_record_v2"
TASK39S5A3_FACTORIAL_16 = "TASK39S5A3_FACTORIAL_16"
TERMINAL_HASH_FIELDS: tuple[str, ...] = (
    "artifact_manifest_sha256",
    "predictions_sha256",
    "final_metrics_sha256",
    "recomputed_metrics_sha256",
    "metric_consistency_sha256",
    "checkpoint_manifest_sha256",
    "diagnostics_summary_sha256",
    "failure_localization_sha256",
    "run_status_sha256",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _record_hash(value: Mapping[str, Any]) -> str:
    bare = dict(value)
    bare.pop("record_hash", None)
    return hashlib.sha256(_canonical_bytes(bare)).hexdigest()


def build_attempt_ledger_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "task39s_2r_attempt_ledger_schema_v1",
        "title": "Task39S.2A hash-chained formal attempt ledger record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "sequence",
            "event",
            "run_id",
            "activation_fingerprint",
            "recorded_at",
            "previous_hash",
            "optimizer_step_count",
            "retry_of",
            "record_hash",
        ],
        "properties": {
            "schema_version": {"const": LEDGER_RECORD_V1},
            "sequence": {"type": "integer", "minimum": 1, "maximum": 32},
            "event": {"enum": ["STARTED", "TERMINAL", "INFRASTRUCTURE_RETRY_STARTED"]},
            "run_id": {"type": "string", "minLength": 1},
            "activation_fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "recorded_at": {"type": "string", "minLength": 1},
            "previous_hash": {"type": ["string", "null"]},
            "optimizer_step_count": {"type": "integer", "minimum": 0},
            "retry_of": {"type": ["integer", "null"]},
            "record_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }


def build_attempt_ledger_schema_v2() -> dict[str, Any]:
    """Return the strict v2 schema while retaining a v1 reader contract."""

    terminal_required = [
        "schema_version",
        "sequence",
        "event",
        "run_id",
        "activation_fingerprint",
        "recorded_at",
        "previous_hash",
        "optimizer_step_count",
        "retry_of",
        "started_record_hash",
        "process_return_code",
        "runtime_completion_status",
        "formal_taxonomy",
        "gate_pass",
        *TERMINAL_HASH_FIELDS,
        "absence_reasons",
        "record_hash",
    ]
    properties: dict[str, Any] = {
        "schema_version": {"const": LEDGER_RECORD_V2},
        "sequence": {"type": "integer", "minimum": 1, "maximum": 32},
        "event": {
            "enum": ["STARTED", "TERMINAL", "INFRASTRUCTURE_RETRY_STARTED"]
        },
        "run_id": {"type": "string", "minLength": 1},
        "activation_fingerprint": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "recorded_at": {"type": "string", "minLength": 1},
        "previous_hash": {"type": ["string", "null"]},
        "optimizer_step_count": {"type": "integer", "minimum": 0},
        "retry_of": {"type": ["integer", "null"]},
        "started_record_hash": {"type": ["string", "null"]},
        "process_return_code": {"type": ["integer", "null"]},
        "runtime_completion_status": {"type": ["string", "null"]},
        "formal_taxonomy": {"type": ["string", "null"]},
        "gate_pass": {"type": ["boolean", "null"]},
        "absence_reasons": {"type": "object"},
        "record_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    }
    properties.update(
        {
            field: {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"}
            for field in TERMINAL_HASH_FIELDS
        }
    )
    base_required = [
        "schema_version",
        "sequence",
        "event",
        "run_id",
        "activation_fingerprint",
        "recorded_at",
        "previous_hash",
        "optimizer_step_count",
        "retry_of",
        "record_hash",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "task39s_2ar_attempt_ledger_schema_v2",
        "title": "Task39S.2A formal attempt ledger record v2",
        "description": "Readers remain backward compatible with the frozen v1 schema.",
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": base_required,
                "properties": properties,
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": terminal_required,
                "properties": properties,
            },
        ],
        "backward_compatible_reader_schema": build_attempt_ledger_schema(),
    }


class AttemptLedger:
    def __init__(
        self,
        path: Path,
        *,
        activation_fingerprint: str,
        authorized_run_ids: Sequence[str],
        total_attempt_limit: int = 8,
        record_schema_version: str = LEDGER_RECORD_V1,
        recovery_mode: bool = False,
        attempt_contract_id: str | None = None,
    ) -> None:
        if recovery_mode and attempt_contract_id is not None:
            raise AttemptLedgerError(
                "recovery and named attempt contracts are mutually exclusive"
            )
        if attempt_contract_id == TASK39S5A3_FACTORIAL_16:
            if total_attempt_limit != 16:
                raise AttemptLedgerError(
                    "Task39S.5A3 factorial ledger requires exactly 16 attempts"
                )
            if len(authorized_run_ids) != 16 or len(set(authorized_run_ids)) != 16:
                raise AttemptLedgerError(
                    "Task39S.5A3 factorial ledger requires exactly 16 unique run IDs"
                )
            if record_schema_version != LEDGER_RECORD_V2:
                raise AttemptLedgerError(
                    "Task39S.5A3 factorial ledger requires v2 record schema"
                )
            self.recovery_mode = False
        elif attempt_contract_id is not None:
            raise AttemptLedgerError("unknown named attempt ledger contract")
        elif recovery_mode:
            # Recovery ledger is an isolated, namespaced ledger for canonical
            # re-execution of cells lost to runtime-resource failure. It reuses
            # the v2 record schema but is bounded by its own activation's
            # attempt limit (currently 2) and its own 2-run namespace.
            if total_attempt_limit != 2:
                raise AttemptLedgerError(
                    "recovery attempt ledger requires exactly 2 attempts"
                )
            if len(set(authorized_run_ids)) != 2:
                raise AttemptLedgerError(
                    "recovery attempt ledger requires exactly 2 unique run IDs"
                )
            if record_schema_version != LEDGER_RECORD_V2:
                raise AttemptLedgerError(
                    "recovery attempt ledger requires v2 record schema"
                )
            self.recovery_mode = True
        else:
            if total_attempt_limit != 8:
                raise AttemptLedgerError("formal canonical attempt limit must remain 8")
            if len(set(authorized_run_ids)) != 8:
                raise AttemptLedgerError("attempt ledger requires exactly 8 unique run IDs")
            self.recovery_mode = False
        self.attempt_contract_id = attempt_contract_id
        self.path = path
        self.activation_fingerprint = activation_fingerprint
        self.authorized_run_ids = frozenset(authorized_run_ids)
        self.total_attempt_limit = total_attempt_limit
        if record_schema_version not in {LEDGER_RECORD_V1, LEDGER_RECORD_V2}:
            raise AttemptLedgerError("unsupported attempt ledger record schema")
        self.record_schema_version = record_schema_version

    @staticmethod
    def _validate_v2_terminal(row: Mapping[str, Any]) -> None:
        required = {
            "started_record_hash",
            "process_return_code",
            "runtime_completion_status",
            "formal_taxonomy",
            "gate_pass",
            "absence_reasons",
            *TERMINAL_HASH_FIELDS,
        }
        missing = sorted(required - set(row))
        if missing:
            raise AttemptLedgerError(
                f"v2 TERMINAL evidence fields missing: {missing}"
            )
        if not isinstance(row.get("process_return_code"), int):
            raise AttemptLedgerError("v2 TERMINAL process return code invalid")
        if not isinstance(row.get("runtime_completion_status"), str):
            raise AttemptLedgerError("v2 TERMINAL runtime status invalid")
        if not isinstance(row.get("formal_taxonomy"), str):
            raise AttemptLedgerError("v2 TERMINAL formal taxonomy invalid")
        if not isinstance(row.get("gate_pass"), bool):
            raise AttemptLedgerError("v2 TERMINAL gate_pass invalid")
        reasons = row.get("absence_reasons")
        if not isinstance(reasons, Mapping):
            raise AttemptLedgerError("v2 TERMINAL absence_reasons invalid")
        for field in TERMINAL_HASH_FIELDS:
            value = row.get(field)
            if value is None:
                if field not in reasons or not str(reasons[field]).strip():
                    raise AttemptLedgerError(
                        f"v2 TERMINAL null {field} requires absence_reason"
                    )
            elif not (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            ):
                raise AttemptLedgerError(f"v2 TERMINAL {field} is not SHA-256")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        previous: str | None = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise AttemptLedgerError(f"blank ledger row at line {line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AttemptLedgerError(
                        f"invalid ledger JSON at line {line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise AttemptLedgerError("ledger row must be an object")
                if row.get("sequence") != line_number:
                    raise AttemptLedgerError("ledger sequence is not contiguous")
                if row.get("previous_hash") != previous:
                    raise AttemptLedgerError("ledger previous-hash chain mismatch")
                if row.get("record_hash") != _record_hash(row):
                    raise AttemptLedgerError("ledger record hash mismatch")
                if row.get("activation_fingerprint") != self.activation_fingerprint:
                    raise AttemptLedgerError("ledger activation fingerprint mismatch")
                if row.get("run_id") not in self.authorized_run_ids:
                    raise AttemptLedgerError("ledger contains unauthorized run ID")
                if row.get("event") not in {
                    "STARTED",
                    "TERMINAL",
                    "INFRASTRUCTURE_RETRY_STARTED",
                }:
                    raise AttemptLedgerError("ledger event invalid")
                schema_version = row.get("schema_version")
                if schema_version not in {LEDGER_RECORD_V1, LEDGER_RECORD_V2}:
                    raise AttemptLedgerError("ledger schema version invalid")
                if schema_version == LEDGER_RECORD_V2 and row.get("event") == "TERMINAL":
                    self._validate_v2_terminal(row)
                previous = str(row["record_hash"])
                records.append(row)
        starts = [
            row
            for row in records
            if row["event"] in {"STARTED", "INFRASTRUCTURE_RETRY_STARTED"}
        ]
        canonical = [row for row in starts if row["event"] == "STARTED"]
        if len(canonical) > self.total_attempt_limit:
            raise AttemptLedgerError(
                "formal canonical attempt total exceeds "
                f"{self.total_attempt_limit}"
            )
        if len({row["run_id"] for row in canonical}) != len(canonical):
            raise AttemptLedgerError("duplicate canonical STARTED record")
        for row in records:
            if row.get("schema_version") != LEDGER_RECORD_V2 or row["event"] != "TERMINAL":
                continue
            starts_for_run = [
                started
                for started in canonical
                if started["run_id"] == row["run_id"]
            ]
            if len(starts_for_run) != 1 or row.get("started_record_hash") != starts_for_run[0].get("record_hash"):
                raise AttemptLedgerError("v2 TERMINAL started-record hash mismatch")
        return records

    def verify(self) -> dict[str, Any]:
        records = self.records()
        starts = [row for row in records if row["event"] == "STARTED"]
        return {
            "verdict": "PASS",
            "record_count": len(records),
            "formal_canonical_attempt_count": len(starts),
            "terminal_count": sum(row["event"] == "TERMINAL" for row in records),
            "total_attempt_limit": self.total_attempt_limit,
            "hash_chain_valid": True,
        }

    def assert_can_start(self, run_id: str) -> None:
        if run_id not in self.authorized_run_ids:
            raise AttemptLedgerError("run ID is not authorized")
        records = self.records()
        starts = [row for row in records if row["event"] == "STARTED"]
        if any(row["run_id"] == run_id for row in starts):
            raise AttemptLedgerError("canonical run already has a STARTED record")
        if len(starts) >= self.total_attempt_limit:
            raise AttemptLedgerError("formal canonical attempt total limit reached")

    def _append(
        self,
        *,
        event: str,
        run_id: str,
        optimizer_step_count: int,
        retry_of: int | None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise AttemptLedgerError("attempt ledger is locked") from exc
        try:
            os.close(descriptor)
            records = self.records()
            if event == "STARTED":
                self.assert_can_start(run_id)
            row: dict[str, Any] = {
                "schema_version": self.record_schema_version,
                "sequence": len(records) + 1,
                "event": event,
                "run_id": run_id,
                "activation_fingerprint": self.activation_fingerprint,
                "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "previous_hash": records[-1]["record_hash"] if records else None,
                "optimizer_step_count": int(optimizer_step_count),
                "retry_of": retry_of,
            }
            if extra:
                overlap = set(row) & set(extra)
                if overlap:
                    raise AttemptLedgerError(
                        f"attempt ledger extra fields overlap base record: {sorted(overlap)}"
                    )
                row.update(dict(extra))
            row["record_hash"] = _record_hash(row)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.records()
            return row
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    def append_started(self, run_id: str) -> dict[str, Any]:
        return self._append(
            event="STARTED", run_id=run_id, optimizer_step_count=0, retry_of=None
        )

    def append_terminal(
        self, run_id: str, *, optimizer_step_count: int
    ) -> dict[str, Any]:
        records = self.records()
        starts = [
            row
            for row in records
            if row["run_id"] == run_id and row["event"] == "STARTED"
        ]
        if len(starts) != 1:
            raise AttemptLedgerError("TERMINAL requires exactly one canonical STARTED")
        if any(
            row["run_id"] == run_id and row["event"] == "TERMINAL"
            for row in records
        ):
            raise AttemptLedgerError("run already has a TERMINAL record")
        return self._append(
            event="TERMINAL",
            run_id=run_id,
            optimizer_step_count=optimizer_step_count,
            retry_of=None,
        )

    def append_terminal_v2(
        self,
        run_id: str,
        *,
        process_return_code: int,
        runtime_completion_status: str,
        formal_taxonomy: str,
        gate_pass: bool,
        optimizer_step_count: int,
        artifact_manifest_sha256: str | None,
        predictions_sha256: str | None,
        final_metrics_sha256: str | None,
        recomputed_metrics_sha256: str | None,
        metric_consistency_sha256: str | None,
        checkpoint_manifest_sha256: str | None,
        diagnostics_summary_sha256: str | None,
        failure_localization_sha256: str | None,
        run_status_sha256: str | None,
        absence_reasons: Mapping[str, str],
    ) -> dict[str, Any]:
        if self.record_schema_version != LEDGER_RECORD_V2:
            raise AttemptLedgerError("v2 TERMINAL requires a v2 ledger instance")
        records = self.records()
        starts = [
            row
            for row in records
            if row["run_id"] == run_id and row["event"] == "STARTED"
        ]
        if len(starts) != 1:
            raise AttemptLedgerError("TERMINAL requires exactly one canonical STARTED")
        if any(
            row["run_id"] == run_id and row["event"] == "TERMINAL"
            for row in records
        ):
            raise AttemptLedgerError("run already has a TERMINAL record")
        extra: dict[str, Any] = {
            "started_record_hash": starts[0]["record_hash"],
            "process_return_code": int(process_return_code),
            "runtime_completion_status": str(runtime_completion_status),
            "formal_taxonomy": str(formal_taxonomy),
            "gate_pass": bool(gate_pass),
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "predictions_sha256": predictions_sha256,
            "final_metrics_sha256": final_metrics_sha256,
            "recomputed_metrics_sha256": recomputed_metrics_sha256,
            "metric_consistency_sha256": metric_consistency_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "diagnostics_summary_sha256": diagnostics_summary_sha256,
            "failure_localization_sha256": failure_localization_sha256,
            "run_status_sha256": run_status_sha256,
            "absence_reasons": dict(absence_reasons),
        }
        self._validate_v2_terminal(extra)
        return self._append(
            event="TERMINAL",
            run_id=run_id,
            optimizer_step_count=optimizer_step_count,
            retry_of=None,
            extra=extra,
        )
