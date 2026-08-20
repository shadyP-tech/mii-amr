"""Effect-injected, no-motion readiness for a sealed first route.

This module is ROS-, prompt-, process-, and write-free. It only invokes the
injected dry runner. Every result payload is backed by validated
``motion_published=false`` evidence; malformed output or reported motion is a
contract error and can never be normalized into a no-motion receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Callable, Mapping

from scripts.aufgabe04.real_robot.autonomous_localization_readiness import (
    LocalizationReadinessDecision,
    evaluate_localization_readiness_retry,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import (
    parse_dry_run_outcome,
)


INITIAL_READINESS_PHASE = "preauthorization_first_leg_readiness"
_RUN_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFETY_FIELDS = {
    "typed_run_requested": False,
    "operator_input_requested": False,
    "motion_authorized": False,
    "motion_published": False,
    "permit_issued": False,
    "reusable_as_motion_permit": False,
    "route_limits_unchanged": True,
}


def _path(value: object, field: str, suffix: str | None = None) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty path")
    if "\x00" in str(value):
        raise ValueError(f"{field} must not contain NUL")
    candidate = Path(value)
    if candidate == Path(".") or ".." in candidate.parts:
        raise ValueError(f"{field} must identify one artifact without traversal")
    if suffix is not None and candidate.suffix != suffix:
        raise ValueError(f"{field} must end in {suffix}")
    return candidate


def _freeze(value: object, field: str = "details") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{field} contains a non-string key")
        return MappingProxyType(
            {key: _freeze(item, f"{field}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{field}[]") for item in value)
    raise ValueError(f"{field} contains non-JSON type {type(value).__name__}")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _normal_file_sha256(path: Path, field: str) -> str:
    """Hash one existing readable normal file without accepting symlinks."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    if not source.is_file():
        raise ValueError(f"{field} must be an existing normal file")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"{field} must be readable") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class SealedRoutePaths:
    route_csv: Path
    diagnostics_json: Path
    route_certificate_json: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_csv",
            _path(self.route_csv, "route_csv", ".csv"),
        )
        object.__setattr__(
            self,
            "diagnostics_json",
            _path(self.diagnostics_json, "diagnostics_json", ".json"),
        )
        object.__setattr__(
            self,
            "route_certificate_json",
            _path(self.route_certificate_json, "route_certificate_json", ".json"),
        )
        if len(set(self.to_mapping().values())) != 3:
            raise ValueError("sealed route artifact paths must be distinct")

    @classmethod
    def from_mapping(cls, sealed: Mapping[str, object]) -> "SealedRoutePaths":
        if not isinstance(sealed, Mapping):
            raise ValueError("sealed route must be a mapping")
        names = ("route_csv", "diagnostics_json", "route_certificate_json")
        missing = [name for name in names if name not in sealed]
        if missing:
            raise ValueError("sealed route is missing: " + ", ".join(missing))
        extras = sorted(set(sealed) - set(names))
        if extras:
            raise ValueError(
                "sealed route contains unsupported fields: " + ", ".join(extras)
            )
        return cls(*(sealed[name] for name in names))  # type: ignore[arg-type]

    def to_mapping(self) -> dict[str, str]:
        return {
            "route_csv": str(self.route_csv),
            "diagnostics_json": str(self.diagnostics_json),
            "route_certificate_json": str(self.route_certificate_json),
        }


@dataclass(frozen=True)
class InitialReadinessDryRequest:
    sealed_route: SealedRoutePaths
    run_id: str
    attempt_index: int
    semantic_log_path: Path
    dry_preflight_path: Path
    dry_odom_certificate_path: Path
    dry_uncertainty_budget_path: Path


@dataclass(frozen=True)
class InitialReadinessAttempt:
    attempt_index: int
    maximum_retry_count: int
    run_id: str
    status: str
    reason: str
    details: Mapping[str, object]
    returncode: int
    semantic_log_path: Path
    semantic_log_sha256: str
    dry_preflight_path: Path
    dry_preflight_sha256: str
    dry_odom_certificate_path: Path
    dry_odom_certificate_sha256: str
    dry_uncertainty_budget_path: Path
    dry_uncertainty_budget_sha256: str
    retry_decision: LocalizationReadinessDecision
    retry_scheduled: bool

    def _fields(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "maximum_retry_count": self.maximum_retry_count,
            "run_id": self.run_id,
            "status": self.status,
            "reason": self.reason,
            "details": _thaw(self.details),
            "returncode": self.returncode,
            "semantic_log_path": str(self.semantic_log_path),
            "semantic_log_sha256": self.semantic_log_sha256,
            "dry_preflight_path": str(self.dry_preflight_path),
            "dry_preflight_sha256": self.dry_preflight_sha256,
            "dry_odom_certificate_path": str(self.dry_odom_certificate_path),
            "dry_odom_certificate_sha256": (
                self.dry_odom_certificate_sha256
            ),
            "dry_uncertainty_budget_path": str(self.dry_uncertainty_budget_path),
            "dry_uncertainty_budget_sha256": (
                self.dry_uncertainty_budget_sha256
            ),
            "retry_decision": {
                "retryable": self.retry_decision.retryable,
                "reason": self.retry_decision.reason,
            },
            "retry_scheduled": self.retry_scheduled,
        }

    def to_event(self) -> dict[str, object]:
        event = "preauthorization_initial_readiness_rejected"
        if self.status == "dry_run_ok":
            event = "preauthorization_initial_readiness_passed"
        elif self.retry_scheduled:
            event = "preauthorization_initial_readiness_retry_scheduled"
        return {
            "schema_version": 1,
            "event": event,
            "failure_phase": INITIAL_READINESS_PHASE,
            **self._fields(),
            **_SAFETY_FIELDS,
        }


@dataclass(frozen=True)
class InitialReadinessResult:
    ready: bool
    reason: str
    run_id_prefix: str
    maximum_retry_count: int
    sealed_route: SealedRoutePaths
    attempts: tuple[InitialReadinessAttempt, ...]

    def __post_init__(self) -> None:
        if type(self.ready) is not bool:
            raise ValueError("ready must be boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty text")
        if not isinstance(self.run_id_prefix, str) or not _RUN_TOKEN.fullmatch(
            self.run_id_prefix
        ):
            raise ValueError("run_id_prefix must be a path-safe token")
        if type(self.maximum_retry_count) is not int or self.maximum_retry_count < 0:
            raise ValueError("maximum_retry_count must be a non-negative integer")
        if not isinstance(self.sealed_route, SealedRoutePaths):
            raise ValueError("sealed_route must be SealedRoutePaths")
        if type(self.attempts) is not tuple or not self.attempts:
            raise ValueError("attempts must be a non-empty tuple")
        if len(self.attempts) > self.maximum_retry_count + 1:
            raise ValueError("attempt count exceeds the configured retry budget")
        for index, attempt in enumerate(self.attempts):
            if not isinstance(attempt, InitialReadinessAttempt):
                raise ValueError("attempts must contain InitialReadinessAttempt")
            if attempt.attempt_index != index:
                raise ValueError("attempt indices must be contiguous from zero")
            if attempt.maximum_retry_count != self.maximum_retry_count:
                raise ValueError("attempt retry budget is inconsistent")
            if attempt.run_id != f"{self.run_id_prefix}_{index:03d}":
                raise ValueError("attempt run_id is inconsistent with its prefix")
            if attempt.retry_scheduled != (index < len(self.attempts) - 1):
                raise ValueError("attempt retry scheduling is inconsistent")
            if attempt.status == "dry_run_ok" and index != len(self.attempts) - 1:
                raise ValueError("dry_run_ok must be the final attempt")
        final_ready = self.attempts[-1].status == "dry_run_ok"
        if self.ready != final_ready:
            raise ValueError("ready must match the final dry-run status")

    @property
    def final_attempt(self) -> InitialReadinessAttempt:
        return self.attempts[-1]

    def to_events(self) -> tuple[dict[str, object], ...]:
        return tuple(attempt.to_event() for attempt in self.attempts)

    def to_evidence(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "event": "preauthorization_initial_readiness_result",
            "failure_phase": INITIAL_READINESS_PHASE,
            "ready": self.ready,
            "readiness_status": "ready" if self.ready else "rejected",
            "readiness_reason": self.reason,
            "run_id_prefix": self.run_id_prefix,
            "attempt_count": len(self.attempts),
            **self.final_attempt._fields(),
            "sealed_route": self.sealed_route.to_mapping(),
            "attempts": [attempt.to_event() for attempt in self.attempts],
            **_SAFETY_FIELDS,
        }

    def to_failure_fields(self) -> dict[str, object]:
        if self.ready:
            raise ValueError("ready initial-readiness result is not a failure")
        final = self.final_attempt
        return {
            "failure_phase": INITIAL_READINESS_PHASE,
            "initial_readiness_attempt_count": len(self.attempts),
            "initial_readiness_maximum_retry_count": self.maximum_retry_count,
            "initial_readiness_reason": self.reason,
            "initial_readiness_last_run_id": final.run_id,
            "initial_readiness_last_status": final.status,
            "initial_readiness_last_reason": final.reason,
            "initial_readiness_last_details": _thaw(final.details),
            "initial_readiness_semantic_log_path": str(final.semantic_log_path),
            "initial_readiness_semantic_log_sha256": final.semantic_log_sha256,
            "initial_readiness_dry_preflight_path": str(final.dry_preflight_path),
            "initial_readiness_dry_preflight_sha256": final.dry_preflight_sha256,
            "initial_readiness_dry_odom_certificate_path": str(
                final.dry_odom_certificate_path
            ),
            "initial_readiness_dry_odom_certificate_sha256": (
                final.dry_odom_certificate_sha256
            ),
            "initial_readiness_dry_uncertainty_budget_path": str(
                final.dry_uncertainty_budget_path
            ),
            "initial_readiness_dry_uncertainty_budget_sha256": (
                final.dry_uncertainty_budget_sha256
            ),
            **_SAFETY_FIELDS,
        }


class InitialReadinessContractError(RuntimeError):
    """Raised when dry-run output cannot prove the no-motion contract."""

    def __init__(
        self,
        reason_code: str,
        *,
        request: InitialReadinessDryRequest,
        attempts: tuple[InitialReadinessAttempt, ...],
        reported_motion_published: bool | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.request = request
        self.attempts = attempts
        self.reported_motion_published = reported_motion_published
        self.cause = cause
        super().__init__(
            f"initial readiness contract failed for {request.run_id}: "
            f"{reason_code}"
        )

    def to_failure_fields(self) -> dict[str, object]:
        return {
            "failure_phase": INITIAL_READINESS_PHASE,
            "typed_run_requested": False,
            "operator_input_requested": False,
            "initial_readiness_attempt_count": len(self.attempts),
            "initial_readiness_reason": self.reason_code,
            "initial_readiness_last_run_id": self.request.run_id,
            "initial_readiness_semantic_log_path": str(self.request.semantic_log_path),
            "initial_readiness_dry_preflight_path": str(
                self.request.dry_preflight_path
            ),
            "initial_readiness_dry_odom_certificate_path": str(
                self.request.dry_odom_certificate_path
            ),
            "initial_readiness_dry_uncertainty_budget_path": str(
                self.request.dry_uncertainty_budget_path
            ),
            "motion_authorized": False,
            "motion_published": self.reported_motion_published,
            "permit_issued": False,
            "reusable_as_motion_permit": False,
            "route_limits_unchanged": True,
        }


class InitialReadinessRejected(RuntimeError):
    """Structured preauthorization stop backed by persisted readiness evidence."""

    def __init__(
        self,
        result: InitialReadinessResult,
        *,
        evidence_path: Path,
        evidence_sha256: str,
    ) -> None:
        if not isinstance(result, InitialReadinessResult) or result.ready:
            raise ValueError("InitialReadinessRejected requires a failed result")
        self.result = result
        self.evidence_path = _path(
            evidence_path,
            "initial readiness evidence path",
            ".json",
        )
        if not isinstance(evidence_sha256, str) or not _SHA256.fullmatch(
            evidence_sha256
        ):
            raise ValueError("initial readiness evidence sha256 is invalid")
        self.evidence_sha256 = evidence_sha256
        super().__init__(
            "first-route readiness failed before typed RUN: "
            f"{result.reason}: {result.final_attempt.reason}"
        )

    def to_failure_fields(self) -> dict[str, object]:
        return {
            **self.result.to_failure_fields(),
            "initial_readiness_json": str(self.evidence_path),
            "initial_readiness_sha256": self.evidence_sha256,
        }


def _request(
    sealed_route: SealedRoutePaths,
    session_root: Path,
    prefix: str,
    attempt_index: int,
) -> InitialReadinessDryRequest:
    run_id = f"{prefix}_{attempt_index:03d}"
    odom_root = session_root / "odom_execution"
    return InitialReadinessDryRequest(
        sealed_route=sealed_route,
        run_id=run_id,
        attempt_index=attempt_index,
        semantic_log_path=session_root / "run_events" / f"{run_id}.jsonl",
        dry_preflight_path=session_root / "preflight" / f"{run_id}_dry.json",
        dry_odom_certificate_path=odom_root / f"{run_id}_dry_certificate.json",
        dry_uncertainty_budget_path=(
            odom_root / f"{run_id}_dry_uncertainty_budget.json"
        ),
    )


def _contract_error(
    reason: str,
    request: InitialReadinessDryRequest,
    attempts: tuple[InitialReadinessAttempt, ...],
    *,
    motion: bool | None = None,
    cause: BaseException | None = None,
) -> InitialReadinessContractError:
    return InitialReadinessContractError(
        reason,
        request=request,
        attempts=attempts,
        reported_motion_published=motion,
        cause=cause,
    )


def _validated_attempt(
    outcome: object,
    request: InitialReadinessDryRequest,
    maximum_retries: int,
    attempts: tuple[InitialReadinessAttempt, ...],
) -> InitialReadinessAttempt:
    names = (
        "run_id",
        "status",
        "stop_reason",
        "stop_details",
        "motion_published",
        "returncode",
        "semantic_log_path",
    )
    try:
        fields = {name: getattr(outcome, name) for name in names}
    except (AttributeError, TypeError) as exc:
        raise _contract_error("outcome_missing_field", request, attempts, cause=exc)

    run_id = fields["run_id"]
    status = fields["status"]
    reason = fields["stop_reason"]
    details = fields["stop_details"]
    motion = fields["motion_published"]
    returncode = fields["returncode"]
    if run_id != request.run_id:
        raise _contract_error("outcome_run_id_mismatch", request, attempts)
    if status not in {"dry_run_ok", "preflight_failed", "stopped"}:
        raise _contract_error("outcome_status_invalid", request, attempts)
    if not isinstance(reason, str) or not isinstance(details, Mapping):
        raise _contract_error("outcome_stop_evidence_malformed", request, attempts)
    if type(motion) is not bool:
        raise _contract_error("outcome_motion_published_not_boolean", request, attempts)
    if motion:
        raise _contract_error(
            "dry_runner_reported_motion_published", request, attempts, motion=True
        )
    if type(returncode) is not int or (status == "dry_run_ok") != (returncode == 0):
        raise _contract_error("outcome_status_returncode_mismatch", request, attempts)
    if (status == "dry_run_ok" and (reason or details)) or (
        status != "dry_run_ok" and not reason
    ):
        raise _contract_error("outcome_stop_evidence_inconsistent", request, attempts)
    try:
        semantic_log = _path(fields["semantic_log_path"], "semantic_log_path", ".jsonl")
        frozen_details = _freeze(details)
    except ValueError as exc:
        raise _contract_error(
            "outcome_evidence_malformed",
            request,
            attempts,
            cause=exc,
        )
    if semantic_log != request.semantic_log_path:
        raise _contract_error("outcome_semantic_log_path_mismatch", request, attempts)
    certificate = getattr(outcome, "odom_execution_certificate_path", None)
    if certificate is not None:
        try:
            certificate = _path(certificate, "odom_execution_certificate_path", ".json")
        except ValueError as exc:
            raise _contract_error(
                "outcome_certificate_path_invalid",
                request,
                attempts,
                cause=exc,
            )
        if certificate != request.dry_odom_certificate_path:
            raise _contract_error(
                "outcome_certificate_path_mismatch",
                request,
                attempts,
            )

    semantic_log_sha256 = ""
    dry_preflight_sha256 = ""
    dry_odom_certificate_sha256 = ""
    dry_uncertainty_budget_sha256 = ""
    if status == "dry_run_ok":
        dry_preflight = getattr(outcome, "dry_preflight_path", None)
        dry_budget = getattr(outcome, "dry_uncertainty_budget_path", None)
        if certificate is None:
            raise _contract_error(
                "outcome_certificate_path_missing",
                request,
                attempts,
            )
        if dry_preflight != request.dry_preflight_path:
            raise _contract_error(
                "outcome_preflight_path_mismatch",
                request,
                attempts,
            )
        if dry_budget != request.dry_uncertainty_budget_path:
            raise _contract_error(
                "outcome_uncertainty_budget_path_mismatch",
                request,
                attempts,
            )
        try:
            parse_dry_run_outcome(
                semantic_log,
                run_id=run_id,
                returncode=returncode,
                start_offset=0,
            )
            semantic_log_sha256 = _normal_file_sha256(
                semantic_log,
                "semantic_log_path",
            )
            dry_preflight_sha256 = _normal_file_sha256(
                request.dry_preflight_path,
                "dry_preflight_path",
            )
            dry_odom_certificate_sha256 = _normal_file_sha256(
                request.dry_odom_certificate_path,
                "dry_odom_certificate_path",
            )
            dry_uncertainty_budget_sha256 = _normal_file_sha256(
                request.dry_uncertainty_budget_path,
                "dry_uncertainty_budget_path",
            )
        except (RuntimeError, ValueError) as exc:
            raise _contract_error(
                "dry_success_evidence_invalid",
                request,
                attempts,
                cause=exc,
            )

    decision = evaluate_localization_readiness_retry(
        status=status,
        stop_reason=reason,
        stop_details=details,
        motion_published=False,
    )
    return InitialReadinessAttempt(
        attempt_index=request.attempt_index,
        maximum_retry_count=maximum_retries,
        run_id=run_id,
        status=status,
        reason=reason,
        details=frozen_details,  # type: ignore[arg-type]
        returncode=returncode,
        semantic_log_path=semantic_log,
        semantic_log_sha256=semantic_log_sha256,
        dry_preflight_path=request.dry_preflight_path,
        dry_preflight_sha256=dry_preflight_sha256,
        dry_odom_certificate_path=request.dry_odom_certificate_path,
        dry_odom_certificate_sha256=dry_odom_certificate_sha256,
        dry_uncertainty_budget_path=request.dry_uncertainty_budget_path,
        dry_uncertainty_budget_sha256=dry_uncertainty_budget_sha256,
        retry_decision=decision,
        retry_scheduled=False,
    )


def run_initial_readiness(
    *,
    sealed_route: SealedRoutePaths,
    session_root: Path,
    run_id_prefix: str,
    maximum_retries: int,
    dry_runner: Callable[[InitialReadinessDryRequest], object],
) -> InitialReadinessResult:
    """Run a bounded sealed-route rehearsal before any typed RUN prompt."""

    if not isinstance(sealed_route, SealedRoutePaths):
        raise ValueError("sealed_route must be SealedRoutePaths")
    session_root = _path(session_root, "session_root")
    if not isinstance(run_id_prefix, str) or not _RUN_TOKEN.fullmatch(run_id_prefix):
        raise ValueError("run_id_prefix must be a non-empty path-safe token")
    if type(maximum_retries) is not int or maximum_retries < 0:
        raise ValueError("maximum_retries must be a non-negative integer")
    if not callable(dry_runner):
        raise ValueError("dry_runner must be callable")

    attempts: tuple[InitialReadinessAttempt, ...] = ()
    for index in range(maximum_retries + 1):
        request = _request(sealed_route, session_root, run_id_prefix, index)
        try:
            outcome = dry_runner(request)
        except InitialReadinessContractError:
            raise
        except Exception as exc:
            raise _contract_error("dry_runner_raised", request, attempts, cause=exc)
        attempt = _validated_attempt(outcome, request, maximum_retries, attempts)
        if attempt.status == "dry_run_ok":
            return InitialReadinessResult(
                True,
                "sealed_route_dry_readiness_passed",
                run_id_prefix,
                maximum_retries,
                sealed_route,
                attempts + (attempt,),
            )
        if not attempt.retry_decision.retryable:
            return InitialReadinessResult(
                False,
                "nonretryable_dry_outcome:" + attempt.retry_decision.reason,
                run_id_prefix,
                maximum_retries,
                sealed_route,
                attempts + (attempt,),
            )
        if index == maximum_retries:
            return InitialReadinessResult(
                False,
                "localization_readiness_retry_budget_exhausted",
                run_id_prefix,
                maximum_retries,
                sealed_route,
                attempts + (attempt,),
            )
        attempts += (replace(attempt, retry_scheduled=True),)
    raise AssertionError("bounded initial-readiness loop did not terminate")


__all__ = [
    "INITIAL_READINESS_PHASE",
    "InitialReadinessAttempt",
    "InitialReadinessContractError",
    "InitialReadinessDryRequest",
    "InitialReadinessRejected",
    "InitialReadinessResult",
    "SealedRoutePaths",
    "run_initial_readiness",
]
