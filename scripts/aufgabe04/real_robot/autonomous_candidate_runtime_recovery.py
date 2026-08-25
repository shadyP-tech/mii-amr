"""Pure, bounded post-motion localization recovery for candidate routines.

This coordinator starts from an already-produced :class:`MotionLegOutcome`.
It deliberately does not run the initial candidate request, which lets it sit
after the existing startup-recovery coordinator without merging their retry
budgets.  Completed and no-motion outcomes remain owned by the caller.  Only
the exact, persisted ``FORCE_ZERO_RESEAL`` contract may enter this loop.

Live localization, same-routine replanning, replacement motion, permit
issuance, and event persistence are injected effects.  Importing this module
therefore cannot connect to ROS, launch a subprocess, prompt an operator, or
authorize motion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
import time
from typing import Callable, Generic, TypeVar

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.runtime_localization_reseal import (
    RuntimeLocalizationResealDecision,
    evaluate_runtime_localization_reseal,
    evaluate_runtime_localization_reseal_budget,
)
from scripts.aufgabe04.real_robot.autonomous_candidate_startup_recovery import (
    CandidateRoutineIdentity,
)
from scripts.aufgabe04.real_robot.autonomous_candidate_recovery_failure import (
    issued_motion_permit_kinds,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PERMIT_FIELDS = (
    (
        "routine_mission_leg",
        "mission_leg_motion_permit_path",
        "mission_leg_motion_permit_sha256",
    ),
    (
        "startup_reseal",
        "startup_reseal_motion_permit_path",
        "startup_reseal_motion_permit_sha256",
    ),
)


@dataclass(frozen=True)
class CandidateRuntimeRecoveryConfig:
    """Immutable runtime-reseal bounds for one committed candidate routine."""

    initial_identity: CandidateRoutineIdentity
    recovery_root: Path
    event_log_path: Path
    max_runtime_reseals: int

    def __post_init__(self) -> None:
        if not isinstance(self.initial_identity, CandidateRoutineIdentity):
            raise TypeError("initial_identity must be CandidateRoutineIdentity")
        if (
            type(self.max_runtime_reseals) is not int
            or self.max_runtime_reseals < 0
        ):
            raise ValueError("max_runtime_reseals must be non-negative")
        for field_name, value in (
            ("recovery_root", self.recovery_root),
            ("event_log_path", self.event_log_path),
        ):
            try:
                Path(value)
            except TypeError as exc:
                raise TypeError(f"{field_name} must be a filesystem path") from exc


@dataclass(frozen=True)
class CandidateRuntimeRecoveryAttempt:
    """Exact immutable handoff for one runtime-localization replacement."""

    identity: CandidateRoutineIdentity
    reseal_index: int
    rejected_outcome: MotionLegOutcome
    runtime_localization_decision: RuntimeLocalizationResealDecision
    fresh_start_pose: Pose2D
    attempt_root: Path
    fresh_localization_evidence_path: Path
    source_root: Path


RequestT = TypeVar("RequestT")
EventSink = Callable[[Path, dict[str, object]], None]


@dataclass(frozen=True)
class CandidateRuntimeRecoveryEffects(Generic[RequestT]):
    """Injected ROS/live effects used by the pure recovery state machine."""

    admit_fresh_stationary_localization: Callable[[Path], Pose2D]
    replan_same_routine: Callable[[CandidateRuntimeRecoveryAttempt], RequestT]
    describe_request: Callable[[RequestT], CandidateRoutineIdentity]
    run_replacement: Callable[
        [RequestT, CandidateRuntimeRecoveryAttempt], MotionLegOutcome
    ]
    event_sink: EventSink
    clock: Callable[[], float] = time.time


@dataclass(frozen=True)
class CandidateRuntimeRejectedOutcome:
    """Structured fail-closed evidence for one rejected child outcome."""

    policy_reason: str
    decision_reason: str
    run_id: str
    status: str
    stop_reason: str
    stop_details: dict[str, object]
    motion_published: object

    @classmethod
    def from_outcome(
        cls,
        outcome: MotionLegOutcome,
        *,
        policy_reason: str,
        decision_reason: str = "",
    ) -> "CandidateRuntimeRejectedOutcome":
        details = (
            dict(outcome.stop_details)
            if isinstance(outcome.stop_details, Mapping)
            else {"malformed_stop_details": repr(outcome.stop_details)}
        )
        return cls(
            policy_reason=policy_reason,
            decision_reason=decision_reason,
            run_id=outcome.run_id,
            status=outcome.status,
            stop_reason=outcome.stop_reason,
            stop_details=details,
            motion_published=outcome.motion_published,
        )

    def to_event_fields(self) -> dict[str, object]:
        return {
            "reason": self.policy_reason,
            "runtime_localization_decision_reason": self.decision_reason,
            "observed_run_id": self.run_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "stop_details": dict(self.stop_details),
            "motion_published": self.motion_published,
        }


class CandidateRuntimeRecoveryError(RuntimeError):
    """Terminal candidate runtime-recovery failure with reportable evidence."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        rejected_child: CandidateRuntimeRejectedOutcome | None = None,
    ) -> None:
        self.phase = phase
        self.rejected_child = rejected_child
        super().__init__(message)

    def to_failure_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "failure_phase": "candidate_runtime_localization_recovery",
            "candidate_runtime_localization_recovery_phase": self.phase,
            "motion_continues_authorized": False,
            "fail_closed": True,
        }
        if self.rejected_child is not None:
            child = self.rejected_child
            fields.update(
                {
                    **child.to_event_fields(),
                    "candidate_runtime_localization_recovery_rejection_reason": (
                        child.policy_reason
                    ),
                    "child_run_id": child.run_id,
                    "child_status": child.status,
                    "rejected_stop_reason": child.stop_reason,
                    "rejected_stop_details": dict(child.stop_details),
                }
            )
        return fields


def _runtime_identity(
    base: CandidateRoutineIdentity,
    reseal_index: int,
) -> CandidateRoutineIdentity:
    if type(reseal_index) is not int or reseal_index <= 0:
        raise ValueError("candidate runtime reseal index must be positive")
    return replace(
        base,
        run_id=(
            f"{base.run_id}_runtime_localization_reseal_{reseal_index:03d}"
        ),
    )


def _attempt_paths(
    config: CandidateRuntimeRecoveryConfig,
    reseal_index: int,
) -> tuple[Path, Path, Path]:
    root = Path(config.recovery_root).resolve(strict=False)
    attempt_root = root / f"runtime_localization_reseal_{reseal_index:03d}"
    return (
        attempt_root,
        attempt_root / "fresh_stationary_localization.json",
        attempt_root / "route_source",
    )


def _event_base(config: CandidateRuntimeRecoveryConfig) -> dict[str, object]:
    identity = config.initial_identity
    return {
        "schema_version": 1,
        "session_id": identity.session_id,
        "semantic_map_id": identity.semantic_map_id,
        "routine_kind": identity.routine_kind,
        "routine_index": identity.routine_index,
        "target_id": identity.target_id,
        "initial_run_id": identity.run_id,
        "maximum_runtime_localization_reseal_count": (
            config.max_runtime_reseals
        ),
        "motion_continues_authorized": False,
    }


def _emit(
    config: CandidateRuntimeRecoveryConfig,
    effects: CandidateRuntimeRecoveryEffects[RequestT],
    payload: dict[str, object],
) -> None:
    try:
        timestamp = effects.clock()
        effects.event_sink(
            Path(config.event_log_path),
            {
                **_event_base(config),
                "timestamp": timestamp,
                **payload,
            },
        )
    except CandidateRuntimeRecoveryError:
        raise
    except Exception as exc:
        raise CandidateRuntimeRecoveryError(
            f"candidate runtime recovery failed during event_sink: {exc}",
            phase="event_sink",
        ) from exc


def _callback_failure(
    config: CandidateRuntimeRecoveryConfig,
    effects: CandidateRuntimeRecoveryEffects[RequestT],
    *,
    phase: str,
    run_id: str,
    reseal_index: int,
    exc: Exception,
) -> CandidateRuntimeRecoveryError:
    _emit(
        config,
        effects,
        {
            "event": "candidate_runtime_localization_reseal_failed",
            "phase": phase,
            "run_id": run_id,
            "runtime_localization_reseal_index": reseal_index,
            "failure": str(exc),
            "fail_closed": True,
        },
    )
    return CandidateRuntimeRecoveryError(
        f"candidate runtime recovery failed during {phase}: {exc}",
        phase=phase,
    )


def _normal_file(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} is unavailable: {candidate}") from exc
    if resolved != candidate or not resolved.is_file():
        raise ValueError(f"{label} must be the exact canonical normal file")
    return resolved


def _existing_normal_file(path: Path, label: str) -> Path:
    """Resolve an existing normal file without prescribing parent aliases."""

    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} is unavailable: {candidate}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must resolve to a normal file")
    return resolved


def _normal_directory(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} is unavailable: {candidate}") from exc
    if resolved != candidate or not resolved.is_dir():
        raise ValueError(f"{label} must be the exact canonical directory")
    return resolved


def _ensure_new_attempt_root(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            f"refusing to reuse candidate runtime recovery attempt: {path}"
        )


def _validate_pose(pose: object) -> Pose2D:
    if not isinstance(pose, Pose2D):
        raise TypeError("fresh stationary localization must return Pose2D")
    values = (pose.x_m, pose.y_m, pose.yaw_rad)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("fresh stationary localization pose must be finite")
    return Pose2D(*(float(value) for value in values))


def _validate_request_identity(
    request: RequestT,
    *,
    expected: CandidateRoutineIdentity,
    effects: CandidateRuntimeRecoveryEffects[RequestT],
) -> None:
    observed = effects.describe_request(request)
    if not isinstance(observed, CandidateRoutineIdentity):
        raise TypeError("describe_request must return CandidateRoutineIdentity")
    if observed != expected:
        raise ValueError(
            "candidate runtime replacement changed the committed routine "
            f"identity: expected={expected!r}, observed={observed!r}"
        )


def _validate_runtime_permit_evidence(
    outcome: MotionLegOutcome,
) -> tuple[Path, str]:
    path = outcome.motion_authorization_permit_path
    digest = outcome.motion_authorization_permit_sha256
    if path is None or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(
            "runtime replacement outcome lacks complete one-use runtime "
            "localization permit evidence"
        )
    permit_kinds = issued_motion_permit_kinds(outcome)
    if permit_kinds != ("runtime_localization",):
        raise ValueError(
            "runtime replacement must report exactly one runtime "
            "localization permit and no other motion permit kind"
        )
    permit_path = _existing_normal_file(
        Path(path),
        "runtime localization permit",
    )
    return permit_path, digest


def _validate_source_motion_permit_evidence(
    outcome: MotionLegOutcome,
) -> tuple[str, Path, str]:
    issued: list[tuple[str, Path, str]] = []
    for kind, path_field, digest_field in _SOURCE_PERMIT_FIELDS:
        path = getattr(outcome, path_field)
        digest = getattr(outcome, digest_field)
        if path is None and not str(digest).strip():
            continue
        if (
            path is None
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError(
                f"source {kind} motion permit evidence is incomplete"
            )
        issued.append(
            (
                kind,
                _normal_file(Path(path), f"source {kind} motion permit"),
                digest,
            )
        )
    if len(issued) != 1:
        raise ValueError(
            "eligible source motion must evidence exactly one routine or "
            "startup one-use permit"
        )
    if outcome.motion_authorization_permit_path is not None or str(
        outcome.motion_authorization_permit_sha256
    ).strip():
        raise ValueError(
            "initial candidate runtime recovery cannot start from another "
            "runtime permit"
        )
    return issued[0]


def _validate_outcome_identity(
    outcome: MotionLegOutcome,
    *,
    expected_run_id: str,
) -> None:
    if outcome.run_id != expected_run_id:
        raise ValueError(
            "motion outcome run identity mismatch: "
            f"expected={expected_run_id}, observed={outcome.run_id}"
        )


def _validate_replacement_outcome_contract(outcome: MotionLegOutcome) -> None:
    if not isinstance(outcome.run_id, str) or not outcome.run_id.strip():
        raise TypeError("replacement outcome run_id must be non-empty text")
    if not isinstance(outcome.status, str) or not outcome.status.strip():
        raise TypeError("replacement outcome status must be non-empty text")
    if not isinstance(outcome.stop_reason, str):
        raise TypeError("replacement outcome stop_reason must be text")
    if not isinstance(outcome.stop_details, Mapping):
        raise TypeError("replacement outcome stop_details must be a mapping")
    if type(outcome.motion_published) is not bool:
        raise TypeError(
            "replacement outcome motion_published must be boolean"
        )
    if type(outcome.returncode) is not int:
        raise TypeError("replacement outcome returncode must be an integer")


def _reject_outcome(
    config: CandidateRuntimeRecoveryConfig,
    effects: CandidateRuntimeRecoveryEffects[RequestT],
    *,
    outcome: MotionLegOutcome,
    expected_run_id: str,
    reseal_index: int,
    reason: str,
    decision_reason: str = "",
) -> None:
    rejection = CandidateRuntimeRejectedOutcome.from_outcome(
        outcome,
        policy_reason=reason,
        decision_reason=decision_reason,
    )
    _emit(
        config,
        effects,
        {
            "event": "candidate_runtime_localization_reseal_rejected",
            "expected_run_id": expected_run_id,
            "runtime_localization_reseal_index": reseal_index,
            **rejection.to_event_fields(),
            "fail_closed": True,
        },
    )
    raise CandidateRuntimeRecoveryError(
        f"candidate runtime recovery rejected {outcome.run_id}: {reason}",
        phase="outcome_rejection",
        rejected_child=rejection,
    )


def _eligible_decision(
    outcome: MotionLegOutcome,
) -> RuntimeLocalizationResealDecision:
    return evaluate_runtime_localization_reseal(
        status=outcome.status,
        motion_published=outcome.motion_published,
        stop_details=outcome.stop_details,
    )


def execute_candidate_runtime_localization_recovery(
    initial_outcome: MotionLegOutcome,
    *,
    config: CandidateRuntimeRecoveryConfig,
    effects: CandidateRuntimeRecoveryEffects[RequestT],
) -> MotionLegOutcome:
    """Recover exact post-motion localization stops for one candidate routine.

    The caller retains completed and no-motion outcomes unchanged.  Once this
    coordinator launches a replacement, every child outcome must report a new
    runtime-localization permit and either complete or present another exact
    eligible ``FORCE_ZERO_RESEAL`` stop.  All other paths are terminal.
    """

    if not isinstance(initial_outcome, MotionLegOutcome):
        exc = TypeError("initial_outcome must be MotionLegOutcome")
        raise _callback_failure(
            config,
            effects,
            phase="initial_outcome_contract",
            run_id=config.initial_identity.run_id,
            reseal_index=0,
            exc=exc,
        ) from exc
    if initial_outcome.status == "completed":
        return initial_outcome
    if initial_outcome.motion_published is False:
        return initial_outcome
    try:
        _validate_outcome_identity(
            initial_outcome,
            expected_run_id=config.initial_identity.run_id,
        )
    except Exception as exc:
        raise _callback_failure(
            config,
            effects,
            phase="initial_outcome_identity",
            run_id=config.initial_identity.run_id,
            reseal_index=0,
            exc=exc,
        ) from exc
    if initial_outcome.motion_published is not True:
        _reject_outcome(
            config,
            effects,
            outcome=initial_outcome,
            expected_run_id=config.initial_identity.run_id,
            reseal_index=0,
            reason="initial outcome motion_published is not boolean",
        )

    outcome = initial_outcome
    expected_identity = config.initial_identity
    decision = _eligible_decision(outcome)
    if not decision.eligible:
        _reject_outcome(
            config,
            effects,
            outcome=outcome,
            expected_run_id=expected_identity.run_id,
            reseal_index=0,
            reason="outcome is not an eligible runtime localization reseal",
            decision_reason=decision.reason,
        )
    try:
        source_permit_kind, source_permit_path, source_permit_sha256 = (
            _validate_source_motion_permit_evidence(outcome)
        )
    except Exception as exc:
        _reject_outcome(
            config,
            effects,
            outcome=outcome,
            expected_run_id=expected_identity.run_id,
            reseal_index=0,
            reason=str(exc),
            decision_reason=decision.reason,
        )

    completed_reseal_count = 0
    used_permit_paths: set[Path] = set()
    used_permit_digests: set[str] = set()
    while True:
        budget = evaluate_runtime_localization_reseal_budget(
            completed_reseal_count=completed_reseal_count,
            maximum_reseal_count=config.max_runtime_reseals,
        )
        if not budget.allowed:
            rejection = CandidateRuntimeRejectedOutcome.from_outcome(
                outcome,
                policy_reason=budget.reason,
                decision_reason=decision.reason,
            )
            _emit(
                config,
                effects,
                {
                    "event": "candidate_runtime_localization_reseal_exhausted",
                    "completed_runtime_localization_reseal_count": (
                        completed_reseal_count
                    ),
                    "rejected_run_id": outcome.run_id,
                    "runtime_localization_reseal_decision": decision.to_evidence(),
                    "runtime_localization_reseal_budget": budget.to_evidence(),
                    **rejection.to_event_fields(),
                    "fail_closed": True,
                },
            )
            raise CandidateRuntimeRecoveryError(
                "candidate runtime localization reseal budget exhausted after "
                f"{completed_reseal_count} replacement attempt(s); last child "
                f"{outcome.run_id}: {outcome.stop_reason}",
                phase="budget_exhausted",
                rejected_child=rejection,
            )

        assert budget.next_reseal_index is not None
        reseal_index = budget.next_reseal_index
        replacement_identity = _runtime_identity(
            config.initial_identity,
            reseal_index,
        )
        attempt_root, evidence_path, source_root = _attempt_paths(
            config,
            reseal_index,
        )
        try:
            _ensure_new_attempt_root(attempt_root)
        except Exception as exc:
            raise _callback_failure(
                config,
                effects,
                phase="attempt_path_admission",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc

        _emit(
            config,
            effects,
            {
                "event": "candidate_runtime_localization_reseal_started",
                "runtime_localization_reseal_index": reseal_index,
                "rejected_run_id": outcome.run_id,
                "replacement_run_id": replacement_identity.run_id,
                "attempt_root": str(attempt_root),
                "fresh_localization_evidence_json": str(evidence_path),
                "replacement_source_root": str(source_root),
                "runtime_localization_reseal_decision": decision.to_evidence(),
                "runtime_localization_reseal_budget": budget.to_evidence(),
                "source_stop_requires_zero_cycle": True,
                "source_motion_permit_kind": source_permit_kind,
                "source_motion_permit_json": str(source_permit_path),
                "source_motion_permit_sha256": source_permit_sha256,
            },
        )
        try:
            fresh_pose = _validate_pose(
                effects.admit_fresh_stationary_localization(evidence_path)
            )
            _normal_file(evidence_path, "fresh stationary localization evidence")
        except Exception as exc:
            raise _callback_failure(
                config,
                effects,
                phase="stationary_localization_admission",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc

        attempt = CandidateRuntimeRecoveryAttempt(
            identity=replacement_identity,
            reseal_index=reseal_index,
            rejected_outcome=outcome,
            runtime_localization_decision=decision,
            fresh_start_pose=fresh_pose,
            attempt_root=attempt_root,
            fresh_localization_evidence_path=evidence_path,
            source_root=source_root,
        )
        _emit(
            config,
            effects,
            {
                "event": "candidate_runtime_localization_admitted",
                "runtime_localization_reseal_index": reseal_index,
                "rejected_run_id": outcome.run_id,
                "replacement_run_id": replacement_identity.run_id,
                "fresh_localization_evidence_json": str(evidence_path),
                "fresh_start_pose": {
                    "x_m": fresh_pose.x_m,
                    "y_m": fresh_pose.y_m,
                    "yaw_rad": fresh_pose.yaw_rad,
                },
            },
        )
        if source_root.exists() or source_root.is_symlink():
            exc = FileExistsError(
                "candidate runtime route source was created before replanning: "
                f"{source_root}"
            )
            raise _callback_failure(
                config,
                effects,
                phase="route_source_admission",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc
        try:
            replacement_request = effects.replan_same_routine(attempt)
            _normal_directory(source_root, "candidate runtime route source")
            _validate_request_identity(
                replacement_request,
                expected=replacement_identity,
                effects=effects,
            )
        except Exception as exc:
            raise _callback_failure(
                config,
                effects,
                phase="same_routine_replan",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc

        _emit(
            config,
            effects,
            {
                "event": "candidate_runtime_localization_route_replanned",
                "runtime_localization_reseal_index": reseal_index,
                "rejected_run_id": outcome.run_id,
                "replacement_run_id": replacement_identity.run_id,
                "replacement_source_root": str(source_root),
                "same_routine_identity_verified": True,
            },
        )
        try:
            replacement_outcome = effects.run_replacement(
                replacement_request,
                attempt,
            )
        except Exception as exc:
            raise _callback_failure(
                config,
                effects,
                phase="replacement_run",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc
        if not isinstance(replacement_outcome, MotionLegOutcome):
            exc = TypeError("replacement callback must return MotionLegOutcome")
            raise _callback_failure(
                config,
                effects,
                phase="replacement_outcome_contract",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc
        try:
            _validate_replacement_outcome_contract(replacement_outcome)
        except Exception as exc:
            raise _callback_failure(
                config,
                effects,
                phase="replacement_outcome_contract",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc
        try:
            _validate_outcome_identity(
                replacement_outcome,
                expected_run_id=replacement_identity.run_id,
            )
        except Exception as exc:
            _reject_outcome(
                config,
                effects,
                outcome=replacement_outcome,
                expected_run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                reason=str(exc),
            )
        try:
            permit_path, permit_digest = _validate_runtime_permit_evidence(
                replacement_outcome
            )
            if (
                permit_path in used_permit_paths
                or permit_digest in used_permit_digests
            ):
                raise ValueError(
                    "runtime replacement reused one-use permit evidence"
                )
        except Exception as exc:
            _reject_outcome(
                config,
                effects,
                outcome=replacement_outcome,
                expected_run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                reason=str(exc),
            )
        used_permit_paths.add(permit_path)
        used_permit_digests.add(permit_digest)
        _emit(
            config,
            effects,
            {
                "event": "candidate_runtime_localization_permit_evidenced",
                "runtime_localization_reseal_index": reseal_index,
                "run_id": replacement_outcome.run_id,
                "runtime_localization_motion_permit_json": str(permit_path),
                "runtime_localization_motion_permit_sha256": permit_digest,
                "one_use_runtime_permit_evidenced": True,
            },
        )

        completed_reseal_count = reseal_index
        if replacement_outcome.status == "completed":
            _emit(
                config,
                effects,
                {
                    "event": "candidate_runtime_localization_reseal_completed",
                    "run_id": replacement_outcome.run_id,
                    "completed_runtime_localization_reseal_count": (
                        completed_reseal_count
                    ),
                    "motion_published": replacement_outcome.motion_published,
                },
            )
            return replacement_outcome
        outcome = replacement_outcome
        expected_identity = replacement_identity
        source_permit_kind = "runtime_localization"
        source_permit_path = permit_path
        source_permit_sha256 = permit_digest
        decision = _eligible_decision(outcome)
        if not decision.eligible:
            _reject_outcome(
                config,
                effects,
                outcome=outcome,
                expected_run_id=expected_identity.run_id,
                reseal_index=completed_reseal_count,
                reason=(
                    "replacement outcome is not an eligible runtime "
                    "localization reseal"
                ),
                decision_reason=decision.reason,
            )


# Compact alias for callers that already describe the initial child outcome.
recover_candidate_runtime_localization = (
    execute_candidate_runtime_localization_recovery
)
execute_candidate_motion_with_runtime_recovery = (
    execute_candidate_runtime_localization_recovery
)


__all__ = [
    "CandidateRuntimeRecoveryAttempt",
    "CandidateRuntimeRecoveryConfig",
    "CandidateRuntimeRecoveryEffects",
    "CandidateRuntimeRecoveryError",
    "CandidateRuntimeRejectedOutcome",
    "execute_candidate_motion_with_runtime_recovery",
    "execute_candidate_runtime_localization_recovery",
    "recover_candidate_runtime_localization",
]
