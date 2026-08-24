"""Bounded, motion-free coordination for candidate startup recovery.

The coordinator in this module owns only the retry state machine.  Live pose
sampling, route planning, motion execution, permit issuance, and event storage
remain injected effects.  Consequently importing this module cannot connect
to ROS, launch a subprocess, publish velocity, or prompt an operator.

Only an exact startup rejection before motion may enter this recovery path.
A child can report the one-use permit it consumed before its live startup gate
failed; that permit is retained as evidence and never reused.  Every
replacement is bound to the original routine identity and receives
deterministic, one-use artifact paths.  Existing attempt paths are never reused.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
import time
from collections.abc import Mapping
from typing import Callable, Generic, TypeVar

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.prestart_localization_reseal import (
    evaluate_prestart_localization_reseal,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH,
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot.autonomous_candidate_recovery_failure import (
    CandidateStartupRecoveryError,
    RejectedChildFailure,
    issued_motion_permit_kinds,
)
from scripts.aufgabe04.real_robot.autonomous_coverage_replanning import (
    is_resealable_startup_mismatch,
)


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SUPPORTED_ROUTINE_KINDS = frozenset(
    {"candidate_preapproach", "opposite_face"}
)
@dataclass(frozen=True)
class CandidateRoutineIdentity:
    """Exact candidate routine identity carried by one motion request."""

    session_id: str
    semantic_map_id: str
    routine_kind: str
    routine_index: int
    target_id: str
    run_id: str

    def __post_init__(self) -> None:
        text_fields = {
            "session_id": self.session_id,
            "semantic_map_id": self.semantic_map_id,
            "routine_kind": self.routine_kind,
            "target_id": self.target_id,
            "run_id": self.run_id,
        }
        for field, value in text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"candidate routine {field} must be non-empty")
        if self.routine_kind not in _SUPPORTED_ROUTINE_KINDS:
            raise ValueError(
                "candidate startup recovery supports only candidate_preapproach "
                "and opposite_face routines"
            )
        if type(self.routine_index) is not int or self.routine_index < 0:
            raise ValueError("candidate routine index must be non-negative")
        if _SAFE_RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("candidate routine run_id must be path-safe")

    def replacement(self, reseal_index: int) -> "CandidateRoutineIdentity":
        """Return the exact same routine with its deterministic retry run ID."""

        if type(reseal_index) is not int or reseal_index <= 0:
            raise ValueError("candidate startup reseal index must be positive")
        return replace(
            self,
            run_id=(
                f"{self.run_id}_startup_reseal_{reseal_index:03d}"
            ),
        )


@dataclass(frozen=True)
class CandidateStartupRecoveryConfig:
    """Immutable bounds and artifact roots for one candidate routine."""

    initial_identity: CandidateRoutineIdentity
    recovery_root: Path
    event_log_path: Path
    max_startup_reseals: int

    def __post_init__(self) -> None:
        if not isinstance(self.initial_identity, CandidateRoutineIdentity):
            raise TypeError("initial_identity must be CandidateRoutineIdentity")
        if (
            type(self.max_startup_reseals) is not int
            or self.max_startup_reseals < 0
        ):
            raise ValueError("max_startup_reseals must be non-negative")


@dataclass(frozen=True)
class CandidateStartupRecoveryAttempt:
    """Exact immutable handoff shared by replanning and replacement motion."""

    identity: CandidateRoutineIdentity
    reseal_index: int
    recovery_source_kind: str
    rejected_outcome: MotionLegOutcome
    fresh_start_pose: Pose2D
    attempt_root: Path
    fresh_localization_evidence_path: Path
    source_root: Path


RequestT = TypeVar("RequestT")
EventSink = Callable[[Path, dict[str, object]], None]


@dataclass(frozen=True)
class CandidateStartupRecoveryEffects(Generic[RequestT]):
    """Injected live and artifact effects used by the ROS-free coordinator.

    The replacement callback is intentionally distinct from the initial
    routine callback.  It can therefore bind a dedicated startup-reseal permit
    without teaching this state machine about a concrete permit schema.
    """

    run_initial: Callable[[RequestT], MotionLegOutcome]
    run_replacement: Callable[
        [RequestT, CandidateStartupRecoveryAttempt], MotionLegOutcome
    ]
    admit_fresh_stationary_localization: Callable[[Path], Pose2D]
    replan_same_routine: Callable[
        [CandidateStartupRecoveryAttempt], RequestT
    ]
    describe_request: Callable[[RequestT], CandidateRoutineIdentity]
    event_sink: EventSink
    clock: Callable[[], float] = time.time


def _attempt_paths(
    config: CandidateStartupRecoveryConfig,
    reseal_index: int,
) -> tuple[Path, Path, Path]:
    root = Path(config.recovery_root).resolve(strict=False)
    attempt_root = root / f"startup_reseal_{reseal_index:03d}"
    return (
        attempt_root,
        attempt_root / "fresh_stationary_localization.json",
        attempt_root / "route_source",
    )


def _event_base(
    config: CandidateStartupRecoveryConfig,
) -> dict[str, object]:
    identity = config.initial_identity
    return {
        "schema_version": 1,
        "session_id": identity.session_id,
        "semantic_map_id": identity.semantic_map_id,
        "routine_kind": identity.routine_kind,
        "routine_index": identity.routine_index,
        "target_id": identity.target_id,
        "initial_run_id": identity.run_id,
        "maximum_startup_reseal_count": config.max_startup_reseals,
        "motion_continues_authorized": False,
    }


def _emit(
    config: CandidateStartupRecoveryConfig,
    effects: CandidateStartupRecoveryEffects[RequestT],
    payload: dict[str, object],
) -> None:
    effects.event_sink(
        Path(config.event_log_path),
        {
            **_event_base(config),
            "timestamp": effects.clock(),
            **payload,
        },
    )


def _fail_callback(
    config: CandidateStartupRecoveryConfig,
    effects: CandidateStartupRecoveryEffects[RequestT],
    *,
    phase: str,
    run_id: str,
    reseal_index: int,
    exc: Exception,
) -> CandidateStartupRecoveryError:
    _emit(
        config,
        effects,
        {
            "event": "candidate_startup_recovery_failed",
            "phase": phase,
            "run_id": run_id,
            "startup_reseal_index": reseal_index,
            "failure": str(exc),
            "fail_closed": True,
        },
    )
    return CandidateStartupRecoveryError(
        f"candidate startup recovery failed during {phase}: {exc}",
        phase=phase,
    )


def _validate_request_identity(
    request: RequestT,
    *,
    expected: CandidateRoutineIdentity,
    effects: CandidateStartupRecoveryEffects[RequestT],
) -> None:
    observed = effects.describe_request(request)
    if not isinstance(observed, CandidateRoutineIdentity):
        raise TypeError(
            "describe_request must return CandidateRoutineIdentity"
        )
    if observed != expected:
        raise ValueError(
            "candidate replacement changed the committed routine identity: "
            f"expected={expected!r}, observed={observed!r}"
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


def _ensure_new_attempt_root(attempt_root: Path) -> None:
    if attempt_root.exists() or attempt_root.is_symlink():
        raise FileExistsError(
            f"refusing to reuse candidate startup recovery attempt: {attempt_root}"
        )


def _ensure_evidence_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "fresh stationary localization callback did not create the exact "
            f"regular evidence file: {path}"
        )


def _ensure_source_root(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(
            "same-routine replan callback did not create the exact source "
            f"directory: {path}"
        )


def _reject_outcome(
    config: CandidateStartupRecoveryConfig,
    effects: CandidateStartupRecoveryEffects[RequestT],
    *,
    outcome: MotionLegOutcome,
    expected_run_id: str,
    reseal_index: int,
    reason: str,
    preserve_child_reason: bool = False,
) -> None:
    rejection = RejectedChildFailure.from_outcome(
        outcome,
        policy_reason=reason,
        preserve_child_reason=preserve_child_reason,
    )
    _emit(
        config,
        effects,
        {
            "event": "candidate_startup_recovery_rejected",
            "expected_run_id": expected_run_id,
            "startup_reseal_index": reseal_index,
            **rejection.to_event_fields(),
            "fail_closed": True,
        },
    )
    raise CandidateStartupRecoveryError(
        rejection.rejection_message(),
        phase="outcome_rejection",
        rejected_child=rejection,
    )


def _recovery_source_kind(outcome: MotionLegOutcome) -> str | None:
    """Classify the two authorized no-motion startup recovery sources."""

    if is_resealable_startup_mismatch(outcome):
        return STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
    decision = evaluate_prestart_localization_reseal(
        status=outcome.status,
        motion_published=outcome.motion_published,
        stop_details=outcome.stop_details,
    )
    if not (
        decision.eligible
        and isinstance(outcome.stop_details, Mapping)
        and outcome.stop_reason == outcome.stop_details.get("reason")
        and decision.motion_published is False
        and decision.requires_fresh_localization
        and decision.requires_new_route_certificate
        and not decision.automatic_motion_authorized
    ):
        return None
    return STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY


def execute_candidate_motion_with_startup_recovery(
    initial_request: RequestT,
    *,
    config: CandidateStartupRecoveryConfig,
    effects: CandidateStartupRecoveryEffects[RequestT],
) -> MotionLegOutcome:
    """Run one candidate routine with bounded, exact-identity startup retries.

    A completed initial or replacement outcome is returned.  Every other
    terminal path raises :class:`CandidateStartupRecoveryError`; no callback
    is invoked for an N+1 attempt after the configured budget is exhausted.
    """

    try:
        _validate_request_identity(
            initial_request,
            expected=config.initial_identity,
            effects=effects,
        )
    except Exception as exc:
        raise _fail_callback(
            config,
            effects,
            phase="initial_request_identity",
            run_id=config.initial_identity.run_id,
            reseal_index=0,
            exc=exc,
        ) from exc

    try:
        outcome = effects.run_initial(initial_request)
    except Exception as exc:
        raise _fail_callback(
            config,
            effects,
            phase="initial_run",
            run_id=config.initial_identity.run_id,
            reseal_index=0,
            exc=exc,
        ) from exc

    expected_identity = config.initial_identity
    completed_reseal_count = 0
    while True:
        if not isinstance(outcome, MotionLegOutcome):
            exc = TypeError("motion callback must return MotionLegOutcome")
            raise _fail_callback(
                config,
                effects,
                phase="motion_outcome_contract",
                run_id=expected_identity.run_id,
                reseal_index=completed_reseal_count,
                exc=exc,
            ) from exc
        if outcome.run_id != expected_identity.run_id:
            _reject_outcome(
                config,
                effects,
                outcome=outcome,
                expected_run_id=expected_identity.run_id,
                reseal_index=completed_reseal_count,
                reason="motion outcome run identity mismatch",
            )
        if outcome.status == "completed":
            if completed_reseal_count:
                _emit(
                    config,
                    effects,
                    {
                        "event": "candidate_startup_recovery_completed",
                        "run_id": outcome.run_id,
                        "completed_startup_reseal_count": (
                            completed_reseal_count
                        ),
                        "motion_published": outcome.motion_published,
                    },
                )
            return outcome

        issued_permits = issued_motion_permit_kinds(outcome)
        if outcome.motion_published:
            _reject_outcome(
                config,
                effects,
                outcome=outcome,
                expected_run_id=expected_identity.run_id,
                reseal_index=completed_reseal_count,
                reason="rejected candidate run published motion",
                preserve_child_reason=True,
            )
        recovery_source_kind = _recovery_source_kind(outcome)
        if recovery_source_kind is None:
            _reject_outcome(
                config,
                effects,
                outcome=outcome,
                expected_run_id=expected_identity.run_id,
                reseal_index=completed_reseal_count,
                reason="outcome is not an eligible startup-segment mismatch",
            )
        if completed_reseal_count >= config.max_startup_reseals:
            rejection = RejectedChildFailure.from_outcome(
                outcome,
                policy_reason="candidate startup reseal budget exhausted",
                preserve_child_reason=True,
            )
            _emit(
                config,
                effects,
                {
                    "event": "candidate_startup_recovery_exhausted",
                    "rejected_run_id": outcome.run_id,
                    "completed_startup_reseal_count": completed_reseal_count,
                    **rejection.to_event_fields(),
                    "motion_published": False,
                    "fail_closed": True,
                },
            )
            raise CandidateStartupRecoveryError(
                "candidate startup reseal budget exhausted after "
                f"{completed_reseal_count} replacement attempt(s); last child "
                f"{outcome.run_id}: {rejection.reported_reason}",
                phase="budget_exhausted",
                rejected_child=rejection,
            )

        reseal_index = completed_reseal_count + 1
        replacement_identity = config.initial_identity.replacement(reseal_index)
        attempt_root, evidence_path, source_root = _attempt_paths(
            config,
            reseal_index,
        )
        try:
            _ensure_new_attempt_root(attempt_root)
        except Exception as exc:
            raise _fail_callback(
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
                "event": "candidate_startup_recovery_started",
                "startup_reseal_index": reseal_index,
                "rejected_run_id": outcome.run_id,
                "replacement_run_id": replacement_identity.run_id,
                "attempt_root": str(attempt_root),
                "fresh_localization_evidence_json": str(evidence_path),
                "replacement_source_root": str(source_root),
                "recovery_source_kind": recovery_source_kind,
                "source_rejection_stop_reason": outcome.stop_reason,
                "source_rejection_stop_details": dict(outcome.stop_details),
                "source_rejection_published_motion": False,
                "source_rejection_issued_motion_permit": bool(issued_permits),
                "source_rejection_issued_motion_permit_kinds": list(
                    issued_permits
                ),
            },
        )
        try:
            fresh_pose = _validate_pose(
                effects.admit_fresh_stationary_localization(evidence_path)
            )
            _ensure_evidence_file(evidence_path)
        except Exception as exc:
            raise _fail_callback(
                config,
                effects,
                phase="stationary_localization_admission",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc

        attempt = CandidateStartupRecoveryAttempt(
            identity=replacement_identity,
            reseal_index=reseal_index,
            recovery_source_kind=recovery_source_kind,
            rejected_outcome=outcome,
            fresh_start_pose=fresh_pose,
            attempt_root=attempt_root,
            fresh_localization_evidence_path=evidence_path,
            source_root=source_root,
        )
        _emit(
            config,
            effects,
            {
                "event": "candidate_startup_localization_admitted",
                "startup_reseal_index": reseal_index,
                "rejected_run_id": outcome.run_id,
                "replacement_run_id": replacement_identity.run_id,
                "recovery_source_kind": recovery_source_kind,
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
                "candidate startup route source was created before replanning: "
                f"{source_root}"
            )
            raise _fail_callback(
                config,
                effects,
                phase="route_source_admission",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc
        try:
            replacement_request = effects.replan_same_routine(attempt)
            _ensure_source_root(source_root)
            _validate_request_identity(
                replacement_request,
                expected=replacement_identity,
                effects=effects,
            )
        except Exception as exc:
            raise _fail_callback(
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
                "event": "candidate_startup_route_replanned",
                "startup_reseal_index": reseal_index,
                "rejected_run_id": outcome.run_id,
                "replacement_run_id": replacement_identity.run_id,
                "recovery_source_kind": recovery_source_kind,
                "fresh_localization_evidence_json": str(evidence_path),
                "replacement_source_root": str(source_root),
                "same_routine_identity_verified": True,
            },
        )
        try:
            outcome = effects.run_replacement(replacement_request, attempt)
        except Exception as exc:
            raise _fail_callback(
                config,
                effects,
                phase="replacement_run",
                run_id=replacement_identity.run_id,
                reseal_index=reseal_index,
                exc=exc,
            ) from exc
        expected_identity = replacement_identity
        completed_reseal_count = reseal_index


__all__ = [
    "CandidateRoutineIdentity",
    "CandidateStartupRecoveryAttempt",
    "CandidateStartupRecoveryConfig",
    "CandidateStartupRecoveryEffects",
    "CandidateStartupRecoveryError",
    "execute_candidate_motion_with_startup_recovery",
    "issued_motion_permit_kinds",
]
