"""Pure failure evidence for autonomous candidate startup recovery.

This module translates an already-completed child motion outcome into
JSON-ready rejection evidence.  It never retries motion, grants authority,
connects to ROS, launches a process, or writes an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
from typing import TYPE_CHECKING

from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)

if TYPE_CHECKING:
    from scripts.aufgabe04.real_robot.candidate.startup_recovery import CandidateRoutineIdentity


_PERMIT_FIELDS = (
    (
        "runtime_localization",
        "motion_authorization_permit_path",
        "motion_authorization_permit_sha256",
    ),
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


def issued_motion_permit_evidence(
    outcome: MotionLegOutcome,
) -> dict[str, dict[str, str | None]]:
    """Return JSON-ready evidence for every permit reported by a child."""

    issued: dict[str, dict[str, str | None]] = {}
    for kind, path_field, digest_field in _PERMIT_FIELDS:
        path = getattr(outcome, path_field)
        digest = getattr(outcome, digest_field)
        if path is not None or (isinstance(digest, str) and digest.strip()):
            issued[kind] = {
                "path": None if path is None else str(path),
                "sha256": digest if isinstance(digest, str) else "",
            }
    return issued


def issued_motion_permit_kinds(outcome: MotionLegOutcome) -> tuple[str, ...]:
    """Return every permit class evidenced by an outcome."""

    return tuple(issued_motion_permit_evidence(outcome))


@dataclass(frozen=True)
class RejectedChildFailure:
    """Structured evidence for a child rejected by recovery policy."""

    policy_reason: str
    reported_reason: str
    run_id: str
    status: str
    stop_reason: str
    stop_details: dict[str, object]
    motion_published: bool
    issued_motion_permit_kinds: tuple[str, ...]
    issued_motion_permits: dict[str, dict[str, str | None]]

    @classmethod
    def from_outcome(
        cls,
        outcome: MotionLegOutcome,
        *,
        policy_reason: str,
        preserve_child_reason: bool,
    ) -> "RejectedChildFailure":
        child_reason = outcome.stop_reason.strip()
        permit_evidence = issued_motion_permit_evidence(outcome)
        reported_reason = (
            child_reason
            if preserve_child_reason and child_reason
            else policy_reason
        )
        return cls(
            policy_reason=policy_reason,
            reported_reason=reported_reason,
            run_id=outcome.run_id,
            status=outcome.status,
            stop_reason=outcome.stop_reason,
            stop_details=dict(outcome.stop_details),
            motion_published=outcome.motion_published,
            issued_motion_permit_kinds=tuple(permit_evidence),
            issued_motion_permits=permit_evidence,
        )

    def rejection_message(
        self,
        *,
        prefix: str = "candidate startup recovery rejected",
    ) -> str:
        """Format a child-first terminal error without hiding policy context."""

        policy_suffix = (
            ""
            if self.reported_reason == self.policy_reason
            else f"; fail-closed policy: {self.policy_reason}"
        )
        diagnostic = self._tf_diagnostic_summary()
        return f"{prefix} {self.run_id}: {self.reported_reason}{diagnostic}{policy_suffix}"

    def _tf_diagnostic_summary(self) -> str:
        """Explain typed TF rejection without changing the child reason contract."""

        details = self.stop_details
        if details.get("source") != "tf_lookup":
            return ""
        fields = []
        reason = details.get("reason")
        if isinstance(reason, str) and reason:
            fields.append(reason)
        for label, value in (("age", details.get("age_sec")),
                             ("limit", details.get("max_age_sec"))):
            if type(value) in (int, float) and math.isfinite(value):
                fields.append(f"{label}={value:.3f}s")
        state = details.get("initial_tf_acquisition")
        if isinstance(state, dict):
            denial = state.get("denial_reason")
            if isinstance(denial, str) and denial:
                fields.append(denial)
            elapsed = state.get("elapsed_sec")
            maximum = state.get("maximum_startup_wait_sec")
            if all(type(value) in (int, float) and math.isfinite(value)
                   for value in (elapsed, maximum)):
                fields.append(f"startup={elapsed:.3f}/{maximum:.3f}s")
        return f" [{'; '.join(fields)}]" if fields else ""

    def to_event_fields(self) -> dict[str, object]:
        return {
            "reason": self.reported_reason,
            "rejection_policy_reason": self.policy_reason,
            "observed_run_id": self.run_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "stop_details": dict(self.stop_details),
            "rejected_stop_reason": self.stop_reason,
            "rejected_stop_details": dict(self.stop_details),
            "motion_published": self.motion_published,
            "issued_motion_permit_kinds": list(
                self.issued_motion_permit_kinds
            ),
            "issued_motion_permits": {
                kind: dict(evidence)
                for kind, evidence in self.issued_motion_permits.items()
            },
        }

    def to_failure_fields(self) -> dict[str, object]:
        return {
            "candidate_startup_recovery_rejection_reason": (
                self.policy_reason
            ),
            "child_run_id": self.run_id,
            "child_status": self.status,
            "stop_reason": self.stop_reason,
            "stop_details": dict(self.stop_details),
            "motion_published": self.motion_published,
            "issued_motion_permit_kinds": list(
                self.issued_motion_permit_kinds
            ),
            "issued_motion_permits": {
                kind: dict(evidence)
                for kind, evidence in self.issued_motion_permits.items()
            },
        }


class CandidateStartupRecoveryError(RuntimeError):
    """Fail-closed terminal error with mission-reporting evidence."""

    def __init__(
        self,
        message: str,
        *,
        phase: str = "coordinator",
        rejected_child: RejectedChildFailure | None = None,
    ) -> None:
        self.phase = phase
        self.rejected_child = rejected_child
        super().__init__(message)

    def to_failure_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "failure_phase": "candidate_startup_recovery",
            "candidate_startup_recovery_phase": self.phase,
            "motion_continues_authorized": False,
            "fail_closed": True,
        }
        if self.rejected_child is not None:
            fields.update(self.rejected_child.to_failure_fields())
        return fields


class CandidateStartupTargetUnavailableError(CandidateStartupRecoveryError):
    """A bound target deferral after every approach child stopped before motion.

    Only the startup coordinator creates this subtype, after validating the
    stopped child history and closing each child's one-use authority. It does
    not authorize a replacement or claim a successful camera observation.
    """

    def __init__(
        self,
        *,
        observation_error: CandidateObservationUnavailableError,
        initial_identity: CandidateRoutineIdentity,
        rejected_child: RejectedChildFailure,
        completed_startup_reseal_count: int,
        startup_reseal_index: int,
        startup_target_deferral_evidence: dict[str, object],
    ) -> None:
        self.initial_identity = initial_identity
        self.initial_run_id = initial_identity.run_id
        self.target_id = initial_identity.target_id
        self.completed_startup_reseal_count = completed_startup_reseal_count
        self.startup_reseal_index = startup_reseal_index
        self._startup_target_deferral_evidence = deepcopy(startup_target_deferral_evidence)
        # Keep the stopped-child lineage even when the parent persists only
        # the normal candidate observation deferral contract.
        self.observation_error = CandidateObservationUnavailableError(
            candidate_uid=observation_error.candidate_uid,
            observation_attempt_index=observation_error.observation_attempt_index,
            reason=observation_error.reason,
            process_evidence={
                **observation_error.process_evidence,
                "candidate_startup_target_deferral": self.startup_target_deferral_evidence,
            },
            status_evidence=observation_error.status_evidence,
        )
        super().__init__(
            f"candidate startup target unavailable before motion for {self.target_id}",
            phase="same_routine_target_admission",
            rejected_child=rejected_child,
        )

    @property
    def startup_target_deferral_evidence(self) -> dict[str, object]:
        return deepcopy(self._startup_target_deferral_evidence)

    def to_failure_fields(self) -> dict[str, object]:
        return {
            **super().to_failure_fields(),
            "candidate_uid": self.target_id,
            "candidate_observation_reason": self.observation_error.reason,
            "observer_process_evidence": self.observation_error.process_evidence,
            "observer_status_evidence": self.observation_error.status_evidence,
            "candidate_startup_target_deferral": self.startup_target_deferral_evidence,
        }


__all__ = [
    "CandidateStartupRecoveryError",
    "CandidateStartupTargetUnavailableError",
    "RejectedChildFailure",
    "issued_motion_permit_evidence",
    "issued_motion_permit_kinds",
]
