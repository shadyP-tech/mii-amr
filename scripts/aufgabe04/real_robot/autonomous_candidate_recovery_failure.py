"""Pure failure evidence for autonomous candidate startup recovery.

This module translates an already-completed child motion outcome into
JSON-ready rejection evidence.  It never retries motion, grants authority,
connects to ROS, launches a process, or writes an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome


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
        return f"{prefix} {self.run_id}: {self.reported_reason}{policy_suffix}"

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


__all__ = [
    "CandidateStartupRecoveryError",
    "RejectedChildFailure",
    "issued_motion_permit_evidence",
    "issued_motion_permit_kinds",
]
