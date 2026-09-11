"""Adapt an exact candidate outcome to the atomic permit disposition effect."""

from __future__ import annotations

from pathlib import Path

from scripts.aufgabe04.navigation.execution.startup_reseal_permit_retirement import (
    retire_odom_startup_rejected_permit,
)
from scripts.aufgabe04.real_robot.candidate.recovery_failure import (
    issued_motion_permit_evidence,
)
from scripts.aufgabe04.real_robot.candidate.startup_recovery import CandidateRoutineIdentity
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome


def retire_candidate_startup_permit(
    outcome: MotionLegOutcome,
    identity: CandidateRoutineIdentity,
    reseal_index: int,
    attempt_root: Path,
) -> Path:
    """Retire the old authority before sampling or planning a replacement.

    The execution helper independently validates the terminal semantic evidence
    and exact permit identity. A dry rejection can prove that no permit existed;
    an execution rejection must retire its actual issued, unclaimed permit.
    """

    if outcome.run_id != identity.run_id or outcome.motion_published is not False:
        raise ValueError("startup permit disposition requires the exact no-motion child")
    issued = issued_motion_permit_evidence(outcome)
    if len(issued) > 1:
        raise ValueError("startup rejection cannot carry multiple motion permits")
    permit_kind = None
    permit_path = None
    digest = ""
    if issued:
        kind, binding = next(iter(issued.items()))
        permit_kind = {
            "routine_mission_leg": "mission_leg",
            "startup_reseal": "startup_reseal",
            "runtime_localization": "runtime_localization",
        }[kind]
        if binding["path"] is None or not binding["sha256"]:
            raise ValueError("startup rejection permit identity is incomplete")
        permit_path = Path(binding["path"])
        digest = binding["sha256"]
    return retire_odom_startup_rejected_permit(
        permit_path=permit_path,
        permit_kind=permit_kind,
        expected_permit_sha256=digest,
        rejected_semantic_log_path=Path(outcome.semantic_log_path),
        session_id=identity.session_id,
        rejected_run_id=identity.run_id,
        mission_leg_kind=identity.routine_kind,
        mission_leg_index=identity.routine_index,
        target_id=identity.target_id,
        reseal_index=reseal_index,
        disposition_path=(
            attempt_root / "rejected_permit_disposition.json"
            if permit_path is None else None
        ),
    )
