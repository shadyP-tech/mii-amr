"""Phase handoffs and cumulative bounds for one candidate motion routine.

Each recovery owner executes and validates its own direct child. A handoff
returns an already stopped outcome to the other owner; it never reruns that
child or changes the permit under which it executed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Literal, TYPE_CHECKING, TypeVar

from scripts.aufgabe04.real_robot.candidate.recovery_failure import (
    issued_motion_permit_evidence,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome

if TYPE_CHECKING:
    from .startup_recovery import CandidateStartupRecoveryConfig, CandidateStartupRecoveryEffects
    from .runtime_recovery import CandidateRuntimeRecoveryConfig, CandidateRuntimeRecoveryEffects

RequestT = TypeVar("RequestT")

@dataclass(frozen=True)
class CandidateRecoveryHandoff:
    outcome: MotionLegOutcome
    next_owner: Literal["startup", "runtime"]


def validate_child_outcome(outcome: object, *, expected_run_id: str) -> None:
    if not isinstance(outcome, MotionLegOutcome):
        raise TypeError("motion callback must return MotionLegOutcome")
    if outcome.run_id != expected_run_id:
        raise ValueError("motion outcome run identity mismatch")
    if not isinstance(outcome.status, str) or not outcome.status.strip():
        raise TypeError("motion outcome status must be non-empty text")
    if not isinstance(outcome.stop_reason, str) or not isinstance(outcome.stop_details, Mapping):
        raise TypeError("motion outcome stop evidence is malformed")
    if type(outcome.motion_published) is not bool or type(outcome.returncode) is not int:
        raise TypeError("motion outcome requires boolean motion and integer returncode")


@dataclass
class CandidateRecoveryState:
    """One state survives every startup/runtime handoff in a routine."""

    startup_reseal_count: int = 0
    runtime_reseal_count: int = 0
    permits_by_run: dict[str, tuple[str, Path, str]] = field(default_factory=dict)

    def remember_permit(self, outcome: MotionLegOutcome, *, kind: str) -> None:
        evidence = issued_motion_permit_evidence(outcome)
        if tuple(evidence) != (kind,):
            raise ValueError(f"{kind} child must report exactly its own one-use permit")
        permit = evidence[kind]
        digest = permit["sha256"]
        if permit["path"] is None or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{kind} child permit evidence is incomplete")
        path = Path(permit["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{kind} child permit must be an existing regular file")
        entry = (kind, path.resolve(strict=True), digest)
        previous = self.permits_by_run.get(outcome.run_id)
        if previous is not None:
            if previous != entry:
                raise ValueError("recovery handoff changed its child permit evidence")
            return  # The same stopped child may be checked by both owners.
        if any(entry[1] == item[1] or entry[2] == item[2] for item in self.permits_by_run.values()):
            raise ValueError("candidate recovery reused one-use permit evidence")
        self.permits_by_run[outcome.run_id] = entry


def execute_candidate_motion_with_recovery(
    initial_request: RequestT, *,
    startup_config: CandidateStartupRecoveryConfig,
    startup_effects: CandidateStartupRecoveryEffects[RequestT],
    runtime_config: CandidateRuntimeRecoveryConfig,
    runtime_effects: CandidateRuntimeRecoveryEffects[RequestT],
) -> MotionLegOutcome:
    """Dispatch exact phase handoffs without resetting either recovery budget."""

    # Local imports keep the owners usable as standalone narrow coordinators.
    from .startup_recovery import execute_candidate_motion_with_startup_recovery
    from .runtime_recovery import execute_candidate_runtime_localization_recovery

    if startup_config.initial_identity != runtime_config.initial_identity:
        raise ValueError("candidate recovery owners must bind the same initial routine")
    if Path(startup_config.event_log_path) != Path(runtime_config.event_log_path):
        raise ValueError("candidate recovery owners must share the routine event log")
    state = CandidateRecoveryState()
    handoff = None
    owner = "startup"
    while True:
        if owner == "startup":
            active_config = startup_config
            if handoff is not None:
                active_config = replace(
                    startup_config,
                    initial_identity=replace(startup_config.initial_identity, run_id=handoff.outcome.run_id),
                )
            result = execute_candidate_motion_with_startup_recovery(
                initial_request, config=active_config, effects=startup_effects,
                recovery_state=state, resumed_handoff=handoff,
            )
        else:
            assert handoff is not None
            active_config = replace(
                runtime_config,
                initial_identity=replace(runtime_config.initial_identity, run_id=handoff.outcome.run_id),
            )
            result = execute_candidate_runtime_localization_recovery(
                handoff.outcome, config=active_config, effects=runtime_effects,
                recovery_state=state,
            )
        if isinstance(result, MotionLegOutcome):
            return result
        if (not isinstance(result, CandidateRecoveryHandoff)
                or result.next_owner not in {"startup", "runtime"}
                or result.next_owner == owner):
            raise RuntimeError("candidate recovery returned an invalid phase handoff")
        handoff, owner = result, result.next_owner


__all__ = ["execute_candidate_motion_with_recovery"]
