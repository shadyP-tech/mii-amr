"""One fresh opposite-route search after a validated post-motion rejection.

This is a replan handoff, not motion authority. The caller must acquire a fresh
stationary frame, reproject the original camera receipt, and disable further
startup/runtime reseals in the extra epoch. Ordinary no-motion retries must
never see its exhaustion as though the robot had not previously moved.
"""

from collections.abc import Callable, Mapping
from typing import TypeVar

from scripts.aufgabe04.real_robot.candidate.inspection_route_search import (
    CandidateInspectionRouteUnavailableError,
)
from scripts.aufgabe04.real_robot.candidate.runtime_recovery import (
    CandidateRuntimeRecoveryError,
    validate_runtime_route_rejection_context,
)
from scripts.aufgabe04.real_robot.candidate.startup_recovery import CandidateRoutineIdentity


Result = TypeVar("Result")


class OppositeRuntimeRouteRejected(RuntimeError):
    """Scope a validated runtime rejection to its exact opposite route call."""

    def __init__(self, error: CandidateRuntimeRecoveryError, *,
                 expected_identity: CandidateRoutineIdentity):
        validate_runtime_route_rejection_context(error, expected_identity=expected_identity)
        self.error = error
        self.expected_identity = expected_identity
        super().__init__(str(error))


def with_opposite_runtime_retry(
    *, attempt: Callable[[], Result], retry: Callable[[], Result], enabled: bool,
    event_sink: Callable[[Mapping[str, object]], None],
) -> Result:
    """Run at most one extra epoch, outside the pre-motion retry coordinator."""
    try:
        return attempt()
    except OppositeRuntimeRouteRejected as rejected:
        if not enabled:
            raise rejected.error from rejected
        try:
            context = validate_runtime_route_rejection_context(
                rejected.error, expected_identity=rejected.expected_identity)
        except (OSError, TypeError, ValueError):
            raise rejected.error from rejected
        evidence = {
            "maximum_post_motion_planning_epochs": 1,
            "original_run_id": context.original_identity.run_id,
            "stopped_run_id": context.attempt.rejected_outcome.run_id,
            "rejected_replacement_run_id": context.replacement_outcome.run_id,
            "runtime_reseal_attempts_consumed": context.attempt.reseal_index,
            "source_motion_published": True,
            "rejected_replacement_motion_published": False,
            "fresh_stationary_frame_required": True,
            "retained_source_orientation_unchanged": True,
            "maximum_additional_startup_reseals": 0,
            "maximum_additional_runtime_reseals": 0,
            "maximum_additional_checkpoints": 1,
            "route_limits_unchanged": True,
            "motion_authorized": False,
            "motion_continues_authorized": False,
        }
        event_sink({"event": "opposite_post_motion_route_search_requested", **evidence})
        try:
            result = retry()
        except CandidateInspectionRouteUnavailableError as exc:
            event_sink({"event": "opposite_post_motion_routes_exhausted", **evidence,
                        "reason": str(exc), "reason_code": exc.reason_code})
            # Keep this terminal error outside all pre-motion-only fallbacks.
            raise CandidateRuntimeRecoveryError(
                f"post-motion opposite routes exhausted: {exc}",
                phase="opposite_routes_exhausted",
                rejected_child=rejected.error.rejected_child,
            ) from exc
        event_sink({"event": "opposite_post_motion_route_search_completed", **evidence})
        return result
